"""
Smart frame extraction with blur-aware sampling.
Extracts more candidate frames, then selects the sharpest ones while maintaining temporal order.
"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
NUM_FRAMES = 32          # Final number of frames to keep
CANDIDATE_MULT = 2.0     # Sample this many more candidates (e.g., 2x = 64 candidates for 32 frames)
IMAGE_SIZE = 224
BLUR_WEIGHT = 0.7        # Balance between sharpness (1.0) and uniform spacing (0.0)


def compute_blur_score(frame):
    """Compute Laplacian variance blur score. Higher = sharper."""
    if frame is None:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_sharpest_frames(frames_with_scores, num_to_keep):
    """
    Select sharpest frames while maintaining rough temporal order.
    Uses a greedy approach: divide into bins and pick sharpest from each.
    """
    n = len(frames_with_scores)
    if n <= num_to_keep:
        return [f for f, s in frames_with_scores]

    # Divide into bins and pick sharpest from each
    bin_size = n / num_to_keep
    selected = []

    for i in range(num_to_keep):
        start_idx = int(i * bin_size)
        end_idx = int((i + 1) * bin_size)
        bin_frames = frames_with_scores[start_idx:end_idx]

        if bin_frames:
            # Pick the sharpest frame in this bin
            best_frame, best_score = max(bin_frames, key=lambda x: x[1])
            selected.append(best_frame)

    return selected


def extract_action_frames(args):
    """Extract frames for a single action with blur-aware sampling."""
    idx, row, video_base_dir, output_dir = args

    participant = row['participant_id']
    video_id = row['video_id']
    start_frame = int(row['start_frame'])
    stop_frame = int(row['stop_frame'])

    # Output directory for this action
    action_id = f"{participant}_{video_id}_{idx:06d}"
    action_dir = output_dir / action_id

    # Skip if already extracted
    if action_dir.exists() and len(list(action_dir.glob("frame_*.jpg"))) == NUM_FRAMES:
        return {"action_id": action_id, "status": "skipped"}

    # Find video file
    video_path = video_base_dir / f"{video_id}.MP4"
    if not video_path.exists():
        video_path = video_base_dir / participant / f"{video_id}.MP4"
    if not video_path.exists():
        return {"action_id": action_id, "status": "failed", "error": "Video not found"}

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"action_id": action_id, "status": "failed", "error": "Could not open video"}

        # Sample more candidate frames
        num_candidates = int(NUM_FRAMES * CANDIDATE_MULT)
        candidate_indices = np.linspace(start_frame, stop_frame, num_candidates, dtype=int)

        # Read candidate frames and compute blur scores
        frames_with_scores = []
        last_frame = None

        for frame_idx in candidate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()

            if ret:
                # Compute blur score on original resolution (more accurate)
                blur_score = compute_blur_score(frame)
                # Resize for storage
                frame_resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
                frames_with_scores.append((frame_resized, blur_score))
                last_frame = frame_resized
            elif last_frame is not None:
                frames_with_scores.append((last_frame.copy(), 0.0))  # Low score for duplicates

        cap.release()

        if not frames_with_scores:
            return {"action_id": action_id, "status": "failed", "error": "No frames read"}

        # Select sharpest frames while maintaining temporal order
        selected_frames = select_sharpest_frames(frames_with_scores, NUM_FRAMES)

        # Pad if needed
        while len(selected_frames) < NUM_FRAMES:
            selected_frames.append(selected_frames[-1] if selected_frames else
                                   np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))

        # Save frames
        action_dir.mkdir(parents=True, exist_ok=True)
        avg_blur = np.mean([s for f, s in frames_with_scores]) if frames_with_scores else 0

        for i, frame in enumerate(selected_frames):
            frame_path = action_dir / f"frame_{i:02d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        return {
            "action_id": action_id,
            "status": "success",
            "avg_blur_score": avg_blur,
            "frames_saved": len(selected_frames)
        }

    except Exception as e:
        return {"action_id": action_id, "status": "failed", "error": str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Annotations CSV")
    parser.add_argument("--video_dir", type=str, required=True, help="Video directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=32)
    args = parser.parse_args()

    global NUM_FRAMES
    NUM_FRAMES = args.num_frames

    csv_path = Path(args.csv)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)

    print(f"Annotations: {csv_path}")
    print(f"Video directory: {video_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Frames per action: {NUM_FRAMES}")
    print(f"Candidate multiplier: {CANDIDATE_MULT}x")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    annotations = pd.read_csv(csv_path)
    print(f"Total actions: {len(annotations)}")

    # Prepare tasks
    tasks = [
        (idx, annotations.iloc[idx], video_dir, output_dir)
        for idx in range(len(annotations))
    ]

    # Extract frames
    results = {"success": 0, "failed": 0, "skipped": 0}
    failed_actions = []
    blur_scores = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_action_frames, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Extracting"):
            result = future.result()
            results[result["status"]] += 1
            if result["status"] == "failed":
                failed_actions.append(result)
            elif result["status"] == "success" and "avg_blur_score" in result:
                blur_scores.append(result["avg_blur_score"])

    print(f"\n{'='*50}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*50}")
    print(f"  Success: {results['success']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  Failed:  {results['failed']}")

    if blur_scores:
        print(f"\nBlur Statistics:")
        print(f"  Mean: {np.mean(blur_scores):.2f}")
        print(f"  Min:  {np.min(blur_scores):.2f}")
        print(f"  Max:  {np.max(blur_scores):.2f}")

    # Save summary
    summary = {
        "config": {
            "num_frames": NUM_FRAMES,
            "candidate_multiplier": CANDIDATE_MULT,
            "image_size": IMAGE_SIZE
        },
        "results": {
            "success": results['success'],
            "skipped": results['skipped'],
            "failed": results['failed']
        },
        "blur_stats": {
            "mean": float(np.mean(blur_scores)) if blur_scores else 0,
            "min": float(np.min(blur_scores)) if blur_scores else 0,
            "max": float(np.max(blur_scores)) if blur_scores else 0
        },
        "failed_actions": failed_actions[:100]
    }

    with open(output_dir / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {output_dir / 'extraction_summary.json'}")


if __name__ == "__main__":
    main()
