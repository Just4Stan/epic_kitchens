"""
Extract action frames from pre-extracted rgb_frames with blur-aware sampling.
Uses existing rgb_frames on scratch instead of videos.

Structure expected:
  /scratch/.../EPIC-KITCHENS/EPIC-KITCHENS/P01/rgb_frames/P01_01/frame_0000000001.jpg
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
CANDIDATE_MULT = 2.0     # Sample this many more candidates
IMAGE_SIZE = 224


def compute_blur_score(frame):
    """Compute Laplacian variance blur score. Higher = sharper."""
    if frame is None:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_sharpest_frames(frames_with_scores, num_to_keep):
    """Select sharpest frames while maintaining temporal order."""
    n = len(frames_with_scores)
    if n <= num_to_keep:
        return [f for f, s in frames_with_scores]

    bin_size = n / num_to_keep
    selected = []

    for i in range(num_to_keep):
        start_idx = int(i * bin_size)
        end_idx = int((i + 1) * bin_size)
        bin_frames = frames_with_scores[start_idx:end_idx]

        if bin_frames:
            best_frame, best_score = max(bin_frames, key=lambda x: x[1])
            selected.append(best_frame)

    return selected


def extract_action_frames(args):
    """Extract frames for a single action from rgb_frames directory."""
    idx, row, rgb_frames_root, output_dir, num_frames = args

    participant = row['participant_id']
    video_id = row['video_id']
    start_frame = int(row['start_frame'])
    stop_frame = int(row['stop_frame'])

    # Output directory for this action
    action_id = f"{participant}_{video_id}_{idx:06d}"
    action_dir = output_dir / action_id

    # Skip if already extracted
    if action_dir.exists() and len(list(action_dir.glob("frame_*.jpg"))) == num_frames:
        return {"action_id": action_id, "status": "skipped"}

    # Find rgb_frames directory for this video
    video_frames_dir = rgb_frames_root / participant / "rgb_frames" / video_id
    if not video_frames_dir.exists():
        # Try alternative structure
        video_frames_dir = rgb_frames_root / participant / video_id
        if not video_frames_dir.exists():
            return {"action_id": action_id, "status": "failed", "error": f"Frames dir not found: {video_frames_dir}"}

    # Get all frame files
    frame_files = sorted(video_frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        frame_files = sorted(video_frames_dir.glob("*.jpg"))
    if not frame_files:
        return {"action_id": action_id, "status": "failed", "error": "No frame files found"}

    # Frame files are named like frame_0000000001.jpg
    # We need to map frame numbers to file indices
    def get_frame_number(f):
        name = f.stem
        if name.startswith("frame_"):
            return int(name.split("_")[1])
        return int(name)

    frame_numbers = {get_frame_number(f): f for f in frame_files}
    available_frames = sorted(frame_numbers.keys())

    if not available_frames:
        return {"action_id": action_id, "status": "failed", "error": "Could not parse frame numbers"}

    # Clamp to available range
    start_frame = max(start_frame, available_frames[0])
    stop_frame = min(stop_frame, available_frames[-1])

    if start_frame >= stop_frame:
        return {"action_id": action_id, "status": "failed", "error": "Invalid frame range"}

    try:
        # Sample candidate frame indices
        num_candidates = int(num_frames * CANDIDATE_MULT)
        candidate_indices = np.linspace(start_frame, stop_frame, num_candidates, dtype=int)

        # Read candidate frames and compute blur scores
        frames_with_scores = []
        last_frame = None

        for frame_idx in candidate_indices:
            # Find closest available frame
            closest_idx = min(available_frames, key=lambda x: abs(x - frame_idx))
            frame_path = frame_numbers.get(closest_idx)

            if frame_path and frame_path.exists():
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    blur_score = compute_blur_score(frame)
                    frame_resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
                    frames_with_scores.append((frame_resized, blur_score))
                    last_frame = frame_resized
                elif last_frame is not None:
                    frames_with_scores.append((last_frame.copy(), 0.0))
            elif last_frame is not None:
                frames_with_scores.append((last_frame.copy(), 0.0))

        if not frames_with_scores:
            return {"action_id": action_id, "status": "failed", "error": "No frames read"}

        # Select sharpest frames
        selected_frames = select_sharpest_frames(frames_with_scores, num_frames)

        # Pad if needed
        while len(selected_frames) < num_frames:
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
    parser.add_argument("--rgb_frames_root", type=str, required=True,
                       help="Root of rgb_frames (e.g., /scratch/.../EPIC-KITCHENS/EPIC-KITCHENS)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=32)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    rgb_frames_root = Path(args.rgb_frames_root)
    output_dir = Path(args.output_dir)
    num_frames = args.num_frames

    print(f"Annotations: {csv_path}")
    print(f"RGB frames root: {rgb_frames_root}")
    print(f"Output directory: {output_dir}")
    print(f"Frames per action: {num_frames}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    annotations = pd.read_csv(csv_path)
    print(f"Total actions: {len(annotations)}")

    # Prepare tasks
    tasks = [
        (idx, annotations.iloc[idx], rgb_frames_root, output_dir, num_frames)
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
            "num_frames": num_frames,
            "candidate_multiplier": CANDIDATE_MULT,
            "image_size": IMAGE_SIZE
        },
        "results": results,
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
