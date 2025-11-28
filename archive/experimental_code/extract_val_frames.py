"""
Extract validation frames from videos for faster validation.
"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Configuration
NUM_FRAMES = 16
IMAGE_SIZE = 224


def extract_action_frames(args):
    """Extract frames for a single action."""
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

    # Find video file - try flat structure first, then participant folder
    video_path = video_base_dir / f"{video_id}.MP4"
    if not video_path.exists():
        video_path = video_base_dir / participant / f"{video_id}.MP4"
    if not video_path.exists():
        return {"action_id": action_id, "status": "failed", "error": "Video not found"}

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"action_id": action_id, "status": "failed", "error": "Could not open video"}

        # Sample frame indices
        indices = np.linspace(start_frame, stop_frame, NUM_FRAMES, dtype=int)

        action_dir.mkdir(parents=True, exist_ok=True)

        frames_saved = 0
        last_frame = None

        for i, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()

            if ret:
                # Resize to save space
                frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
                last_frame = frame
            elif last_frame is not None:
                frame = last_frame
            else:
                frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

            frame_path = action_dir / f"frame_{i:02d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frames_saved += 1

        cap.release()

        if frames_saved == NUM_FRAMES:
            return {"action_id": action_id, "status": "success"}
        else:
            return {"action_id": action_id, "status": "failed", "error": f"Only saved {frames_saved} frames"}

    except Exception as e:
        return {"action_id": action_id, "status": "failed", "error": str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    val_csv = Path(args.val_csv)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)

    print(f"Validation CSV: {val_csv}")
    print(f"Video directory: {video_dir}")
    print(f"Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    annotations = pd.read_csv(val_csv)
    print(f"Total validation actions: {len(annotations)}")

    # Prepare arguments for parallel processing
    tasks = [
        (idx, annotations.iloc[idx], video_dir, output_dir)
        for idx in range(len(annotations))
    ]

    # Extract frames in parallel
    results = {"success": 0, "failed": 0, "skipped": 0}
    failed_actions = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_action_frames, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Extracting"):
            result = future.result()
            results[result["status"]] += 1
            if result["status"] == "failed":
                failed_actions.append(result)

    print(f"\n{'='*50}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*50}")
    print(f"  Success: {results['success']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  Failed:  {results['failed']}")
    print(f"{'='*50}")

    # Save summary
    summary = {
        "total_success": results['success'],
        "total_skipped": results['skipped'],
        "total_failed": results['failed'],
        "failed_actions": failed_actions[:100]  # Only save first 100 failures
    }

    with open(output_dir / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {output_dir / 'extraction_summary.json'}")


if __name__ == "__main__":
    main()
