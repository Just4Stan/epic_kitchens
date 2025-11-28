#!/usr/bin/env python3
"""
Analyze blur distribution in EPIC-KITCHENS frames.
Computes Laplacian variance for sampled frames and saves examples.
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

# Paths
FRAMES_ROOT = "/scratch/leuven/380/vsc38064/EPIC-KITCHENS/EPIC-KITCHENS"
OUTPUT_DIR = "/scratch/leuven/380/vsc38064/blur_analysis"
SAMPLES_PER_VIDEO = 10  # Sample frames per video
MAX_VIDEOS = 100  # Limit videos to analyze

def compute_blur_score(image_path):
    """Compute Laplacian variance blur score. Higher = sharper."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(score)
    except Exception as e:
        return None

def analyze_video(video_path):
    """Analyze blur scores for sampled frames from one video."""
    frames = sorted(Path(video_path).glob("*.jpg"))
    if len(frames) == 0:
        return None

    # Sample frames evenly
    indices = np.linspace(0, len(frames)-1, min(SAMPLES_PER_VIDEO, len(frames)), dtype=int)
    sampled_frames = [frames[i] for i in indices]

    results = []
    for frame_path in sampled_frames:
        score = compute_blur_score(frame_path)
        if score is not None:
            results.append({
                'path': str(frame_path),
                'score': score,
                'video': video_path.name
            })
    return results

def main():
    print("EPIC-KITCHENS Blur Analysis")
    print("=" * 50)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/sharp", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/blurry", exist_ok=True)

    # Find all video directories
    all_videos = []
    for participant_dir in sorted(Path(FRAMES_ROOT).iterdir()):
        if participant_dir.is_dir() and participant_dir.name.startswith("P"):
            rgb_dir = participant_dir / "rgb_frames"
            if rgb_dir.exists():
                for video_dir in sorted(rgb_dir.iterdir()):
                    if video_dir.is_dir():
                        all_videos.append(video_dir)

    print(f"Found {len(all_videos)} videos")

    # Sample videos
    if len(all_videos) > MAX_VIDEOS:
        random.seed(42)
        videos_to_analyze = random.sample(all_videos, MAX_VIDEOS)
    else:
        videos_to_analyze = all_videos

    print(f"Analyzing {len(videos_to_analyze)} videos...")

    # Analyze in parallel
    all_results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(analyze_video, v): v for v in videos_to_analyze}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                all_results.extend(result)
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(videos_to_analyze)} videos...")

    print(f"\nAnalyzed {len(all_results)} frames")

    # Compute statistics
    scores = [r['score'] for r in all_results]
    scores_arr = np.array(scores)

    stats = {
        'total_frames': len(scores),
        'mean': float(np.mean(scores_arr)),
        'std': float(np.std(scores_arr)),
        'min': float(np.min(scores_arr)),
        'max': float(np.max(scores_arr)),
        'percentiles': {
            '5': float(np.percentile(scores_arr, 5)),
            '25': float(np.percentile(scores_arr, 25)),
            '50': float(np.percentile(scores_arr, 50)),
            '75': float(np.percentile(scores_arr, 75)),
            '95': float(np.percentile(scores_arr, 95))
        }
    }

    print("\n" + "=" * 50)
    print("BLUR SCORE STATISTICS (higher = sharper)")
    print("=" * 50)
    print(f"Total frames analyzed: {stats['total_frames']}")
    print(f"Mean:   {stats['mean']:.2f}")
    print(f"Std:    {stats['std']:.2f}")
    print(f"Min:    {stats['min']:.2f}")
    print(f"Max:    {stats['max']:.2f}")
    print(f"\nPercentiles:")
    print(f"  5th:  {stats['percentiles']['5']:.2f} (very blurry)")
    print(f"  25th: {stats['percentiles']['25']:.2f}")
    print(f"  50th: {stats['percentiles']['50']:.2f} (median)")
    print(f"  75th: {stats['percentiles']['75']:.2f}")
    print(f"  95th: {stats['percentiles']['95']:.2f} (very sharp)")

    # Suggested threshold
    suggested_threshold = stats['percentiles']['25']
    below_threshold = sum(1 for s in scores if s < suggested_threshold)
    print(f"\nSuggested blur threshold: {suggested_threshold:.2f}")
    print(f"Frames below threshold: {below_threshold} ({100*below_threshold/len(scores):.1f}%)")

    # Sort by score
    sorted_results = sorted(all_results, key=lambda x: x['score'])

    # Copy 10 blurriest and 10 sharpest
    print("\nCopying example frames...")

    blurry_examples = sorted_results[:10]
    sharp_examples = sorted_results[-10:]

    for i, item in enumerate(blurry_examples):
        src = item['path']
        dst = f"{OUTPUT_DIR}/blurry/{i+1:02d}_score{item['score']:.0f}_{Path(src).name}"
        cv2.imwrite(dst, cv2.imread(src))
        print(f"  Blurry {i+1}: score={item['score']:.2f}")

    for i, item in enumerate(sharp_examples):
        src = item['path']
        dst = f"{OUTPUT_DIR}/sharp/{i+1:02d}_score{item['score']:.0f}_{Path(src).name}"
        cv2.imwrite(dst, cv2.imread(src))
        print(f"  Sharp {i+1}: score={item['score']:.2f}")

    # Save full results
    with open(f"{OUTPUT_DIR}/blur_stats.json", 'w') as f:
        json.dump({
            'statistics': stats,
            'blurry_examples': blurry_examples,
            'sharp_examples': sharp_examples
        }, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("Done!")

if __name__ == "__main__":
    main()
