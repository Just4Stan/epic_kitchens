"""
Test model on custom POV video
Upload your own kitchen footage and see what the model predicts
"""

import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
import pandas as pd
from torchvision import transforms
import sys

sys.path.append(str(Path(__file__).parent.parent))

from common.config import Config
from phase1.model import get_model


def load_verb_noun_mappings(config):
    """Load verb and noun class names."""
    verb_csv = pd.read_csv(config.VERB_CLASSES_CSV)
    noun_csv = pd.read_csv(config.NOUN_CLASSES_CSV)

    verb_map = dict(zip(verb_csv['id'], verb_csv['key']))
    noun_map = dict(zip(noun_csv['id'], noun_csv['key']))

    return verb_map, noun_map


def extract_frames_from_video(video_path, num_frames=8, start_time=0, duration=None):
    """
    Extract frames from a video.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        start_time: Start time in seconds (default: 0)
        duration: Duration in seconds (default: entire video)

    Returns:
        frames: List of numpy arrays (H, W, 3)
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame range
    start_frame = int(start_time * fps)
    if duration is None:
        end_frame = total_frames
    else:
        end_frame = min(int((start_time + duration) * fps), total_frames)

    print(f"Video info:")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Extracting from frame {start_frame} to {end_frame}")

    # Sample frames uniformly
    frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)

    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"Warning: Could not read frame {frame_idx}")
            continue

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    print(f"Extracted {len(frames)} frames")
    return frames


def preprocess_frames(frames, image_size=224):
    """Preprocess frames for model input."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    processed = []
    for frame in frames:
        processed.append(transform(frame))

    # Stack into (num_frames, 3, H, W)
    return torch.stack(processed, dim=0)


def test_custom_video(video_path, checkpoint_path, config, start_time=0, duration=None, top_k=5):
    """
    Test model on custom video.

    Args:
        video_path: Path to your video file
        checkpoint_path: Path to model checkpoint
        config: Config object
        start_time: Start time in seconds
        duration: Duration to analyze in seconds (None = entire video)
        top_k: Number of top predictions to show
    """
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: {device} (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = get_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model from epoch {checkpoint['epoch']}")

    # Load class mappings
    print("\nLoading class names...")
    verb_map, noun_map = load_verb_noun_mappings(config)

    # Extract frames from video
    print(f"\nExtracting frames from {video_path}...")
    frames = extract_frames_from_video(video_path, config.NUM_FRAMES, start_time, duration)

    if len(frames) == 0:
        print("ERROR: No frames extracted!")
        return

    # Preprocess
    print("Preprocessing frames...")
    frames_tensor = preprocess_frames(frames, config.IMAGE_SIZE)

    # Add batch dimension: (1, num_frames, 3, H, W)
    frames_tensor = frames_tensor.unsqueeze(0).to(device)

    print(f"Input shape: {frames_tensor.shape}")

    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        verb_logits, noun_logits = model(frames_tensor)

        # Get probabilities
        verb_probs = torch.softmax(verb_logits, dim=1)
        noun_probs = torch.softmax(noun_logits, dim=1)

        # Get top-k predictions
        verb_top_probs, verb_top_indices = verb_probs[0].topk(top_k)
        noun_top_probs, noun_top_indices = noun_probs[0].topk(top_k)

    # Print results
    print("\n" + "="*60)
    print("PREDICTIONS FOR YOUR VIDEO")
    print("="*60)

    print(f"\nTop {top_k} VERB predictions:")
    for i, (idx, prob) in enumerate(zip(verb_top_indices, verb_top_probs), 1):
        verb_name = verb_map.get(idx.item(), f"unknown_{idx.item()}")
        print(f"  {i}. {verb_name:30s} {prob.item()*100:6.2f}%")

    print(f"\nTop {top_k} NOUN predictions:")
    for i, (idx, prob) in enumerate(zip(noun_top_indices, noun_top_probs), 1):
        noun_name = noun_map.get(idx.item(), f"unknown_{idx.item()}")
        print(f"  {i}. {noun_name:30s} {prob.item()*100:6.2f}%")

    # Most likely action
    top_verb = verb_map.get(verb_top_indices[0].item(), "unknown")
    top_noun = noun_map.get(noun_top_indices[0].item(), "unknown")

    print("\n" + "="*60)
    print(f"MOST LIKELY ACTION: {top_verb} {top_noun}")
    print(f"Confidence: Verb {verb_top_probs[0].item()*100:.1f}%, Noun {noun_top_probs[0].item()*100:.1f}%")
    print("="*60)

    # Interpretation
    print("\nINTERPRETATION:")
    if verb_top_probs[0] < 0.3 or noun_top_probs[0] < 0.3:
        print("⚠️  Low confidence predictions!")
        print("   The model is uncertain - might not recognize this video.")
        print("   This could indicate overfitting to EPIC-KITCHENS dataset.")
    elif verb_top_probs[0] > 0.7 and noun_top_probs[0] > 0.7:
        print("✓ High confidence predictions!")
        print("  The model seems confident about the action.")
        print("  Check if the prediction makes sense for what you filmed.")
    else:
        print("~ Moderate confidence.")
        print("  The model has some idea but isn't very sure.")

    print("\nNote: Since the model was trained only on EPIC-KITCHENS data,")
    print("it may struggle with different kitchens, lighting, or camera angles.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model on your custom video')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to your video file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--start', type=float, default=0,
                        help='Start time in seconds (default: 0)')
    parser.add_argument('--duration', type=float, default=None,
                        help='Duration in seconds (default: entire video)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to show (default: 5)')

    args = parser.parse_args()

    config = Config()

    test_custom_video(
        args.video,
        args.checkpoint,
        config,
        start_time=args.start,
        duration=args.duration,
        top_k=args.top_k
    )
