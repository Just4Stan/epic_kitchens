"""
Create demo video showing action recognition on EPIC-KITCHENS footage.
Shows model predictions alongside ground truth annotations.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd
from torchvision import transforms
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models import ActionModel
from config import Config as cfg


def create_demo(video_path, output_path, checkpoint_path, annotations_csv, video_id):
    """Create annotated demo video."""

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading model: {Path(checkpoint_path).name}")
    model = ActionModel(
        num_verb_classes=cfg.NUM_VERB_CLASSES,
        num_noun_classes=cfg.NUM_NOUN_CLASSES,
        backbone=cfg.BACKBONE,
        temporal_model=cfg.TEMPORAL_MODEL,
        dropout=cfg.DROPOUT,
        num_frames=cfg.NUM_FRAMES
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()

    # Class mappings
    epic_dir = Path(__file__).parent.parent
    verb_csv = pd.read_csv(epic_dir / cfg.VERB_CLASSES_CSV)
    noun_csv = pd.read_csv(epic_dir / cfg.NOUN_CLASSES_CSV)
    verb_map = dict(zip(verb_csv['id'], verb_csv['key']))
    noun_map = dict(zip(noun_csv['id'], noun_csv['key']))

    # Load annotations for this video
    all_annotations = pd.read_csv(annotations_csv)
    video_annotations = all_annotations[all_annotations['video_id'] == video_id].copy()
    print(f"Found {len(video_annotations)} annotated actions for {video_id}")

    # Transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    # Side panel width
    panel_width = 350
    output_width = width + panel_width

    # Output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, height))

    # Frame buffer for temporal model
    frame_buffer = []
    buffer_size = 60  # 2 seconds at 30fps

    frame_idx = 0
    current_pred = None
    pred_every = 5  # Predict every N frames

    print(f"Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess and buffer
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = transform(rgb)
        frame_buffer.append(processed)
        if len(frame_buffer) > buffer_size:
            frame_buffer.pop(0)

        # Run prediction
        if frame_idx % pred_every == 0 and len(frame_buffer) >= 16:
            with torch.no_grad():
                # Sample 16 frames from buffer
                indices = np.linspace(0, len(frame_buffer) - 1, 16, dtype=int)
                frames = [frame_buffer[i] for i in indices]
                x = torch.stack(frames).unsqueeze(0).to(device)

                verb_logits, noun_logits = model(x)
                verb_probs = F.softmax(verb_logits, dim=1)[0]
                noun_probs = F.softmax(noun_logits, dim=1)[0]

                v_conf, v_idx = verb_probs.max(0)
                n_conf, n_idx = noun_probs.max(0)

                current_pred = {
                    'verb': verb_map.get(v_idx.item(), "?"),
                    'noun': noun_map.get(n_idx.item(), "?"),
                    'verb_conf': v_conf.item(),
                    'noun_conf': n_conf.item(),
                }

        # Find ground truth for current frame
        gt_action = None
        for _, row in video_annotations.iterrows():
            if row['start_frame'] <= frame_idx <= row['stop_frame']:
                gt_action = row['narration']
                break

        # Create output frame with side panel
        display = np.zeros((height, output_width, 3), dtype=np.uint8)

        # Put original video on left
        display[:, :width] = frame

        # Side panel (dark background)
        panel_x = width
        cv2.rectangle(display, (panel_x, 0), (output_width, height), (30, 30, 30), -1)

        # Title
        cv2.putText(display, "Action Recognition", (panel_x + 15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Demo: {video_id}", (panel_x + 15, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Divider
        cv2.line(display, (panel_x + 15, 75), (output_width - 15, 75), (80, 80, 80), 1)

        # Model prediction section
        cv2.putText(display, "MODEL PREDICTION", (panel_x + 15, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

        if current_pred:
            conf = (current_pred['verb_conf'] + current_pred['noun_conf']) / 2
            color = (0, 255, 0) if conf > 0.5 else (0, 165, 255) if conf > 0.3 else (0, 100, 255)

            cv2.putText(display, current_pred['verb'], (panel_x + 15, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(display, current_pred['noun'], (panel_x + 15, 175),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Confidence bar
            bar_w = int(conf * (panel_width - 80))
            cv2.rectangle(display, (panel_x + 15, 195), (panel_x + 15 + bar_w, 215), color, -1)
            cv2.rectangle(display, (panel_x + 15, 195), (panel_x + panel_width - 60, 215), (80, 80, 80), 2)
            cv2.putText(display, f"{conf*100:.0f}%", (panel_x + panel_width - 55, 212),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Divider
        cv2.line(display, (panel_x + 15, 235), (output_width - 15, 235), (80, 80, 80), 1)

        # Ground truth section
        cv2.putText(display, "GROUND TRUTH", (panel_x + 15, 265),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

        if gt_action:
            # Split narration into words
            words = gt_action.split()
            cv2.putText(display, words[0] if words else "", (panel_x + 15, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 100), 2)
            cv2.putText(display, ' '.join(words[1:]) if len(words) > 1 else "", (panel_x + 15, 335),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 100), 2)

            # Match indicator
            if current_pred and current_pred['verb'].lower() in gt_action.lower():
                cv2.putText(display, "MATCH!", (panel_x + 15, 370),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display, "(no action)", (panel_x + 15, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

        # Frame counter at bottom
        cv2.putText(display, f"Frame {frame_idx}/{total_frames}", (panel_x + 15, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # Write frame
        out.write(display)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"\nSaved demo video to: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='P03_21')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    epic_dir = Path(__file__).parent.parent

    # Find video
    video_id = args.video
    participant = video_id.split('_')[0]
    video_path = epic_dir / f"EPIC-KITCHENS/videos_640x360/{participant}/{video_id}.MP4"

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return

    # Checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = epic_dir / "outputs_exp15_h100/checkpoints/best_model.pth"

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return

    # Output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = epic_dir / f"outputs/demo_{video_id}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Annotations
    annotations_csv = epic_dir / cfg.VAL_CSV

    create_demo(video_path, output_path, checkpoint_path, annotations_csv, video_id)


if __name__ == '__main__':
    main()
