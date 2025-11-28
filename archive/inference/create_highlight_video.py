"""
Create highlight video showing CORRECT action predictions only.
Selects best clips from multiple participants with context.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd
from torchvision import transforms
from pathlib import Path
import sys
import random

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models import ActionModel
from config import Config as cfg


def create_highlights(participants, output_path, checkpoint_path, annotations_csv, max_clips=20, context_frames=30):
    """Create highlight video from multiple participants."""

    epic_dir = Path(__file__).parent.parent

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading model...")
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
    verb_csv = pd.read_csv(epic_dir / cfg.VERB_CLASSES_CSV)
    noun_csv = pd.read_csv(epic_dir / cfg.NOUN_CLASSES_CSV)
    verb_map = dict(zip(verb_csv['id'], verb_csv['key']))
    noun_map = dict(zip(noun_csv['id'], noun_csv['key']))

    # Load all annotations
    all_annotations = pd.read_csv(annotations_csv)

    # Transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # First pass: collect all matching clips with confidence
    all_clips = []

    for participant in participants:
        video_dir = epic_dir / f"EPIC-KITCHENS/videos_640x360/{participant}"
        if not video_dir.exists():
            continue

        video_paths = sorted(video_dir.glob("*.MP4"))
        print(f"\nScanning {participant} ({len(video_paths)} videos)...")

        for video_path in video_paths:
            video_id = video_path.stem
            video_annotations = all_annotations[all_annotations['video_id'] == video_id]

            if len(video_annotations) == 0:
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for _, row in video_annotations.iterrows():
                start_frame = max(0, int(row['start_frame']) - context_frames)
                stop_frame = min(total_frames - 1, int(row['stop_frame']) + context_frames)
                action_start = int(row['start_frame'])
                action_stop = int(row['stop_frame'])
                gt_verb = row['verb']
                gt_narration = row['narration']

                # Read frames for prediction
                cap.set(cv2.CAP_PROP_POS_FRAMES, action_start)
                frame_buffer = []

                for _ in range(action_stop - action_start + 1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed = transform(rgb)
                    frame_buffer.append(processed)

                if len(frame_buffer) < 16:
                    continue

                # Run prediction
                with torch.no_grad():
                    indices = np.linspace(0, len(frame_buffer) - 1, 16, dtype=int)
                    frames = [frame_buffer[i] for i in indices]
                    x = torch.stack(frames).unsqueeze(0).to(device)

                    verb_logits, noun_logits = model(x)
                    verb_probs = F.softmax(verb_logits, dim=1)[0]
                    noun_probs = F.softmax(noun_logits, dim=1)[0]

                    v_conf, v_idx = verb_probs.max(0)
                    n_conf, n_idx = noun_probs.max(0)

                    pred_verb = verb_map.get(v_idx.item(), "?")
                    pred_noun = noun_map.get(n_idx.item(), "?")
                    conf = (v_conf.item() + n_conf.item()) / 2

                # Check if verb matches
                if pred_verb.lower() == gt_verb.lower() and conf > 0.4:
                    all_clips.append({
                        'video_path': video_path,
                        'video_id': video_id,
                        'participant': participant,
                        'start_frame': start_frame,
                        'stop_frame': stop_frame,
                        'action_start': action_start,
                        'action_stop': action_stop,
                        'gt_verb': gt_verb,
                        'gt_narration': gt_narration,
                        'pred_verb': pred_verb,
                        'pred_noun': pred_noun,
                        'conf': conf,
                    })

            cap.release()

    print(f"\nFound {len(all_clips)} matching clips total")

    # Select diverse high-confidence clips
    all_clips.sort(key=lambda x: x['conf'], reverse=True)

    # Pick variety - different verbs and participants
    selected = []
    used_verbs = set()
    used_participants = set()

    # First pass: get variety
    for clip in all_clips:
        if len(selected) >= max_clips:
            break
        verb = clip['gt_verb']
        part = clip['participant']

        # Prefer unseen verbs and participants
        if verb not in used_verbs or part not in used_participants:
            selected.append(clip)
            used_verbs.add(verb)
            used_participants.add(part)

    # Fill remaining with highest confidence
    for clip in all_clips:
        if len(selected) >= max_clips:
            break
        if clip not in selected:
            selected.append(clip)

    print(f"Selected {len(selected)} clips for highlight reel")

    # Output settings
    panel_width = 350
    out_width = 640 + panel_width
    out_height = 360

    # Get FPS from first clip's source video
    first_cap = cv2.VideoCapture(str(selected[0]['video_path']))
    out_fps = first_cap.get(cv2.CAP_PROP_FPS)
    first_cap.release()
    print(f"Output FPS: {out_fps:.1f}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, out_fps, (out_width, out_height))

    total_frames_written = 0

    for i, clip in enumerate(selected):
        print(f"  [{i+1}/{len(selected)}] {clip['gt_narration']} -> {clip['pred_verb']} {clip['pred_noun']} ({clip['conf']*100:.0f}%)")

        cap = cv2.VideoCapture(str(clip['video_path']))
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip['start_frame'])

        for f_idx in range(clip['start_frame'], clip['stop_frame'] + 1):
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (640, 360))

            # Create display
            display = np.zeros((out_height, out_width, 3), dtype=np.uint8)
            display[:, :640] = frame_resized

            # Side panel
            panel_x = 640
            cv2.rectangle(display, (panel_x, 0), (out_width, out_height), (30, 30, 30), -1)

            # Title
            cv2.putText(display, "CORRECT PREDICTION", (panel_x + 15, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(display, f"{clip['participant']} - {clip['video_id']}", (panel_x + 15, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            cv2.line(display, (panel_x + 15, 70), (out_width - 15, 70), (80, 80, 80), 1)

            # Model prediction
            cv2.putText(display, "MODEL:", (panel_x + 15, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

            color = (0, 255, 0)
            cv2.putText(display, clip['pred_verb'], (panel_x + 15, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(display, clip['pred_noun'], (panel_x + 15, 165),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Confidence bar
            bar_w = int(clip['conf'] * 250)
            cv2.rectangle(display, (panel_x + 15, 185), (panel_x + 15 + bar_w, 205), color, -1)
            cv2.rectangle(display, (panel_x + 15, 185), (panel_x + 275, 205), (80, 80, 80), 2)
            cv2.putText(display, f"{clip['conf']*100:.0f}%", (panel_x + 285, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.line(display, (panel_x + 15, 225), (out_width - 15, 225), (80, 80, 80), 1)

            # Ground truth
            cv2.putText(display, "GROUND TRUTH:", (panel_x + 15, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

            words = clip['gt_narration'].split()
            cv2.putText(display, words[0] if words else "", (panel_x + 15, 285),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 2)
            cv2.putText(display, ' '.join(words[1:3]) if len(words) > 1 else "", (panel_x + 15, 320),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)

            # Clip counter
            cv2.putText(display, f"Clip {i+1}/{len(selected)}", (panel_x + 15, out_height - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            out.write(display)
            total_frames_written += 1

        cap.release()

    out.release()

    duration = total_frames_written / out_fps
    print(f"\n{'='*50}")
    print(f"HIGHLIGHT VIDEO CREATED")
    print(f"{'='*50}")
    print(f"  Clips: {len(selected)}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Output: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_clips', type=int, default=15)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    epic_dir = Path(__file__).parent.parent

    # Multiple participants for variety
    participants = ['P01', 'P03', 'P04', 'P05', 'P06', 'P08', 'P10', 'P12', 'P22']

    checkpoint_path = epic_dir / "outputs_exp15_h100/checkpoints/best_model.pth"
    annotations_csv = epic_dir / cfg.VAL_CSV

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = epic_dir / "outputs/highlights_best.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

    create_highlights(participants, output_path, checkpoint_path, annotations_csv,
                     max_clips=args.max_clips, context_frames=30)


if __name__ == '__main__':
    main()
