"""
Validate EPIC-KITCHENS checkpoint locally.
"""

import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from models import ActionModel
from config import Config as cfg


class LocalValDataset(Dataset):
    """Validation dataset - handles local video structure (videos in participant folders)."""

    def __init__(self, annotations_csv, video_base_dir, num_frames=16, image_size=224):
        self.annotations = pd.read_csv(annotations_csv)
        self.video_base_dir = Path(video_base_dir)
        self.num_frames = num_frames
        self.image_size = image_size

        # Filter to available videos
        self.valid_indices = []
        for idx in range(len(self.annotations)):
            row = self.annotations.iloc[idx]
            # Videos are in participant folders: P01/P01_11.MP4
            participant = row['participant_id']
            video_path = self.video_base_dir / participant / f"{row['video_id']}.MP4"
            if video_path.exists():
                self.valid_indices.append(idx)

        print(f"Validation: {len(self.valid_indices)} actions (from {len(self.annotations)} total)")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _extract_frames(self, video_path, start_frame, stop_frame):
        """Extract frames from video segment."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None

            indices = np.linspace(start_frame, stop_frame, self.num_frames, dtype=int)
            frames = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                elif frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))

            cap.release()
            return frames if len(frames) == self.num_frames else None
        except:
            return None

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.annotations.iloc[real_idx]

        participant = row['participant_id']
        video_path = self.video_base_dir / participant / f"{row['video_id']}.MP4"
        frames = self._extract_frames(video_path, int(row['start_frame']), int(row['stop_frame']))

        if frames is None:
            frames = [np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)] * self.num_frames

        frames_tensor = torch.stack([self.transform(f) for f in frames])
        return frames_tensor, int(row['verb_class']), int(row['noun_class'])


@torch.no_grad()
def validate(model, loader, device):
    """Run validation and return metrics."""
    model.eval()

    verb_correct, noun_correct, action_correct, total = 0, 0, 0, 0

    pbar = tqdm(loader, desc="Validating")
    for frames, verb_labels, noun_labels in pbar:
        frames = frames.to(device)
        verb_labels = verb_labels.to(device)
        noun_labels = noun_labels.to(device)

        verb_logits, noun_logits = model(frames)

        verb_pred = verb_logits.argmax(1)
        noun_pred = noun_logits.argmax(1)

        verb_correct += (verb_pred == verb_labels).sum().item()
        noun_correct += (noun_pred == noun_labels).sum().item()
        action_correct += ((verb_pred == verb_labels) & (noun_pred == noun_labels)).sum().item()
        total += frames.size(0)

        pbar.set_postfix({
            'verb': f'{100*verb_correct/total:.1f}%',
            'noun': f'{100*noun_correct/total:.1f}%',
            'action': f'{100*action_correct/total:.2f}%'
        })

    return {
        'verb_acc': 100 * verb_correct / total,
        'noun_acc': 100 * noun_correct / total,
        'action_acc': 100 * action_correct / total,
        'total': total
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (smaller for local)")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Paths - relative to epic_kitchens directory
    epic_dir = Path(__file__).parent.parent
    val_csv = epic_dir / cfg.VAL_CSV
    video_dir = epic_dir / "EPIC-KITCHENS" / "videos_640x360"

    print(f"Validation CSV: {val_csv}")
    print(f"Video directory: {video_dir}")
    print(f"Checkpoint: {args.checkpoint}")

    # Dataset
    val_dataset = LocalValDataset(
        annotations_csv=val_csv,
        video_base_dir=video_dir,
        num_frames=cfg.NUM_FRAMES,
        image_size=cfg.IMAGE_SIZE
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # Model
    model = ActionModel(
        num_verb_classes=cfg.NUM_VERB_CLASSES,
        num_noun_classes=cfg.NUM_NOUN_CLASSES,
        backbone=cfg.BACKBONE,
        temporal_model=cfg.TEMPORAL_MODEL,
        dropout=cfg.DROPOUT,
        num_frames=cfg.NUM_FRAMES
    ).to(device)

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    print("Checkpoint loaded successfully!")

    # Validate
    print(f"\nValidating on {len(val_dataset)} actions...")
    metrics = validate(model, val_loader, device)

    print(f"\n{'='*50}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"  Verb Accuracy:   {metrics['verb_acc']:.2f}%")
    print(f"  Noun Accuracy:   {metrics['noun_acc']:.2f}%")
    print(f"  Action Accuracy: {metrics['action_acc']:.2f}%")
    print(f"  Total samples:   {metrics['total']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
