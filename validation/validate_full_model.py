#!/usr/bin/env python3
"""
Validate full EPIC-KITCHENS model locally using videos.
Model architecture matches train_full.py from VSC training.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm


class ActionModel(nn.Module):
    """Action recognition model - matches train_full.py architecture."""

    def __init__(self, num_verb_classes=97, num_noun_classes=300,
                 backbone='resnet50', temporal_model='lstm',
                 dropout=0.5, num_frames=16):
        super().__init__()

        # Backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feature_dim = 2048
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feature_dim = 512
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Temporal model
        self.temporal_model = temporal_model
        if temporal_model == 'lstm':
            self.temporal = nn.LSTM(
                self.feature_dim, self.feature_dim // 2,
                num_layers=2, batch_first=True, bidirectional=True, dropout=0.3
            )
            self.temporal_dim = self.feature_dim
        elif temporal_model == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.feature_dim, nhead=8, dim_feedforward=2048,
                dropout=0.1, batch_first=True
            )
            self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.temporal_dim = self.feature_dim
        else:
            self.temporal = None
            self.temporal_dim = self.feature_dim

        # Classification heads
        self.dropout = nn.Dropout(dropout)
        self.verb_head = nn.Linear(self.temporal_dim, num_verb_classes)
        self.noun_head = nn.Linear(self.temporal_dim, num_noun_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)
        features = features.view(B, T, -1)

        if self.temporal_model == 'lstm':
            temporal_out, _ = self.temporal(features)
            pooled = temporal_out.mean(dim=1)
        elif self.temporal_model == 'transformer':
            temporal_out = self.temporal(features)
            pooled = temporal_out.mean(dim=1)
        else:
            pooled = features.mean(dim=1)

        pooled = self.dropout(pooled)
        verb_logits = self.verb_head(pooled)
        noun_logits = self.noun_head(pooled)
        return verb_logits, noun_logits


class LocalValDataset(Dataset):
    """Validation dataset using local videos."""

    def __init__(self, annotations_csv, video_base_dir, num_frames=16, image_size=224):
        self.annotations = pd.read_csv(annotations_csv)
        self.video_base_dir = Path(video_base_dir)
        self.num_frames = num_frames
        self.image_size = image_size

        # Filter to available videos
        self.valid_indices = []
        missing_videos = set()

        for idx in range(len(self.annotations)):
            row = self.annotations.iloc[idx]
            participant = row['participant_id']
            video_path = self.video_base_dir / participant / f"{row['video_id']}.MP4"
            if video_path.exists():
                self.valid_indices.append(idx)
            else:
                missing_videos.add(row['video_id'])

        print(f"Validation: {len(self.valid_indices)}/{len(self.annotations)} actions available")
        if missing_videos:
            print(f"  Missing {len(missing_videos)} videos")

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

            total_frames = stop_frame - start_frame + 1
            if total_frames <= self.num_frames:
                indices = list(range(start_frame, stop_frame + 1))
                while len(indices) < self.num_frames:
                    indices.append(stop_frame)
            else:
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
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=16)
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Paths
    epic_dir = Path(__file__).parent.parent
    val_csv = epic_dir / "EPIC-KITCHENS" / "epic-kitchens-100-annotations-master" / "EPIC_100_validation.csv"
    video_dir = epic_dir / "EPIC-KITCHENS" / "videos_640x360"

    print(f"Validation CSV: {val_csv}")
    print(f"Video directory: {video_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num frames: {args.num_frames}")

    # Dataset
    val_dataset = LocalValDataset(
        annotations_csv=val_csv,
        video_base_dir=video_dir,
        num_frames=args.num_frames,
        image_size=224
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # Model - matches train_full.py
    model = ActionModel(
        num_verb_classes=97,
        num_noun_classes=300,
        backbone='resnet50',
        temporal_model='lstm',
        dropout=0.5,
        num_frames=args.num_frames
    ).to(device)

    # Load checkpoint
    print(f"\nLoading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    print("Checkpoint loaded!")

    # Validate
    print(f"\nValidating on {len(val_dataset)} actions...")
    metrics = validate(model, val_loader, device)

    print(f"\n{'='*50}")
    print(f"VALIDATION RESULTS (Local Videos)")
    print(f"{'='*50}")
    print(f"  Verb Accuracy:   {metrics['verb_acc']:.2f}%")
    print(f"  Noun Accuracy:   {metrics['noun_acc']:.2f}%")
    print(f"  Action Accuracy: {metrics['action_acc']:.2f}%")
    print(f"  Total samples:   {metrics['total']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
