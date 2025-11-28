#!/usr/bin/env python3
"""
Ensemble validation - combines 16-frame and 32-frame models.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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

        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feature_dim = 2048
            self.backbone.fc = nn.Identity()

        self.temporal_model = temporal_model
        if temporal_model == 'lstm':
            self.temporal = nn.LSTM(
                self.feature_dim, self.feature_dim // 2,
                num_layers=2, batch_first=True, bidirectional=True, dropout=0.3
            )
            self.temporal_dim = self.feature_dim

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
        else:
            pooled = features.mean(dim=1)

        pooled = self.dropout(pooled)
        return self.verb_head(pooled), self.noun_head(pooled)


class LocalValDataset(Dataset):
    """Validation dataset using local videos."""

    def __init__(self, annotations_csv, video_base_dir, num_frames=16, image_size=224):
        self.annotations = pd.read_csv(annotations_csv)
        self.video_base_dir = Path(video_base_dir)
        self.num_frames = num_frames
        self.image_size = image_size

        self.valid_indices = []
        for idx in range(len(self.annotations)):
            row = self.annotations.iloc[idx]
            participant = row['participant_id']
            video_path = self.video_base_dir / participant / f"{row['video_id']}.MP4"
            if video_path.exists():
                self.valid_indices.append(idx)

        print(f"Validation: {len(self.valid_indices)}/{len(self.annotations)} actions")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _extract_frames(self, video_path, start_frame, stop_frame, num_frames):
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None

            total = stop_frame - start_frame + 1
            if total <= num_frames:
                indices = list(range(start_frame, stop_frame + 1))
                while len(indices) < num_frames:
                    indices.append(stop_frame)
            else:
                indices = np.linspace(start_frame, stop_frame, num_frames, dtype=int)

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
            return frames if len(frames) == num_frames else None
        except:
            return None

    def __len__(self):
        return len(self.valid_indices)

    def get_item_with_frames(self, idx, num_frames):
        """Get item with specific number of frames."""
        real_idx = self.valid_indices[idx]
        row = self.annotations.iloc[real_idx]

        participant = row['participant_id']
        video_path = self.video_base_dir / participant / f"{row['video_id']}.MP4"
        frames = self._extract_frames(video_path, int(row['start_frame']), int(row['stop_frame']), num_frames)

        if frames is None:
            frames = [np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)] * num_frames

        frames_tensor = torch.stack([self.transform(f) for f in frames])
        return frames_tensor, int(row['verb_class']), int(row['noun_class'])


def load_model(checkpoint_path, num_frames, device):
    model = ActionModel(
        num_verb_classes=97,
        num_noun_classes=300,
        backbone='resnet50',
        temporal_model='lstm',
        dropout=0.0,
        num_frames=num_frames
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
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

    checkpoint_16 = epic_dir / "outputs" / "full_a100_v3" / "checkpoints" / "best_model.pth"
    checkpoint_32 = epic_dir / "outputs" / "full_32frames_v1" / "checkpoints" / "best_model.pth"

    print(f"\nLoading models...")
    print(f"  16-frame: {checkpoint_16}")
    print(f"  32-frame: {checkpoint_32}")

    model_16 = load_model(checkpoint_16, 16, device)
    model_32 = load_model(checkpoint_32, 32, device)
    print("Models loaded!")

    # Dataset
    dataset = LocalValDataset(val_csv, video_dir, num_frames=32, image_size=224)

    # Metrics
    verb_correct_16, noun_correct_16, action_correct_16 = 0, 0, 0
    verb_correct_32, noun_correct_32, action_correct_32 = 0, 0, 0
    verb_correct_ens, noun_correct_ens, action_correct_ens = 0, 0, 0
    total = 0

    print(f"\nValidating {len(dataset)} samples...")

    for idx in tqdm(range(len(dataset))):
        # Get frames for both models
        frames_16, verb_label, noun_label = dataset.get_item_with_frames(idx, 16)
        frames_32, _, _ = dataset.get_item_with_frames(idx, 32)

        frames_16 = frames_16.unsqueeze(0).to(device)
        frames_32 = frames_32.unsqueeze(0).to(device)

        with torch.no_grad():
            # 16-frame model
            verb_logits_16, noun_logits_16 = model_16(frames_16)
            verb_probs_16 = torch.softmax(verb_logits_16, dim=1)
            noun_probs_16 = torch.softmax(noun_logits_16, dim=1)

            # 32-frame model
            verb_logits_32, noun_logits_32 = model_32(frames_32)
            verb_probs_32 = torch.softmax(verb_logits_32, dim=1)
            noun_probs_32 = torch.softmax(noun_logits_32, dim=1)

            # Ensemble (average probabilities)
            verb_probs_ens = (verb_probs_16 + verb_probs_32) / 2
            noun_probs_ens = (noun_probs_16 + noun_probs_32) / 2

        # Predictions
        verb_pred_16 = verb_probs_16.argmax(1).item()
        noun_pred_16 = noun_probs_16.argmax(1).item()
        verb_pred_32 = verb_probs_32.argmax(1).item()
        noun_pred_32 = noun_probs_32.argmax(1).item()
        verb_pred_ens = verb_probs_ens.argmax(1).item()
        noun_pred_ens = noun_probs_ens.argmax(1).item()

        # 16-frame accuracy
        if verb_pred_16 == verb_label:
            verb_correct_16 += 1
        if noun_pred_16 == noun_label:
            noun_correct_16 += 1
        if verb_pred_16 == verb_label and noun_pred_16 == noun_label:
            action_correct_16 += 1

        # 32-frame accuracy
        if verb_pred_32 == verb_label:
            verb_correct_32 += 1
        if noun_pred_32 == noun_label:
            noun_correct_32 += 1
        if verb_pred_32 == verb_label and noun_pred_32 == noun_label:
            action_correct_32 += 1

        # Ensemble accuracy
        if verb_pred_ens == verb_label:
            verb_correct_ens += 1
        if noun_pred_ens == noun_label:
            noun_correct_ens += 1
        if verb_pred_ens == verb_label and noun_pred_ens == noun_label:
            action_correct_ens += 1

        total += 1

    print(f"\n{'='*60}")
    print(f"ENSEMBLE VALIDATION RESULTS ({total} samples)")
    print(f"{'='*60}")
    print(f"\n16-Frame Model:")
    print(f"  Verb:   {100*verb_correct_16/total:.2f}%")
    print(f"  Noun:   {100*noun_correct_16/total:.2f}%")
    print(f"  Action: {100*action_correct_16/total:.2f}%")
    print(f"\n32-Frame Model:")
    print(f"  Verb:   {100*verb_correct_32/total:.2f}%")
    print(f"  Noun:   {100*noun_correct_32/total:.2f}%")
    print(f"  Action: {100*action_correct_32/total:.2f}%")
    print(f"\nENSEMBLE (avg probs):")
    print(f"  Verb:   {100*verb_correct_ens/total:.2f}%")
    print(f"  Noun:   {100*noun_correct_ens/total:.2f}%")
    print(f"  Action: {100*action_correct_ens/total:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
