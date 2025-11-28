#!/usr/bin/env python3
"""
Generate CodaBench ensemble submission combining 16-frame and 32-frame models.

Submission format: submission.pt containing list of dicts with verb_output and noun_output tensors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
import zipfile


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


class SubmissionDataset(Dataset):
    """Dataset for validation set submission - only processes segments with available videos."""

    def __init__(self, annotations_csv, video_base_dir, image_size=224):
        self.annotations = pd.read_csv(annotations_csv)
        self.video_base_dir = Path(video_base_dir)
        self.image_size = image_size

        # Find valid indices (videos that exist)
        self.valid_indices = []
        for idx in range(len(self.annotations)):
            row = self.annotations.iloc[idx]
            participant = row['participant_id']
            video_id = row['video_id']
            video_path = self.video_base_dir / participant / f"{video_id}.MP4"
            if video_path.exists():
                self.valid_indices.append(idx)

        print(f"Found {len(self.valid_indices)}/{len(self.annotations)} valid segments")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.valid_indices)

    def get_narration_id(self, idx):
        real_idx = self.valid_indices[idx]
        return self.annotations.iloc[real_idx]['narration_id']

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

    def get_item_with_frames(self, idx, num_frames):
        """Get item with specific number of frames."""
        real_idx = self.valid_indices[idx]
        row = self.annotations.iloc[real_idx]

        narration_id = row['narration_id']
        participant = row['participant_id']
        video_path = self.video_base_dir / participant / f"{row['video_id']}.MP4"

        frames = self._extract_frames(video_path, int(row['start_frame']), int(row['stop_frame']), num_frames)

        if frames is None:
            # Use black frames as fallback for extraction errors
            frames = [np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)] * num_frames

        frames_tensor = torch.stack([self.transform(f) for f in frames])
        return narration_id, frames_tensor


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
    # Use validation set since we have those videos locally
    test_csv = epic_dir / "EPIC-KITCHENS" / "epic-kitchens-100-annotations-master" / "EPIC_100_validation.csv"
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
    dataset = SubmissionDataset(test_csv, video_dir, image_size=224)

    # Generate predictions
    predictions = []
    print(f"\nGenerating ensemble predictions for {len(dataset)} samples...")

    for idx in tqdm(range(len(dataset))):
        narration_id, frames_16 = dataset.get_item_with_frames(idx, 16)
        _, frames_32 = dataset.get_item_with_frames(idx, 32)

        frames_16 = frames_16.unsqueeze(0).to(device)
        frames_32 = frames_32.unsqueeze(0).to(device)

        with torch.no_grad():
            # 16-frame model
            verb_logits_16, noun_logits_16 = model_16(frames_16)
            verb_probs_16 = F.softmax(verb_logits_16, dim=1)
            noun_probs_16 = F.softmax(noun_logits_16, dim=1)

            # 32-frame model
            verb_logits_32, noun_logits_32 = model_32(frames_32)
            verb_probs_32 = F.softmax(verb_logits_32, dim=1)
            noun_probs_32 = F.softmax(noun_logits_32, dim=1)

            # Ensemble (average probabilities)
            verb_probs = ((verb_probs_16 + verb_probs_32) / 2)[0].cpu()
            noun_probs = ((noun_probs_16 + noun_probs_32) / 2)[0].cpu()

        predictions.append({
            'narration_id': narration_id,
            'verb_output': verb_probs,
            'noun_output': noun_probs,
        })

    print(f"\nGenerated {len(predictions)} predictions")

    # Save submission
    output_dir = epic_dir / "outputs" / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_pt = output_dir / "submission.pt"
    torch.save(predictions, submission_pt)
    print(f"\nSaved {len(predictions)} predictions to {submission_pt}")

    # Verify format
    print("\nVerifying submission format...")
    loaded = torch.load(submission_pt, weights_only=False)
    print(f"  Total predictions: {len(loaded)}")
    if len(loaded) > 0:
        sample = loaded[0]
        print(f"  Sample narration_id: {sample['narration_id']}")
        print(f"  verb_output shape: {sample['verb_output'].shape}")
        print(f"  noun_output shape: {sample['noun_output'].shape}")

    # Create zip
    submission_zip = output_dir / "ensemble_submission.zip"
    with zipfile.ZipFile(submission_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_pt, "submission.pt")

    print(f"\nCreated submission zip: {submission_zip}")
    print(f"Ready to upload to CodaBench!")


if __name__ == "__main__":
    main()
