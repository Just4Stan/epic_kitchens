"""
EPIC-KITCHENS-100 Action Recognition - Datasets
================================================
Dataset classes for training and validation.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import pandas as pd
import cv2
import numpy as np


class TrainDataset(Dataset):
    """Training dataset using pre-extracted frames."""

    def __init__(self, annotations_csv, frames_dir, num_frames=16,
                 image_size=224, augmentation='medium'):
        self.annotations = pd.read_csv(annotations_csv)
        self.frames_dir = Path(frames_dir)
        self.num_frames = num_frames
        self.image_size = image_size

        # Filter to actions with extracted frames
        self.valid_indices = []
        for idx in range(len(self.annotations)):
            row = self.annotations.iloc[idx]
            action_id = f"{row['participant_id']}_{row['video_id']}_{idx:06d}"
            if (self.frames_dir / action_id).exists():
                self.valid_indices.append(idx)

        print(f"Training: {len(self.valid_indices)} actions")

        # Augmentation transforms
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        aug_configs = {
            'none': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]),
            'light': transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                normalize,
            ]),
            'medium': transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                normalize,
            ]),
            'heavy': transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(5)], p=0.3),
                transforms.ToTensor(),
                normalize,
            ]),
        }
        self.transform = aug_configs.get(augmentation, aug_configs['medium'])

    def _load_frames(self, action_dir):
        """Load and sample frames from action directory."""
        frame_files = sorted(action_dir.glob("frame_*.jpg"))
        if not frame_files:
            return None

        n = len(frame_files)
        if n <= self.num_frames:
            indices = list(range(n)) + [n-1] * (self.num_frames - n)
        else:
            indices = np.linspace(0, n - 1, self.num_frames, dtype=int)

        frames = []
        for i in indices:
            frame = cv2.imread(str(frame_files[i]))
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        return frames if len(frames) == self.num_frames else None

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.annotations.iloc[real_idx]

        action_id = f"{row['participant_id']}_{row['video_id']}_{real_idx:06d}"
        action_dir = self.frames_dir / action_id

        frames = self._load_frames(action_dir)
        if frames is None:
            frames = [np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)] * self.num_frames

        frames_tensor = torch.stack([self.transform(f) for f in frames])
        return frames_tensor, int(row['verb_class']), int(row['noun_class'])


class ValDataset(Dataset):
    """Validation dataset - uses pre-extracted frames if available, falls back to video."""

    def __init__(self, annotations_csv, video_dir, num_frames=16, image_size=224, frames_dir=None):
        self.annotations = pd.read_csv(annotations_csv)
        self.video_dir = Path(video_dir) if video_dir else None
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.num_frames = num_frames
        self.image_size = image_size

        # Filter to available actions
        self.valid_indices = []
        self.use_frames = []  # Track whether to use pre-extracted frames for each action

        for idx in range(len(self.annotations)):
            row = self.annotations.iloc[idx]
            action_id = f"{row['participant_id']}_{row['video_id']}_{idx:06d}"

            # Check pre-extracted frames first
            if self.frames_dir and (self.frames_dir / action_id).exists():
                self.valid_indices.append(idx)
                self.use_frames.append(True)
            # Fall back to video
            elif self.video_dir:
                video_path = self.video_dir / f"{row['video_id']}.MP4"
                if video_path.exists():
                    self.valid_indices.append(idx)
                    self.use_frames.append(False)

        frames_count = sum(self.use_frames)
        video_count = len(self.use_frames) - frames_count
        print(f"Validation: {len(self.valid_indices)} actions ({frames_count} from frames, {video_count} from video)")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_frames(self, action_dir):
        """Load frames from pre-extracted directory."""
        frame_files = sorted(action_dir.glob("frame_*.jpg"))
        if not frame_files:
            return None

        n = len(frame_files)
        if n <= self.num_frames:
            indices = list(range(n)) + [n-1] * (self.num_frames - n)
        else:
            indices = np.linspace(0, n - 1, self.num_frames, dtype=int)

        frames = []
        for i in indices:
            frame = cv2.imread(str(frame_files[i]))
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        return frames if len(frames) == self.num_frames else None

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

        if self.use_frames[idx]:
            # Use pre-extracted frames
            action_id = f"{row['participant_id']}_{row['video_id']}_{real_idx:06d}"
            action_dir = self.frames_dir / action_id
            frames = self._load_frames(action_dir)
        else:
            # Fall back to video extraction
            video_path = self.video_dir / f"{row['video_id']}.MP4"
            frames = self._extract_frames(video_path, int(row['start_frame']), int(row['stop_frame']))

        if frames is None:
            frames = [np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)] * self.num_frames

        frames_tensor = torch.stack([self.transform(f) for f in frames])
        return frames_tensor, int(row['verb_class']), int(row['noun_class'])
