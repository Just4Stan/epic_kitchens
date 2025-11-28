"""
EPIC-KITCHENS-100 - Full Dataset Loading
=========================================
Loads directly from official pre-extracted RGB frames.
Format: {data_dir}/{participant}/rgb_frames/{video_id}/frame_{:010d}.jpg
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import pandas as pd
import cv2
import numpy as np


class FullTrainDataset(Dataset):
    """Training dataset using full pre-extracted RGB frames."""

    def __init__(self, annotations_csv, frames_dir, num_frames=16,
                 image_size=224, augmentation='medium'):
        """
        Args:
            annotations_csv: Path to EPIC_100_train.csv
            frames_dir: Path to root of RGB frames (e.g., /scratch/.../EPIC-KITCHENS/EPIC-KITCHENS)
            num_frames: Number of frames to sample per action
            image_size: Output image size
            augmentation: Augmentation level ('none', 'light', 'medium', 'heavy')
        """
        self.annotations = pd.read_csv(annotations_csv)
        self.frames_dir = Path(frames_dir)
        self.num_frames = num_frames
        self.image_size = image_size

        # Filter to actions where frames exist
        self.valid_indices = []
        missing_videos = set()

        for idx in range(len(self.annotations)):
            row = self.annotations.iloc[idx]
            participant = row['participant_id']
            video_id = row['video_id']
            video_frames_dir = self.frames_dir / participant / "rgb_frames" / video_id

            if video_frames_dir.exists():
                self.valid_indices.append(idx)
            else:
                missing_videos.add(video_id)

        print(f"Training: {len(self.valid_indices)}/{len(self.annotations)} actions")
        if missing_videos:
            print(f"  Missing {len(missing_videos)} videos: {list(missing_videos)[:5]}...")

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

    def _load_frames(self, video_frames_dir, start_frame, stop_frame):
        """Load and sample frames from start to stop frame."""
        # Sample frame indices uniformly
        total_frames = stop_frame - start_frame + 1
        if total_frames <= self.num_frames:
            indices = list(range(start_frame, stop_frame + 1))
            # Pad with last frame if needed
            while len(indices) < self.num_frames:
                indices.append(stop_frame)
        else:
            indices = np.linspace(start_frame, stop_frame, self.num_frames, dtype=int)

        frames = []
        for frame_idx in indices:
            frame_path = video_frames_dir / f"frame_{frame_idx:010d}.jpg"
            if frame_path.exists():
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
            else:
                frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))

        return frames

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.annotations.iloc[real_idx]

        participant = row['participant_id']
        video_id = row['video_id']
        start_frame = int(row['start_frame'])
        stop_frame = int(row['stop_frame'])

        video_frames_dir = self.frames_dir / participant / "rgb_frames" / video_id
        frames = self._load_frames(video_frames_dir, start_frame, stop_frame)

        frames_tensor = torch.stack([self.transform(f) for f in frames])
        return frames_tensor, int(row['verb_class']), int(row['noun_class'])


class FullValDataset(Dataset):
    """Validation dataset using full pre-extracted RGB frames."""

    def __init__(self, annotations_csv, frames_dir, num_frames=16, image_size=224):
        """
        Args:
            annotations_csv: Path to EPIC_100_validation.csv
            frames_dir: Path to root of RGB frames
            num_frames: Number of frames to sample per action
            image_size: Output image size
        """
        self.annotations = pd.read_csv(annotations_csv)
        self.frames_dir = Path(frames_dir)
        self.num_frames = num_frames
        self.image_size = image_size

        # Filter to actions where frames exist
        self.valid_indices = []
        missing_videos = set()

        for idx in range(len(self.annotations)):
            row = self.annotations.iloc[idx]
            participant = row['participant_id']
            video_id = row['video_id']
            video_frames_dir = self.frames_dir / participant / "rgb_frames" / video_id

            if video_frames_dir.exists():
                self.valid_indices.append(idx)
            else:
                missing_videos.add(video_id)

        print(f"Validation: {len(self.valid_indices)}/{len(self.annotations)} actions")
        if missing_videos:
            print(f"  Missing {len(missing_videos)} videos: {list(missing_videos)[:5]}...")

        # Validation transform (no augmentation)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def _load_frames(self, video_frames_dir, start_frame, stop_frame):
        """Load and sample frames from start to stop frame."""
        total_frames = stop_frame - start_frame + 1
        if total_frames <= self.num_frames:
            indices = list(range(start_frame, stop_frame + 1))
            while len(indices) < self.num_frames:
                indices.append(stop_frame)
        else:
            indices = np.linspace(start_frame, stop_frame, self.num_frames, dtype=int)

        frames = []
        for frame_idx in indices:
            frame_path = video_frames_dir / f"frame_{frame_idx:010d}.jpg"
            if frame_path.exists():
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
            else:
                frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))

        return frames

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.annotations.iloc[real_idx]

        participant = row['participant_id']
        video_id = row['video_id']
        start_frame = int(row['start_frame'])
        stop_frame = int(row['stop_frame'])

        video_frames_dir = self.frames_dir / participant / "rgb_frames" / video_id
        frames = self._load_frames(video_frames_dir, start_frame, stop_frame)

        frames_tensor = torch.stack([self.transform(f) for f in frames])
        return frames_tensor, int(row['verb_class']), int(row['noun_class'])
