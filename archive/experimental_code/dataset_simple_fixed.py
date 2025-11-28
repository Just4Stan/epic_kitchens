"""
Fixed Dataset Loader with Minimal Augmentation
Fixes bugs from phase2/dataset_improved.py and reduces aggressive augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
import warnings

warnings.filterwarnings('ignore')


class SimpleFixedDataset(Dataset):
    """
    EPIC-KITCHENS dataset with:
    - Fixed frame sampling (uses stop_frame - 1 correctly)
    - Minimal, appropriate augmentation
    - No aggressive cropping, blurring, or grayscale
    """

    def __init__(self, annotations_csv, video_dir, num_frames=8,
                 image_size=224, mode='train'):
        """
        Args:
            annotations_csv: Path to CSV with annotations
            video_dir: Directory containing video files
            num_frames: Number of frames to sample per video
            image_size: Size to resize frames to
            mode: 'train' or 'val'
        """
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.image_size = image_size
        self.mode = mode

        print(f"Loading {'training' if mode == 'train' else 'validation'} dataset...")

        # Load annotations
        self.annotations = pd.read_csv(annotations_csv)
        print(f"Found {len(self.annotations)} action segments")

        # Filter out missing videos
        print("Checking which videos exist...")
        valid_indices = []
        missing_videos = set()

        for idx, row in self.annotations.iterrows():
            video_id = row['video_id']
            participant = video_id.split('_')[0]
            video_path = self.video_dir / participant / f"{video_id}.MP4"

            if video_path.exists():
                valid_indices.append(idx)
            else:
                missing_videos.add(str(video_path))

        self.annotations = self.annotations.iloc[valid_indices].reset_index(drop=True)

        if missing_videos:
            print(f"⚠ Warning: {len(missing_videos)} videos not found (skipped)")

        print(f"Using {len(self.annotations)} valid action segments\n")

        # Create appropriate transforms
        if mode == 'train':
            self.transform = self._get_train_transforms()
        else:
            self.transform = self._get_val_transforms()

    def _get_train_transforms(self):
        """
        Minimal, appropriate augmentation for egocentric action recognition.

        Rationale:
        - Horizontal flip: hands can be on either side
        - Small color jitter: lighting variations
        - Small random crop: camera movement
        - NO grayscale, NO blur, NO aggressive erasing
        """
        return transforms.Compose([
            transforms.ToPILImage(),

            # Resize with small random crop (mild augmentation)
            transforms.RandomResizedCrop(
                self.image_size,
                scale=(0.85, 1.0),  # Mild crop (not 0.6!)
                ratio=(0.9, 1.1)    # Keep aspect ratio close to original
            ),

            # Horizontal flip (natural for egocentric videos)
            transforms.RandomHorizontalFlip(p=0.5),

            # Mild color jitter (lighting variations)
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),

            transforms.ToTensor(),

            # ImageNet normalization
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),

            # Very mild random erasing (occlusion robustness)
            transforms.RandomErasing(
                p=0.1,  # Only 10% of images (not 30%!)
                scale=(0.02, 0.1),  # Small patches only
                ratio=(0.5, 2.0)
            )
        ])

    def _get_val_transforms(self):
        """Standard transforms for validation (no augmentation)."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Returns:
            frames: Tensor of shape (num_frames, 3, H, W)
            verb_label: Integer verb class
            noun_label: Integer noun class
        """
        row = self.annotations.iloc[idx]

        # Get video path
        video_id = row['video_id']
        participant = video_id.split('_')[0]
        video_path = self.video_dir / participant / f"{video_id}.MP4"

        # Get frame range and labels
        start_frame = int(row['start_frame'])
        stop_frame = int(row['stop_frame'])
        verb_label = int(row['verb_class'])
        noun_label = int(row['noun_class'])

        try:
            # Load video segment
            frames = self._load_video_segment(video_path, start_frame, stop_frame)
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Return dummy data on error
            frames = torch.zeros(self.num_frames, 3, self.image_size, self.image_size)
            verb_label = 0
            noun_label = 0

        return frames, verb_label, noun_label

    def _load_video_segment(self, video_path, start_frame, stop_frame):
        """
        Load and sample frames from a video segment.

        FIXED: Uses stop_frame - 1 (not stop_frame) to avoid sampling
               frames after the action ends.

        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            stop_frame: Ending frame index

        Returns:
            frames: Tensor of shape (num_frames, 3, H, W)
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Calculate frame indices to sample (uniformly spaced)
        total_frames = stop_frame - start_frame

        if total_frames <= 0:
            raise ValueError(f"Invalid frame range: {start_frame} to {stop_frame}")

        # FIXED: Use stop_frame - 1 (not stop_frame)
        # This ensures we don't sample frames after the action ends
        if total_frames < self.num_frames:
            # If segment is shorter than num_frames, allow some repetition
            frame_indices = np.linspace(start_frame, stop_frame - 1,
                                       self.num_frames, dtype=int)
        else:
            # Uniformly sample num_frames from the segment
            frame_indices = np.linspace(start_frame, stop_frame - 1,
                                       self.num_frames, dtype=int)

        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                # If frame read fails, use last successful frame or black frame
                if len(frames) > 0:
                    frames.append(frames[-1])
                else:
                    frames.append(torch.zeros(3, self.image_size, self.image_size))
                continue

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply transforms
            frame = self.transform(frame)
            frames.append(frame)

        cap.release()

        # Stack frames: (num_frames, 3, H, W)
        frames = torch.stack(frames, dim=0)
        return frames


def get_simple_dataloaders(config):
    """
    Create simple train and validation dataloaders.

    Args:
        config: Configuration object

    Returns:
        train_loader, val_loader
    """
    print("=" * 70)
    print("Creating Simple Fixed Dataloaders")
    print("=" * 70)

    # Training dataset
    train_dataset = SimpleFixedDataset(
        annotations_csv=config.TRAIN_CSV,
        video_dir=config.VIDEO_DIR,
        num_frames=config.NUM_FRAMES,
        image_size=config.IMAGE_SIZE,
        mode='train'
    )

    # Validation dataset
    val_dataset = SimpleFixedDataset(
        annotations_csv=config.VAL_CSV,
        video_dir=config.VIDEO_DIR,
        num_frames=config.NUM_FRAMES,
        image_size=config.IMAGE_SIZE,
        mode='val'
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    print(f"Train dataset: {len(train_dataset)} segments")
    print(f"Val dataset:   {len(val_dataset)} segments")
    print(f"Batch size:    {config.BATCH_SIZE}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print("=" * 70)
    print()

    return train_loader, val_loader


# Test the dataset
if __name__ == "__main__":
    from common.config import Config

    config = Config()

    print("Testing SimpleFixedDataset...")
    dataset = SimpleFixedDataset(
        annotations_csv=config.TRAIN_CSV,
        video_dir=config.VIDEO_DIR,
        num_frames=8,
        image_size=224,
        mode='train'
    )

    print(f"\nDataset size: {len(dataset)}")

    if len(dataset) > 0:
        print("\nLoading first sample...")
        frames, verb_label, noun_label = dataset[0]

        print(f"Frames shape: {frames.shape}")  # (8, 3, 224, 224)
        print(f"Verb label:   {verb_label}")
        print(f"Noun label:   {noun_label}")
        print("\n✓ Dataset test passed!")
    else:
        print("\n⚠ Warning: Dataset is empty!")
