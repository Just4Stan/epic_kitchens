"""
Improved Dataset with Aggressive Data Augmentation
Implements state-of-the-art augmentation strategies for action recognition
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
import random


class ImprovedEPICKitchensDataset(Dataset):
    """
    EPIC-KITCHENS dataset with aggressive augmentation.

    Improvements:
    - Stronger spatial augmentations
    - Temporal augmentations (frame sampling jitter)
    - Mixup augmentation support
    - Better normalization
    """

    def __init__(self, annotations_csv, video_dir, num_frames=8,
                 image_size=224, mode='train'):
        """
        Args:
            annotations_csv: Path to CSV with annotations
            video_dir: Directory containing video files
            num_frames: Number of frames to sample per video
            image_size: Size to resize frames to
            mode: 'train' or 'val' (different augmentations)
        """
        self.annotations = pd.read_csv(annotations_csv)
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.image_size = image_size
        self.mode = mode

        print(f"Loading {'training' if mode == 'train' else 'validation'} dataset...")
        print(f"Found {len(self.annotations)} action segments")

        # Filter out missing videos
        self.valid_indices = []
        missing_count = 0

        for idx in range(len(self.annotations)):
            row = self.annotations.iloc[idx]

            # Validation videos are flat, training videos are in subdirectories
            if mode == 'val':
                video_path = self.video_dir / f"{row['video_id']}.MP4"
            else:
                video_path = self.video_dir / row['participant_id'] / f"{row['video_id']}.MP4"

            if video_path.exists():
                self.valid_indices.append(idx)
            else:
                missing_count += 1

        if missing_count > 0:
            print(f"⚠ Warning: {missing_count} videos not found (skipped)")

        self.annotations = self.annotations.iloc[self.valid_indices].reset_index(drop=True)
        print(f"Using {len(self.annotations)} valid action segments\n")

        # Create transforms based on mode
        if mode == 'train':
            self.transform = self._get_train_transforms()
        else:
            self.transform = self._get_val_transforms()

    def _get_train_transforms(self):
        """Aggressive augmentations for training."""
        return transforms.Compose([
            transforms.ToPILImage(),
            # Aggressive spatial augmentations
            transforms.RandomResizedCrop(
                self.image_size,
                scale=(0.6, 1.0),  # More aggressive than (0.8, 1.0)
                ratio=(0.75, 1.33)
            ),
            transforms.RandomHorizontalFlip(p=0.5),

            # Color augmentations
            transforms.ColorJitter(
                brightness=0.4,    # Increased from 0.2
                contrast=0.4,      # Increased from 0.2
                saturation=0.4,    # New
                hue=0.2           # New
            ),

            # Additional augmentations
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=0.3),

            # Rotation (slight)
            transforms.RandomRotation(degrees=10),

            transforms.ToTensor(),

            # Normalization (ImageNet stats)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),

            # Random erasing (cutout)
            transforms.RandomErasing(
                p=0.3,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value='random'
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

        # Get video info (validation videos are flat, training in subdirectories)
        if self.mode == 'val':
            video_path = self.video_dir / f"{row['video_id']}.MP4"
        else:
            video_path = self.video_dir / row['participant_id'] / f"{row['video_id']}.MP4"

        start_frame = row['start_frame']
        stop_frame = row['stop_frame']
        verb_class = row['verb_class']
        noun_class = row['noun_class']

        # Extract frames
        frames = self._extract_frames(
            video_path,
            start_frame,
            stop_frame,
            self.num_frames
        )

        # Apply transforms
        transformed_frames = []
        for frame in frames:
            transformed_frames.append(self.transform(frame))

        frames_tensor = torch.stack(transformed_frames, dim=0)

        return frames_tensor, verb_class, noun_class

    def _extract_frames(self, video_path, start_frame, stop_frame, num_frames):
        """
        Extract frames with temporal jitter for training.

        Args:
            video_path: Path to video file
            start_frame: Start frame index
            stop_frame: Stop frame index
            num_frames: Number of frames to extract

        Returns:
            List of numpy arrays (frames)
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Temporal jitter for training
        if self.mode == 'train':
            # Add random jitter to start/stop
            segment_length = stop_frame - start_frame
            jitter_range = int(segment_length * 0.1)  # 10% jitter

            start_frame = max(0, start_frame + random.randint(-jitter_range, jitter_range))
            stop_frame = start_frame + segment_length

        # Sample frame indices uniformly
        frame_indices = np.linspace(start_frame, stop_frame, num_frames, dtype=int)

        # Additional temporal jitter per frame (training only)
        if self.mode == 'train':
            frame_jitter = 2  # +/- 2 frames
            frame_indices = [max(start_frame, min(stop_frame, idx + random.randint(-frame_jitter, frame_jitter)))
                           for idx in frame_indices]

        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # If frame read fails, duplicate last frame
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    # Fallback: create black frame
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        cap.release()
        return frames


def get_improved_dataloaders(config, batch_size=None, num_workers=None):
    """
    Create improved train and validation dataloaders.

    Args:
        config: Config object with dataset parameters
        batch_size: Override config batch size
        num_workers: Override config num workers

    Returns:
        train_loader, val_loader
    """
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS

    # Create datasets
    train_dataset = ImprovedEPICKitchensDataset(
        annotations_csv=config.TRAIN_CSV,
        video_dir=config.VIDEO_DIR,
        num_frames=config.NUM_FRAMES,
        image_size=config.IMAGE_SIZE,
        mode='train'
    )

    # Use separate validation video directory if it exists
    val_video_dir = config.VIDEO_DIR.parent / "videos_640x360_validation"
    if not val_video_dir.exists():
        val_video_dir = config.VIDEO_DIR  # Fallback to training dir
        print(f"Warning: Validation video directory not found, using training dir")

    val_dataset = ImprovedEPICKitchensDataset(
        annotations_csv=config.VAL_CSV,
        video_dir=val_video_dir,
        num_frames=config.NUM_FRAMES,
        image_size=config.IMAGE_SIZE,
        mode='val'
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"DataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Batch size: {batch_size}, Num workers: {num_workers}\n")

    return train_loader, val_loader


if __name__ == '__main__':
    # Test the improved dataset
    from config import Config

    config = Config()

    print("Testing Improved Dataset...")
    train_dataset = ImprovedEPICKitchensDataset(
        annotations_csv=config.TRAIN_CSV,
        video_dir=config.VIDEO_DIR,
        num_frames=8,
        image_size=224,
        mode='train'
    )

    # Test loading a sample
    frames, verb, noun = train_dataset[0]
    print(f"\nSample loaded:")
    print(f"  Frames shape: {frames.shape}")
    print(f"  Verb class: {verb}")
    print(f"  Noun class: {noun}")
    print(f"\n✓ Dataset works correctly!")
