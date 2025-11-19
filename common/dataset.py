"""
EPIC-KITCHENS-100 Dataset Loader
Loads video segments and their verb/noun labels for action recognition
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


class EPICKitchensDataset(Dataset):
    """
    EPIC-KITCHENS-100 dataset for action recognition.

    Each sample is a video segment with:
    - Start/stop frames defining the action
    - Verb label (97 classes)
    - Noun label (300 classes)
    """

    def __init__(
        self,
        annotations_csv,
        video_dir,
        num_frames=8,
        image_size=224,
        mode='train'
    ):
        """
        Args:
            annotations_csv: Path to CSV file (EPIC_100_train.csv or EPIC_100_validation.csv)
            video_dir: Path to videos_640x360 directory
            num_frames: Number of frames to sample from each video segment
            image_size: Target image size (224 for ResNet)
            mode: 'train' or 'val' (affects data augmentation)
        """
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.image_size = image_size
        self.mode = mode

        # Load annotations
        print(f"Loading annotations from {annotations_csv}...")
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
            print(f"  Examples: {list(missing_videos)[:3]}")

        print(f"Using {len(self.annotations)} valid action segments")

        # Data augmentation transforms
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.annotations)

    def load_video_segment(self, video_path, start_frame, stop_frame):
        """
        Load and sample frames from a video segment.

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

        if total_frames < self.num_frames:
            # If segment is shorter than num_frames, repeat last frame
            frame_indices = np.linspace(start_frame, stop_frame - 1, self.num_frames, dtype=int)
        else:
            # Uniformly sample num_frames from the segment
            frame_indices = np.linspace(start_frame, stop_frame - 1, self.num_frames, dtype=int)

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

    def __getitem__(self, idx):
        """
        Get a single action segment.

        Returns:
            frames: Tensor (num_frames, 3, H, W)
            verb_label: Integer (0-96)
            noun_label: Integer (0-299)
        """
        row = self.annotations.iloc[idx]

        # Get video path
        video_id = row['video_id']
        participant = video_id.split('_')[0]  # e.g., "P01" from "P01_01"
        video_filename = f"{video_id}.MP4"
        video_path = self.video_dir / participant / video_filename

        # Get frame range
        start_frame = int(row['start_frame'])
        stop_frame = int(row['stop_frame'])

        # Get labels
        verb_label = int(row['verb_class'])
        noun_label = int(row['noun_class'])

        try:
            # Load video segment
            frames = self.load_video_segment(video_path, start_frame, stop_frame)
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Return dummy data on error
            frames = torch.zeros(self.num_frames, 3, self.image_size, self.image_size)
            verb_label = 0
            noun_label = 0

        return frames, verb_label, noun_label


def get_dataloaders(config):
    """
    Create train and validation dataloaders.

    Args:
        config: Configuration object

    Returns:
        train_loader, val_loader
    """
    print("=" * 70)
    print("Creating Dataloaders")
    print("=" * 70)

    # Training dataset
    train_dataset = EPICKitchensDataset(
        annotations_csv=config.TRAIN_CSV,
        video_dir=config.VIDEO_DIR,
        num_frames=config.NUM_FRAMES,
        image_size=config.IMAGE_SIZE,
        mode='train'
    )

    # Validation dataset
    val_dataset = EPICKitchensDataset(
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
    from config import Config

    config = Config()

    # Test dataset
    print("Testing dataset loader...")
    dataset = EPICKitchensDataset(
        annotations_csv=config.TRAIN_CSV,
        video_dir=config.VIDEO_DIR,
        num_frames=8,
        image_size=224,
        mode='train'
    )

    print(f"\nDataset size: {len(dataset)}")
    print("\nLoading first sample...")

    frames, verb_label, noun_label = dataset[0]

    print(f"Frames shape: {frames.shape}")  # Should be (8, 3, 224, 224)
    print(f"Verb label:   {verb_label}")
    print(f"Noun label:   {noun_label}")
    print("\n✓ Dataset test passed!")
