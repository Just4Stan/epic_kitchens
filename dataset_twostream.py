"""
Two-Stream Dataset with Optical Flow Computation
Loads RGB frames + computes optical flow on-the-fly
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
import random


class TwoStreamEPICKitchensDataset(Dataset):
    """
    EPIC-KITCHENS dataset for two-stream model.
    Returns RGB frames + optical flow.
    """

    def __init__(self, annotations_csv, video_dir, num_frames=8,
                 image_size=224, mode='train'):
        """
        Args:
            annotations_csv: Path to CSV with annotations
            video_dir: Directory containing video files
            num_frames: Number of RGB frames to sample
            image_size: Size to resize frames to
            mode: 'train' or 'val'
        """
        self.annotations = pd.read_csv(annotations_csv)
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.image_size = image_size
        self.mode = mode

        print(f"Loading {'training' if mode == 'train' else 'validation'} dataset (Two-Stream)...")
        print(f"Found {len(self.annotations)} action segments")

        # Filter out missing videos
        self.valid_indices = []
        missing_count = 0

        for idx in range(len(self.annotations)):
            row = self.annotations.iloc[idx]
            video_path = self.video_dir / row['participant_id'] / f"{row['video_id']}.MP4"

            if video_path.exists():
                self.valid_indices.append(idx)
            else:
                missing_count += 1

        if missing_count > 0:
            print(f"⚠ Warning: {missing_count} videos not found (skipped)")

        self.annotations = self.annotations.iloc[self.valid_indices].reset_index(drop=True)
        print(f"Using {len(self.annotations)} valid action segments\\n")

        # Create transforms
        if mode == 'train':
            self.rgb_transform = self._get_train_transforms()
        else:
            self.rgb_transform = self._get_val_transforms()

        # Flow normalization (mean=0, normalize to [-1, 1])
        self.flow_mean = 0.0
        self.flow_std = 20.0  # Typical flow magnitude

    def _get_train_transforms(self):
        """Aggressive augmentations for RGB training."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(
                self.image_size,
                scale=(0.6, 1.0),
                ratio=(0.75, 1.33)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(
                p=0.3,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value='random'
            )
        ])

    def _get_val_transforms(self):
        """Standard transforms for validation."""
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
            rgb_frames: Tensor of shape (num_frames, 3, H, W)
            flow_stack: Tensor of shape (10, H, W) - stacked optical flow
            verb_label: Integer verb class
            noun_label: Integer noun class
        """
        row = self.annotations.iloc[idx]

        # Get video info
        video_path = self.video_dir / row['participant_id'] / f"{row['video_id']}.MP4"
        start_frame = row['start_frame']
        stop_frame = row['stop_frame']
        verb_class = row['verb_class']
        noun_class = row['noun_class']

        # Extract frames (raw, before augmentation)
        frames_raw = self._extract_frames(
            video_path,
            start_frame,
            stop_frame,
            self.num_frames
        )

        # Compute optical flow BEFORE augmentation (more stable)
        flow_stack = self._compute_and_stack_flow(frames_raw)

        # Apply RGB transforms
        rgb_frames = []
        for frame in frames_raw:
            rgb_frames.append(self.rgb_transform(frame))

        rgb_tensor = torch.stack(rgb_frames, dim=0)  # (num_frames, 3, H, W)

        return rgb_tensor, flow_stack, verb_class, noun_class

    def _extract_frames(self, video_path, start_frame, stop_frame, num_frames):
        """
        Extract frames with temporal jitter for training.

        Args:
            video_path: Path to video file
            start_frame: Start frame index
            stop_frame: Stop frame index
            num_frames: Number of frames to extract

        Returns:
            List of numpy arrays (frames) in RGB
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Temporal jitter for training
        if self.mode == 'train':
            segment_length = stop_frame - start_frame
            jitter_range = int(segment_length * 0.1)
            start_frame = max(0, start_frame + random.randint(-jitter_range, jitter_range))
            stop_frame = start_frame + segment_length

        # Sample frame indices uniformly
        frame_indices = np.linspace(start_frame, stop_frame, num_frames, dtype=int)

        # Per-frame jitter (training only)
        if self.mode == 'train':
            frame_jitter = 2
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
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        cap.release()
        return frames

    def _compute_and_stack_flow(self, frames):
        """
        Compute optical flow between consecutive frames and stack.

        Args:
            frames: List of RGB frames (H, W, 3)

        Returns:
            flow_stack: Tensor (10, H, W) - 5 flows * 2 (x, y)
        """
        # Compute flow between consecutive frames
        flows = []

        for i in range(len(frames) - 1):
            # Convert to grayscale
            prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)

            # Compute optical flow (Farneback)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            flows.append(flow)

        # Sample 5 flows uniformly from the computed flows
        num_flows = 5
        if len(flows) >= num_flows:
            indices = np.linspace(0, len(flows)-1, num_flows, dtype=int)
            selected_flows = [flows[i] for i in indices]
        else:
            # Repeat last flow if not enough
            selected_flows = flows + [flows[-1]] * (num_flows - len(flows))

        # Stack flows: [flow1_x, flow1_y, flow2_x, flow2_y, ...]
        flow_channels = []
        for flow in selected_flows:
            flow_x = flow[:, :, 0]
            flow_y = flow[:, :, 1]

            # Normalize flow
            flow_x = (flow_x - self.flow_mean) / self.flow_std
            flow_y = (flow_y - self.flow_mean) / self.flow_std

            # Clip to reasonable range
            flow_x = np.clip(flow_x, -1, 1)
            flow_y = np.clip(flow_y, -1, 1)

            # Resize to target size
            flow_x = cv2.resize(flow_x, (self.image_size, self.image_size))
            flow_y = cv2.resize(flow_y, (self.image_size, self.image_size))

            flow_channels.append(flow_x)
            flow_channels.append(flow_y)

        # Stack to (10, H, W)
        flow_stack = np.stack(flow_channels, axis=0)  # (10, H, W)

        return torch.from_numpy(flow_stack).float()


def get_twostream_dataloaders(config, batch_size=None, num_workers=None):
    """
    Create two-stream train and validation dataloaders.

    Args:
        config: Config object
        batch_size: Override config batch size
        num_workers: Override config num workers

    Returns:
        train_loader, val_loader
    """
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS

    # Create datasets
    train_dataset = TwoStreamEPICKitchensDataset(
        annotations_csv=config.TRAIN_CSV,
        video_dir=config.VIDEO_DIR,
        num_frames=config.NUM_FRAMES,
        image_size=config.IMAGE_SIZE,
        mode='train'
    )

    val_dataset = TwoStreamEPICKitchensDataset(
        annotations_csv=config.VAL_CSV,
        video_dir=config.VIDEO_DIR,
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
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Two-Stream DataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Batch size: {batch_size}, Num workers: {num_workers}\\n")

    return train_loader, val_loader


if __name__ == '__main__':
    # Test the dataset
    from config import Config

    config = Config()

    print("Testing Two-Stream Dataset...")
    train_dataset = TwoStreamEPICKitchensDataset(
        annotations_csv=config.TRAIN_CSV,
        video_dir=config.VIDEO_DIR,
        num_frames=8,
        image_size=224,
        mode='train'
    )

    # Test loading a sample
    rgb, flow, verb, noun = train_dataset[0]
    print(f"\\nSample loaded:")
    print(f"  RGB frames: {rgb.shape}")
    print(f"  Flow stack: {flow.shape}")
    print(f"  Verb class: {verb}")
    print(f"  Noun class: {noun}")
    print(f"\\n✓ Two-Stream dataset works correctly!")
