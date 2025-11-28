"""
EPIC-KITCHENS-100 Action Recognition - V2 Datasets
===================================================
Enhanced data pipeline with 320x320 resolution and all augmentations.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
import random


# =============================================================================
# AUGMENTATION TRANSFORMS
# =============================================================================

class RandomErasing:
    """
    Random Erasing augmentation.
    Randomly erases a rectangular region in the image.
    """

    def __init__(self, p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img):
        if random.random() > self.p:
            return img

        h, w = img.shape[-2:]
        area = h * w

        for _ in range(10):
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            eh = int(round(np.sqrt(target_area * aspect_ratio)))
            ew = int(round(np.sqrt(target_area / aspect_ratio)))

            if eh < h and ew < w:
                i = random.randint(0, h - eh)
                j = random.randint(0, w - ew)
                img[:, i:i+eh, j:j+ew] = self.value
                return img

        return img


class TemporalConsistentTransform:
    """
    Apply the same random transform to all frames in a video.
    Ensures temporal consistency for augmentations.
    """

    def __init__(self, image_size=320, augmentation='heavy'):
        self.image_size = image_size
        self.augmentation = augmentation

        # Normalization (CLIP stats - critical for proper CLIP feature extraction!)
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )

        # Color jitter settings
        self.color_jitter = {
            'none': None,
            'light': transforms.ColorJitter(brightness=0.2, contrast=0.2),
            'medium': transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            'heavy': transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15)
        }.get(augmentation)

    def __call__(self, frames):
        """
        Args:
            frames: List of numpy arrays (H, W, C) in RGB

        Returns:
            Tensor of shape (T, C, H, W)
        """
        T = len(frames)

        # Convert to PIL for transforms
        pil_frames = [transforms.ToPILImage()(f) for f in frames]

        # Get original size
        orig_w, orig_h = pil_frames[0].size

        if self.augmentation == 'none':
            # Just resize and normalize
            transformed = []
            for frame in pil_frames:
                frame = transforms.Resize((self.image_size, self.image_size))(frame)
                frame = transforms.ToTensor()(frame)
                frame = self.normalize(frame)
                transformed.append(frame)
            return torch.stack(transformed)

        # ===== Random parameters (same for all frames) =====

        # Random resized crop parameters
        scale = random.uniform(0.6, 1.0) if self.augmentation in ['medium', 'heavy'] else random.uniform(0.8, 1.0)
        ratio = random.uniform(0.75, 1.33)

        # Calculate crop size
        area = orig_h * orig_w
        target_area = area * scale
        w = int(round(np.sqrt(target_area * ratio)))
        h = int(round(np.sqrt(target_area / ratio)))

        if w > orig_w:
            w = orig_w
            h = int(w / ratio)
        if h > orig_h:
            h = orig_h
            w = int(h * ratio)

        # Random crop position
        i = random.randint(0, max(0, orig_h - h))
        j = random.randint(0, max(0, orig_w - w))

        # Random horizontal flip
        do_hflip = random.random() < 0.5

        # Random grayscale
        do_grayscale = random.random() < (0.2 if self.augmentation == 'heavy' else 0.1)

        # Color jitter parameters (if enabled)
        if self.color_jitter is not None:
            # Get random jitter values
            fn_idx = torch.randperm(4)
            b = random.uniform(max(0, 1 - 0.4), 1 + 0.4) if self.augmentation == 'heavy' else random.uniform(max(0, 1 - 0.3), 1 + 0.3)
            c = random.uniform(max(0, 1 - 0.4), 1 + 0.4) if self.augmentation == 'heavy' else random.uniform(max(0, 1 - 0.3), 1 + 0.3)
            s = random.uniform(max(0, 1 - 0.3), 1 + 0.3) if self.augmentation == 'heavy' else random.uniform(max(0, 1 - 0.2), 1 + 0.2)
            hue = random.uniform(-0.15, 0.15) if self.augmentation == 'heavy' else random.uniform(-0.1, 0.1)

        # ===== Apply same transforms to all frames =====
        transformed = []
        for frame in pil_frames:
            # Crop
            frame = TF.crop(frame, i, j, h, w)

            # Resize to target size
            frame = TF.resize(frame, (self.image_size, self.image_size))

            # Horizontal flip
            if do_hflip:
                frame = TF.hflip(frame)

            # Color jitter (apply same parameters)
            if self.color_jitter is not None:
                for fn_id in fn_idx:
                    if fn_id == 0:
                        frame = TF.adjust_brightness(frame, b)
                    elif fn_id == 1:
                        frame = TF.adjust_contrast(frame, c)
                    elif fn_id == 2:
                        frame = TF.adjust_saturation(frame, s)
                    elif fn_id == 3:
                        frame = TF.adjust_hue(frame, hue)

            # Grayscale
            if do_grayscale:
                frame = TF.rgb_to_grayscale(frame, num_output_channels=3)

            # To tensor and normalize
            frame = transforms.ToTensor()(frame)
            frame = self.normalize(frame)

            transformed.append(frame)

        return torch.stack(transformed)


class ValTransform:
    """Validation transform - just resize and normalize."""

    def __init__(self, image_size=320):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, frames):
        return torch.stack([self.transform(f) for f in frames])


# =============================================================================
# CUTMIX AND MIXUP
# =============================================================================

def cutmix_temporal(frames, verb_labels, noun_labels, alpha=1.0):
    """
    CutMix with temporal consistency - same spatial region across all frames.

    Args:
        frames: (B, T, C, H, W)
        verb_labels, noun_labels: (B,)
        alpha: Beta distribution parameter

    Returns:
        mixed_frames, verb_a, verb_b, noun_a, noun_b, lam
    """
    batch_size = frames.size(0)
    lam = np.random.beta(alpha, alpha)

    rand_index = torch.randperm(batch_size).to(frames.device)

    _, T, C, H, W = frames.shape

    # Calculate cut size
    cut_rat = np.sqrt(1 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Random center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Bounding box (same for ALL frames)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply CutMix
    frames_mixed = frames.clone()
    frames_mixed[:, :, :, y1:y2, x1:x2] = frames[rand_index, :, :, y1:y2, x1:x2]

    # Adjust lambda to actual cut ratio
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

    return (frames_mixed,
            verb_labels, verb_labels[rand_index],
            noun_labels, noun_labels[rand_index],
            lam)


def mixup_temporal(frames, verb_labels, noun_labels, alpha=0.2):
    """
    MixUp for video - blends entire frames.

    Args:
        frames: (B, T, C, H, W)
        verb_labels, noun_labels: (B,)
        alpha: Beta distribution parameter

    Returns:
        mixed_frames, verb_a, verb_b, noun_a, noun_b, lam
    """
    batch_size = frames.size(0)
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure primary dominates

    rand_index = torch.randperm(batch_size).to(frames.device)
    frames_mixed = lam * frames + (1 - lam) * frames[rand_index]

    return (frames_mixed,
            verb_labels, verb_labels[rand_index],
            noun_labels, noun_labels[rand_index],
            lam)


def apply_random_erasing(frames, p=0.25, scale=(0.02, 0.33)):
    """
    Apply random erasing to video frames.
    Same region erased across all frames for temporal consistency.
    """
    if random.random() > p:
        return frames

    B, T, C, H, W = frames.shape

    # Random erasing parameters (same for all frames in batch)
    for b in range(B):
        area = H * W
        target_area = random.uniform(scale[0], scale[1]) * area
        aspect_ratio = random.uniform(0.3, 3.3)

        eh = int(round(np.sqrt(target_area * aspect_ratio)))
        ew = int(round(np.sqrt(target_area / aspect_ratio)))

        if eh < H and ew < W:
            i = random.randint(0, H - eh)
            j = random.randint(0, W - ew)
            frames[b, :, :, i:i+eh, j:j+ew] = 0  # Erase same region across all frames

    return frames


# =============================================================================
# DATASETS
# =============================================================================

class TrainDataset(Dataset):
    """
    Training dataset with full augmentation pipeline.
    Uses pre-extracted frames from official EPIC-KITCHENS RGB frames.
    """

    def __init__(
        self,
        annotations_csv,
        frames_dir,
        num_frames=32,
        image_size=320,
        augmentation='heavy',
        frame_interval=3
    ):
        self.annotations = pd.read_csv(annotations_csv)
        self.frames_dir = Path(frames_dir)
        self.num_frames = num_frames
        self.image_size = image_size
        self.frame_interval = frame_interval

        # Build valid action list
        self.valid_actions = []
        for idx in range(len(self.annotations)):
            row = self.annotations.iloc[idx]
            participant_id = row['participant_id']
            video_id = row['video_id']

            # Check if frames exist
            frame_dir = self.frames_dir / participant_id / "rgb_frames" / video_id
            if frame_dir.exists():
                self.valid_actions.append(idx)

        print(f"Training: {len(self.valid_actions)} / {len(self.annotations)} actions with frames")

        # Transform
        self.transform = TemporalConsistentTransform(
            image_size=image_size,
            augmentation=augmentation
        )

    def _load_frames(self, participant_id, video_id, start_frame, stop_frame):
        """Load frames from EPIC-KITCHENS format."""
        frame_dir = self.frames_dir / participant_id / "rgb_frames" / video_id

        # Calculate frame indices with interval
        total_frames = stop_frame - start_frame + 1
        if total_frames <= self.num_frames:
            indices = list(range(start_frame, stop_frame + 1))
            # Pad with last frame
            while len(indices) < self.num_frames:
                indices.append(stop_frame)
        else:
            # Sample with interval
            step = max(1, total_frames // self.num_frames)
            indices = list(range(start_frame, stop_frame + 1, step))[:self.num_frames]
            # Pad if needed
            while len(indices) < self.num_frames:
                indices.append(indices[-1])

        # Load frames
        frames = []
        for frame_idx in indices:
            # EPIC-KITCHENS frame format: frame_0000000001.jpg
            frame_path = frame_dir / f"frame_{frame_idx:010d}.jpg"
            if frame_path.exists():
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

            if not frames:
                # Fallback: try to find any frame
                any_frames = sorted(frame_dir.glob("frame_*.jpg"))
                if any_frames:
                    frame = cv2.imread(str(any_frames[0]))
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)

        # Pad with last frame or zeros
        while len(frames) < self.num_frames:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((456, 256, 3), dtype=np.uint8))

        return frames[:self.num_frames]

    def __len__(self):
        return len(self.valid_actions)

    def __getitem__(self, idx):
        real_idx = self.valid_actions[idx]
        row = self.annotations.iloc[real_idx]

        participant_id = row['participant_id']
        video_id = row['video_id']
        start_frame = int(row['start_frame'])
        stop_frame = int(row['stop_frame'])

        frames = self._load_frames(participant_id, video_id, start_frame, stop_frame)
        frames_tensor = self.transform(frames)

        return frames_tensor, int(row['verb_class']), int(row['noun_class'])


class ValDataset(Dataset):
    """Validation dataset with minimal augmentation."""

    def __init__(
        self,
        annotations_csv,
        frames_dir,
        num_frames=32,
        image_size=320
    ):
        self.annotations = pd.read_csv(annotations_csv)
        self.frames_dir = Path(frames_dir)
        self.num_frames = num_frames
        self.image_size = image_size

        # Build valid action list
        self.valid_actions = []
        for idx in range(len(self.annotations)):
            row = self.annotations.iloc[idx]
            participant_id = row['participant_id']
            video_id = row['video_id']

            frame_dir = self.frames_dir / participant_id / "rgb_frames" / video_id
            if frame_dir.exists():
                self.valid_actions.append(idx)

        print(f"Validation: {len(self.valid_actions)} / {len(self.annotations)} actions with frames")

        self.transform = ValTransform(image_size=image_size)

    def _load_frames(self, participant_id, video_id, start_frame, stop_frame):
        """Load frames from EPIC-KITCHENS format."""
        frame_dir = self.frames_dir / participant_id / "rgb_frames" / video_id

        total_frames = stop_frame - start_frame + 1
        if total_frames <= self.num_frames:
            indices = list(range(start_frame, stop_frame + 1))
            while len(indices) < self.num_frames:
                indices.append(stop_frame)
        else:
            indices = np.linspace(start_frame, stop_frame, self.num_frames, dtype=int).tolist()

        frames = []
        for frame_idx in indices:
            frame_path = frame_dir / f"frame_{frame_idx:010d}.jpg"
            if frame_path.exists():
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

            if not frames:
                any_frames = sorted(frame_dir.glob("frame_*.jpg"))
                if any_frames:
                    frame = cv2.imread(str(any_frames[0]))
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)

        while len(frames) < self.num_frames:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((456, 256, 3), dtype=np.uint8))

        return frames[:self.num_frames]

    def __len__(self):
        return len(self.valid_actions)

    def __getitem__(self, idx):
        real_idx = self.valid_actions[idx]
        row = self.annotations.iloc[real_idx]

        participant_id = row['participant_id']
        video_id = row['video_id']
        start_frame = int(row['start_frame'])
        stop_frame = int(row['stop_frame'])

        frames = self._load_frames(participant_id, video_id, start_frame, stop_frame)
        frames_tensor = self.transform(frames)

        return frames_tensor, int(row['verb_class']), int(row['noun_class'])


# =============================================================================
# CLASS COUNTS FOR LOSS WEIGHTING
# =============================================================================

def get_class_counts(csv_path, num_verb_classes=97, num_noun_classes=300):
    """Count class occurrences for computing loss weights."""
    df = pd.read_csv(csv_path)

    verb_counts = torch.zeros(num_verb_classes)
    noun_counts = torch.zeros(num_noun_classes)

    for _, row in df.iterrows():
        verb_counts[int(row['verb_class'])] += 1
        noun_counts[int(row['noun_class'])] += 1

    # Add 1 to avoid division by zero
    verb_counts = verb_counts + 1
    noun_counts = noun_counts + 1

    return verb_counts, noun_counts
