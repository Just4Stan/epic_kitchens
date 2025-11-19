"""
Validation script using heavily augmented training videos
Tests model robustness and overfitting by applying transformations not seen during training
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from torchvision import transforms
import sys

sys.path.append(str(Path(__file__).parent.parent))

from common.config import Config
from phase1.model import get_model


class AugmentedEPICKitchensDataset(Dataset):
    """
    EPIC-KITCHENS dataset with HEAVY augmentations not seen during training.
    This simulates out-of-distribution data to test overfitting.
    """

    def __init__(self, annotations_csv, video_dir, num_frames=8, image_size=224):
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.image_size = image_size

        # Load annotations
        self.annotations = pd.read_csv(annotations_csv)

        # Filter out missing videos
        valid_indices = []
        for idx, row in self.annotations.iterrows():
            video_id = row['video_id']
            participant = video_id.split('_')[0]
            video_path = self.video_dir / participant / f"{video_id}.MP4"
            if video_path.exists():
                valid_indices.append(idx)

        self.annotations = self.annotations.iloc[valid_indices].reset_index(drop=True)
        print(f"Using {len(self.annotations)} segments for augmented validation")

        # EXTREME augmentations to test robustness
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            # Much stronger augmentations than training
            transforms.RandomHorizontalFlip(p=1.0),  # Always flip
            transforms.RandomRotation(degrees=15),  # Rotation (not in training)
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),  # Stronger color jitter
            transforms.RandomGrayscale(p=0.3),  # Sometimes grayscale
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # Blur (not in training)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),  # Random erasing
        ])

    def __len__(self):
        return len(self.annotations)

    def load_video_segment(self, video_path, start_frame, stop_frame):
        """Load and sample frames from a video segment with heavy augmentation."""
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = stop_frame - start_frame
        if total_frames <= 0:
            raise ValueError(f"Invalid frame range: {start_frame} to {stop_frame}")

        frame_indices = np.linspace(start_frame, stop_frame - 1, self.num_frames, dtype=int)

        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                if len(frames) > 0:
                    frames.append(frames[-1])
                else:
                    frames.append(torch.zeros(3, self.image_size, self.image_size))
                continue

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply extreme transforms
            frame = self.transform(frame)
            frames.append(frame)

        cap.release()
        frames = torch.stack(frames, dim=0)
        return frames

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        video_id = row['video_id']
        participant = video_id.split('_')[0]
        video_filename = f"{video_id}.MP4"
        video_path = self.video_dir / participant / video_filename

        start_frame = int(row['start_frame'])
        stop_frame = int(row['stop_frame'])
        verb_label = int(row['verb_class'])
        noun_label = int(row['noun_class'])

        try:
            frames = self.load_video_segment(video_path, start_frame, stop_frame)
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            frames = torch.zeros(self.num_frames, 3, self.image_size, self.image_size)
            verb_label = 0
            noun_label = 0

        return frames, verb_label, noun_label


def validate_checkpoint(checkpoint_path, config, num_samples=500):
    """
    Validate checkpoint on heavily augmented data.

    Args:
        checkpoint_path: Path to checkpoint
        config: Config object
        num_samples: Number of samples to test (default 500)
    """
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: {device} (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    print(f"Checkpoint from epoch: {checkpoint['epoch']}")
    if 'best_acc' in checkpoint:
        print(f"Best accuracy so far: {checkpoint['best_acc']:.2f}%")

    # Initialize model
    print("\nInitializing model...")
    model = get_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create heavily augmented dataset
    print("\nCreating heavily augmented validation dataset...")
    print("Augmentations applied:")
    print("  - Always horizontal flip")
    print("  - Random rotation (±15°)")
    print("  - Strong color jitter")
    print("  - Random grayscale (30%)")
    print("  - Gaussian blur")
    print("  - Random erasing (30%)")

    aug_dataset = AugmentedEPICKitchensDataset(
        annotations_csv=config.TRAIN_CSV,
        video_dir=config.VIDEO_DIR,
        num_frames=config.NUM_FRAMES,
        image_size=config.IMAGE_SIZE
    )

    # Limit to num_samples
    if len(aug_dataset) > num_samples:
        indices = np.random.choice(len(aug_dataset), num_samples, replace=False)
        aug_dataset.annotations = aug_dataset.annotations.iloc[indices].reset_index(drop=True)
        print(f"Randomly selected {num_samples} samples for testing")

    aug_loader = DataLoader(
        aug_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    print(f"Testing on {len(aug_dataset)} heavily augmented samples")

    # Loss functions
    criterion_verb = nn.CrossEntropyLoss()
    criterion_noun = nn.CrossEntropyLoss()

    # Validate
    print("\nRunning validation on HEAVILY AUGMENTED data...")
    total_loss = 0
    verb_correct = 0
    noun_correct = 0
    action_correct = 0
    total_samples = 0

    verb_top5_correct = 0
    noun_top5_correct = 0

    with torch.no_grad():
        pbar = tqdm(aug_loader, desc="Validating")

        for frames, verb_labels, noun_labels in pbar:
            frames = frames.to(device)
            verb_labels = verb_labels.to(device)
            noun_labels = noun_labels.to(device)

            # Forward pass
            if config.USE_AMP:
                with autocast():
                    verb_logits, noun_logits = model(frames)
                    loss_verb = criterion_verb(verb_logits, verb_labels)
                    loss_noun = criterion_noun(noun_logits, noun_labels)
                    loss = config.VERB_LOSS_WEIGHT * loss_verb + config.NOUN_LOSS_WEIGHT * loss_noun
            else:
                verb_logits, noun_logits = model(frames)
                loss_verb = criterion_verb(verb_logits, verb_labels)
                loss_noun = criterion_noun(noun_logits, noun_labels)
                loss = config.VERB_LOSS_WEIGHT * loss_verb + config.NOUN_LOSS_WEIGHT * loss_noun

            # Top-1 predictions
            verb_pred = verb_logits.argmax(dim=1)
            noun_pred = noun_logits.argmax(dim=1)

            # Top-5 predictions
            _, verb_top5_pred = verb_logits.topk(5, dim=1)
            _, noun_top5_pred = noun_logits.topk(5, dim=1)

            # Calculate accuracies
            verb_correct += (verb_pred == verb_labels).sum().item()
            noun_correct += (noun_pred == noun_labels).sum().item()
            action_correct += ((verb_pred == verb_labels) & (noun_pred == noun_labels)).sum().item()

            # Top-5 accuracy
            verb_top5_correct += sum([1 for i, label in enumerate(verb_labels) if label in verb_top5_pred[i]])
            noun_top5_correct += sum([1 for i, label in enumerate(noun_labels) if label in noun_top5_pred[i]])

            total_loss += loss.item()
            total_samples += frames.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'verb_acc': f'{100 * verb_correct / total_samples:.2f}%',
                'noun_acc': f'{100 * noun_correct / total_samples:.2f}%'
            })

    # Final statistics
    avg_loss = total_loss / len(aug_loader)
    verb_acc = 100 * verb_correct / total_samples
    noun_acc = 100 * noun_correct / total_samples
    action_acc = 100 * action_correct / total_samples
    verb_top5_acc = 100 * verb_top5_correct / total_samples
    noun_top5_acc = 100 * noun_top5_correct / total_samples

    # Print results
    print("\n" + "="*60)
    print("VALIDATION RESULTS (HEAVILY AUGMENTED DATA)")
    print("="*60)
    print(f"Validation Loss:    {avg_loss:.4f}")
    print(f"\nTop-1 Accuracy:")
    print(f"  Verb:             {verb_acc:.2f}%")
    print(f"  Noun:             {noun_acc:.2f}%")
    print(f"  Action (V+N):     {action_acc:.2f}%")
    print(f"\nTop-5 Accuracy:")
    print(f"  Verb:             {verb_top5_acc:.2f}%")
    print(f"  Noun:             {noun_top5_acc:.2f}%")
    print("="*60)

    # Overfitting analysis
    print("\nOVERFITTING / ROBUSTNESS ANALYSIS:")
    print("  Training accuracy at epoch 11: ~76% verb, ~78% noun")
    print(f"  Augmented data accuracy: {verb_acc:.2f}% verb, {noun_acc:.2f}% noun")

    estimated_train_verb = 76.0
    estimated_train_noun = 78.0

    verb_drop = estimated_train_verb - verb_acc
    noun_drop = estimated_train_noun - noun_acc

    print(f"\n  Performance drop on augmented data:")
    print(f"    Verb: {estimated_train_verb:.1f}% → {verb_acc:.2f}% ({verb_drop:+.2f}%)")
    print(f"    Noun: {estimated_train_noun:.1f}% → {noun_acc:.2f}% ({noun_drop:+.2f}%)")

    if verb_drop > 30 or noun_drop > 30:
        print(f"\n⚠️  SEVERE overfitting detected!")
        print(f"  Model fails badly on augmented data - memorizing training set.")
        print(f"  The model is NOT robust to variations.")
    elif verb_drop > 20 or noun_drop > 20:
        print(f"\n⚠️  WARNING: Significant overfitting detected!")
        print(f"  Model struggles with augmented data.")
        print(f"  Consider: More augmentation during training, regularization.")
    elif verb_drop > 10 or noun_drop > 10:
        print(f"\n⚠️  Moderate overfitting detected.")
        print(f"  Some performance drop on augmented data - expected behavior.")
    else:
        print(f"\n✓ Excellent robustness - model handles augmentations well!")

    print("\nNote: Performance drop is EXPECTED on heavily augmented data.")
    print("This tests if the model learned robust features vs. memorizing pixels.")
    print("="*60)

    return {
        'val_loss': avg_loss,
        'val_verb_acc': verb_acc,
        'val_noun_acc': noun_acc,
        'val_action_acc': action_acc,
        'val_verb_top5_acc': verb_top5_acc,
        'val_noun_top5_acc': noun_top5_acc,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate checkpoint on augmented data')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for validation')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples to test (default 500)')

    args = parser.parse_args()

    # Load config
    config = Config()
    config.BATCH_SIZE = args.batch_size

    # Validate
    validate_checkpoint(args.checkpoint, config, args.num_samples)
