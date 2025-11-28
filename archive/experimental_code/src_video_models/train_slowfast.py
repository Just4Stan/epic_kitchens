"""
SlowFast Networks for EPIC-KITCHENS-100
Efficient 3D CNN with two pathways (slow + fast)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
import argparse
from tqdm import tqdm
import wandb
from torchvision import transforms
from typing import Tuple
import torchvision.models.video as video_models


class EPICKitchensSlowFastDataset(Dataset):
    """EPIC-KITCHENS dataset for SlowFast"""

    def __init__(
        self,
        annotations_file: str,
        frames_dir: str,
        num_frames: int = 32,  # Total frames to sample
        alpha: int = 8,  # Slow/fast ratio
        is_train: bool = True
    ):
        self.annotations = pd.read_csv(annotations_file)
        self.frames_dir = Path(frames_dir)
        self.num_frames = num_frames
        self.alpha = alpha
        self.is_train = is_train

        # Transforms
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
            ])

        # Filter out invalid samples
        self.annotations = self.annotations[
            self.annotations['stop_frame'] - self.annotations['start_frame'] >= num_frames
        ].reset_index(drop=True)

        print(f"Loaded {len(self.annotations)} samples from {annotations_file}")

    def __len__(self):
        return len(self.annotations)

    def _load_frames(self, video_id: str, start_frame: int, stop_frame: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load frames for slow and fast pathways"""
        participant_id = video_id.split('_')[0]
        video_dir = self.frames_dir / participant_id / video_id

        # Sample frames
        total_frames = stop_frame - start_frame
        if total_frames < self.num_frames:
            indices = np.linspace(start_frame, stop_frame - 1, self.num_frames, dtype=int)
        else:
            # Dense temporal sampling
            if self.is_train:
                # Random temporal crop
                max_start = total_frames - self.num_frames
                temporal_start = np.random.randint(0, max_start + 1)
                indices = np.arange(start_frame + temporal_start, start_frame + temporal_start + self.num_frames)
            else:
                # Center crop
                indices = np.linspace(start_frame, stop_frame - 1, self.num_frames, dtype=int)

        # Load frames
        frames = []
        for idx in indices:
            frame_path = video_dir / f"frame_{idx:010d}.jpg"
            try:
                frame = Image.open(frame_path).convert('RGB')
                frame = self.transform(frame)
                frames.append(frame)
            except:
                # If frame doesn't exist, use previous frame or black frame
                if frames:
                    frames.append(frames[-1])
                else:
                    black = torch.zeros(3, 224, 224)
                    frames.append(black)

        frames = torch.stack(frames)  # (T, C, H, W)

        # Create slow and fast pathways
        # Slow: every alpha frames
        slow_indices = torch.arange(0, self.num_frames, self.alpha)
        slow_frames = frames[slow_indices]  # (T//alpha, C, H, W)

        # Fast: all frames
        fast_frames = frames  # (T, C, H, W)

        return slow_frames, fast_frames

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        # Load slow and fast frames
        slow_frames, fast_frames = self._load_frames(
            row['video_id'],
            row['start_frame'],
            row['stop_frame']
        )

        return {
            'slow_frames': slow_frames.transpose(0, 1),  # (C, T, H, W)
            'fast_frames': fast_frames.transpose(0, 1),  # (C, T, H, W)
            'verb_class': torch.tensor(row['verb_class'], dtype=torch.long),
            'noun_class': torch.tensor(row['noun_class'], dtype=torch.long),
        }


class SlowFastDualHead(nn.Module):
    """SlowFast with dual classification heads"""

    def __init__(
        self,
        num_verb_classes: int = 97,
        num_noun_classes: int = 300,
        dropout: float = 0.5,
        pretrained: bool = True
    ):
        super().__init__()

        # Load pretrained SlowFast from torchvision
        # Note: torchvision doesn't have SlowFast, so we'll use R(2+1)D as a strong baseline
        # R(2+1)D is similar in spirit - uses factorized spatiotemporal convolutions
        print("Loading pretrained R(2+1)D (efficient 3D CNN baseline)...")
        self.backbone = video_models.r2plus1d_18(pretrained=pretrained)

        # Get feature dim
        feature_dim = self.backbone.fc.in_features

        # Remove original fc layer
        self.backbone.fc = nn.Identity()

        # Dual classification heads
        self.verb_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_verb_classes)
        )

        self.noun_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_noun_classes)
        )

        self._print_model_info()

    def _print_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "="*60)
        print("R(2+1)D DUAL HEAD MODEL")
        print("="*60)
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("="*60 + "\n")

    def forward(self, slow_frames, fast_frames):
        """
        Args:
            slow_frames: (B, C, T_slow, H, W)
            fast_frames: (B, C, T_fast, H, W)
        """
        # For R(2+1)D, we'll use the fast pathway (more frames)
        # In a true SlowFast implementation, you'd fuse both pathways

        # Extract features
        features = self.backbone(fast_frames)  # (B, feature_dim)

        # Dual heads
        verb_logits = self.verb_head(features)
        noun_logits = self.noun_head(features)

        return verb_logits, noun_logits


def calculate_accuracy(verb_preds, noun_preds, verb_labels, noun_labels):
    """Calculate verb, noun, and action accuracy"""
    verb_correct = (verb_preds == verb_labels).float().sum()
    noun_correct = (noun_preds == noun_labels).float().sum()
    action_correct = ((verb_preds == verb_labels) & (noun_preds == noun_labels)).float().sum()

    total = len(verb_labels)

    return {
        'verb': (verb_correct / total * 100).item(),
        'noun': (noun_correct / total * 100).item(),
        'action': (action_correct / total * 100).item()
    }


def train_epoch(model, dataloader, optimizer, device, verb_weight=0.4, noun_weight=0.6):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_verb_preds, all_noun_preds = [], []
    all_verb_labels, all_noun_labels = [], []

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        slow_frames = batch['slow_frames'].to(device)
        fast_frames = batch['fast_frames'].to(device)
        verb_labels = batch['verb_class'].to(device)
        noun_labels = batch['noun_class'].to(device)

        # Forward
        verb_logits, noun_logits = model(slow_frames, fast_frames)

        # Loss
        verb_loss = F.cross_entropy(verb_logits, verb_labels)
        noun_loss = F.cross_entropy(noun_logits, noun_labels)
        loss = verb_weight * verb_loss + noun_weight * noun_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        verb_preds = verb_logits.argmax(dim=1)
        noun_preds = noun_logits.argmax(dim=1)

        all_verb_preds.append(verb_preds.cpu())
        all_noun_preds.append(noun_preds.cpu())
        all_verb_labels.append(verb_labels.cpu())
        all_noun_labels.append(noun_labels.cpu())

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    # Calculate accuracy
    all_verb_preds = torch.cat(all_verb_preds)
    all_noun_preds = torch.cat(all_noun_preds)
    all_verb_labels = torch.cat(all_verb_labels)
    all_noun_labels = torch.cat(all_noun_labels)

    acc = calculate_accuracy(all_verb_preds, all_noun_preds, all_verb_labels, all_noun_labels)

    return total_loss / len(dataloader), acc


@torch.no_grad()
def validate(model, dataloader, device, verb_weight=0.4, noun_weight=0.6):
    """Validate"""
    model.eval()
    total_loss = 0
    all_verb_preds, all_noun_preds = [], []
    all_verb_labels, all_noun_labels = [], []

    pbar = tqdm(dataloader, desc="Validation")
    for batch in pbar:
        slow_frames = batch['slow_frames'].to(device)
        fast_frames = batch['fast_frames'].to(device)
        verb_labels = batch['verb_class'].to(device)
        noun_labels = batch['noun_class'].to(device)

        # Forward
        verb_logits, noun_logits = model(slow_frames, fast_frames)

        # Loss
        verb_loss = F.cross_entropy(verb_logits, verb_labels)
        noun_loss = F.cross_entropy(noun_logits, noun_labels)
        loss = verb_weight * verb_loss + noun_weight * noun_loss

        total_loss += loss.item()
        verb_preds = verb_logits.argmax(dim=1)
        noun_preds = noun_logits.argmax(dim=1)

        all_verb_preds.append(verb_preds.cpu())
        all_noun_preds.append(noun_preds.cpu())
        all_verb_labels.append(verb_labels.cpu())
        all_noun_labels.append(noun_labels.cpu())

    # Calculate accuracy
    all_verb_preds = torch.cat(all_verb_preds)
    all_noun_preds = torch.cat(all_noun_preds)
    all_verb_labels = torch.cat(all_verb_labels)
    all_noun_labels = torch.cat(all_noun_labels)

    acc = calculate_accuracy(all_verb_preds, all_noun_preds, all_verb_labels, all_noun_labels)

    return total_loss / len(dataloader), acc


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # WandB
    if args.wandb:
        wandb.init(
            project="epic-kitchens-slowfast",
            name=args.exp_name,
            config=vars(args)
        )

    # Datasets
    train_dataset = EPICKitchensSlowFastDataset(
        annotations_file=f"{args.annotations_dir}/EPIC_100_train.csv",
        frames_dir=args.frames_dir,
        num_frames=args.num_frames,
        alpha=args.alpha,
        is_train=True
    )

    val_dataset = EPICKitchensSlowFastDataset(
        annotations_file=f"{args.annotations_dir}/EPIC_100_validation.csv",
        frames_dir=args.frames_dir,
        num_frames=args.num_frames,
        alpha=args.alpha,
        is_train=False
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Model
    model = SlowFastDualHead(
        num_verb_classes=97,
        num_noun_classes=300,
        dropout=args.dropout,
        pretrained=args.pretrained
    ).to(device)

    # Optimizer (SGD with momentum, as per EPIC-KITCHENS baselines)
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )

    # Scheduler (step decay at epochs 20 and 40)
    scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

    # Training
    best_action_acc = 0
    output_dir = Path(f"outputs/{args.exp_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            args.verb_weight, args.noun_weight
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, device,
            args.verb_weight, args.noun_weight
        )

        # Scheduler step
        scheduler.step()

        # Log
        print(f"Train - Loss: {train_loss:.4f}, Verb: {train_acc['verb']:.2f}%, "
              f"Noun: {train_acc['noun']:.2f}%, Action: {train_acc['action']:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f}, Verb: {val_acc['verb']:.2f}%, "
              f"Noun: {val_acc['noun']:.2f}%, Action: {val_acc['action']:.2f}%")

        if args.wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/verb_acc': train_acc['verb'],
                'train/noun_acc': train_acc['noun'],
                'train/action_acc': train_acc['action'],
                'val/loss': val_loss,
                'val/verb_acc': val_acc['verb'],
                'val/noun_acc': val_acc['noun'],
                'val/action_acc': val_acc['action'],
                'lr': optimizer.param_groups[0]['lr']
            })

        # Save best model
        if val_acc['action'] > best_action_acc:
            best_action_acc = val_acc['action']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args)
            }, output_dir / 'best_model.pth')
            print(f"✓ Saved best model (action acc: {best_action_acc:.2f}%)")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')

    print(f"\nTraining complete! Best action accuracy: {best_action_acc:.2f}%")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--frames_dir', type=str, required=True)
    parser.add_argument('--annotations_dir', type=str, required=True)
    parser.add_argument('--num_frames', type=int, default=32)
    parser.add_argument('--alpha', type=int, default=8, help='Slow/fast ratio')

    # Model
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--pretrained', action='store_true', default=True)

    # Training (EPIC-KITCHENS baseline config)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--verb_weight', type=float, default=0.4)
    parser.add_argument('--noun_weight', type=float, default=0.6)

    # System
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--wandb', action='store_true')

    args = parser.parse_args()
    main(args)
