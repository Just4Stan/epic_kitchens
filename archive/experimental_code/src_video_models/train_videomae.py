"""
VideoMAE Fine-tuning for EPIC-KITCHENS-100
Uses pretrained VideoMAE-Base from HuggingFace
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
import argparse
from tqdm import tqdm
import wandb
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from typing import Tuple, Optional
import json


class EPICKitchensVideoDataset(Dataset):
    """EPIC-KITCHENS dataset for VideoMAE"""

    def __init__(
        self,
        annotations_file: str,
        frames_dir: str,
        processor: VideoMAEImageProcessor,
        num_frames: int = 16,
        is_train: bool = True
    ):
        self.annotations = pd.read_csv(annotations_file)
        self.frames_dir = Path(frames_dir)
        self.processor = processor
        self.num_frames = num_frames
        self.is_train = is_train

        # Filter out invalid samples
        self.annotations = self.annotations[
            self.annotations['stop_frame'] - self.annotations['start_frame'] >= num_frames
        ].reset_index(drop=True)

        print(f"Loaded {len(self.annotations)} samples from {annotations_file}")

    def __len__(self):
        return len(self.annotations)

    def _load_frames(self, video_id: str, start_frame: int, stop_frame: int) -> list:
        """Load and uniformly sample frames from video"""
        participant_id = video_id.split('_')[0]
        video_dir = self.frames_dir / participant_id / video_id

        # Uniformly sample frames
        total_frames = stop_frame - start_frame
        if total_frames < self.num_frames:
            indices = np.linspace(start_frame, stop_frame - 1, self.num_frames, dtype=int)
        else:
            indices = np.linspace(start_frame, stop_frame - 1, self.num_frames, dtype=int)

        frames = []
        for idx in indices:
            frame_path = video_dir / f"frame_{idx:010d}.jpg"
            try:
                frame = Image.open(frame_path).convert('RGB')
                frames.append(frame)
            except:
                # If frame doesn't exist, use previous frame or black frame
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(Image.new('RGB', (456, 256), (0, 0, 0)))

        return frames

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        # Load frames
        frames = self._load_frames(
            row['video_id'],
            row['start_frame'],
            row['stop_frame']
        )

        # Process frames using VideoMAE processor
        inputs = self.processor(frames, return_tensors="pt")

        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),  # (num_frames, 3, H, W)
            'verb_class': torch.tensor(row['verb_class'], dtype=torch.long),
            'noun_class': torch.tensor(row['noun_class'], dtype=torch.long),
        }


class VideoMAEDualHead(nn.Module):
    """VideoMAE with dual classification heads for verb and noun"""

    def __init__(
        self,
        num_verb_classes: int = 97,
        num_noun_classes: int = 300,
        pretrained_model: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        dropout: float = 0.5,
        freeze_backbone: bool = False
    ):
        super().__init__()

        # Load pretrained VideoMAE
        print(f"Loading pretrained VideoMAE from {pretrained_model}...")
        self.videomae = VideoMAEForVideoClassification.from_pretrained(
            pretrained_model,
            ignore_mismatched_sizes=True,
            num_labels=num_verb_classes  # Temporary, we'll replace the head
        )

        # Get hidden size
        hidden_size = self.videomae.config.hidden_size

        # Remove the original classification head
        self.videomae.classifier = nn.Identity()

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.videomae.videomae.parameters():
                param.requires_grad = False
            print("✓ Backbone frozen")

        # Dual classification heads
        self.verb_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_verb_classes)
        )

        self.noun_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_noun_classes)
        )

        # Print model info
        self._print_model_info()

    def _print_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "="*60)
        print("VIDEOMAE DUAL HEAD MODEL")
        print("="*60)
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters:    {total_params - trainable_params:,}")
        print("="*60 + "\n")

    def forward(self, pixel_values):
        # Extract features from VideoMAE
        outputs = self.videomae.videomae(pixel_values)
        sequence_output = outputs.last_hidden_state  # (B, num_patches, hidden_size)

        # Global average pooling
        features = sequence_output.mean(dim=1)  # (B, hidden_size)

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
        pixel_values = batch['pixel_values'].to(device)
        verb_labels = batch['verb_class'].to(device)
        noun_labels = batch['noun_class'].to(device)

        # Forward
        verb_logits, noun_logits = model(pixel_values)

        # Loss
        verb_loss = F.cross_entropy(verb_logits, verb_labels)
        noun_loss = F.cross_entropy(noun_logits, noun_labels)
        loss = verb_weight * verb_loss + noun_weight * noun_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        pixel_values = batch['pixel_values'].to(device)
        verb_labels = batch['verb_class'].to(device)
        noun_labels = batch['noun_class'].to(device)

        # Forward
        verb_logits, noun_logits = model(pixel_values)

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
            project="epic-kitchens-videomae",
            name=args.exp_name,
            config=vars(args)
        )

    # Processor
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

    # Datasets
    train_dataset = EPICKitchensVideoDataset(
        annotations_file=f"{args.annotations_dir}/EPIC_100_train.csv",
        frames_dir=args.frames_dir,
        processor=processor,
        num_frames=args.num_frames,
        is_train=True
    )

    val_dataset = EPICKitchensVideoDataset(
        annotations_file=f"{args.annotations_dir}/EPIC_100_validation.csv",
        frames_dir=args.frames_dir,
        processor=processor,
        num_frames=args.num_frames,
        is_train=False
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Model
    model = VideoMAEDualHead(
        num_verb_classes=97,
        num_noun_classes=300,
        pretrained_model=args.pretrained_model,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone
    ).to(device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

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
    parser.add_argument('--num_frames', type=int, default=16)

    # Model
    parser.add_argument('--pretrained_model', type=str, default='MCG-NJU/videomae-base-finetuned-kinetics')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--freeze_backbone', action='store_true')

    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--verb_weight', type=float, default=0.4)
    parser.add_argument('--noun_weight', type=float, default=0.6)

    # System
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--wandb', action='store_true')

    args = parser.parse_args()
    main(args)
