#!/usr/bin/env python3
"""
EPIC-KITCHENS-100 - Full Dataset Training
==========================================
Train on full pre-extracted RGB frames from official dataset.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from tqdm import tqdm

from datasets_full import FullTrainDataset, FullValDataset


# =============================================================================
# MODEL
# =============================================================================

class ActionModel(nn.Module):
    """Action recognition model with temporal modeling."""

    def __init__(self, num_verb_classes=97, num_noun_classes=300,
                 backbone='resnet50', temporal_model='lstm',
                 dropout=0.5, num_frames=16):
        super().__init__()

        import torchvision.models as models

        # Backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feature_dim = 2048
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feature_dim = 512
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Temporal model
        self.temporal_model = temporal_model
        if temporal_model == 'lstm':
            self.temporal = nn.LSTM(
                self.feature_dim, self.feature_dim // 2,
                num_layers=2, batch_first=True, bidirectional=True, dropout=0.3
            )
            self.temporal_dim = self.feature_dim
        elif temporal_model == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.feature_dim, nhead=8, dim_feedforward=2048,
                dropout=0.1, batch_first=True
            )
            self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.temporal_dim = self.feature_dim
        else:
            self.temporal = None
            self.temporal_dim = self.feature_dim

        # Classification heads
        self.dropout = nn.Dropout(dropout)
        self.verb_head = nn.Linear(self.temporal_dim, num_verb_classes)
        self.noun_head = nn.Linear(self.temporal_dim, num_noun_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)
        features = features.view(B, T, -1)

        if self.temporal_model == 'lstm':
            temporal_out, _ = self.temporal(features)
            pooled = temporal_out.mean(dim=1)
        elif self.temporal_model == 'transformer':
            temporal_out = self.temporal(features)
            pooled = temporal_out.mean(dim=1)
        else:
            pooled = features.mean(dim=1)

        pooled = self.dropout(pooled)
        verb_logits = self.verb_head(pooled)
        noun_logits = self.noun_head(pooled)
        return verb_logits, noun_logits


# =============================================================================
# LOSSES
# =============================================================================

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        one_hot = F.one_hot(target, n_classes).float()
        smooth_target = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        log_probs = F.log_softmax(pred, dim=1)
        loss = -(smooth_target * log_probs).sum(dim=1).mean()
        return loss


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# =============================================================================
# CUTMIX
# =============================================================================

def cutmix_temporal(frames, verb_labels, noun_labels, alpha=1.0):
    """CutMix with temporal consistency."""
    if alpha <= 0:
        return frames, verb_labels, verb_labels, noun_labels, noun_labels, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = frames.size(0)
    index = torch.randperm(batch_size).to(frames.device)

    _, T, C, H, W = frames.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_frames = frames.clone()
    mixed_frames[:, :, :, y1:y2, x1:x2] = frames[index, :, :, y1:y2, x1:x2]

    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

    return mixed_frames, verb_labels, verb_labels[index], noun_labels, noun_labels[index], lam


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, scaler, verb_criterion, noun_criterion,
                device, cutmix_alpha=0.0):
    model.train()
    total_loss = 0
    verb_correct = 0
    noun_correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for frames, verb_labels, noun_labels in pbar:
        frames = frames.to(device)
        verb_labels = verb_labels.to(device)
        noun_labels = noun_labels.to(device)

        # CutMix
        if cutmix_alpha > 0 and np.random.random() > 0.5:
            frames, verb_a, verb_b, noun_a, noun_b, lam = cutmix_temporal(
                frames, verb_labels, noun_labels, cutmix_alpha
            )
            use_cutmix = True
        else:
            use_cutmix = False

        optimizer.zero_grad()

        with autocast():
            verb_logits, noun_logits = model(frames)

            if use_cutmix:
                verb_loss = lam * verb_criterion(verb_logits, verb_a) + (1 - lam) * verb_criterion(verb_logits, verb_b)
                noun_loss = lam * noun_criterion(noun_logits, noun_a) + (1 - lam) * noun_criterion(noun_logits, noun_b)
            else:
                verb_loss = verb_criterion(verb_logits, verb_labels)
                noun_loss = noun_criterion(noun_logits, noun_labels)

            loss = verb_loss + noun_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        verb_pred = verb_logits.argmax(dim=1)
        noun_pred = noun_logits.argmax(dim=1)
        verb_correct += (verb_pred == verb_labels).sum().item()
        noun_correct += (noun_pred == noun_labels).sum().item()
        total += verb_labels.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'verb': f'{100*verb_correct/total:.1f}%',
            'noun': f'{100*noun_correct/total:.1f}%'
        })

    return total_loss / len(loader), verb_correct / total, noun_correct / total


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    verb_correct = 0
    noun_correct = 0
    action_correct = 0
    total = 0

    for frames, verb_labels, noun_labels in tqdm(loader, desc="Validating"):
        frames = frames.to(device)
        verb_labels = verb_labels.to(device)
        noun_labels = noun_labels.to(device)

        verb_logits, noun_logits = model(frames)

        verb_pred = verb_logits.argmax(dim=1)
        noun_pred = noun_logits.argmax(dim=1)

        verb_match = (verb_pred == verb_labels)
        noun_match = (noun_pred == noun_labels)

        verb_correct += verb_match.sum().item()
        noun_correct += noun_match.sum().item()
        action_correct += (verb_match & noun_match).sum().item()
        total += verb_labels.size(0)

    return {
        'verb_acc': verb_correct / total,
        'noun_acc': noun_correct / total,
        'action_acc': action_correct / total
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--frames_dir', type=str, required=True,
                       help='Path to RGB frames (e.g., /scratch/.../EPIC-KITCHENS/EPIC-KITCHENS)')
    parser.add_argument('--annotations_dir', type=str, required=True,
                       help='Path to annotations')
    parser.add_argument('--output_dir', type=str, default='outputs')

    # Training params
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--warmup_epochs', type=int, default=3)

    # Model params
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--temporal_model', type=str, default='lstm')
    parser.add_argument('--dropout', type=float, default=0.5)

    # Regularization
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--cutmix_alpha', type=float, default=1.0)

    # Training options
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--augmentation', type=str, default='medium')

    # W&B
    parser.add_argument('--wandb', action='store_true')

    args = parser.parse_args()

    # Setup output
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Paths
    frames_dir = Path(args.frames_dir)
    annotations_dir = Path(args.annotations_dir)
    train_csv = annotations_dir / "EPIC_100_train.csv"
    val_csv = annotations_dir / "EPIC_100_validation.csv"

    # Datasets
    print("\nLoading datasets...")
    train_dataset = FullTrainDataset(
        train_csv, frames_dir,
        num_frames=args.num_frames,
        augmentation=args.augmentation
    )
    val_dataset = FullValDataset(
        val_csv, frames_dir,
        num_frames=args.num_frames
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True
    )

    # Model
    print("\nCreating model...")
    model = ActionModel(
        num_verb_classes=97,
        num_noun_classes=300,
        backbone=args.backbone,
        temporal_model=args.temporal_model,
        dropout=args.dropout,
        num_frames=args.num_frames
    ).to(device)

    # Loss
    verb_criterion = LabelSmoothingLoss(args.label_smoothing)
    noun_criterion = LabelSmoothingLoss(args.label_smoothing)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler with warmup
    def warmup_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, warmup_lambda)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    scaler = GradScaler()

    # Early stopping
    early_stopper = EarlyStopping(patience=args.patience) if args.early_stopping else None

    # W&B
    if args.wandb:
        import wandb
        wandb.init(project="epic-kitchens-full", name=args.exp_name, config=vars(args))

    # Training loop
    best_action_acc = 0
    print(f"\n{'='*60}")
    print(f"Starting training: {args.exp_name}")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)

        train_loss, train_verb, train_noun = train_epoch(
            model, train_loader, optimizer, scaler,
            verb_criterion, noun_criterion, device,
            cutmix_alpha=args.cutmix_alpha
        )

        val_metrics = validate(model, val_loader, device)

        # Scheduler step
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        # Log
        print(f"\nTrain Loss: {train_loss:.4f} | Verb: {train_verb*100:.2f}% | Noun: {train_noun*100:.2f}%")
        print(f"Val   Verb: {val_metrics['verb_acc']*100:.2f}% | Noun: {val_metrics['noun_acc']*100:.2f}% | Action: {val_metrics['action_acc']*100:.2f}%")

        if args.wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_verb_acc': train_verb,
                'train_noun_acc': train_noun,
                'val_verb_acc': val_metrics['verb_acc'],
                'val_noun_acc': val_metrics['noun_acc'],
                'val_action_acc': val_metrics['action_acc'],
                'lr': optimizer.param_groups[0]['lr']
            })

        # Save best
        if val_metrics['action_acc'] > best_action_acc:
            best_action_acc = val_metrics['action_acc']
            torch.save(model.state_dict(), output_dir / "checkpoints" / "best_model.pth")
            print(f"  -> New best! Action: {best_action_acc*100:.2f}%")

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_action_acc': best_action_acc,
        }, output_dir / "checkpoints" / "last_checkpoint.pth")

        # Early stopping
        if early_stopper:
            early_stopper(val_metrics['action_acc'])
            if early_stopper.early_stop:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Best Action Accuracy: {best_action_acc*100:.2f}%")
    print(f"{'='*60}")

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
