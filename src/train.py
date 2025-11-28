#!/usr/bin/env python3
"""
EPIC-KITCHENS-100 Action Recognition - Training Script
=======================================================
Unified training script for all experiments.

Usage:
    python src/train.py --exp_name exp29_test --epochs 30 --batch_size 64

See --help for all options.
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

# Local imports
from config import Config, get_output_dir
from datasets import TrainDataset, ValDataset
from models import ActionModel, LabelSmoothingLoss, EarlyStopping


# =============================================================================
# LOGIT ADJUSTMENT LOSS (for long-tail class imbalance)
# =============================================================================

class LogitAdjustedLoss(nn.Module):
    """
    Logit Adjustment Loss from "Long-Tail Learning via Logit Adjustment" (ICLR 2021).
    Adjusts logits based on class frequency to handle class imbalance.
    """
    def __init__(self, class_counts, tau=1.0, smoothing=0.1):
        super().__init__()
        # Compute log prior from class counts
        class_freq = class_counts / class_counts.sum()
        log_prior = tau * torch.log(class_freq + 1e-8)
        self.register_buffer('log_prior', log_prior)
        self.smoothing = smoothing

    def forward(self, logits, targets):
        # Adjust logits by subtracting log prior
        adjusted_logits = logits - self.log_prior.unsqueeze(0)

        if self.smoothing > 0:
            n_classes = logits.size(1)
            one_hot = F.one_hot(targets, n_classes).float()
            smooth_targets = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
            log_probs = F.log_softmax(adjusted_logits, dim=1)
            loss = -(smooth_targets * log_probs).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(adjusted_logits, targets)
        return loss


def get_class_counts(csv_path, num_verb_classes=97, num_noun_classes=300):
    """Count class occurrences from training CSV."""
    df = pd.read_csv(csv_path)
    verb_counts = torch.zeros(num_verb_classes)
    noun_counts = torch.zeros(num_noun_classes)

    for _, row in df.iterrows():
        verb_counts[row['verb_class']] += 1
        noun_counts[row['noun_class']] += 1

    # Add small constant to avoid log(0)
    verb_counts = verb_counts + 1
    noun_counts = noun_counts + 1

    return verb_counts, noun_counts


# =============================================================================
# CUTMIX WITH TEMPORAL CONSISTENCY
# =============================================================================

def cutmix_temporal(frames, verb_labels, noun_labels, alpha=1.0):
    """
    CutMix with temporal consistency - same spatial region cut across ALL frames.
    This is important for video because random regions per frame would break temporal coherence.

    Args:
        frames: (B, T, C, H, W) video tensor
        verb_labels, noun_labels: class labels
        alpha: Beta distribution parameter

    Returns:
        mixed_frames, verb_a, verb_b, noun_a, noun_b, lam
    """
    batch_size = frames.size(0)

    # Generate mixing ratio from beta distribution
    lam = np.random.beta(alpha, alpha)

    # Random permutation for mixing partners
    rand_index = torch.randperm(batch_size)

    # Get spatial dimensions
    _, _, _, H, W = frames.shape

    # Calculate cut dimensions
    cut_rat = np.sqrt(1 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Random center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Bounding box (same for ALL frames - temporal consistency)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply CutMix - SAME region across all frames
    frames_mixed = frames.clone()
    frames_mixed[:, :, :, bby1:bby2, bbx1:bbx2] = frames[rand_index, :, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda to actual cut ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    return (frames_mixed, verb_labels, verb_labels[rand_index],
            noun_labels, noun_labels[rand_index], lam)


# =============================================================================
# MIXUP (blends entire frames - softer than CutMix)
# =============================================================================

def mixup_temporal(frames, verb_labels, noun_labels, alpha=0.4):
    """
    MixUp for video - blends entire frames.
    Softer augmentation than CutMix, may work better for small datasets.
    """
    batch_size = frames.size(0)

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure primary sample dominates

    rand_index = torch.randperm(batch_size)
    frames_mixed = lam * frames + (1 - lam) * frames[rand_index]

    return (frames_mixed, verb_labels, verb_labels[rand_index],
            noun_labels, noun_labels[rand_index], lam)


# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(model, loader, optimizer, criterion_verb, criterion_noun, device, scaler, epoch, total, cutmix_alpha=0.0, mixup_alpha=0.0):
    """Train for one epoch with optional CutMix or MixUp."""
    model.train()
    total_loss, verb_correct, noun_correct, n = 0, 0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total} [Train]")
    for frames, verb_labels, noun_labels in pbar:
        frames = frames.to(device)
        verb_labels = verb_labels.to(device)
        noun_labels = noun_labels.to(device)

        # Apply CutMix or MixUp with 50% probability if enabled
        use_mixing = False
        if cutmix_alpha > 0 and np.random.random() < 0.5:
            frames, verb_a, verb_b, noun_a, noun_b, lam = cutmix_temporal(
                frames, verb_labels, noun_labels, alpha=cutmix_alpha
            )
            use_mixing = True
        elif mixup_alpha > 0 and np.random.random() < 0.5:
            frames, verb_a, verb_b, noun_a, noun_b, lam = mixup_temporal(
                frames, verb_labels, noun_labels, alpha=mixup_alpha
            )
            use_mixing = True

        optimizer.zero_grad()
        with autocast():
            verb_logits, noun_logits = model(frames)

            if use_mixing:
                # Mixed loss for CutMix/MixUp
                loss_verb = lam * criterion_verb(verb_logits, verb_a) + (1 - lam) * criterion_verb(verb_logits, verb_b)
                loss_noun = lam * criterion_noun(noun_logits, noun_a) + (1 - lam) * criterion_noun(noun_logits, noun_b)
                loss = 0.5 * loss_verb + 0.5 * loss_noun
            else:
                loss = 0.5 * criterion_verb(verb_logits, verb_labels) + \
                       0.5 * criterion_noun(noun_logits, noun_labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * frames.size(0)
        # For accuracy, use primary labels
        target_verb = verb_a if use_mixing else verb_labels
        target_noun = noun_a if use_mixing else noun_labels
        verb_correct += (verb_logits.argmax(1) == target_verb).sum().item()
        noun_correct += (noun_logits.argmax(1) == target_noun).sum().item()
        n += frames.size(0)

        pbar.set_postfix({
            'loss': f'{total_loss/n:.4f}',
            'verb': f'{100*verb_correct/n:.1f}%'
        })

    return total_loss/n, 100*verb_correct/n, 100*noun_correct/n


def validate(model, loader, criterion_verb, criterion_noun, device, epoch, total):
    """Validate the model."""
    model.eval()
    total_loss = 0
    verb_correct, noun_correct, action_correct = 0, 0, 0
    verb_top5, noun_top5 = 0, 0
    n = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{total} [Val]")
        for frames, verb_labels, noun_labels in pbar:
            frames = frames.to(device)
            verb_labels = verb_labels.to(device)
            noun_labels = noun_labels.to(device)

            verb_logits, noun_logits = model(frames)
            loss = 0.5 * criterion_verb(verb_logits, verb_labels) + \
                   0.5 * criterion_noun(noun_logits, noun_labels)

            total_loss += loss.item() * frames.size(0)

            # Top-1 accuracy
            v_pred = verb_logits.argmax(1)
            n_pred = noun_logits.argmax(1)
            verb_correct += (v_pred == verb_labels).sum().item()
            noun_correct += (n_pred == noun_labels).sum().item()
            action_correct += ((v_pred == verb_labels) & (n_pred == noun_labels)).sum().item()

            # Top-5 accuracy
            v5 = verb_logits.topk(5, dim=1)[1]
            n5 = noun_logits.topk(5, dim=1)[1]
            verb_top5 += (v5 == verb_labels.unsqueeze(1)).any(1).sum().item()
            noun_top5 += (n5 == noun_labels.unsqueeze(1)).any(1).sum().item()

            n += frames.size(0)
            pbar.set_postfix({'action': f'{100*action_correct/n:.1f}%'})

    return {
        'loss': total_loss/n,
        'verb_top1': 100*verb_correct/n,
        'noun_top1': 100*noun_correct/n,
        'action_top1': 100*action_correct/n,
        'verb_top5': 100*verb_top5/n,
        'noun_top5': 100*noun_top5/n,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="EPIC-KITCHENS Action Recognition Training")

    # Required
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name (e.g., exp29_mixup)")

    # Training params
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=14)
    parser.add_argument("--warmup_epochs", type=int, default=3)

    # Model params
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--backbone", type=str, default="resnet50",
                       choices=["resnet50", "resnet18", "efficientnet_b0", "efficientnet_b3"])
    parser.add_argument("--temporal_model", type=str, default="lstm",
                       choices=["lstm", "transformer", "mean"])
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--freeze_backbone", type=str, default="none",
                       choices=["none", "all", "early", "bn"])

    # Regularization
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--augmentation", type=str, default="medium",
                       choices=["none", "light", "medium", "heavy"])

    # Advanced features (ablation study)
    parser.add_argument("--logit_adjust_tau", type=float, default=0.0,
                       help="Logit adjustment tau (0=disabled, 1.0=recommended)")
    parser.add_argument("--cutmix_alpha", type=float, default=0.0,
                       help="CutMix alpha (0=disabled, 1.0=recommended)")
    parser.add_argument("--mixup_alpha", type=float, default=0.0,
                       help="MixUp alpha (0=disabled, 0.4=recommended)")

    # Early stopping
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=7)

    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--start_epoch", type=int, default=0,
                       help="Epoch to start from when resuming")

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Output directory
    output_dir = get_output_dir(args.exp_name)
    print(f"Output: {output_dir}")

    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Wandb
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        os.environ["WANDB_API_KEY"] = Config.WANDB_API_KEY
        wandb.init(
            project=Config.WANDB_PROJECT,
            name=args.exp_name,
            config=vars(args),
            dir=str(output_dir)
        )
        print(f"W&B: {wandb.run.url}")

    # Print config
    print("\n" + "="*60)
    print(f"EXPERIMENT: {args.exp_name}")
    print("="*60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("="*60 + "\n")

    # Datasets
    cfg = Config()
    train_dataset = TrainDataset(
        cfg.TRAIN_CSV, cfg.EXTRACTED_FRAMES_DIR,
        num_frames=args.num_frames, augmentation=args.augmentation
    )
    val_dataset = ValDataset(
        cfg.VAL_CSV, cfg.VAL_VIDEO_DIR,
        num_frames=args.num_frames,
        frames_dir=cfg.VAL_FRAMES_DIR
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Model
    model = ActionModel(
        num_verb_classes=cfg.NUM_VERB_CLASSES,
        num_noun_classes=cfg.NUM_NOUN_CLASSES,
        backbone=args.backbone,
        temporal_model=args.temporal_model,
        dropout=args.dropout,
        num_frames=args.num_frames,
        freeze_backbone=args.freeze_backbone
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable:,} trainable\n")

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint, starting from epoch {args.start_epoch + 1}")

    # Loss, optimizer, scheduler
    if args.logit_adjust_tau > 0:
        # Use Logit Adjusted Loss for class imbalance
        verb_counts, noun_counts = get_class_counts(cfg.TRAIN_CSV)
        criterion_verb = LogitAdjustedLoss(verb_counts.to(device), tau=args.logit_adjust_tau, smoothing=args.label_smoothing)
        criterion_noun = LogitAdjustedLoss(noun_counts.to(device), tau=args.logit_adjust_tau, smoothing=args.label_smoothing)
        print(f"Using Logit Adjusted Loss (tau={args.logit_adjust_tau})")
    else:
        criterion_verb = LabelSmoothingLoss(args.label_smoothing)
        criterion_noun = LabelSmoothingLoss(args.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Warmup + cosine schedule
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    scaler = GradScaler()

    # Early stopping
    early_stopper = EarlyStopping(patience=args.patience) if args.early_stopping else None

    # Training loop
    best_acc = 0
    history = []

    if args.cutmix_alpha > 0:
        print(f"Using CutMix with temporal consistency (alpha={args.cutmix_alpha})")
    if args.mixup_alpha > 0:
        print(f"Using MixUp (alpha={args.mixup_alpha})")

    for epoch in range(args.start_epoch, args.epochs):
        # Train
        train_loss, train_verb, train_noun = train_epoch(
            model, train_loader, optimizer, criterion_verb, criterion_noun,
            device, scaler, epoch+1, args.epochs,
            cutmix_alpha=args.cutmix_alpha, mixup_alpha=args.mixup_alpha
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion_verb, criterion_noun,
            device, epoch+1, args.epochs
        )

        # Update scheduler
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        # Log
        print(f"\n  Train: Loss={train_loss:.4f}, Verb={train_verb:.1f}%, Noun={train_noun:.1f}%")
        print(f"  Val:   Verb={val_metrics['verb_top1']:.1f}%, Noun={val_metrics['noun_top1']:.1f}%, Action={val_metrics['action_top1']:.1f}%")
        print(f"  Top-5: Verb={val_metrics['verb_top5']:.1f}%, Noun={val_metrics['noun_top5']:.1f}%")

        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_verb': train_verb,
                'train_noun': train_noun,
                **{f'val_{k}': v for k, v in val_metrics.items()}
            })

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_verb': train_verb,
            **val_metrics
        })

        # Save best model
        if val_metrics['action_top1'] > best_acc:
            best_acc = val_metrics['action_top1']
            torch.save(model.state_dict(), output_dir / "checkpoints" / "best_model.pth")
            print(f"  >>> New best! Action={best_acc:.2f}%")

        # Early stopping
        if early_stopper:
            early_stopper(val_metrics['action_top1'], model)
            if early_stopper.should_stop:
                print(f"\nEarly stopping at epoch {epoch+1}")
                model.load_state_dict(early_stopper.best_state)
                break

        print()

    # Save history
    with open(output_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Best action accuracy: {best_acc:.2f}%")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
