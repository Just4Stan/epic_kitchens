#!/usr/bin/env python3
"""
EPIC-KITCHENS-100 Action Recognition - V2 Moonshot Training
============================================================
All-in-one training script combining all proven improvements:
- Two-Stream architecture (CLIP spatial + LSTM temporal)
- 320x320 resolution
- CutMix + MixUp + RandomErasing augmentation
- Focal Loss for long-tail
- Cosine classifier for nouns
- Class-balanced loss weighting

Usage:
    python src_v2/train.py \
        --exp_name moonshot_v1 \
        --frames_dir /path/to/EPIC-KITCHENS \
        --annotations_dir /path/to/annotations \
        --model twostream \
        --epochs 50

VSC Usage:
    sbatch jobs/v2_moonshot.slurm
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm

from config import Config, get_output_dir
from datasets import (
    TrainDataset, ValDataset,
    cutmix_temporal, mixup_temporal, apply_random_erasing,
    get_class_counts
)
from models import (
    TwoStreamModel, BaselineModel,
    FocalLoss, LabelSmoothingLoss,
    EarlyStopping, get_class_weights, count_parameters
)
from models_improved import ImprovedTwoStreamModel


# Optional W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(
    model, loader, optimizer, scaler,
    criterion_verb, criterion_noun, device, epoch, total_epochs,
    cutmix_alpha=0.0, mixup_alpha=0.0, random_erasing_prob=0.0,
    cutmix_prob=0.5, mixup_prob=0.3,
    verb_weight=0.4, noun_weight=0.6,
    grad_accum=1
):
    """Train for one epoch with all augmentations."""
    model.train()
    total_loss = 0
    verb_correct, noun_correct, action_correct = 0, 0, 0
    n = 0

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")

    for batch_idx, (frames, verb_labels, noun_labels) in enumerate(pbar):
        frames = frames.to(device)
        verb_labels = verb_labels.to(device)
        noun_labels = noun_labels.to(device)

        # ===== Apply augmentations =====
        use_mixing = False
        r = np.random.random()

        if cutmix_alpha > 0 and r < cutmix_prob:
            # CutMix
            frames, verb_a, verb_b, noun_a, noun_b, lam = cutmix_temporal(
                frames, verb_labels, noun_labels, alpha=cutmix_alpha
            )
            use_mixing = True
        elif mixup_alpha > 0 and r < cutmix_prob + mixup_prob:
            # MixUp
            frames, verb_a, verb_b, noun_a, noun_b, lam = mixup_temporal(
                frames, verb_labels, noun_labels, alpha=mixup_alpha
            )
            use_mixing = True

        # Random Erasing (after mixing)
        if random_erasing_prob > 0:
            frames = apply_random_erasing(frames, p=random_erasing_prob)

        # ===== Forward pass =====
        with autocast():
            verb_logits, noun_logits = model(frames)

            if use_mixing:
                loss_verb = lam * criterion_verb(verb_logits, verb_a) + \
                           (1 - lam) * criterion_verb(verb_logits, verb_b)
                loss_noun = lam * criterion_noun(noun_logits, noun_a) + \
                           (1 - lam) * criterion_noun(noun_logits, noun_b)
            else:
                loss_verb = criterion_verb(verb_logits, verb_labels)
                loss_noun = criterion_noun(noun_logits, noun_labels)

            loss = verb_weight * loss_verb + noun_weight * loss_noun
            loss = loss / grad_accum  # Scale for gradient accumulation

        # ===== Backward pass =====
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # ===== Metrics =====
        total_loss += loss.item() * grad_accum * frames.size(0)
        target_verb = verb_a if use_mixing else verb_labels
        target_noun = noun_a if use_mixing else noun_labels

        v_pred = verb_logits.argmax(1)
        n_pred = noun_logits.argmax(1)
        verb_correct += (v_pred == target_verb).sum().item()
        noun_correct += (n_pred == target_noun).sum().item()
        action_correct += ((v_pred == target_verb) & (n_pred == target_noun)).sum().item()
        n += frames.size(0)

        pbar.set_postfix({
            'loss': f'{total_loss/n:.4f}',
            'verb': f'{100*verb_correct/n:.1f}%',
            'noun': f'{100*noun_correct/n:.1f}%',
            'action': f'{100*action_correct/n:.1f}%'
        })

    return {
        'loss': total_loss / n,
        'verb_acc': 100 * verb_correct / n,
        'noun_acc': 100 * noun_correct / n,
        'action_acc': 100 * action_correct / n
    }


@torch.no_grad()
def validate(model, loader, criterion_verb, criterion_noun, device, epoch, total_epochs):
    """Validate the model."""
    model.eval()
    total_loss = 0
    verb_correct, noun_correct, action_correct = 0, 0, 0
    verb_top5, noun_top5 = 0, 0
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Val]")
    for frames, verb_labels, noun_labels in pbar:
        frames = frames.to(device)
        verb_labels = verb_labels.to(device)
        noun_labels = noun_labels.to(device)

        verb_logits, noun_logits = model(frames)

        loss_verb = criterion_verb(verb_logits, verb_labels)
        loss_noun = criterion_noun(noun_logits, noun_labels)
        loss = 0.5 * loss_verb + 0.5 * loss_noun

        total_loss += loss.item() * frames.size(0)

        # Top-1
        v_pred = verb_logits.argmax(1)
        n_pred = noun_logits.argmax(1)
        verb_correct += (v_pred == verb_labels).sum().item()
        noun_correct += (n_pred == noun_labels).sum().item()
        action_correct += ((v_pred == verb_labels) & (n_pred == noun_labels)).sum().item()

        # Top-5
        v5 = verb_logits.topk(5, dim=1)[1]
        n5 = noun_logits.topk(5, dim=1)[1]
        verb_top5 += (v5 == verb_labels.unsqueeze(1)).any(1).sum().item()
        noun_top5 += (n5 == noun_labels.unsqueeze(1)).any(1).sum().item()

        n += frames.size(0)
        pbar.set_postfix({'action': f'{100*action_correct/n:.1f}%'})

    return {
        'loss': total_loss / n,
        'verb_top1': 100 * verb_correct / n,
        'noun_top1': 100 * noun_correct / n,
        'action_top1': 100 * action_correct / n,
        'verb_top5': 100 * verb_top5 / n,
        'noun_top5': 100 * noun_top5 / n
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="EPIC-KITCHENS V2 Moonshot Training")

    # Required paths
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--frames_dir", type=str, required=True,
                       help="Path to EPIC-KITCHENS RGB frames")
    parser.add_argument("--annotations_dir", type=str, required=True,
                       help="Path to EPIC-100 annotations")
    parser.add_argument("--output_dir", type=str, default="outputs")

    # Model
    parser.add_argument("--model", type=str, default="twostream",
                       choices=["twostream", "improved", "baseline"])
    parser.add_argument("--backbone", type=str, default="resnet50",
                       choices=["resnet50", "resnet18"],
                       help="Backbone for baseline model")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=8,
                       help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--warmup_epochs", type=int, default=5)

    # Data
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--augmentation", type=str, default="heavy",
                       choices=["none", "light", "medium", "heavy"])

    # Augmentation
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--cutmix_prob", type=float, default=0.5)
    parser.add_argument("--mixup_prob", type=float, default=0.3)
    parser.add_argument("--random_erasing", type=float, default=0.25)

    # Loss
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                       help="Focal loss gamma (0 = standard CE)")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--verb_weight", type=float, default=0.4)
    parser.add_argument("--noun_weight", type=float, default=0.6)
    parser.add_argument("--class_weight_power", type=float, default=0.5,
                       help="Class weight power (0 = no weighting, 0.5 = sqrt)")

    # Model specific
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use_cosine_noun", action="store_true", default=True)
    parser.add_argument("--cosine_temp", type=float, default=0.05)
    parser.add_argument("--use_cross_attention", action="store_true", default=True)
    parser.add_argument("--freeze_backbone", action="store_true", default=False,
                       help="Freeze backbone (CLIP/ResNet). Default: False (train all layers)")

    # Training options
    parser.add_argument("--early_stopping", action="store_true", default=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=5)

    # Logging
    parser.add_argument("--wandb", action="store_true")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # ===== Setup =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"EPIC-KITCHENS V2 MOONSHOT TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Output directory
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Output: {output_dir}")

    # ===== Wandb =====
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        os.environ["WANDB_API_KEY"] = Config.WANDB_API_KEY
        wandb.init(
            project="epic-kitchens-moonshot",
            name=args.exp_name,
            config=vars(args),
            dir=str(output_dir)
        )
        print(f"W&B: {wandb.run.url}")

    # ===== Print config =====
    print(f"\n{'='*60}")
    print("CONFIGURATION")
    print(f"{'='*60}")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    # ===== Paths =====
    frames_dir = Path(args.frames_dir)
    annotations_dir = Path(args.annotations_dir)
    train_csv = annotations_dir / "EPIC_100_train.csv"
    val_csv = annotations_dir / "EPIC_100_validation.csv"

    # ===== Datasets =====
    print("Loading datasets...")
    train_dataset = TrainDataset(
        train_csv, frames_dir,
        num_frames=args.num_frames,
        image_size=args.image_size,
        augmentation=args.augmentation
    )
    val_dataset = ValDataset(
        val_csv, frames_dir,
        num_frames=args.num_frames,
        image_size=args.image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers // 2,
        pin_memory=True
    )

    # ===== Model =====
    print(f"\nCreating {args.model} model...")
    if args.model == "twostream":
        model = TwoStreamModel(
            num_verb_classes=97,
            num_noun_classes=300,
            proj_dim=512,
            lstm_hidden=512,
            num_frames=args.num_frames,
            dropout=args.dropout,
            use_cross_attention=args.use_cross_attention,
            use_cosine_noun=args.use_cosine_noun,
            cosine_temp=args.cosine_temp,
            freeze_clip=args.freeze_backbone
        ).to(device)
    elif args.model == "improved":
        model = ImprovedTwoStreamModel(
            num_verb_classes=97,
            num_noun_classes=300,
            num_frames=args.num_frames,
            proj_dim=512,
            dropout=args.dropout,
            use_adapters=True,
            freeze_clip=args.freeze_backbone,
            use_cosine_noun=args.use_cosine_noun,
            cosine_temp=args.cosine_temp
        ).to(device)
    else:
        model = BaselineModel(
            num_verb_classes=97,
            num_noun_classes=300,
            backbone=args.backbone,
            dropout=args.dropout,
            num_frames=args.num_frames
        ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ===== Loss Functions =====
    print("\nSetting up loss functions...")

    # Get class weights if enabled
    verb_weights, noun_weights = None, None
    if args.class_weight_power > 0:
        verb_counts, noun_counts = get_class_counts(train_csv)
        verb_weights = get_class_weights(verb_counts, power=args.class_weight_power).to(device)
        noun_weights = get_class_weights(noun_counts, power=args.class_weight_power).to(device)
        print(f"Using class weights (power={args.class_weight_power})")

    # Focal or standard CE
    if args.focal_gamma > 0:
        criterion_verb = FocalLoss(
            gamma=args.focal_gamma,
            alpha=verb_weights,
            smoothing=args.label_smoothing
        )
        criterion_noun = FocalLoss(
            gamma=args.focal_gamma,
            alpha=noun_weights,
            smoothing=args.label_smoothing
        )
        print(f"Using Focal Loss (gamma={args.focal_gamma})")
    else:
        criterion_verb = LabelSmoothingLoss(args.label_smoothing)
        criterion_noun = LabelSmoothingLoss(args.label_smoothing)
        print(f"Using Label Smoothing Loss (smoothing={args.label_smoothing})")

    # ===== Optimizer & Scheduler =====
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Warmup + Cosine schedule
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=1e-6
    )

    scaler = GradScaler()

    # ===== Early Stopping =====
    early_stopper = EarlyStopping(patience=args.patience) if args.early_stopping else None

    # ===== Resume =====
    start_epoch = 0
    best_acc = 0
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_acc = checkpoint.get('best_action_acc', 0)
        else:
            model.load_state_dict(checkpoint)
        print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%")

    # ===== Training Loop =====
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"{'='*60}\n")

    history = []

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler,
            criterion_verb, criterion_noun, device,
            epoch + 1, args.epochs,
            cutmix_alpha=args.cutmix_alpha,
            mixup_alpha=args.mixup_alpha,
            random_erasing_prob=args.random_erasing,
            cutmix_prob=args.cutmix_prob,
            mixup_prob=args.mixup_prob,
            verb_weight=args.verb_weight,
            noun_weight=args.noun_weight,
            grad_accum=args.grad_accum
        )

        # Validate
        val_metrics = validate(
            model, val_loader,
            criterion_verb, criterion_noun, device,
            epoch + 1, args.epochs
        )

        # Update scheduler
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # ===== Log =====
        print(f"\n  Train: Loss={train_metrics['loss']:.4f}, "
              f"Verb={train_metrics['verb_acc']:.1f}%, "
              f"Noun={train_metrics['noun_acc']:.1f}%, "
              f"Action={train_metrics['action_acc']:.1f}%")
        print(f"  Val:   Verb={val_metrics['verb_top1']:.1f}%, "
              f"Noun={val_metrics['noun_top1']:.1f}%, "
              f"Action={val_metrics['action_top1']:.1f}%")
        print(f"  Top-5: Verb={val_metrics['verb_top5']:.1f}%, "
              f"Noun={val_metrics['noun_top5']:.1f}%")
        print(f"  LR: {current_lr:.2e}")

        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'lr': current_lr,
                'train/loss': train_metrics['loss'],
                'train/verb_acc': train_metrics['verb_acc'],
                'train/noun_acc': train_metrics['noun_acc'],
                'train/action_acc': train_metrics['action_acc'],
                'val/loss': val_metrics['loss'],
                'val/verb_top1': val_metrics['verb_top1'],
                'val/noun_top1': val_metrics['noun_top1'],
                'val/action_top1': val_metrics['action_top1'],
                'val/verb_top5': val_metrics['verb_top5'],
                'val/noun_top5': val_metrics['noun_top5']
            })

        history.append({
            'epoch': epoch + 1,
            'lr': current_lr,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()}
        })

        # ===== Save Best =====
        if val_metrics['action_top1'] > best_acc:
            best_acc = val_metrics['action_top1']
            torch.save(model.state_dict(), output_dir / "checkpoints" / "best_model.pth")
            print(f"  >>> New best! Action={best_acc:.2f}%")

        # ===== Save Checkpoint =====
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_action_acc': best_acc,
                'config': vars(args)
            }, output_dir / "checkpoints" / f"checkpoint_epoch_{epoch+1}.pth")
            print(f"  Saved checkpoint at epoch {epoch+1}")

        # Always save last checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_action_acc': best_acc,
            'config': vars(args)
        }, output_dir / "checkpoints" / "last_checkpoint.pth")

        # ===== Early Stopping =====
        if early_stopper:
            early_stopper(val_metrics['action_top1'], model)
            if early_stopper.should_stop:
                print(f"\nEarly stopping at epoch {epoch+1}")
                # Restore best model
                model.load_state_dict(early_stopper.best_state)
                break

        print()

    # ===== Save History =====
    with open(output_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    # ===== Final Summary =====
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Action Accuracy: {best_acc:.2f}%")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")

    if use_wandb:
        wandb.log({'best_action_acc': best_acc})
        wandb.finish()


if __name__ == "__main__":
    main()
