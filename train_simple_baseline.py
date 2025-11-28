"""
Simple Baseline Training Script
Clean, straightforward training following best practices
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
import json
from pathlib import Path
import time

from common.config import Config
from dataset_simple_fixed import get_simple_dataloaders
from model_simple_baseline import get_simple_model


def train_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    verb_correct = 0
    noun_correct = 0
    action_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

    for frames, verb_labels, noun_labels in pbar:
        frames = frames.to(device)
        verb_labels = verb_labels.to(device)
        noun_labels = noun_labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            verb_logits, noun_logits = model(frames)

            # Simple cross-entropy loss (no label smoothing)
            loss_verb = criterion(verb_logits, verb_labels)
            loss_noun = criterion(noun_logits, noun_labels)
            loss = 0.5 * loss_verb + 0.5 * loss_noun

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        total_loss += loss.item() * frames.size(0)

        _, verb_pred = verb_logits.max(1)
        _, noun_pred = noun_logits.max(1)

        verb_correct += (verb_pred == verb_labels).sum().item()
        noun_correct += (noun_pred == noun_labels).sum().item()
        action_correct += ((verb_pred == verb_labels) & (noun_pred == noun_labels)).sum().item()

        total_samples += frames.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss/total_samples:.4f}',
            'V': f'{100.*verb_correct/total_samples:.1f}%',
            'N': f'{100.*noun_correct/total_samples:.1f}%',
            'A': f'{100.*action_correct/total_samples:.1f}%'
        })

    avg_loss = total_loss / total_samples
    verb_acc = 100.0 * verb_correct / total_samples
    noun_acc = 100.0 * noun_correct / total_samples
    action_acc = 100.0 * action_correct / total_samples

    return avg_loss, verb_acc, noun_acc, action_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    verb_correct_top1 = 0
    verb_correct_top5 = 0
    noun_correct_top1 = 0
    noun_correct_top5 = 0
    action_correct = 0
    total_samples = 0

    with torch.no_grad():
        for frames, verb_labels, noun_labels in tqdm(val_loader, desc="Validating"):
            frames = frames.to(device)
            verb_labels = verb_labels.to(device)
            noun_labels = noun_labels.to(device)

            # Forward pass
            verb_logits, noun_logits = model(frames)

            # Loss
            loss_verb = criterion(verb_logits, verb_labels)
            loss_noun = criterion(noun_logits, noun_labels)
            loss = 0.5 * loss_verb + 0.5 * loss_noun

            total_loss += loss.item() * frames.size(0)

            # Top-1 and Top-5 accuracy
            _, verb_pred_top1 = verb_logits.topk(1, dim=1)
            _, verb_pred_top5 = verb_logits.topk(5, dim=1)
            _, noun_pred_top1 = noun_logits.topk(1, dim=1)
            _, noun_pred_top5 = noun_logits.topk(5, dim=1)

            verb_correct_top1 += (verb_pred_top1.squeeze() == verb_labels).sum().item()
            verb_correct_top5 += sum([(verb_labels[i] in verb_pred_top5[i]) for i in range(len(verb_labels))])
            noun_correct_top1 += (noun_pred_top1.squeeze() == noun_labels).sum().item()
            noun_correct_top5 += sum([(noun_labels[i] in noun_pred_top5[i]) for i in range(len(noun_labels))])

            # Action accuracy (both verb and noun must be correct)
            action_correct += ((verb_pred_top1.squeeze() == verb_labels) &
                             (noun_pred_top1.squeeze() == noun_labels)).sum().item()

            total_samples += frames.size(0)

    avg_loss = total_loss / total_samples
    metrics = {
        'loss': avg_loss,
        'verb_acc_top1': 100.0 * verb_correct_top1 / total_samples,
        'verb_acc_top5': 100.0 * verb_correct_top5 / total_samples,
        'noun_acc_top1': 100.0 * noun_correct_top1 / total_samples,
        'noun_acc_top5': 100.0 * noun_correct_top5 / total_samples,
        'action_acc': 100.0 * action_correct / total_samples
    }

    return metrics


def main(args):
    # Configuration
    config = Config()

    # Override config with args
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.num_frames:
        config.NUM_FRAMES = args.num_frames

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create dataloaders
    train_loader, val_loader = get_simple_dataloaders(config)

    # Create model
    model = get_simple_model(config).to(device)

    # Simple cross-entropy loss (no label smoothing)
    criterion = nn.CrossEntropyLoss()

    # Optimizer - standard AdamW with moderate weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler - reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Mixed precision training
    scaler = GradScaler()

    # Create output directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        'train_loss': [],
        'train_verb_acc': [],
        'train_noun_acc': [],
        'train_action_acc': [],
        'val_loss': [],
        'val_verb_acc': [],
        'val_noun_acc': [],
        'val_action_acc': []
    }

    best_action_acc = 0.0
    best_epoch = 0

    print(f"\n{'='*70}")
    print(f"Starting Training - Simple Baseline")
    print(f"{'='*70}")
    print(f"Model:         ResNet-18 + Temporal Pooling")
    print(f"Epochs:        {config.EPOCHS}")
    print(f"Batch size:    {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Frames:        {config.NUM_FRAMES}")
    print(f"Dropout:       0.3")
    print(f"Regularization: Minimal (no label smoothing, no drop path)")
    print(f"{'='*70}\n")

    start_time = time.time()

    for epoch in range(config.EPOCHS):
        # Train
        train_loss, train_verb_acc, train_noun_acc, train_action_acc = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_metrics['action_acc'])

        # Save history
        history['train_loss'].append(train_loss)
        history['train_verb_acc'].append(train_verb_acc)
        history['train_noun_acc'].append(train_noun_acc)
        history['train_action_acc'].append(train_action_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_verb_acc'].append(val_metrics['verb_acc_top1'])
        history['val_noun_acc'].append(val_metrics['noun_acc_top1'])
        history['val_action_acc'].append(val_metrics['action_acc'])

        # Calculate overfitting gap
        verb_gap = train_verb_acc - val_metrics['verb_acc_top1']
        noun_gap = train_noun_acc - val_metrics['noun_acc_top1']
        action_gap = train_action_acc - val_metrics['action_acc']

        # Print epoch summary
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{config.EPOCHS} Summary")
        print(f"{'='*70}")
        print(f"Train: Loss={train_loss:.4f} | V={train_verb_acc:.1f}% N={train_noun_acc:.1f}% A={train_action_acc:.1f}%")
        print(f"Val:   Loss={val_metrics['loss']:.4f} | V={val_metrics['verb_acc_top1']:.1f}% N={val_metrics['noun_acc_top1']:.1f}% A={val_metrics['action_acc']:.1f}%")
        print(f"Gap:   V={verb_gap:+.1f}% N={noun_gap:+.1f}% A={action_gap:+.1f}%")
        print(f"Top-5: V={val_metrics['verb_acc_top5']:.1f}% N={val_metrics['noun_acc_top5']:.1f}%")
        print(f"{'='*70}\n")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
                'history': history
            }, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}\n")

        # Save best model
        if val_metrics['action_acc'] > best_action_acc:
            best_action_acc = val_metrics['action_acc']
            best_epoch = epoch + 1
            best_path = checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'history': history
            }, best_path)
            print(f"★ New best model! Action Acc: {best_action_acc:.2f}%\n")

    # Training complete
    total_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Total time:        {total_time/3600:.2f} hours")
    print(f"Best action acc:   {best_action_acc:.2f}% (epoch {best_epoch})")
    print(f"Best verb acc:     {max(history['val_verb_acc']):.2f}%")
    print(f"Best noun acc:     {max(history['val_noun_acc']):.2f}%")
    print(f"{'='*70}\n")

    # Save final history
    history_path = log_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"History saved to: {history_path}")

    # Compare to previous results
    print(f"\n{'='*70}")
    print(f"Comparison to Previous Results")
    print(f"{'='*70}")
    print(f"Phase 1 Baseline:  ~30% verb, ~25% noun, ~15-20% action")
    print(f"Phase 2 Improved:  ~36% verb, ~19% noun, ~10% action")
    print(f"Simple Baseline:   {max(history['val_verb_acc']):.1f}% verb, {max(history['val_noun_acc']):.1f}% noun, {best_action_acc:.1f}% action")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Simple Baseline Model')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--num_frames', type=int, default=8,
                       help='Number of frames per clip (default: 8)')
    parser.add_argument('--output_dir', type=str, default='outputs_simple_baseline',
                       help='Output directory (default: outputs_simple_baseline)')

    args = parser.parse_args()

    main(args)
