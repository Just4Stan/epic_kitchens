"""
Improved Training Script for EPIC-KITCHENS
Incorporates all best practices for reducing overfitting
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config import Config
from phase2.dataset_improved import get_improved_dataloaders
from phase2.model_improved import get_improved_model


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy with label smoothing.
    Prevents overconfident predictions and reduces overfitting.
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = nn.functional.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, train_loader, optimizer, criterion_verb, criterion_noun,
                device, scaler, epoch, config):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    verb_correct = 0
    noun_correct = 0
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

            # Compute losses
            loss_verb = criterion_verb(verb_logits, verb_labels)
            loss_noun = criterion_noun(noun_logits, noun_labels)
            loss = 0.5 * loss_verb + 0.5 * loss_noun

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Metrics
        total_loss += loss.item() * frames.size(0)
        _, verb_pred = verb_logits.max(1)
        _, noun_pred = noun_logits.max(1)
        verb_correct += (verb_pred == verb_labels).sum().item()
        noun_correct += (noun_pred == noun_labels).sum().item()
        total_samples += frames.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss/total_samples:.4f}',
            'verb_acc': f'{100.*verb_correct/total_samples:.1f}%',
            'noun_acc': f'{100.*noun_correct/total_samples:.1f}%'
        })

    avg_loss = total_loss / total_samples
    verb_acc = 100.0 * verb_correct / total_samples
    noun_acc = 100.0 * noun_correct / total_samples

    return avg_loss, verb_acc, noun_acc


def validate(model, val_loader, criterion_verb, criterion_noun, device):
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
            loss_verb = criterion_verb(verb_logits, verb_labels)
            loss_noun = criterion_noun(noun_logits, noun_labels)
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

            # Action accuracy
            action_correct += ((verb_pred_top1.squeeze() == verb_labels) & (noun_pred_top1.squeeze() == noun_labels)).sum().item()

            total_samples += frames.size(0)

    # Handle empty validation set
    if total_samples == 0:
        print("WARNING: Validation set is empty!")
        return {
            'loss': 0.0,
            'verb_acc_top1': 0.0,
            'verb_acc_top5': 0.0,
            'noun_acc_top1': 0.0,
            'noun_acc_top5': 0.0,
            'action_acc': 0.0
        }

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
    config = Config()

    # Override config with args
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create dataloaders
    train_loader, val_loader = get_improved_dataloaders(config)

    # Create model
    model = get_improved_model(config).to(device)

    # Loss functions with label smoothing
    criterion_verb = LabelSmoothingCrossEntropy(smoothing=0.1)
    criterion_noun = LabelSmoothingCrossEntropy(smoothing=0.1)

    # Optimizer with higher weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=1e-4,  # Increased from 1e-5
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler with warmup
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    # Mixed precision training
    scaler = GradScaler()

    # Early stopping - DISABLED (train all 30 epochs)
    # early_stopping = EarlyStopping(patience=7, min_delta=0.001)

    # Create output directories
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    log_dir = Path(args.output_dir) / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        'train_loss': [],
        'train_verb_acc': [],
        'train_noun_acc': [],
        'val_loss': [],
        'val_verb_acc': [],
        'val_noun_acc': [],
        'val_action_acc': []
    }

    best_val_acc = 0.0

    print(f"\n{'='*70}")
    print(f"Starting Training - Improved Model")
    print(f"{'='*70}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Weight decay: 1e-4")
    print(f"Label smoothing: 0.1")
    print(f"Dropout: 0.5")
    print(f"Drop path: 0.1")
    print(f"{'='*70}\n")

    for epoch in range(config.EPOCHS):
        # Train
        train_loss, train_verb_acc, train_noun_acc = train_epoch(
            model, train_loader, optimizer, criterion_verb, criterion_noun,
            device, scaler, epoch, config
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion_verb, criterion_noun, device)

        # Update learning rate
        scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['train_verb_acc'].append(train_verb_acc)
        history['train_noun_acc'].append(train_noun_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_verb_acc'].append(val_metrics['verb_acc_top1'])
        history['val_noun_acc'].append(val_metrics['noun_acc_top1'])
        history['val_action_acc'].append(val_metrics['action_acc'])

        # Print epoch summary
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{config.EPOCHS} Summary")
        print(f"{'='*70}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc:  Verb={train_verb_acc:.2f}%, Noun={train_noun_acc:.2f}%")
        print(f"\nVal Loss:   {val_metrics['loss']:.4f}")
        print(f"Val Acc:    Verb={val_metrics['verb_acc_top1']:.2f}%, Noun={val_metrics['noun_acc_top1']:.2f}%, Action={val_metrics['action_acc']:.2f}%")
        print(f"Val Top-5:  Verb={val_metrics['verb_acc_top5']:.2f}%, Noun={val_metrics['noun_acc_top5']:.2f}%")

        # Calculate overfitting gap
        verb_gap = train_verb_acc - val_metrics['verb_acc_top1']
        noun_gap = train_noun_acc - val_metrics['noun_acc_top1']
        print(f"\nOverfitting Gap: Verb={verb_gap:.1f}%, Noun={noun_gap:.1f}%")
        print(f"{'='*70}\n")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_metrics': val_metrics,
                'history': history
            }, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}\n")

        # Save best model
        val_acc_avg = (val_metrics['verb_acc_top1'] + val_metrics['noun_acc_top1']) / 2
        if val_acc_avg > best_val_acc:
            best_val_acc = val_acc_avg
            best_path = checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'history': history
            }, best_path)
            print(f"✓ New best model saved: {best_path} (Val Acc: {val_acc_avg:.2f}%)\n")

        # Early stopping check - DISABLED
        # early_stopping(val_metrics['loss'])
        # if early_stopping.early_stop:
        #     print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
        #     break

    # Save final history
    history_path = log_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"History saved to: {history_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Improved EPIC-KITCHENS Model')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='outputs_improved',
                       help='Output directory for checkpoints and logs')

    args = parser.parse_args()

    main(args)
