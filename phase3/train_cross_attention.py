"""
Phase 3: Cross-Attention Model Training Script
Trains the cross-task attention model for EPIC-KITCHENS action recognition
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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config import Config
from phase2.dataset_improved import get_improved_dataloaders
from phase3.model_cross_attention import get_cross_attention_model


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing."""
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


def train_epoch(model, train_loader, optimizer, criterion_verb, criterion_noun,
                device, scaler, epoch):
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

        with autocast():
            verb_logits, noun_logits = model(frames)
            loss_verb = criterion_verb(verb_logits, verb_labels)
            loss_noun = criterion_noun(noun_logits, noun_labels)
            loss = 0.5 * loss_verb + 0.5 * loss_noun

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Calculate accuracy
        verb_pred = verb_logits.argmax(dim=1)
        noun_pred = noun_logits.argmax(dim=1)
        verb_correct += (verb_pred == verb_labels).sum().item()
        noun_correct += (noun_pred == noun_labels).sum().item()
        total_samples += frames.size(0)
        total_loss += loss.item() * frames.size(0)

        pbar.set_postfix({
            'loss': f'{total_loss/total_samples:.4f}',
            'v_acc': f'{100*verb_correct/total_samples:.2f}%',
            'n_acc': f'{100*noun_correct/total_samples:.2f}%'
        })

    return {
        'loss': total_loss / total_samples,
        'verb_acc': 100 * verb_correct / total_samples,
        'noun_acc': 100 * noun_correct / total_samples
    }


def validate_epoch(model, val_loader, criterion_verb, criterion_noun, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    verb_correct = 0
    verb_correct_top5 = 0
    noun_correct = 0
    noun_correct_top5 = 0
    action_correct = 0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for frames, verb_labels, noun_labels in pbar:
            frames = frames.to(device)
            verb_labels = verb_labels.to(device)
            noun_labels = noun_labels.to(device)

            verb_logits, noun_logits = model(frames)

            loss_verb = criterion_verb(verb_logits, verb_labels)
            loss_noun = criterion_noun(noun_logits, noun_labels)
            loss = 0.5 * loss_verb + 0.5 * loss_noun

            # Top-1 accuracy
            verb_pred = verb_logits.argmax(dim=1)
            noun_pred = noun_logits.argmax(dim=1)
            verb_correct += (verb_pred == verb_labels).sum().item()
            noun_correct += (noun_pred == noun_labels).sum().item()
            action_correct += ((verb_pred == verb_labels) & (noun_pred == noun_labels)).sum().item()

            # Top-5 accuracy
            verb_top5 = verb_logits.topk(5, dim=1)[1]
            noun_top5 = noun_logits.topk(5, dim=1)[1]
            for i in range(frames.size(0)):
                if verb_labels[i] in verb_top5[i]:
                    verb_correct_top5 += 1
                if noun_labels[i] in noun_top5[i]:
                    noun_correct_top5 += 1

            total_samples += frames.size(0)
            total_loss += loss.item() * frames.size(0)

            pbar.set_postfix({
                'loss': f'{total_loss/total_samples:.4f}',
                'v_acc': f'{100*verb_correct/total_samples:.2f}%',
                'n_acc': f'{100*noun_correct/total_samples:.2f}%'
            })

    return {
        'loss': total_loss / total_samples,
        'verb_acc': 100 * verb_correct / total_samples,
        'verb_acc_top5': 100 * verb_correct_top5 / total_samples,
        'noun_acc': 100 * noun_correct / total_samples,
        'noun_acc_top5': 100 * noun_correct_top5 / total_samples,
        'action_acc': 100 * action_correct / total_samples
    }


def main():
    parser = argparse.ArgumentParser(description='Train Cross-Attention Model (Phase 3)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--output_dir', type=str, default='outputs_cross_attention')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = Config()
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Data loaders
    print("\nLoading datasets...")
    train_loader, val_loader = get_improved_dataloaders(
        config,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS
    )

    # Model
    print("\nInitializing cross-attention model...")
    model = get_cross_attention_model(config, dropout=args.dropout)
    model = model.to(device)

    # Loss and optimizer
    criterion_verb = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    criterion_noun = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler()

    # Training loop
    print("\n" + "="*70)
    print("Starting Training - Cross-Attention Model (Phase 3)")
    print("="*70)

    best_val_acc = 0.0
    history = {'train': [], 'val': []}

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion_verb, criterion_noun,
            device, scaler, epoch
        )

        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion_verb, criterion_noun, device
        )

        scheduler.step()

        # Save metrics
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # Print epoch summary
        print("\n" + "="*70)
        print(f"Epoch {epoch+1}/{args.epochs} Summary")
        print("="*70)
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Train Acc:  Verb={train_metrics['verb_acc']:.2f}%, Noun={train_metrics['noun_acc']:.2f}%")
        print(f"\nVal Loss:   {val_metrics['loss']:.4f}")
        print(f"Val Acc:    Verb={val_metrics['verb_acc']:.2f}%, Noun={val_metrics['noun_acc']:.2f}%, Action={val_metrics['action_acc']:.2f}%")
        print(f"Val Top-5:  Verb={val_metrics['verb_acc_top5']:.2f}%, Noun={val_metrics['noun_acc_top5']:.2f}%")
        print("="*70)

        # Save best model
        val_acc = (val_metrics['verb_acc'] + val_metrics['noun_acc']) / 2
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_metrics': val_metrics
            }, checkpoint_dir / 'best_model.pth')
            print(f"✓ New best model saved: {checkpoint_dir}/best_model.pth (Val Acc: {val_acc:.2f}%)\n")

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: {checkpoint_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
