"""
Training script for baseline model with train/val split
Uses dataset_split.py to create validation set from training data
"""

import sys
sys.path.insert(0, '.')

from common.config import Config
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json

# Import split dataset instead of regular dataset
from utils import dataset_split
from phase1.model import get_model


def train_epoch(model, train_loader, criterion_verb, criterion_noun, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    verb_correct = 0
    noun_correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (frames, verb_labels, noun_labels) in enumerate(pbar):
        frames = frames.to(device)
        verb_labels = verb_labels.to(device)
        noun_labels = noun_labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        verb_logits, noun_logits = model(frames)

        # Compute losses
        verb_loss = criterion_verb(verb_logits, verb_labels)
        noun_loss = criterion_noun(noun_logits, noun_labels)
        total_loss = verb_loss + noun_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Statistics
        running_loss += total_loss.item()
        verb_pred = verb_logits.argmax(dim=1)
        noun_pred = noun_logits.argmax(dim=1)
        verb_correct += (verb_pred == verb_labels).sum().item()
        noun_correct += (noun_pred == noun_labels).sum().item()
        total += verb_labels.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / (batch_idx + 1):.4f}',
            'verb_acc': f'{100.0 * verb_correct / total:.2f}%',
            'noun_acc': f'{100.0 * noun_correct / total:.2f}%'
        })

    return running_loss / len(train_loader), verb_correct / total, noun_correct / total


def validate(model, val_loader, criterion_verb, criterion_noun, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    verb_correct = 0
    noun_correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for frames, verb_labels, noun_labels in pbar:
            frames = frames.to(device)
            verb_labels = verb_labels.to(device)
            noun_labels = noun_labels.to(device)

            # Forward pass
            verb_logits, noun_logits = model(frames)

            # Compute losses
            verb_loss = criterion_verb(verb_logits, verb_labels)
            noun_loss = criterion_noun(noun_logits, noun_labels)
            total_loss = verb_loss + noun_loss

            # Statistics
            running_loss += total_loss.item()
            verb_pred = verb_logits.argmax(dim=1)
            noun_pred = noun_logits.argmax(dim=1)
            verb_correct += (verb_pred == verb_labels).sum().item()
            noun_correct += (noun_pred == noun_labels).sum().item()
            total += verb_labels.size(0)

    return running_loss / len(val_loader), verb_correct / total, noun_correct / total


def train(config):
    """Main training function"""
    print("=" * 70)
    print("EPIC-KITCHENS Action Recognition - Baseline Model with Train/Val Split")
    print("=" * 70)
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Number of frames: {config.NUM_FRAMES}")
    print()

    # Create output directories
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Get dataloaders with 90/10 split
    train_loader, val_loader = dataset_split.get_dataloaders(config, val_split=0.1, random_seed=42)

    # Get model
    device = torch.device(config.DEVICE)
    model = get_model(config).to(device)

    # Loss functions
    criterion_verb = nn.CrossEntropyLoss()
    criterion_noun = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'train_verb_acc': [],
        'train_noun_acc': [],
        'val_loss': [],
        'val_verb_acc': [],
        'val_noun_acc': []
    }

    best_val_acc = 0.0

    # Training loop
    print("=" * 70)
    print("Starting Training")
    print("=" * 70)

    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        print("-" * 70)

        # Train
        train_loss, train_verb_acc, train_noun_acc = train_epoch(
            model, train_loader, criterion_verb, criterion_noun, optimizer, device
        )

        # Validate
        val_loss, val_verb_acc, val_noun_acc = validate(
            model, val_loader, criterion_verb, criterion_noun, device
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_verb_acc'].append(train_verb_acc)
        history['train_noun_acc'].append(train_noun_acc)
        history['val_loss'].append(val_loss)
        history['val_verb_acc'].append(val_verb_acc)
        history['val_noun_acc'].append(val_noun_acc)

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train Verb Acc: {train_verb_acc * 100:.2f}% | Val Verb Acc: {val_verb_acc * 100:.2f}%")
        print(f"  Train Noun Acc: {train_noun_acc * 100:.2f}% | Val Noun Acc: {val_noun_acc * 100:.2f}%")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_verb_acc': train_verb_acc,
            'train_noun_acc': train_noun_acc,
            'val_verb_acc': val_verb_acc,
            'val_noun_acc': val_noun_acc
        }

        # Save latest checkpoint
        torch.save(checkpoint, config.CHECKPOINT_DIR / 'latest_checkpoint.pth')

        # Save best checkpoint
        val_acc = (val_verb_acc + val_noun_acc) / 2
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, config.CHECKPOINT_DIR / 'best_checkpoint.pth')
            print(f"  ✓ Best model saved! (avg val acc: {val_acc * 100:.2f}%)")

        # Save history
        with open(config.OUTPUT_DIR / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best validation accuracy: {best_val_acc * 100:.2f}%")
    print(f"Checkpoints saved to: {config.CHECKPOINT_DIR}")
    print(f"Training history saved to: {config.OUTPUT_DIR / 'training_history.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Baseline Model with Train/Val Split')
    parser.add_argument('--data_dir', type=str, default='EPIC-KITCHENS')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()

    # Create config
    config = Config(
        DATA_DIR=Path(args.data_dir),
        EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        LEARNING_RATE=args.lr,
        NUM_FRAMES=args.num_frames,
        DEVICE=args.device,
        NUM_WORKERS=args.num_workers,
        OUTPUT_DIR=Path('outputs_baseline_split'),
        CHECKPOINT_DIR=Path('outputs_baseline_split/checkpoints')
    )

    # Train
    train(config)
