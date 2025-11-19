"""
Training script for EPIC-KITCHENS-100 Action Recognition
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import time
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config import Config
from common.dataset import get_dataloaders
from phase1.model import get_model


def train_epoch(model, train_loader, criterion_verb, criterion_noun, optimizer, scaler, device, epoch, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    verb_correct = 0
    noun_correct = 0
    action_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")

    for batch_idx, (frames, verb_labels, noun_labels) in enumerate(pbar):
        # Move to device
        frames = frames.to(device)
        verb_labels = verb_labels.to(device)
        noun_labels = noun_labels.to(device)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        if config.USE_AMP:
            with autocast():
                verb_logits, noun_logits = model(frames)
                loss_verb = criterion_verb(verb_logits, verb_labels)
                loss_noun = criterion_noun(noun_logits, noun_labels)
                loss = config.VERB_LOSS_WEIGHT * loss_verb + config.NOUN_LOSS_WEIGHT * loss_noun

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            verb_logits, noun_logits = model(frames)
            loss_verb = criterion_verb(verb_logits, verb_labels)
            loss_noun = criterion_noun(noun_logits, noun_labels)
            loss = config.VERB_LOSS_WEIGHT * loss_verb + config.NOUN_LOSS_WEIGHT * loss_noun

            loss.backward()
            optimizer.step()

        # Calculate accuracies
        verb_pred = verb_logits.argmax(dim=1)
        noun_pred = noun_logits.argmax(dim=1)

        verb_correct += (verb_pred == verb_labels).sum().item()
        noun_correct += (noun_pred == noun_labels).sum().item()
        action_correct += ((verb_pred == verb_labels) & (noun_pred == noun_labels)).sum().item()

        total_loss += loss.item()
        total_samples += frames.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'verb_acc': f'{100 * verb_correct / total_samples:.2f}%',
            'noun_acc': f'{100 * noun_correct / total_samples:.2f}%'
        })

    # Epoch statistics
    avg_loss = total_loss / len(train_loader)
    verb_acc = 100 * verb_correct / total_samples
    noun_acc = 100 * noun_correct / total_samples
    action_acc = 100 * action_correct / total_samples

    return avg_loss, verb_acc, noun_acc, action_acc


def validate(model, val_loader, criterion_verb, criterion_noun, device, config):
    """Validate the model."""
    model.eval()
    total_loss = 0
    verb_correct = 0
    noun_correct = 0
    action_correct = 0
    total_samples = 0

    verb_top5_correct = 0
    noun_top5_correct = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")

        for frames, verb_labels, noun_labels in pbar:
            frames = frames.to(device)
            verb_labels = verb_labels.to(device)
            noun_labels = noun_labels.to(device)

            # Forward pass
            verb_logits, noun_logits = model(frames)

            # Calculate loss
            loss_verb = criterion_verb(verb_logits, verb_labels)
            loss_noun = criterion_noun(noun_logits, noun_labels)
            loss = config.VERB_LOSS_WEIGHT * loss_verb + config.NOUN_LOSS_WEIGHT * loss_noun

            total_loss += loss.item()

            # Top-1 accuracy
            verb_pred = verb_logits.argmax(dim=1)
            noun_pred = noun_logits.argmax(dim=1)

            verb_correct += (verb_pred == verb_labels).sum().item()
            noun_correct += (noun_pred == noun_labels).sum().item()
            action_correct += ((verb_pred == verb_labels) & (noun_pred == noun_labels)).sum().item()

            # Top-5 accuracy
            verb_top5_pred = verb_logits.topk(5, dim=1)[1]
            noun_top5_pred = noun_logits.topk(5, dim=1)[1]

            verb_top5_correct += (verb_top5_pred == verb_labels.unsqueeze(1)).any(dim=1).sum().item()
            noun_top5_correct += (noun_top5_pred == noun_labels.unsqueeze(1)).any(dim=1).sum().item()

            total_samples += frames.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'verb_acc': f'{100 * verb_correct / total_samples:.2f}%',
                'noun_acc': f'{100 * noun_correct / total_samples:.2f}%'
            })

    # Validation statistics
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    verb_acc = 100 * verb_correct / total_samples if total_samples > 0 else 0
    noun_acc = 100 * noun_correct / total_samples if total_samples > 0 else 0
    action_acc = 100 * action_correct / total_samples if total_samples > 0 else 0
    verb_top5_acc = 100 * verb_top5_correct / total_samples if total_samples > 0 else 0
    noun_top5_acc = 100 * noun_top5_correct / total_samples if total_samples > 0 else 0

    return avg_loss, verb_acc, noun_acc, action_acc, verb_top5_acc, noun_top5_acc


def save_checkpoint(model, optimizer, epoch, best_acc, config, filename='checkpoint.pth'):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'config': vars(config)
    }
    checkpoint_path = config.CHECKPOINT_DIR / filename
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")


def train(config):
    """Main training loop."""

    print("\n" + "=" * 70)
    print("EPIC-KITCHENS-100 Action Recognition Training")
    print("=" * 70)
    print(config)
    print()

    # Set device
    if config.DEVICE == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Apple Silicon GPU (MPS)\n")
    elif config.DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        device = torch.device("cpu")
        print("⚠ Using CPU (training will be slow)\n")

    # Create dataloaders
    train_loader, val_loader = get_dataloaders(config)

    # Create model
    model = get_model(config, use_3d=False)
    model = model.to(device)

    # Loss functions
    criterion_verb = nn.CrossEntropyLoss()
    criterion_noun = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # Mixed precision scaler
    scaler = GradScaler() if config.USE_AMP else None

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'verb_acc': [],
        'noun_acc': [],
        'action_acc': [],
        'verb_top5_acc': [],
        'noun_top5_acc': []
    }

    best_action_acc = 0
    patience_counter = 0

    # Training loop
    print("=" * 70)
    print("Starting Training")
    print("=" * 70)
    print()

    start_time = time.time()

    for epoch in range(config.EPOCHS):
        # Train
        train_loss, train_verb_acc, train_noun_acc, train_action_acc = train_epoch(
            model, train_loader, criterion_verb, criterion_noun, optimizer, scaler, device, epoch, config
        )

        # Validate
        val_loss, val_verb_acc, val_noun_acc, val_action_acc, val_verb_top5, val_noun_top5 = validate(
            model, val_loader, criterion_verb, criterion_noun, device, config
        )

        # Update scheduler
        scheduler.step(val_action_acc)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['verb_acc'].append(val_verb_acc)
        history['noun_acc'].append(val_noun_acc)
        history['action_acc'].append(val_action_acc)
        history['verb_top5_acc'].append(val_verb_top5)
        history['noun_top5_acc'].append(val_noun_top5)

        # Print epoch summary
        print()
        print("=" * 70)
        print(f"Epoch {epoch+1}/{config.EPOCHS} Summary")
        print("=" * 70)
        print(f"Train Loss:       {train_loss:.4f}")
        print(f"Val Loss:         {val_loss:.4f}")
        print(f"Verb Acc (Top-1): {val_verb_acc:.2f}%  (Top-5): {val_verb_top5:.2f}%")
        print(f"Noun Acc (Top-1): {val_noun_acc:.2f}%  (Top-5): {val_noun_top5:.2f}%")
        print(f"Action Acc:       {val_action_acc:.2f}%")
        print("=" * 70)
        print()

        # Save checkpoint
        if (epoch + 1) % config.SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, epoch, best_action_acc, config, f'checkpoint_epoch_{epoch+1}.pth')

        # Save best model
        if val_action_acc > best_action_acc:
            best_action_acc = val_action_acc
            save_checkpoint(model, optimizer, epoch, best_action_acc, config, 'best_model.pth')
            patience_counter = 0
            print(f"★ New best action accuracy: {best_action_acc:.2f}%\n")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {config.PATIENCE} epochs)")
            break

    # Training complete
    total_time = time.time() - start_time
    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Total time:       {total_time / 3600:.2f} hours")
    print(f"Best action acc:  {best_action_acc:.2f}%")
    print("=" * 70)

    # Save training history
    history_path = config.OUTPUT_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved: {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train EPIC-KITCHENS-100 Action Recognition')

    parser.add_argument('--data_dir', type=str, default='EPIC-KITCHENS', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_frames', type=int, default=8, help='Frames per video')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'], help='Device')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')

    args = parser.parse_args()

    # Create config
    config = Config(
        DATA_DIR=Path(args.data_dir),
        EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        LEARNING_RATE=args.lr,
        NUM_FRAMES=args.num_frames,
        DEVICE=args.device,
        NUM_WORKERS=args.num_workers
    )

    # Train
    train(config)
