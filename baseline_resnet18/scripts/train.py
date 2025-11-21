"""
Training Script for Simple ResNet-18 Baseline

Clean, minimal training loop:
- AdamW optimizer
- ReduceLROnPlateau scheduler
- Mixed precision training
- Validation every epoch
- Save best model based on action accuracy
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import time

from configs.config import Config
from models.resnet18_simple import create_model
from data.dataset import create_dataloaders


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    # Make cuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, config):
    """Train for one epoch."""
    model.train()

    running_loss = 0.0
    verb_correct = 0
    noun_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")

    for batch_idx, (frames, verb_labels, noun_labels) in enumerate(pbar):
        # Move to device
        frames = frames.to(device)
        verb_labels = verb_labels.to(device)
        noun_labels = noun_labels.to(device)

        # Forward pass with mixed precision
        optimizer.zero_grad()

        if config.MIXED_PRECISION:
            with autocast():
                verb_logits, noun_logits = model(frames)
                verb_loss = criterion(verb_logits, verb_labels)
                noun_loss = criterion(noun_logits, noun_labels)
                loss = verb_loss + noun_loss

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            verb_logits, noun_logits = model(frames)
            verb_loss = criterion(verb_logits, verb_labels)
            noun_loss = criterion(noun_logits, noun_labels)
            loss = verb_loss + noun_loss

            loss.backward()
            optimizer.step()

        # Statistics
        batch_size = frames.size(0)
        running_loss += loss.item() * batch_size

        _, verb_preds = verb_logits.max(1)
        _, noun_preds = noun_logits.max(1)

        verb_correct += (verb_preds == verb_labels).sum().item()
        noun_correct += (noun_preds == noun_labels).sum().item()
        total_samples += batch_size

        # Update progress bar
        if (batch_idx + 1) % config.LOG_INTERVAL == 0 or (batch_idx + 1) == len(dataloader):
            avg_loss = running_loss / total_samples
            verb_acc = 100.0 * verb_correct / total_samples
            noun_acc = 100.0 * noun_correct / total_samples
            action_acc = 100.0 * sum((verb_preds == verb_labels) & (noun_preds == noun_labels)).item() / batch_size

            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'verb': f'{verb_acc:.2f}%',
                'noun': f'{noun_acc:.2f}%',
                'action': f'{action_acc:.2f}%'
            })

    # Epoch statistics
    avg_loss = running_loss / total_samples
    verb_acc = 100.0 * verb_correct / total_samples
    noun_acc = 100.0 * noun_correct / total_samples

    return avg_loss, verb_acc, noun_acc


def validate(model, dataloader, criterion, device, epoch, config):
    """Validate the model."""
    model.eval()

    running_loss = 0.0
    verb_correct = 0
    noun_correct = 0
    action_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]  ")

    with torch.no_grad():
        for frames, verb_labels, noun_labels in pbar:
            # Move to device
            frames = frames.to(device)
            verb_labels = verb_labels.to(device)
            noun_labels = noun_labels.to(device)

            # Forward pass
            if config.MIXED_PRECISION:
                with autocast():
                    verb_logits, noun_logits = model(frames)
                    verb_loss = criterion(verb_logits, verb_labels)
                    noun_loss = criterion(noun_logits, noun_labels)
                    loss = verb_loss + noun_loss
            else:
                verb_logits, noun_logits = model(frames)
                verb_loss = criterion(verb_logits, verb_labels)
                noun_loss = criterion(noun_logits, noun_labels)
                loss = verb_loss + noun_loss

            # Statistics
            batch_size = frames.size(0)
            running_loss += loss.item() * batch_size

            _, verb_preds = verb_logits.max(1)
            _, noun_preds = noun_logits.max(1)

            verb_correct += (verb_preds == verb_labels).sum().item()
            noun_correct += (noun_preds == noun_labels).sum().item()
            action_correct += ((verb_preds == verb_labels) & (noun_preds == noun_labels)).sum().item()
            total_samples += batch_size

            # Update progress bar
            verb_acc = 100.0 * verb_correct / total_samples
            noun_acc = 100.0 * noun_correct / total_samples
            action_acc = 100.0 * action_correct / total_samples

            pbar.set_postfix({
                'loss': f'{running_loss/total_samples:.4f}',
                'verb': f'{verb_acc:.2f}%',
                'noun': f'{noun_acc:.2f}%',
                'action': f'{action_acc:.2f}%'
            })

    # Epoch statistics
    avg_loss = running_loss / total_samples
    verb_acc = 100.0 * verb_correct / total_samples
    noun_acc = 100.0 * noun_correct / total_samples
    action_acc = 100.0 * action_correct / total_samples

    return avg_loss, verb_acc, noun_acc, action_acc


def save_checkpoint(model, optimizer, scheduler, epoch, best_action_acc, config, filename):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_action_acc': best_action_acc,
    }
    torch.save(checkpoint, config.CHECKPOINT_DIR / filename)
    print(f"  Saved: {filename}")


def main():
    """Main training function."""
    # Configuration
    config = Config()
    print(config)

    # Set seed
    set_seed(config.SEED)
    print(f"Random seed: {config.SEED}\n")

    # Device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Create model
    model = create_model(config)
    model = model.to(device)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)

    # Loss function (no label smoothing!)
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler (reduce LR on plateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize action accuracy
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE,
        verbose=True
    )

    # Mixed precision scaler
    scaler = GradScaler() if config.MIXED_PRECISION else None

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    best_action_acc = 0.0
    start_time = time.time()

    # Results log
    results_file = config.OUTPUT_DIR / "training_log.txt"
    with open(results_file, 'w') as f:
        f.write("Epoch,Train_Loss,Train_Verb_Acc,Train_Noun_Acc,"
                "Val_Loss,Val_Verb_Acc,Val_Noun_Acc,Val_Action_Acc,LR\n")

    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss, train_verb_acc, train_noun_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, config
        )

        # Validate
        val_loss, val_verb_acc, val_noun_acc, val_action_acc = validate(
            model, val_loader, criterion, device, epoch, config
        )

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
        print(f"  Train - Loss: {train_loss:.4f} | Verb: {train_verb_acc:.2f}% | Noun: {train_noun_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f} | Verb: {val_verb_acc:.2f}% | Noun: {val_noun_acc:.2f}% | Action: {val_action_acc:.2f}%")

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning rate: {current_lr:.6f}")

        # Log results
        with open(results_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_verb_acc:.2f},{train_noun_acc:.2f},"
                   f"{val_loss:.4f},{val_verb_acc:.2f},{val_noun_acc:.2f},{val_action_acc:.2f},{current_lr:.6f}\n")

        # Save checkpoint
        if (epoch + 1) % config.SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_action_acc,
                          config, f"checkpoint_epoch_{epoch+1}.pth")

        # Save best model
        if val_action_acc > best_action_acc:
            best_action_acc = val_action_acc
            save_checkpoint(model, optimizer, scheduler, epoch, best_action_acc,
                          config, "best_model.pth")
            print(f"  New best action accuracy: {best_action_acc:.2f}%")

        # Update learning rate
        scheduler.step(val_action_acc)

        print()

    # Training complete
    elapsed = time.time() - start_time
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Total time: {elapsed/3600:.2f} hours")
    print(f"Best action accuracy: {best_action_acc:.2f}%")
    print(f"Results saved to: {results_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
