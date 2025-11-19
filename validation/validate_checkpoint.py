"""
Validation script to evaluate a saved checkpoint on validation data
This helps detect overfitting by comparing training vs validation performance
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))

from common.config import Config
from utils.dataset_split import get_dataloaders
from phase1.model import get_model


def validate_checkpoint(checkpoint_path, config):
    """
    Load a checkpoint and evaluate it on the validation set.

    Args:
        checkpoint_path: Path to the checkpoint.pth file
        config: Config object with hyperparameters
    """
    # Use MPS for M3 Pro, CUDA for GPU, or CPU as fallback
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: {device} (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")
    print(f"Device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Print checkpoint info
    print(f"Checkpoint from epoch: {checkpoint['epoch']}")
    if 'best_acc' in checkpoint:
        print(f"Best accuracy so far: {checkpoint['best_acc']:.2f}%")
    if 'train_loss' in checkpoint:
        print(f"Training loss: {checkpoint['train_loss']:.4f}")
    if 'train_verb_acc' in checkpoint:
        print(f"Training verb acc: {checkpoint['train_verb_acc']:.2f}%")
    if 'train_noun_acc' in checkpoint:
        print(f"Training noun acc: {checkpoint['train_noun_acc']:.2f}%")

    # Initialize model
    print("\nInitializing model...")
    model = get_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load data with validation split
    print("\nLoading validation data...")
    train_loader, val_loader = get_dataloaders(
        config,
        val_split=0.1,  # 10% validation split
        random_seed=42
    )

    if len(val_loader) == 0:
        print("ERROR: No validation data available!")
        return

    print(f"Validation set size: {len(val_loader.dataset)} samples")

    # Loss functions
    criterion_verb = nn.CrossEntropyLoss()
    criterion_noun = nn.CrossEntropyLoss()

    # Validate
    print("\nRunning validation...")
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
            if config.USE_AMP:
                with autocast():
                    verb_logits, noun_logits = model(frames)
                    loss_verb = criterion_verb(verb_logits, verb_labels)
                    loss_noun = criterion_noun(noun_logits, noun_labels)
                    loss = config.VERB_LOSS_WEIGHT * loss_verb + config.NOUN_LOSS_WEIGHT * loss_noun
            else:
                verb_logits, noun_logits = model(frames)
                loss_verb = criterion_verb(verb_logits, verb_labels)
                loss_noun = criterion_noun(noun_logits, noun_labels)
                loss = config.VERB_LOSS_WEIGHT * loss_verb + config.NOUN_LOSS_WEIGHT * loss_noun

            # Top-1 predictions
            verb_pred = verb_logits.argmax(dim=1)
            noun_pred = noun_logits.argmax(dim=1)

            # Top-5 predictions
            _, verb_top5_pred = verb_logits.topk(5, dim=1)
            _, noun_top5_pred = noun_logits.topk(5, dim=1)

            # Calculate accuracies
            verb_correct += (verb_pred == verb_labels).sum().item()
            noun_correct += (noun_pred == noun_labels).sum().item()
            action_correct += ((verb_pred == verb_labels) & (noun_pred == noun_labels)).sum().item()

            # Top-5 accuracy
            verb_top5_correct += sum([1 for i, label in enumerate(verb_labels) if label in verb_top5_pred[i]])
            noun_top5_correct += sum([1 for i, label in enumerate(noun_labels) if label in noun_top5_pred[i]])

            total_loss += loss.item()
            total_samples += frames.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'verb_acc': f'{100 * verb_correct / total_samples:.2f}%',
                'noun_acc': f'{100 * noun_correct / total_samples:.2f}%'
            })

    # Final statistics
    avg_loss = total_loss / len(val_loader)
    verb_acc = 100 * verb_correct / total_samples
    noun_acc = 100 * noun_correct / total_samples
    action_acc = 100 * action_correct / total_samples
    verb_top5_acc = 100 * verb_top5_correct / total_samples
    noun_top5_acc = 100 * noun_top5_correct / total_samples

    # Print results
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"Validation Loss:    {avg_loss:.4f}")
    print(f"\nTop-1 Accuracy:")
    print(f"  Verb:             {verb_acc:.2f}%")
    print(f"  Noun:             {noun_acc:.2f}%")
    print(f"  Action (V+N):     {action_acc:.2f}%")
    print(f"\nTop-5 Accuracy:")
    print(f"  Verb:             {verb_top5_acc:.2f}%")
    print(f"  Noun:             {noun_top5_acc:.2f}%")
    print("="*60)

    # Compare to training metrics (if available)
    # Note: Since checkpoint doesn't save training metrics, user should compare manually
    # with the training logs showing ~76% verb and ~78% noun at epoch 11
    print("\nOVERFITTING ANALYSIS:")
    print("  Compare these validation results with your training logs.")
    print("  Training accuracy at epoch 11: ~76% verb, ~78% noun")
    print(f"  Validation accuracy: {verb_acc:.2f}% verb, {noun_acc:.2f}% noun")

    # Rough estimate based on user's reported training accuracy
    estimated_train_verb = 76.0  # From user's terminal output
    estimated_train_noun = 78.0

    verb_drop = estimated_train_verb - verb_acc
    noun_drop = estimated_train_noun - noun_acc

    print(f"\n  Estimated accuracy drop:")
    print(f"    Verb: {estimated_train_verb:.1f}% → {verb_acc:.2f}% ({verb_drop:.2f}% decrease)")
    print(f"    Noun: {estimated_train_noun:.1f}% → {noun_acc:.2f}% ({noun_drop:.2f}% decrease)")

    if verb_drop > 10 or noun_drop > 10:
        print(f"\n⚠️  WARNING: Significant overfitting detected!")
        print(f"  Training accuracy is much higher than validation accuracy.")
        print(f"  Consider: early stopping, regularization, or data augmentation.")
    elif verb_drop > 5 or noun_drop > 5:
        print(f"\n⚠️  Moderate overfitting detected.")
        print(f"  Some generalization gap, but within acceptable range.")
    else:
        print(f"\n✓ Good generalization - minimal overfitting.")

    print("="*60)

    return {
        'val_loss': avg_loss,
        'val_verb_acc': verb_acc,
        'val_noun_acc': noun_acc,
        'val_action_acc': action_acc,
        'val_verb_top5_acc': verb_top5_acc,
        'val_noun_top5_acc': noun_top5_acc,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate a checkpoint on validation data')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (e.g., outputs/checkpoint_epoch_10.pth)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for validation')

    args = parser.parse_args()

    # Load config
    config = Config()
    config.BATCH_SIZE = args.batch_size

    # Validate
    validate_checkpoint(args.checkpoint, config)
