"""
Validation script with participant-based holdout
Tests true generalization by evaluating on participants the model never saw during training
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
import argparse
from pathlib import Path
import pandas as pd
import sys

sys.path.append(str(Path(__file__).parent.parent))

from common.config import Config
from common.dataset import EPICKitchensDataset
from torch.utils.data import DataLoader
from phase1.model import get_model


def get_holdout_dataloaders(config, holdout_participants=['P29', 'P30', 'P31']):
    """
    Create train and holdout validation loaders based on participant split.

    Args:
        config: Config object
        holdout_participants: List of participant IDs to hold out for validation

    Returns:
        train_loader, val_loader
    """
    print("=" * 70)
    print("Creating Participant-Based Holdout Validation Split")
    print("=" * 70)

    # Load training annotations
    train_annotations = pd.read_csv(config.TRAIN_CSV)

    # Filter by participant
    train_annotations['participant'] = train_annotations['video_id'].str.split('_').str[0]

    train_data = train_annotations[~train_annotations['participant'].isin(holdout_participants)]
    val_data = train_annotations[train_annotations['participant'].isin(holdout_participants)]

    print(f"\nHoldout participants: {holdout_participants}")
    print(f"Training participants: {sorted(train_data['participant'].unique())}")
    print(f"Training segments: {len(train_data)}")
    print(f"Validation segments: {len(val_data)}")

    # Create datasets
    train_dataset = EPICKitchensDataset(
        annotations_csv=config.TRAIN_CSV,
        video_dir=config.VIDEO_DIR,
        num_frames=config.NUM_FRAMES,
        image_size=config.IMAGE_SIZE,
        mode='train'
    )
    train_dataset.annotations = train_data.reset_index(drop=True)

    val_dataset = EPICKitchensDataset(
        annotations_csv=config.TRAIN_CSV,
        video_dir=config.VIDEO_DIR,
        num_frames=config.NUM_FRAMES,
        image_size=config.IMAGE_SIZE,
        mode='val'
    )
    val_dataset.annotations = val_data.reset_index(drop=True)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    print(f"Batch size:    {config.BATCH_SIZE}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print("=" * 70)
    print()

    return train_loader, val_loader


def validate_checkpoint(checkpoint_path, config, holdout_participants):
    """Validate checkpoint on holdout participants."""

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: {device} (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    print(f"Checkpoint from epoch: {checkpoint['epoch']}")
    if 'best_acc' in checkpoint:
        print(f"Best accuracy so far: {checkpoint['best_acc']:.2f}%")

    # Initialize model
    print("\nInitializing model...")
    model = get_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load data with participant holdout
    print("\nLoading validation data (holdout participants)...")
    train_loader, val_loader = get_holdout_dataloaders(config, holdout_participants)

    if len(val_loader) == 0:
        print("ERROR: No validation data available for holdout participants!")
        return

    print(f"Validation set size: {len(val_loader.dataset)} samples")

    # Loss functions
    criterion_verb = nn.CrossEntropyLoss()
    criterion_noun = nn.CrossEntropyLoss()

    # Validate
    print("\nRunning validation on UNSEEN participants...")
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
    print("VALIDATION RESULTS (UNSEEN PARTICIPANTS)")
    print("="*60)
    print(f"Holdout Participants: {holdout_participants}")
    print(f"Validation Loss:    {avg_loss:.4f}")
    print(f"\nTop-1 Accuracy:")
    print(f"  Verb:             {verb_acc:.2f}%")
    print(f"  Noun:             {noun_acc:.2f}%")
    print(f"  Action (V+N):     {action_acc:.2f}%")
    print(f"\nTop-5 Accuracy:")
    print(f"  Verb:             {verb_top5_acc:.2f}%")
    print(f"  Noun:             {noun_top5_acc:.2f}%")
    print("="*60)

    # Overfitting analysis
    print("\nOVERFITTING ANALYSIS:")
    print("  Training accuracy at epoch 11: ~76% verb, ~78% noun")
    print(f"  Validation accuracy (unseen participants): {verb_acc:.2f}% verb, {noun_acc:.2f}% noun")

    estimated_train_verb = 76.0
    estimated_train_noun = 78.0

    verb_drop = estimated_train_verb - verb_acc
    noun_drop = estimated_train_noun - noun_acc

    print(f"\n  Accuracy drop:")
    print(f"    Verb: {estimated_train_verb:.1f}% → {verb_acc:.2f}% ({verb_drop:+.2f}%)")
    print(f"    Noun: {estimated_train_noun:.1f}% → {noun_acc:.2f}% ({noun_drop:+.2f}%)")

    if verb_drop > 15 or noun_drop > 15:
        print(f"\n⚠️  SEVERE overfitting detected!")
        print(f"  Model does NOT generalize to unseen participants.")
        print(f"  Recommendations: More data, better regularization, or simpler model.")
    elif verb_drop > 10 or noun_drop > 10:
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
    parser = argparse.ArgumentParser(description='Validate checkpoint on holdout participants')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for validation')
    parser.add_argument('--holdout', type=str, nargs='+', default=['P29', 'P30', 'P31'],
                        help='Participant IDs to hold out (default: P29 P30 P31)')

    args = parser.parse_args()

    # Load config
    config = Config()
    config.BATCH_SIZE = args.batch_size

    # Validate
    validate_checkpoint(args.checkpoint, config, args.holdout)
