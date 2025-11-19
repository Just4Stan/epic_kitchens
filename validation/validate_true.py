"""
Validation on TRUE validation set from EPIC_100_validation.csv
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from common.config import Config
from common.dataset import EPICKitchensDataset
from torch.utils.data import DataLoader
from phase1.model import get_model


def validate_on_true_val(checkpoint_path, config):
    """Validate on actual validation set."""

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: {device} (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Checkpoint from epoch: {checkpoint['epoch']}")

    # Load model
    model = get_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load ACTUAL validation dataset
    print("\nLoading TRUE validation dataset...")
    val_dataset = EPICKitchensDataset(
        annotations_csv=config.VAL_CSV,
        video_dir=config.VIDEO_DIR,
        num_frames=config.NUM_FRAMES,
        image_size=config.IMAGE_SIZE,
        mode='val'
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=False  # Disable for MPS
    )

    print(f"Validation set: {len(val_dataset)} segments")

    # Loss functions
    criterion_verb = nn.CrossEntropyLoss()
    criterion_noun = nn.CrossEntropyLoss()

    # Validate
    print("\nRunning validation on TRUE validation set...")
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
            loss_verb = criterion_verb(verb_logits, verb_labels)
            loss_noun = criterion_noun(noun_logits, noun_labels)
            loss = config.VERB_LOSS_WEIGHT * loss_verb + config.NOUN_LOSS_WEIGHT * loss_noun

            # Top-1
            verb_pred = verb_logits.argmax(dim=1)
            noun_pred = noun_logits.argmax(dim=1)

            # Top-5
            _, verb_top5_pred = verb_logits.topk(5, dim=1)
            _, noun_top5_pred = noun_logits.topk(5, dim=1)

            verb_correct += (verb_pred == verb_labels).sum().item()
            noun_correct += (noun_pred == noun_labels).sum().item()
            action_correct += ((verb_pred == verb_labels) & (noun_pred == noun_labels)).sum().item()

            verb_top5_correct += sum([1 for i, label in enumerate(verb_labels) if label in verb_top5_pred[i]])
            noun_top5_correct += sum([1 for i, label in enumerate(noun_labels) if label in noun_top5_pred[i]])

            total_loss += loss.item()
            total_samples += frames.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'verb_acc': f'{100 * verb_correct / total_samples:.2f}%',
                'noun_acc': f'{100 * noun_correct / total_samples:.2f}%'
            })

    # Results
    avg_loss = total_loss / len(val_loader)
    verb_acc = 100 * verb_correct / total_samples
    noun_acc = 100 * noun_correct / total_samples
    action_acc = 100 * action_correct / total_samples
    verb_top5_acc = 100 * verb_top5_correct / total_samples
    noun_top5_acc = 100 * noun_top5_correct / total_samples

    print("\n" + "="*60)
    print("VALIDATION RESULTS (TRUE VALIDATION SET)")
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

    # Overfitting analysis
    print("\nOVERFITTING ANALYSIS:")
    print("  Training accuracy at epoch 11: ~76% verb, ~78% noun")
    print(f"  TRUE validation accuracy: {verb_acc:.2f}% verb, {noun_acc:.2f}% noun")

    verb_drop = 76.0 - verb_acc
    noun_drop = 78.0 - noun_acc

    print(f"\n  Performance drop:")
    print(f"    Verb: 76.0% → {verb_acc:.2f}% ({verb_drop:+.2f}%)")
    print(f"    Noun: 78.0% → {noun_acc:.2f}% ({noun_drop:+.2f}%)")

    if verb_drop > 20 or noun_drop > 20:
        print(f"\n⚠️  SEVERE overfitting detected!")
    elif verb_drop > 10 or noun_drop > 10:
        print(f"\n⚠️  Significant overfitting detected!")
    elif verb_drop > 5 or noun_drop > 5:
        print(f"\n⚠️  Moderate overfitting.")
    else:
        print(f"\n✓ Good generalization!")

    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    config = Config()
    config.BATCH_SIZE = args.batch_size

    validate_on_true_val(args.checkpoint, config)
