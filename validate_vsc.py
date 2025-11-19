"""
Validation script for VSC - validates a checkpoint on true validation set
Optimized for A100 GPU
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
from pathlib import Path

from common.config import Config
from common.dataset import EPICKitchensDataset
from phase1.model import get_model


def validate_model(checkpoint_path, config, batch_size=32):
    """Validate a model checkpoint on true validation set."""

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Checkpoint from epoch: {epoch}")

    # Create model
    model = get_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load validation dataset
    print(f"\nLoading validation dataset...")
    val_dataset = EPICKitchensDataset(
        annotations_csv=config.VAL_CSV,
        video_dir=config.VIDEO_DIR,
        num_frames=config.NUM_FRAMES,
        image_size=config.IMAGE_SIZE,
        mode='val'
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Validation set: {len(val_dataset)} segments")

    # Validation metrics
    criterion_verb = nn.CrossEntropyLoss()
    criterion_noun = nn.CrossEntropyLoss()

    total_loss = 0.0
    verb_correct_top1 = 0
    verb_correct_top5 = 0
    noun_correct_top1 = 0
    noun_correct_top5 = 0
    action_correct = 0
    total_samples = 0

    print(f"\nRunning validation...")

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

            # Action accuracy (both verb and noun correct)
            action_correct += ((verb_pred_top1.squeeze() == verb_labels) & (noun_pred_top1.squeeze() == noun_labels)).sum().item()

            total_samples += frames.size(0)

    # Calculate metrics
    avg_loss = total_loss / total_samples
    verb_acc_top1 = 100.0 * verb_correct_top1 / total_samples
    verb_acc_top5 = 100.0 * verb_correct_top5 / total_samples
    noun_acc_top1 = 100.0 * noun_correct_top1 / total_samples
    noun_acc_top5 = 100.0 * noun_correct_top5 / total_samples
    action_acc = 100.0 * action_correct / total_samples

    # Print results
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Checkpoint: {Path(checkpoint_path).name}")
    print(f"Epoch: {epoch}")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"\nTop-1 Accuracy:")
    print(f"  Verb:   {verb_acc_top1:.2f}%")
    print(f"  Noun:   {noun_acc_top1:.2f}%")
    print(f"  Action: {action_acc:.2f}%")
    print(f"\nTop-5 Accuracy:")
    print(f"  Verb:   {verb_acc_top5:.2f}%")
    print(f"  Noun:   {noun_acc_top5:.2f}%")
    print(f"{'='*60}\n")

    # Save results
    results = {
        'checkpoint': str(checkpoint_path),
        'epoch': epoch,
        'val_loss': avg_loss,
        'verb_acc_top1': verb_acc_top1,
        'verb_acc_top5': verb_acc_top5,
        'noun_acc_top1': noun_acc_top1,
        'noun_acc_top5': noun_acc_top5,
        'action_acc': action_acc,
        'total_samples': total_samples
    }

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate EPIC-KITCHENS model on VSC')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for validation')
    parser.add_argument('--output', type=str, default='validation_results.json',
                       help='Output JSON file for results')

    args = parser.parse_args()

    config = Config()

    # Run validation
    results = validate_model(args.checkpoint, config, args.batch_size)

    # Save results to JSON
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")
