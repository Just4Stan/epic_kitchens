"""
Validation script for Simple ResNet-18 Baseline
"""

import sys
import os
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from tqdm import tqdm

from configs.config import Config
from models.resnet18_simple import create_model
from data.dataset import create_dataloaders


def validate(model, dataloader, device):
    """Validate model and compute detailed metrics."""
    model.eval()

    verb_correct = 0
    noun_correct = 0
    action_correct = 0
    total_samples = 0

    # For per-class accuracy
    verb_class_correct = {}
    verb_class_total = {}
    noun_class_correct = {}
    noun_class_total = {}

    print("\nRunning validation...")
    pbar = tqdm(dataloader, desc="Validating")

    with torch.no_grad():
        for frames, verb_labels, noun_labels in pbar:
            # Move to device
            frames = frames.to(device)
            verb_labels = verb_labels.to(device)
            noun_labels = noun_labels.to(device)

            # Forward pass
            verb_logits, noun_logits = model(frames)

            # Predictions
            _, verb_preds = verb_logits.max(1)
            _, noun_preds = noun_logits.max(1)

            # Overall accuracy
            verb_correct += (verb_preds == verb_labels).sum().item()
            noun_correct += (noun_preds == noun_labels).sum().item()
            action_correct += ((verb_preds == verb_labels) & (noun_preds == noun_labels)).sum().item()
            total_samples += frames.size(0)

            # Per-class accuracy
            for i in range(len(verb_labels)):
                v_label = verb_labels[i].item()
                n_label = noun_labels[i].item()

                # Verb
                if v_label not in verb_class_total:
                    verb_class_total[v_label] = 0
                    verb_class_correct[v_label] = 0
                verb_class_total[v_label] += 1
                if verb_preds[i] == v_label:
                    verb_class_correct[v_label] += 1

                # Noun
                if n_label not in noun_class_total:
                    noun_class_total[n_label] = 0
                    noun_class_correct[n_label] = 0
                noun_class_total[n_label] += 1
                if noun_preds[i] == n_label:
                    noun_class_correct[n_label] += 1

    # Compute accuracies
    verb_acc = 100.0 * verb_correct / total_samples
    noun_acc = 100.0 * noun_correct / total_samples
    action_acc = 100.0 * action_correct / total_samples

    # Results
    results = {
        'total_samples': total_samples,
        'verb_accuracy': verb_acc,
        'noun_accuracy': noun_acc,
        'action_accuracy': action_acc,
        'verb_class_acc': {k: 100.0 * verb_class_correct[k] / verb_class_total[k]
                          for k in verb_class_total.keys()},
        'noun_class_acc': {k: 100.0 * noun_class_correct[k] / noun_class_total[k]
                          for k in noun_class_total.keys()}
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate ResNet-18 baseline")
    parser.add_argument('--checkpoint', type=str, default='outputs/checkpoints/best_model.pth',
                       help='Path to checkpoint')
    parser.add_argument('--output', type=str, default='validation_results.txt',
                       help='Output file for results')
    args = parser.parse_args()

    # Configuration
    config = Config()
    print(config)

    # Device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    # Create model
    model = create_model(config)
    model = model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")
    print(f"Best training action accuracy: {checkpoint['best_action_acc']:.2f}%\n")

    # Create validation dataloader
    print("Creating validation dataloader...")
    _, val_loader = create_dataloaders(config)

    # Validate
    results = validate(model, val_loader, device)

    # Print results
    print("\n" + "=" * 70)
    print("Validation Results")
    print("=" * 70)
    print(f"Total samples:    {results['total_samples']:,}")
    print(f"Verb accuracy:    {results['verb_accuracy']:.2f}%")
    print(f"Noun accuracy:    {results['noun_accuracy']:.2f}%")
    print(f"Action accuracy:  {results['action_accuracy']:.2f}%")
    print("=" * 70)

    # Save results
    with open(args.output, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Validation Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Total samples:    {results['total_samples']:,}\n")
        f.write(f"Verb accuracy:    {results['verb_accuracy']:.2f}%\n")
        f.write(f"Noun accuracy:    {results['noun_accuracy']:.2f}%\n")
        f.write(f"Action accuracy:  {results['action_accuracy']:.2f}%\n")
        f.write("=" * 70 + "\n")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
