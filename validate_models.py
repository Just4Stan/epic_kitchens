"""
Validation Script for All Trained Models
Evaluates each model's checkpoints on the validation set and generates comparison report
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse
from collections import defaultdict
import sys

# Add current directory to path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

# Import model architectures from phase directories
from phase1.model_lstm import ActionRecognitionLSTM
from phase1.model_transformer import ActionRecognitionTransformer
from phase1.model_advanced import EfficientNetLSTM, EfficientNetTransformer
# Add this if you have a 3D CNN model:
# from phase1.model_3dcnn import CNN3DActionRecognizer

from common.dataset import EPICKitchensDataset
from common.config import Config


def load_model(model_type, checkpoint_path, device, config):
    """Load a model from checkpoint."""
    print(f"\nLoading {model_type} from {checkpoint_path}...")

    # Initialize model based on type
    if model_type == 'lstm':
        model = ActionRecognitionLSTM(
            num_verbs=config.NUM_VERB_CLASSES,
            num_nouns=config.NUM_NOUN_CLASSES,
            hidden_dim=512,
            num_layers=2,
            dropout=0.5
        )
    elif model_type == 'transformer':
        model = ActionRecognitionTransformer(
            num_verbs=config.NUM_VERB_CLASSES,
            num_nouns=config.NUM_NOUN_CLASSES,
            d_model=512,
            nhead=8,
            num_layers=6,
            dropout=0.3
        )
    elif model_type == '3dcnn':
        # You need to import CNN3DActionRecognizer from the appropriate module
        from phase1.model_3dcnn import CNN3DActionRecognizer
        model = CNN3DActionRecognizer(
            num_verbs=config.NUM_VERB_CLASSES,
            num_nouns=config.NUM_NOUN_CLASSES
        )
    elif model_type == 'efficientnet_lstm':
        model = EfficientNetLSTM(
            num_verbs=config.NUM_VERB_CLASSES,
            num_nouns=config.NUM_NOUN_CLASSES
        )
    elif model_type == 'efficientnet_transformer':
        model = EfficientNetTransformer(
            num_verbs=config.NUM_VERB_CLASSES,
            num_nouns=config.NUM_NOUN_CLASSES
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loaded checkpoint from epoch {epoch}")

    return model, epoch


def evaluate_model(model, val_loader, device, config):
    """Evaluate model on validation set."""
    verb_correct_top1 = 0
    verb_correct_top5 = 0
    noun_correct_top1 = 0
    noun_correct_top5 = 0
    action_correct = 0
    total_samples = 0

    all_verb_preds = []
    all_noun_preds = []
    all_verb_labels = []
    all_noun_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating")
        for frames, verb_labels, noun_labels in pbar:
            frames = frames.to(device)
            verb_labels = verb_labels.to(device)
            noun_labels = noun_labels.to(device)

            # Forward pass
            verb_logits, noun_logits = model(frames)

            batch_size = frames.size(0)
            total_samples += batch_size

            # Top-1 predictions
            verb_pred_top1 = verb_logits.argmax(dim=1)
            noun_pred_top1 = noun_logits.argmax(dim=1)

            # Top-5 predictions
            verb_pred_top5 = verb_logits.topk(5, dim=1)[1]
            noun_pred_top5 = noun_logits.topk(5, dim=1)[1]

            # Top-1 accuracy
            verb_correct_top1 += (verb_pred_top1 == verb_labels).sum().item()
            noun_correct_top1 += (noun_pred_top1 == noun_labels).sum().item()

            # Top-5 accuracy
            for i in range(batch_size):
                if verb_labels[i] in verb_pred_top5[i]:
                    verb_correct_top5 += 1
                if noun_labels[i] in noun_pred_top5[i]:
                    noun_correct_top5 += 1

            # Action accuracy (both verb and noun correct)
            action_correct += ((verb_pred_top1 == verb_labels) & (noun_pred_top1 == noun_labels)).sum().item()

            # Store predictions for per-class analysis
            all_verb_preds.extend(verb_pred_top1.cpu().numpy())
            all_noun_preds.extend(noun_pred_top1.cpu().numpy())
            all_verb_labels.extend(verb_labels.cpu().numpy())
            all_noun_labels.extend(noun_labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'Verb Acc': f"{100*verb_correct_top1/total_samples:.2f}%",
                'Noun Acc': f"{100*noun_correct_top1/total_samples:.2f}%",
                'Action Acc': f"{100*action_correct/total_samples:.2f}%"
            })

    # Calculate final metrics
    results = {
        'total_samples': total_samples,
        'verb_top1_acc': 100 * verb_correct_top1 / total_samples,
        'verb_top5_acc': 100 * verb_correct_top5 / total_samples,
        'noun_top1_acc': 100 * noun_correct_top1 / total_samples,
        'noun_top5_acc': 100 * noun_correct_top5 / total_samples,
        'action_acc': 100 * action_correct / total_samples,
        'predictions': {
            'verb_preds': all_verb_preds,
            'noun_preds': all_noun_preds,
            'verb_labels': all_verb_labels,
            'noun_labels': all_noun_labels
        }
    }

    return results


def find_best_checkpoint(output_dir):
    """Find the checkpoint with the highest epoch number."""
    checkpoint_dir = Path(output_dir) / 'checkpoints'
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    if not checkpoints:
        return None

    # Sort by epoch number and get the last one
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
    return checkpoints[-1]


def main():
    parser = argparse.ArgumentParser(description='Validate all trained models')
    parser.add_argument('--val_csv', type=str, required=True, help='Path to validation CSV')
    parser.add_argument('--val_video_dir', type=str, required=True, help='Path to validation videos')
    parser.add_argument('--output_file', type=str, default='validation_results.json', help='Output JSON file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for validation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = Config()

    # Define models to evaluate
    models_to_eval = [
        ('lstm', 'outputs_lstm'),
        ('transformer', 'outputs_transformer'),
        ('3dcnn', 'outputs_3dcnn'),
        ('efficientnet_lstm', 'outputs_efficientnet_lstm'),
        ('efficientnet_transformer', 'outputs_efficientnet_transformer'),
    ]

    # Create validation dataset
    print("\n" + "="*80)
    print("Setting up validation dataset...")
    print("="*80)

    val_dataset = EPICKitchensDataset(
        annotations_csv=args.val_csv,
        video_dir=args.val_video_dir,
        num_frames=config.NUM_FRAMES,
        image_size=config.IMAGE_SIZE,
        mode='val'
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Validation set: {len(val_dataset)} samples")

    # Evaluate each model
    all_results = {}

    for model_type, output_dir in models_to_eval:
        print("\n" + "="*80)
        print(f"Evaluating: {model_type.upper()}")
        print("="*80)

        # Find best checkpoint
        checkpoint_path = find_best_checkpoint(output_dir)
        if checkpoint_path is None:
            print(f"⚠️  No checkpoint found for {model_type}, skipping...")
            continue

        try:
            # Load model
            model, epoch = load_model(model_type, checkpoint_path, device, config)

            # Evaluate
            results = evaluate_model(model, val_loader, device, config)

            # Store results (without predictions to save space)
            all_results[model_type] = {
                'checkpoint': str(checkpoint_path),
                'epoch': epoch,
                'verb_top1_acc': results['verb_top1_acc'],
                'verb_top5_acc': results['verb_top5_acc'],
                'noun_top1_acc': results['noun_top1_acc'],
                'noun_top5_acc': results['noun_top5_acc'],
                'action_acc': results['action_acc'],
                'total_samples': results['total_samples']
            }

            print(f"\n✓ Results for {model_type}:")
            print(f"  Verb Accuracy:   Top-1: {results['verb_top1_acc']:.2f}%  Top-5: {results['verb_top5_acc']:.2f}%")
            print(f"  Noun Accuracy:   Top-1: {results['noun_top1_acc']:.2f}%  Top-5: {results['noun_top5_acc']:.2f}%")
            print(f"  Action Accuracy: {results['action_acc']:.2f}%")

        except Exception as e:
            print(f"❌ Error evaluating {model_type}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_path = Path(args.output_file)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\n✓ Results saved to: {output_path}")

    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"{'Model':<25} {'Verb Top-1':<12} {'Noun Top-1':<12} {'Action Acc':<12}")
    print("-" * 80)

    sorted_models = sorted(all_results.items(),
                          key=lambda x: x[1]['action_acc'],
                          reverse=True)

    for model_name, results in sorted_models:
        print(f"{model_name:<25} {results['verb_top1_acc']:>10.2f}%  {results['noun_top1_acc']:>10.2f}%  {results['action_acc']:>10.2f}%")

    print("="*80)


if __name__ == '__main__':
    main()
