"""
Ensemble validation - combine predictions from multiple models.
"""

import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models import ActionModel
from config import Config as cfg
from validate import LocalValDataset


@torch.no_grad()
def get_predictions(model, loader, device):
    """Get softmax predictions from a model."""
    model.eval()
    all_verb_probs = []
    all_noun_probs = []
    all_verb_labels = []
    all_noun_labels = []

    for frames, verb_labels, noun_labels in tqdm(loader, desc="Getting predictions"):
        frames = frames.to(device)
        verb_logits, noun_logits = model(frames)

        verb_probs = F.softmax(verb_logits, dim=1)
        noun_probs = F.softmax(noun_logits, dim=1)

        all_verb_probs.append(verb_probs.cpu())
        all_noun_probs.append(noun_probs.cpu())
        all_verb_labels.append(verb_labels)
        all_noun_labels.append(noun_labels)

    return {
        'verb_probs': torch.cat(all_verb_probs),
        'noun_probs': torch.cat(all_noun_probs),
        'verb_labels': torch.cat(all_verb_labels),
        'noun_labels': torch.cat(all_noun_labels),
    }


def evaluate_predictions(verb_probs, noun_probs, verb_labels, noun_labels):
    """Evaluate accuracy from probability predictions."""
    verb_pred = verb_probs.argmax(1)
    noun_pred = noun_probs.argmax(1)

    verb_correct = (verb_pred == verb_labels).sum().item()
    noun_correct = (noun_pred == noun_labels).sum().item()
    action_correct = ((verb_pred == verb_labels) & (noun_pred == noun_labels)).sum().item()
    total = len(verb_labels)

    return {
        'verb_acc': 100 * verb_correct / total,
        'noun_acc': 100 * noun_correct / total,
        'action_acc': 100 * action_correct / total,
        'total': total
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, nargs='+', required=True,
                       help="Paths to checkpoint files")
    parser.add_argument("--weights", type=float, nargs='+', default=None,
                       help="Weights for each model (default: equal)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Weights
    n_models = len(args.checkpoints)
    if args.weights is None:
        weights = [1.0 / n_models] * n_models
    else:
        assert len(args.weights) == n_models
        total = sum(args.weights)
        weights = [w / total for w in args.weights]

    print(f"\nEnsemble of {n_models} models:")
    for i, (ckpt, w) in enumerate(zip(args.checkpoints, weights)):
        print(f"  [{i+1}] {ckpt} (weight={w:.2f})")

    # Paths
    epic_dir = Path(__file__).parent.parent
    val_csv = epic_dir / cfg.VAL_CSV
    video_dir = epic_dir / "EPIC-KITCHENS" / "videos_640x360"

    # Dataset
    val_dataset = LocalValDataset(
        annotations_csv=val_csv,
        video_base_dir=video_dir,
        num_frames=cfg.NUM_FRAMES,
        image_size=cfg.IMAGE_SIZE
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    # Get predictions from each model
    all_predictions = []
    for i, ckpt_path in enumerate(args.checkpoints):
        print(f"\n[{i+1}/{n_models}] Loading {ckpt_path}...")

        model = ActionModel(
            num_verb_classes=cfg.NUM_VERB_CLASSES,
            num_noun_classes=cfg.NUM_NOUN_CLASSES,
            backbone=cfg.BACKBONE,
            temporal_model=cfg.TEMPORAL_MODEL,
            dropout=cfg.DROPOUT,
            num_frames=cfg.NUM_FRAMES
        ).to(device)

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint)

        preds = get_predictions(model, val_loader, device)
        all_predictions.append(preds)

        # Individual model accuracy
        metrics = evaluate_predictions(
            preds['verb_probs'], preds['noun_probs'],
            preds['verb_labels'], preds['noun_labels']
        )
        print(f"  Individual: Verb={metrics['verb_acc']:.2f}%, Noun={metrics['noun_acc']:.2f}%, Action={metrics['action_acc']:.2f}%")

        # Clear model from memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Ensemble predictions (weighted average)
    print("\nComputing ensemble predictions...")
    ensemble_verb_probs = sum(w * p['verb_probs'] for w, p in zip(weights, all_predictions))
    ensemble_noun_probs = sum(w * p['noun_probs'] for w, p in zip(weights, all_predictions))

    # Evaluate ensemble
    metrics = evaluate_predictions(
        ensemble_verb_probs, ensemble_noun_probs,
        all_predictions[0]['verb_labels'], all_predictions[0]['noun_labels']
    )

    print(f"\n{'='*50}")
    print(f"ENSEMBLE RESULTS ({n_models} models)")
    print(f"{'='*50}")
    print(f"  Verb Accuracy:   {metrics['verb_acc']:.2f}%")
    print(f"  Noun Accuracy:   {metrics['noun_acc']:.2f}%")
    print(f"  Action Accuracy: {metrics['action_acc']:.2f}%")
    print(f"  Total samples:   {metrics['total']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
