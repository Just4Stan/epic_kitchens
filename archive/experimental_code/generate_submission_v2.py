"""
Generate CodaBench submission file for EPIC-KITCHENS-100.
Version 2: Simplified, single-model or ensemble with proper format.

Submission format: submission.pt containing list of dicts:
[
    {
        'narration_id': 'P01_101_0',
        'verb_output': tensor[97],  # logits or probabilities
        'noun_output': tensor[300]
    },
    ...
]
"""

import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from models import ActionModel
from config import Config as cfg


class SubmissionDataset(Dataset):
    """Dataset for generating submissions - returns narration_id with frames."""

    def __init__(self, annotations_csv, video_base_dir, num_frames=16, image_size=224):
        self.annotations = pd.read_csv(annotations_csv)
        self.video_base_dir = Path(video_base_dir)
        self.num_frames = num_frames
        self.image_size = image_size

        # Find valid videos
        self.valid_indices = []
        for idx in range(len(self.annotations)):
            row = self.annotations.iloc[idx]
            participant = row['participant_id']
            video_id = row['video_id']

            # Try both flat and participant folder structures
            video_path = self.video_base_dir / f"{video_id}.MP4"
            if not video_path.exists():
                video_path = self.video_base_dir / participant / f"{video_id}.MP4"

            if video_path.exists():
                self.valid_indices.append(idx)

        print(f"Found {len(self.valid_indices)}/{len(self.annotations)} valid segments")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.valid_indices)

    def _get_video_path(self, participant, video_id):
        video_path = self.video_base_dir / f"{video_id}.MP4"
        if not video_path.exists():
            video_path = self.video_base_dir / participant / f"{video_id}.MP4"
        return video_path

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.annotations.iloc[real_idx]

        narration_id = row['narration_id']
        participant = row['participant_id']
        video_id = row['video_id']
        start_frame = int(row['start_frame'])
        stop_frame = int(row['stop_frame'])

        video_path = self._get_video_path(participant, video_id)

        # Extract frames
        frames = self._extract_frames(video_path, start_frame, stop_frame)
        if frames is None:
            frames = [np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)] * self.num_frames

        frames_tensor = torch.stack([self.transform(f) for f in frames])
        return narration_id, frames_tensor

    def _extract_frames(self, video_path, start_frame, stop_frame):
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None

            indices = np.linspace(start_frame, stop_frame, self.num_frames, dtype=int)
            frames = []
            last_frame = None

            for frame_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    last_frame = frame
                    frames.append(frame)
                elif last_frame is not None:
                    frames.append(last_frame)
                else:
                    frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))

            cap.release()
            return frames if len(frames) == self.num_frames else None
        except:
            return None


@torch.no_grad()
def generate_single_model_predictions(model, loader, device, use_softmax=True):
    """Generate predictions from a single model."""
    model.eval()
    predictions = []

    for narration_ids, frames in tqdm(loader, desc="Generating predictions"):
        frames = frames.to(device)
        batch_size = frames.size(0)

        verb_logits, noun_logits = model(frames)

        if use_softmax:
            verb_out = F.softmax(verb_logits, dim=1)
            noun_out = F.softmax(noun_logits, dim=1)
        else:
            verb_out = verb_logits
            noun_out = noun_logits

        for i in range(batch_size):
            predictions.append({
                'narration_id': narration_ids[i],
                'verb_output': verb_out[i].cpu().float(),
                'noun_output': noun_out[i].cpu().float(),
            })

    return predictions


@torch.no_grad()
def generate_ensemble_predictions(models, weights, loader, device, use_softmax=True):
    """Generate ensemble predictions - average logits then optionally softmax."""
    for model in models:
        model.eval()

    predictions = []

    for narration_ids, frames in tqdm(loader, desc="Generating predictions"):
        frames = frames.to(device)
        batch_size = frames.size(0)

        # Average LOGITS (not softmax) for proper ensemble
        ensemble_verb = torch.zeros(batch_size, cfg.NUM_VERB_CLASSES, device=device)
        ensemble_noun = torch.zeros(batch_size, cfg.NUM_NOUN_CLASSES, device=device)

        for model, weight in zip(models, weights):
            verb_logits, noun_logits = model(frames)
            ensemble_verb += weight * verb_logits
            ensemble_noun += weight * noun_logits

        if use_softmax:
            verb_out = F.softmax(ensemble_verb, dim=1)
            noun_out = F.softmax(ensemble_noun, dim=1)
        else:
            verb_out = ensemble_verb
            noun_out = ensemble_noun

        for i in range(batch_size):
            predictions.append({
                'narration_id': narration_ids[i],
                'verb_output': verb_out[i].cpu().float(),
                'noun_output': noun_out[i].cpu().float(),
            })

    return predictions


def verify_submission(predictions, verbose=True):
    """Verify submission format and values."""
    issues = []

    if not isinstance(predictions, list):
        issues.append(f"predictions is not a list: {type(predictions)}")
        return issues

    if len(predictions) == 0:
        issues.append("predictions is empty")
        return issues

    for i, pred in enumerate(predictions):
        if not isinstance(pred, dict):
            issues.append(f"[{i}] not a dict: {type(pred)}")
            continue

        # Check keys
        required_keys = {'narration_id', 'verb_output', 'noun_output'}
        missing = required_keys - set(pred.keys())
        if missing:
            issues.append(f"[{i}] missing keys: {missing}")
            continue

        # Check narration_id is string
        if not isinstance(pred['narration_id'], str):
            issues.append(f"[{i}] narration_id not string: {type(pred['narration_id'])}")

        # Check verb_output
        verb = pred['verb_output']
        if not isinstance(verb, torch.Tensor):
            issues.append(f"[{i}] verb_output not tensor: {type(verb)}")
        elif verb.shape != (cfg.NUM_VERB_CLASSES,):
            issues.append(f"[{i}] verb_output shape: {verb.shape} (expected {cfg.NUM_VERB_CLASSES})")
        elif torch.isnan(verb).any():
            issues.append(f"[{i}] verb_output has NaN")
        elif torch.isinf(verb).any():
            issues.append(f"[{i}] verb_output has Inf")

        # Check noun_output
        noun = pred['noun_output']
        if not isinstance(noun, torch.Tensor):
            issues.append(f"[{i}] noun_output not tensor: {type(noun)}")
        elif noun.shape != (cfg.NUM_NOUN_CLASSES,):
            issues.append(f"[{i}] noun_output shape: {noun.shape} (expected {cfg.NUM_NOUN_CLASSES})")
        elif torch.isnan(noun).any():
            issues.append(f"[{i}] noun_output has NaN")
        elif torch.isinf(noun).any():
            issues.append(f"[{i}] noun_output has Inf")

    if verbose:
        if issues:
            print(f"\n❌ Found {len(issues)} issues:")
            for issue in issues[:20]:
                print(f"  - {issue}")
            if len(issues) > 20:
                print(f"  ... and {len(issues) - 20} more")
        else:
            print("\n✓ Submission format verified OK")

            # Print sample stats
            sample = predictions[0]
            verb = sample['verb_output']
            noun = sample['noun_output']
            print(f"  Total predictions: {len(predictions)}")
            print(f"  Sample narration_id: {sample['narration_id']}")
            print(f"  verb_output: shape={verb.shape}, dtype={verb.dtype}, sum={verb.sum():.4f}")
            print(f"  noun_output: shape={noun.shape}, dtype={noun.dtype}, sum={noun.sum():.4f}")
            print(f"  verb range: [{verb.min():.4f}, {verb.max():.4f}]")
            print(f"  noun range: [{noun.min():.4f}, {noun.max():.4f}]")

    return issues


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to single checkpoint file")
    parser.add_argument("--checkpoint2", type=str, default=None,
                       help="Optional second checkpoint for ensemble")
    parser.add_argument("--weight1", type=float, default=0.5,
                       help="Weight for first model (only used with ensemble)")
    parser.add_argument("--annotations", type=str, default=None,
                       help="Path to annotations CSV (default: validation)")
    parser.add_argument("--video_dir", type=str, default=None,
                       help="Path to video directory")
    parser.add_argument("--output", type=str, default="submission.pt",
                       help="Output filename")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_softmax", action="store_true",
                       help="Output raw logits instead of softmax")
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Paths
    epic_dir = Path(__file__).parent.parent
    if args.annotations:
        annotations_csv = Path(args.annotations)
    else:
        annotations_csv = epic_dir / cfg.VAL_CSV

    if args.video_dir:
        video_dir = Path(args.video_dir)
    else:
        video_dir = epic_dir / "EPIC-KITCHENS" / "videos_640x360"

    print(f"\nAnnotations: {annotations_csv}")
    print(f"Video dir: {video_dir}")
    print(f"Output mode: {'logits' if args.no_softmax else 'softmax probabilities'}")

    # Dataset
    dataset = SubmissionDataset(
        annotations_csv=annotations_csv,
        video_base_dir=video_dir,
        num_frames=cfg.NUM_FRAMES,
        image_size=cfg.IMAGE_SIZE
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    # Load model(s)
    def load_model(ckpt_path):
        print(f"Loading {ckpt_path}...")
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
        return model

    model1 = load_model(args.checkpoint)

    if args.checkpoint2:
        model2 = load_model(args.checkpoint2)
        models = [model1, model2]
        weights = [args.weight1, 1.0 - args.weight1]
        print(f"\nEnsemble mode: weights = {weights}")
        predictions = generate_ensemble_predictions(
            models, weights, loader, device,
            use_softmax=not args.no_softmax
        )
    else:
        print("\nSingle model mode")
        predictions = generate_single_model_predictions(
            model1, loader, device,
            use_softmax=not args.no_softmax
        )

    # Verify format
    print("\nVerifying submission format...")
    issues = verify_submission(predictions)

    if issues:
        print(f"\n⚠️  {len(issues)} issues found - submission may be invalid")

    # Save submission
    output_path = Path(args.output)
    torch.save(predictions, output_path)
    print(f"\n✓ Saved {len(predictions)} predictions to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Reload and re-verify
    print("\nRe-loading and verifying saved file...")
    loaded = torch.load(output_path, weights_only=False)
    verify_submission(loaded)

    print(f"\nTo create submission zip:")
    print(f"  zip submission.zip {output_path}")


if __name__ == "__main__":
    main()
