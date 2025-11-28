"""
Generate CodaBench submission file for EPIC-KITCHENS-100.

Submission format: submission.pt containing list of dicts:
[
    {
        'narration_id': 'P01_101_0',
        'verb_output': tensor[97],
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
def generate_predictions(models, weights, loader, device):
    """Generate ensemble predictions for submission."""
    for model in models:
        model.eval()

    predictions = []

    for narration_ids, frames in tqdm(loader, desc="Generating predictions"):
        frames = frames.to(device)
        batch_size = frames.size(0)

        # Get weighted ensemble predictions
        ensemble_verb = torch.zeros(batch_size, cfg.NUM_VERB_CLASSES, device=device)
        ensemble_noun = torch.zeros(batch_size, cfg.NUM_NOUN_CLASSES, device=device)

        for model, weight in zip(models, weights):
            verb_logits, noun_logits = model(frames)
            # Use softmax probabilities for ensemble
            ensemble_verb += weight * F.softmax(verb_logits, dim=1)
            ensemble_noun += weight * F.softmax(noun_logits, dim=1)

        # Store predictions for each sample
        for i in range(batch_size):
            predictions.append({
                'narration_id': narration_ids[i],
                'verb_output': ensemble_verb[i].cpu(),
                'noun_output': ensemble_noun[i].cpu(),
            })

    return predictions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, nargs='+', required=True,
                       help="Paths to checkpoint files")
    parser.add_argument("--weights", type=float, nargs='+', default=None,
                       help="Weights for each model (default: equal)")
    parser.add_argument("--annotations", type=str, default=None,
                       help="Path to annotations CSV (default: validation)")
    parser.add_argument("--video_dir", type=str, default=None,
                       help="Path to video directory")
    parser.add_argument("--output", type=str, default="submission.pt",
                       help="Output filename")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
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

    # Load models
    models = []
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
        models.append(model)

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = generate_predictions(models, weights, loader, device)

    # Save submission
    output_path = Path(args.output)
    torch.save(predictions, output_path)
    print(f"\nSaved {len(predictions)} predictions to {output_path}")

    # Verify format
    print("\nVerifying submission format...")
    loaded = torch.load(output_path, weights_only=False)
    sample = loaded[0]
    print(f"  Total predictions: {len(loaded)}")
    print(f"  Sample narration_id: {sample['narration_id']}")
    print(f"  verb_output shape: {sample['verb_output'].shape}")
    print(f"  noun_output shape: {sample['noun_output'].shape}")

    print(f"\nTo create submission zip:")
    print(f"  zip submission.zip {output_path}")


if __name__ == "__main__":
    main()
