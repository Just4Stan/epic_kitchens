"""
Training script for VideoMAE (State-of-the-Art) model
"""

import sys
sys.path.insert(0, '.')

from phase1.train import train
from common.config import Config
import argparse
from pathlib import Path
# Note: model_videomae was in deleted models/ directory - need to check if it exists elsewhere
# import model_videomae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VideoMAE Model (SOTA)')
    parser.add_argument('--data_dir', type=str, default='EPIC-KITCHENS')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)  # Smaller due to larger model
    parser.add_argument('--lr', type=float, default=5e-5)  # Lower LR for fine-tuning
    parser.add_argument('--num_frames', type=int, default=16)  # VideoMAE uses 16 frames
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()

    # Create config
    config = Config(
        DATA_DIR=Path(args.data_dir),
        EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        LEARNING_RATE=args.lr,
        NUM_FRAMES=args.num_frames,
        DEVICE=args.device,
        NUM_WORKERS=args.num_workers,
        OUTPUT_DIR=Path('outputs_videomae'),
        CHECKPOINT_DIR=Path('outputs_videomae/checkpoints')
    )

    # Override model getter to use VideoMAE
    def get_model_videomae(cfg):
        return model_videomae.get_model(cfg, pretrained=True)

    from phase1 import model as original_model
    original_model.get_model = get_model_videomae

    # Train
    train(config)
