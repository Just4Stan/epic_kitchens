"""
Training script for 3D CNN model
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from main train.py
from phase1.train import train
from common.config import Config
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train 3D CNN Model')
    parser.add_argument('--data_dir', type=str, default='EPIC-KITCHENS')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)  # Smaller due to memory
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_frames', type=int, default=8)
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
        OUTPUT_DIR=Path('outputs_3dcnn'),
        CHECKPOINT_DIR=Path('outputs_3dcnn/checkpoints')
    )

    # Import and override model getter to use 3D
    from phase1.model import get_model as original_get_model

    def get_model_3d(cfg):
        return original_get_model(cfg, use_3d=True)

    from phase1 import model as model_module
    model_module.get_model = get_model_3d

    # Train
    train(config)
