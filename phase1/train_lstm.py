"""
Training script for LSTM model
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from main train.py
from phase1.train import train
from common.config import Config
import argparse

# Import LSTM model
from phase1 import model_lstm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM Model')
    parser.add_argument('--data_dir', type=str, default='EPIC-KITCHENS')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
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
        OUTPUT_DIR=Path('outputs_lstm'),
        CHECKPOINT_DIR=Path('outputs_lstm/checkpoints')
    )

    # Override model getter
    from phase1 import model as original_model
    original_model.get_model = model_lstm.get_model

    # Train
    train(config)
