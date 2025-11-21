"""
Configuration for Simple ResNet-18 Baseline
Minimal, clean configuration following friend's approach
"""

import os
from pathlib import Path


class Config:
    """Simple baseline configuration."""

    # =========================================================================
    # Data Paths (VSC)
    # =========================================================================
    DATA_ROOT = Path("/vsc-hard-mounts/leuven-data/380/vsc38064/EPIC-KITCHENS")

    TRAIN_CSV = DATA_ROOT / "EPIC_100_train.csv"
    VAL_CSV = DATA_ROOT / "EPIC_100_validation.csv"
    VIDEO_DIR = DATA_ROOT / "videos"

    # =========================================================================
    # Model Architecture
    # =========================================================================
    NUM_FRAMES = 8              # Match friend's setup
    IMAGE_SIZE = 224            # Standard ImageNet size
    DROPOUT = 0.3               # Minimal dropout only

    # EPIC-KITCHENS classes
    NUM_VERB_CLASSES = 97
    NUM_NOUN_CLASSES = 300

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32             # Standard batch size
    LEARNING_RATE = 1e-4        # Standard for AdamW + vision
    WEIGHT_DECAY = 1e-4         # L2 regularization
    NUM_EPOCHS = 30             # Sufficient for convergence

    # Scheduler (ReduceLROnPlateau)
    LR_PATIENCE = 3             # Reduce LR after 3 epochs without improvement
    LR_FACTOR = 0.5             # Reduce LR by 50%

    # =========================================================================
    # Data Loading
    # =========================================================================
    NUM_WORKERS = 4             # VSC: 4 CPU cores per GPU
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = "cuda"             # GPU training
    MIXED_PRECISION = True      # FP16 for faster training

    # =========================================================================
    # Checkpointing
    # =========================================================================
    OUTPUT_DIR = Path("outputs")
    LOG_DIR = Path("logs")
    CHECKPOINT_DIR = Path("outputs/checkpoints")

    # Save checkpoints
    SAVE_EVERY = 5              # Save every 5 epochs
    SAVE_BEST_ONLY = True       # Keep only best model

    # =========================================================================
    # Logging
    # =========================================================================
    LOG_INTERVAL = 50           # Print every 50 batches
    VAL_EVERY = 1               # Validate every epoch

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    def __init__(self):
        """Create output directories."""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def __str__(self):
        """Print configuration."""
        lines = ["=" * 70, "Configuration", "=" * 70]

        # Model
        lines.append(f"Model:           ResNet-18 + Temporal Pooling")
        lines.append(f"Frames:          {self.NUM_FRAMES}")
        lines.append(f"Image size:      {self.IMAGE_SIZE}")
        lines.append(f"Dropout:         {self.DROPOUT}")

        # Training
        lines.append(f"Batch size:      {self.BATCH_SIZE}")
        lines.append(f"Learning rate:   {self.LEARNING_RATE}")
        lines.append(f"Weight decay:    {self.WEIGHT_DECAY}")
        lines.append(f"Epochs:          {self.NUM_EPOCHS}")

        # Classes
        lines.append(f"Verb classes:    {self.NUM_VERB_CLASSES}")
        lines.append(f"Noun classes:    {self.NUM_NOUN_CLASSES}")

        # Hardware
        lines.append(f"Device:          {self.DEVICE}")
        lines.append(f"Mixed precision: {self.MIXED_PRECISION}")
        lines.append(f"Num workers:     {self.NUM_WORKERS}")

        lines.append("=" * 70)
        return "\n".join(lines)


# Create default config instance
default_config = Config()


if __name__ == "__main__":
    # Test configuration
    config = Config()
    print(config)

    # Check paths
    print(f"\nData paths:")
    print(f"  Train CSV:  {config.TRAIN_CSV}")
    print(f"  Val CSV:    {config.VAL_CSV}")
    print(f"  Video dir:  {config.VIDEO_DIR}")

    print(f"\nOutput paths:")
    print(f"  Output:     {config.OUTPUT_DIR}")
    print(f"  Logs:       {config.LOG_DIR}")
    print(f"  Checkpoint: {config.CHECKPOINT_DIR}")

    print(f"\n✓ Configuration loaded successfully!")
