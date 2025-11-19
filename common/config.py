"""
Configuration for EPIC-KITCHENS-100 Action Recognition
"""

import os
from pathlib import Path


class Config:
    """Configuration for training and evaluation."""

    # Dataset paths
    DATA_DIR = Path("EPIC-KITCHENS")
    VIDEO_DIR = DATA_DIR / "videos_640x360"
    ANNOTATION_DIR = DATA_DIR / "epic-kitchens-100-annotations-master"

    TRAIN_CSV = ANNOTATION_DIR / "EPIC_100_train.csv"
    VAL_CSV = ANNOTATION_DIR / "EPIC_100_validation.csv"
    TEST_CSV = ANNOTATION_DIR / "EPIC_100_test_timestamps.csv"

    VERB_CLASSES_CSV = ANNOTATION_DIR / "EPIC_100_verb_classes.csv"
    NOUN_CLASSES_CSV = ANNOTATION_DIR / "EPIC_100_noun_classes.csv"

    # Model parameters
    NUM_VERB_CLASSES = 97
    NUM_NOUN_CLASSES = 300

    # Video processing
    NUM_FRAMES = 8  # Number of frames to sample per video segment
    IMAGE_SIZE = 224  # Input image size (224 for ResNet)
    FPS = 30  # Target FPS for frame extraction

    # Training parameters
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Device
    DEVICE = "cuda"  # "cuda", "mps", or "cpu"

    # Output
    OUTPUT_DIR = Path("outputs")
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    LOG_DIR = OUTPUT_DIR / "logs"

    # Mixed precision training
    USE_AMP = True  # Automatic Mixed Precision for faster training

    # Data augmentation
    HORIZONTAL_FLIP_PROB = 0.5
    COLOR_JITTER = True

    # Loss weights
    VERB_LOSS_WEIGHT = 1.0
    NOUN_LOSS_WEIGHT = 1.0

    # Early stopping
    PATIENCE = 999  # Effectively disabled (no validation set available)

    # Checkpointing
    SAVE_EVERY = 5  # Save checkpoint every N epochs
    SAVE_BEST = True  # Save best model based on validation accuracy

    def __init__(self, **kwargs):
        """Allow overriding config values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Create output directories
        self.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        self.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        self.LOG_DIR.mkdir(exist_ok=True, parents=True)

    def __repr__(self):
        """Print configuration."""
        config_str = "=" * 70 + "\n"
        config_str += "EPIC-KITCHENS-100 Configuration\n"
        config_str += "=" * 70 + "\n"

        for key, value in vars(self).items():
            if not key.startswith("_"):
                config_str += f"{key:25s}: {value}\n"

        config_str += "=" * 70
        return config_str


# Default configuration
default_config = Config()


if __name__ == "__main__":
    # Print default configuration
    print(default_config)
