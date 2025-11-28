"""
EPIC-KITCHENS-100 Action Recognition - Configuration
=====================================================
Central configuration for all experiments.
"""

from pathlib import Path


class Config:
    """Configuration for EPIC-KITCHENS training."""

    # ==========================================================================
    # PATHS (relative to epic_kitchens root)
    # ==========================================================================
    DATA_DIR = Path("EPIC-KITCHENS")
    ANNOTATION_DIR = DATA_DIR / "epic-kitchens-100-annotations-master"

    # Training data
    TRAIN_CSV = ANNOTATION_DIR / "EPIC_100_train.csv"
    VAL_CSV = ANNOTATION_DIR / "EPIC_100_validation.csv"
    TEST_CSV = ANNOTATION_DIR / "EPIC_100_test_timestamps.csv"

    # Class mappings
    VERB_CLASSES_CSV = ANNOTATION_DIR / "EPIC_100_verb_classes.csv"
    NOUN_CLASSES_CSV = ANNOTATION_DIR / "EPIC_100_noun_classes.csv"

    # Pre-extracted frames (much faster than video loading)
    EXTRACTED_FRAMES_DIR = Path("extracted_frames/train")
    VAL_FRAMES_DIR = Path("extracted_frames/val")

    # Validation videos (loaded on-the-fly if frames not available)
    VAL_VIDEO_DIR = DATA_DIR / "videos_640x360_validation"

    # ==========================================================================
    # CLASSES
    # ==========================================================================
    NUM_VERB_CLASSES = 97
    NUM_NOUN_CLASSES = 300

    # ==========================================================================
    # DEFAULT TRAINING PARAMS
    # ==========================================================================
    BATCH_SIZE = 64
    NUM_WORKERS = 14
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # ==========================================================================
    # MODEL DEFAULTS
    # ==========================================================================
    NUM_FRAMES = 16
    IMAGE_SIZE = 224
    BACKBONE = "resnet50"
    TEMPORAL_MODEL = "lstm"
    DROPOUT = 0.5

    # ==========================================================================
    # REGULARIZATION
    # ==========================================================================
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.0        # 0 = disabled
    CUTMIX_ALPHA = 0.0       # 0 = disabled

    # ==========================================================================
    # TRAINING OPTIONS
    # ==========================================================================
    USE_AMP = True           # Mixed precision
    WARMUP_EPOCHS = 3
    EARLY_STOPPING = True
    PATIENCE = 7

    # ==========================================================================
    # WANDB
    # ==========================================================================
    WANDB_PROJECT = "epic-kitchens-action"
    WANDB_API_KEY = "a050122e318cf57511f2c745aa871735df7c6de8"


def get_output_dir(exp_name: str) -> Path:
    """Get output directory for an experiment."""
    output_dir = Path("outputs") / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    return output_dir
