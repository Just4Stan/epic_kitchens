"""
EPIC-KITCHENS-100 Action Recognition - V2 Configuration
========================================================
Moonshot configuration combining all proven improvements.
"""

from pathlib import Path


class Config:
    """Optimal configuration for moonshot training."""

    # ==========================================================================
    # PATHS (VSC cluster paths)
    # ==========================================================================
    # These will be overridden by CLI args for VSC
    DATA_DIR = Path("EPIC-KITCHENS")
    ANNOTATION_DIR = DATA_DIR / "epic-kitchens-100-annotations-master"

    TRAIN_CSV = ANNOTATION_DIR / "EPIC_100_train.csv"
    VAL_CSV = ANNOTATION_DIR / "EPIC_100_validation.csv"
    TEST_CSV = ANNOTATION_DIR / "EPIC_100_test_timestamps.csv"

    VERB_CLASSES_CSV = ANNOTATION_DIR / "EPIC_100_verb_classes.csv"
    NOUN_CLASSES_CSV = ANNOTATION_DIR / "EPIC_100_noun_classes.csv"

    # ==========================================================================
    # CLASSES
    # ==========================================================================
    NUM_VERB_CLASSES = 97
    NUM_NOUN_CLASSES = 300

    # ==========================================================================
    # MODEL - MOONSHOT OPTIMAL
    # ==========================================================================
    # Resolution: 320x320 provides +2.4% over 224x224 (research proven)
    IMAGE_SIZE = 320

    # Frames: 32 frames at interval 3 covers ~1.6s of action
    NUM_FRAMES = 32

    # Model architecture
    MODEL_TYPE = "twostream"  # "twostream", "slowfast", "baseline"
    BACKBONE = "clip"  # "clip", "resnet50", "slowfast"
    TEMPORAL_MODEL = "lstm"  # "lstm", "transformer"

    # Feature dimensions
    CLIP_DIM = 768  # CLIP ViT-B/16 output
    LSTM_HIDDEN = 512
    PROJ_DIM = 512

    # ==========================================================================
    # TRAINING - OPTIMAL FOR A100 40GB
    # ==========================================================================
    # Batch size conservative for 320x320 + 32 frames + two-stream
    BATCH_SIZE = 8
    GRAD_ACCUM = 8  # Effective batch = 64

    # Training params
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 16

    # Warmup + Cosine schedule
    WARMUP_EPOCHS = 5
    LR_MIN = 1e-6

    # ==========================================================================
    # AUGMENTATION - ALL PROVEN IMPROVEMENTS (+3.2% combined)
    # ==========================================================================
    # MixUp + CutMix combination
    CUTMIX_ALPHA = 1.0  # Beta distribution parameter
    MIXUP_ALPHA = 0.2   # Softer mixing
    CUTMIX_PROB = 0.5   # Probability of applying CutMix
    MIXUP_PROB = 0.3    # Probability of applying MixUp (when no CutMix)

    # RandomErasing
    RANDOM_ERASING_PROB = 0.25
    RANDOM_ERASING_SCALE = (0.02, 0.33)

    # Standard augmentation
    AUGMENTATION_LEVEL = "heavy"  # "none", "light", "medium", "heavy"

    # ==========================================================================
    # LOSS FUNCTIONS - LONG-TAIL HANDLING
    # ==========================================================================
    # Focal Loss for class imbalance (research: helps tail classes)
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 2.0  # Focus parameter (higher = more focus on hard examples)
    FOCAL_ALPHA = None  # Per-class weights (computed from data if None)

    # Class balancing
    USE_CLASS_WEIGHTS = True
    CLASS_WEIGHT_POWER = 0.5  # sqrt scaling (0.5) is recommended

    # Label smoothing
    LABEL_SMOOTHING = 0.1

    # Loss weights for verb vs noun (noun is harder, weight it more)
    VERB_LOSS_WEIGHT = 0.4
    NOUN_LOSS_WEIGHT = 0.6

    # ==========================================================================
    # COSINE CLASSIFIER (for noun head - handles imbalance better)
    # ==========================================================================
    USE_COSINE_CLASSIFIER = True
    COSINE_TEMPERATURE = 0.05  # Scaling factor

    # ==========================================================================
    # TWO-STREAM SPECIFIC
    # ==========================================================================
    # Freeze pretrained backbones (CLIP, VideoMAE)
    FREEZE_SPATIAL = True
    FREEZE_TEMPORAL = False  # LSTM is trained from scratch

    # Cross-attention fusion
    USE_CROSS_ATTENTION = True
    CROSS_ATTENTION_HEADS = 8
    CROSS_ATTENTION_DIM = 512

    # ==========================================================================
    # REGULARIZATION
    # ==========================================================================
    DROPOUT = 0.3  # Lower than baseline (0.5) since we have strong augmentation

    # ==========================================================================
    # TRAINING OPTIONS
    # ==========================================================================
    USE_AMP = True  # Mixed precision
    GRADIENT_CLIP = 1.0

    # Early stopping
    EARLY_STOPPING = True
    PATIENCE = 10  # More patience for longer training

    # Checkpointing
    SAVE_EVERY = 5  # Save checkpoint every N epochs

    # ==========================================================================
    # WANDB
    # ==========================================================================
    WANDB_PROJECT = "epic-kitchens-moonshot"
    WANDB_API_KEY = "a050122e318cf57511f2c745aa871735df7c6de8"


def get_output_dir(exp_name: str) -> Path:
    """Get output directory for an experiment."""
    output_dir = Path("outputs") / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    return output_dir
