"""
EPIC-KITCHENS-100 Action Recognition - V2 Moonshot
===================================================
All improvements combined for maximum accuracy.
"""

from .config import Config, get_output_dir
from .models import (
    TwoStreamModel, BaselineModel,
    FocalLoss, LabelSmoothingLoss, CosineClassifier,
    EarlyStopping, get_class_weights
)
from .datasets import (
    TrainDataset, ValDataset,
    cutmix_temporal, mixup_temporal, apply_random_erasing,
    get_class_counts
)

__all__ = [
    'Config', 'get_output_dir',
    'TwoStreamModel', 'BaselineModel',
    'FocalLoss', 'LabelSmoothingLoss', 'CosineClassifier',
    'EarlyStopping', 'get_class_weights',
    'TrainDataset', 'ValDataset',
    'cutmix_temporal', 'mixup_temporal', 'apply_random_erasing',
    'get_class_counts'
]
