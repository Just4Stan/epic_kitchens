"""
Phase 2: Hyperparameter Optimization

This phase performs systematic hyperparameter search with:
- Improved model architecture with regularization
- Aggressive data augmentation
- Label smoothing and dropout
- Drop path for better generalization
- Frozen ResNet layers to reduce overfitting
"""

from .model_improved import get_improved_model
from .dataset_improved import get_improved_dataloaders

__all__ = ['get_improved_model', 'get_improved_dataloaders']
