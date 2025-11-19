# EPIC-KITCHENS Action Recognition

Deep learning models for egocentric action recognition on the EPIC-KITCHENS-100 dataset.

## Project Overview

This project implements a **three-phase research approach** to action recognition:

1. **Phase 1**: Architecture Comparison (Completed)
2. **Phase 2**: Hyperparameter Optimization (In Progress)
3. **Phase 3**: Cross-Task Attention (Ready to Train)

## Current Status (2025-11-19)

### Phase 1: Architecture Comparison
- **6 models trained** successfully on VSC cluster
- Models: Baseline, LSTM, Transformer, 3D CNN, EfficientNet+LSTM, EfficientNet+Transformer
- Training epochs: 20-35 depending on model
- **Key Finding**: All models exhibit overfitting (~15% gap between train/val)

### Phase 2: Hyperparameter Optimization
- **3 out of 8 configurations** completed
- Significant overfitting reduction: 15% → 5%
- Best model: Config 1 (Val Acc: ~27%)
- Remaining configs queued for training

### Phase 3: Cross-Attention
- Model implemented with bidirectional verb-noun attention
- Training script created
- Awaiting Phase 2 completion for hyperparameter selection

## Quick Start

### Training Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Phase 1: Train baseline model
cd phase1
python train.py

# Phase 2: Train improved model
cd ..
python train_improved.py --epochs 30 --batch_size 24

# Phase 3: Train cross-attention
cd phase3
python train_cross_attention.py
```

### Training on VSC Cluster

```bash
# SSH to cluster
ssh vsc
cd $VSC_DATA/epic_kitchens

# Submit Phase 2 jobs
cd phase2
sbatch train_config1.slurm

# Check job status
squeue -u $USER
```

## Project Structure

```
epic_kitchens/
├── common/                   # Shared utilities
│   ├── config.py            # Configuration
│   └── dataset.py           # Dataset loader
├── phase1/                  # Architecture comparison
│   ├── model.py             # Baseline ResNet-50
│   ├── model_lstm.py        # LSTM model
│   ├── model_transformer.py # Transformer model
│   ├── model_advanced.py    # EfficientNet models
│   └── train.py             # Training script
├── phase2/                  # Hyperparameter optimization
│   ├── model_improved.py    # Improved model (frozen layers, dropout)
│   ├── dataset_improved.py  # Aggressive augmentation
│   └── train_improved.py    # Training with validation monitoring
├── phase3/                  # Cross-attention
│   ├── model_cross_attention.py  # Cross-task attention model
│   └── train_cross_attention.py  # Training script
├── validation/              # Validation scripts
│   └── validate_*.py        # Various validation modes
├── inference/               # Real-time inference
│   └── realtime_webcam.py   # Webcam demo
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md      # Detailed architecture docs
│   └── TRAINING_GUIDE.md    # Training instructions
└── README.md                # This file
```

## Results Summary

| Phase | Model | Params | Val Verb Acc | Val Noun Acc | Overfitting Gap |
|-------|-------|--------|--------------|--------------|-----------------|
| 1 | Baseline | 25M | ~30% | ~25% | ~15% |
| 1 | LSTM | 27M | ~32% | ~26% | ~16% |
| 1 | Transformer | 30M | ~35% | ~28% | ~15% |
| 2 | Improved | 25M | ~36% | ~19% | ~5% (Good) |
| 3 | Cross-Attn | 26M | TBD | TBD | TBD |

*Note: Phase 2 shows lower accuracy but much better generalization (reduced overfitting)*

## Key Features

### Phase 1
- Multiple architecture types (CNN, LSTM, Transformer, 3D CNN)
- ResNet-50 and EfficientNet backbones
- Temporal aggregation strategies

### Phase 2
- **Frozen backbone layers** (reduce overfitting)
- **Heavy data augmentation** (flip, rotate, color jitter, erasing)
- **Label smoothing** (0.1-0.2)
- **Dropout** (0.3-0.6) and **Drop Path** (0.05-0.15)
- **Cosine annealing** learning rate schedule
- **Mixed precision** training (FP16)

### Phase 3
- **Cross-task attention** between verbs and nouns
- **Multi-scale temporal modeling** (4, 8, 16 frame windows)
- **Bidirectional attention** for semantic dependencies
- Maintains Phase 2 regularization strategies

## Configuration

Key hyperparameters in `common/config.py`:

```python
NUM_FRAMES = 16           # Frames per video clip
IMAGE_SIZE = 224          # Spatial resolution
BATCH_SIZE = 24-32        # Batch size
LEARNING_RATE = 5e-5      # Learning rate
NUM_VERB_CLASSES = 97     # EPIC-KITCHENS verbs
NUM_NOUN_CLASSES = 300    # EPIC-KITCHENS nouns
```

## Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Detailed model architectures and design decisions
- **[TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)**: Step-by-step training instructions
- **[phase3/README.md](phase3/README.md)**: Cross-attention architecture details

## Validation

```bash
# Validate single model
python validate_vsc.py --checkpoint outputs/checkpoints/best_model.pth

# Validate all Phase 1 models
python validate_models.py \
  --val_csv EPIC_100_validation.csv \
  --val_video_dir EPIC-KITCHENS/videos_640x360
```

## VSC Cluster Training Status

**Currently on VSC** (`/vsc-hard-mounts/leuven-data/380/vsc38064/epic_kitchens`):

- Phase 1: All 6 models trained with checkpoints
  - `outputs/` (baseline, 25 epochs)
  - `outputs_lstm/` (25 epochs)
  - `outputs_transformer/` (35 epochs)
  - `outputs_3dcnn/` (20 epochs)
  - `outputs_efficientnet_lstm/` (20 epochs)
  - `outputs_efficientnet_transformer/` (20 epochs)

- Phase 2: 3 models partially trained
  - `outputs_model1/` (best_model.pth available)
  - `outputs_model2/` (best_model.pth available)
  - `outputs_model3/` (best_model.pth available)
  - `outputs_model4/` (incomplete)

- Phase 3: Ready to train once Phase 2 completes

**No jobs currently running** - ready for new submissions

## Dataset

**EPIC-KITCHENS-100**:
- 67,217 training action segments
- 9,668 validation action segments
- 97 verb classes, 300 noun classes
- Multi-task learning (verb + noun)
- Download from: https://epic-kitchens.github.io/

## References

- **Dataset**: Damen et al., "Rescaling Egocentric Vision" (IJCV 2022)
- **Repo**: https://github.com/epic-kitchens/epic-kitchens-100-annotations

## TODO

- [ ] Complete Phase 2 hyperparameter sweep (5 configs remaining)
- [ ] Analyze Phase 2 results and select best configuration
- [ ] Train Phase 3 cross-attention model
- [ ] Prepare final presentation with results comparison
- [ ] Document final findings and insights

## Development

This project was developed as part of computer vision research for action recognition in egocentric videos.

**Last Updated**: November 19, 2025
