# EPIC-KITCHENS-100 Action Recognition Project

## Overview

This project implements action recognition on the EPIC-KITCHENS-100 dataset, predicting verb (97 classes) and noun (300 classes) labels from video clips of kitchen activities.

**Best Result: 23.98% action accuracy** (exp15: ResNet50 + LSTM, batch_size=64)

---

## Quick Reference

### SSH & VSC Commands

```bash
# Connect to VSC
ssh vsc

# Check job queue
squeue -u vsc38064 --clusters=wice

# Submit job
cd $VSC_DATA/epic_kitchens
sbatch jobs/exp29_baseline.slurm

# Cancel job
scancel <job_id> --clusters=wice

# Check job details
scontrol show job <job_id> --clusters=wice

# Watch output in real-time
tail -f logs/exp29_*.out

# Check GPU usage on compute node
ssh <node_name> nvidia-smi
```

### File Transfer

```bash
# Copy files to VSC
scp local_file vsc:/data/leuven/380/vsc38064/epic_kitchens/

# Sync directory
rsync -avz local_dir/ vsc:/data/leuven/380/vsc38064/epic_kitchens/remote_dir/

# Get VSC_DATA path (for scripting)
ssh vsc "bash -l -c 'echo \$VSC_DATA'"
# Result: /data/leuven/380/vsc38064
```

### SLURM Template

```bash
#!/bin/bash
#SBATCH --job-name=exp_name
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge
module load cluster/wice/batch
module load Python/3.11.3-GCCcore-12.3.0

source $VSC_DATA/epic_kitchens/epic_env/bin/activate
export WANDB_API_KEY="a050122e318cf57511f2c745aa871735df7c6de8"
cd $VSC_DATA/epic_kitchens

mkdir -p logs

python src/train.py \
    --exp_name experiment_name \
    --epochs 40 \
    --batch_size 64 \
    --wandb
```

---

## Experiment Results Summary

### Best Experiments (Action Top-1 Accuracy)

| Rank | Experiment | Config | Accuracy | Notes |
|------|------------|--------|----------|-------|
| 1 | exp15_h100 | ResNet50+LSTM, batch=64, dropout=0.5 | **23.98%** | Best overall |
| 2 | exp11_lstmNF | ResNet50+LSTM, no freeze | 23.90% | Close second |
| 3 | exp17_h100 | ResNet50+LSTM, dropout=0.3 | 23.76% | Lower dropout helped |
| 4 | exp9_lstmLR | ResNet50+LSTM, lr=2e-4 | 22.49% | Higher LR |
| 5 | exp16_h100 | EfficientNet-B3+LSTM | 22.42% | Different backbone |
| 6 | exp20_h100 | ResNet50+LSTM, high reg | 21.75% | Too much regularization |
| 7 | exp21_h100 | ResNet50+LSTM+CrossAttn | 21.08% | Added complexity hurt |
| 8 | exp10_lstmCA | ResNet50+LSTM+CrossAttn | 20.04% | Cross-attention didn't help |
| 9 | exp12_lstm16 | ResNet50+LSTM, 16 frames | 19.90% | Same as 8 frames |
| 10 | exp22_h100 | VideoMAE | 18.56% | Pre-trained video model |
| 11 | exp8_slowfast | SlowFast | 14.18% | Poor performance |

### What Worked

1. **LSTM over Transformer**: LSTM temporal model consistently outperformed transformer
2. **No backbone freezing**: Training full backbone gave best results
3. **Batch size 64-80**: Larger batches improved training stability
4. **Dropout 0.3-0.5**: Moderate dropout, 0.3 slightly better than 0.5
5. **Medium augmentation**: RandomResizedCrop(0.6-1.0), ColorJitter, HorizontalFlip
6. **Label smoothing 0.1**: Helped generalization
7. **Warmup + Cosine LR**: 3 epoch warmup with cosine annealing
8. **Early stopping**: Patience=7 prevented overfitting

### What Didn't Work

1. **Transformer temporal model**: Underperformed LSTM by ~3-5%
2. **Cross-attention heads**: Added complexity without improvement
3. **SlowFast**: Only 14.18%, possibly due to implementation issues
4. **VideoMAE**: 18.56%, pretrained model didn't transfer well
5. **High regularization**: dropout=0.7, weight_decay=0.05 hurt performance
6. **Frozen backbone**: Reduced accuracy significantly
7. **Heavy augmentation**: Too aggressive, caused underfitting
8. **MixUp/CutMix**: Phase4 experiments crashed before completion

### Experiments That Crashed

- exp24-26: Directory creation bug (fixed in new src/train.py)
- exp27: SlowFast frame count error (needed 8 slow frames, not 4)
- exp28: VideoMAE initially crashed, restarted

---

## Project Structure

```
epic_kitchens/
├── src/                          # Clean unified codebase
│   ├── config.py                 # Centralized configuration
│   ├── datasets.py               # TrainDataset, ValDataset
│   ├── models.py                 # ActionModel, temporal models
│   └── train.py                  # Unified training script
├── jobs/                         # SLURM job scripts
│   ├── exp29_baseline.slurm
│   ├── exp30_transformer.slurm
│   ├── exp31_efficientnet.slurm
│   └── exp32_heavy_aug.slurm
├── outputs/                      # Training outputs (auto-created)
│   └── {exp_name}/
│       ├── checkpoints/
│       │   └── best_model.pth
│       ├── config.json
│       └── history.json
├── logs/                         # SLURM logs
├── results/                      # Compiled results
│   └── best_models/
├── EPIC-KITCHENS/               # Dataset
│   ├── epic-kitchens-100-annotations-master/
│   │   ├── EPIC_100_train.csv
│   │   └── EPIC_100_validation.csv
│   └── videos_640x360_validation/
├── extracted_frames/            # Pre-extracted training frames
│   └── train/
│       └── {participant}_{video}_{idx}/
│           ├── frame_000000.jpg
│           └── ...
└── PROJECT.md                   # This file
```

---

## Code Architecture

### Model Architecture

```
Input: (B, T, C, H, W) - Batch of T frames
    │
    ▼
┌─────────────────┐
│  ResNet50       │  Pre-trained ImageNet backbone
│  (per frame)    │  Output: (B*T, 2048)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature Proj   │  Linear(2048→512) + LayerNorm + GELU
│                 │  Output: (B, T, 512)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Temporal LSTM  │  Bidirectional LSTM
│  (2 layers)     │  Output: (B, 512)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│ Verb  │ │ Noun  │
│ Head  │ │ Head  │  Linear(512→512) + LayerNorm + GELU + Dropout + Linear
│ (97)  │ │ (300) │
└───────┘ └───────┘
```

### Training Pipeline

1. **Data Loading**: Pre-extracted frames for training (fast I/O), video decoding for validation
2. **Augmentation**: RandomResizedCrop, HorizontalFlip, ColorJitter, optional Grayscale/Blur
3. **Mixed Precision**: FP16 training with GradScaler
4. **Optimization**: AdamW with warmup + cosine LR schedule
5. **Regularization**: Label smoothing, dropout, weight decay
6. **Early Stopping**: Monitor action accuracy, patience=7

### Key Hyperparameters (Best Config)

```python
{
    "backbone": "resnet50",
    "temporal_model": "lstm",
    "num_frames": 16,
    "batch_size": 64,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "dropout": 0.5,
    "label_smoothing": 0.1,
    "augmentation": "medium",
    "warmup_epochs": 3,
    "freeze_backbone": "none",
    "early_stopping": true,
    "patience": 7
}
```

---

## Training Commands

### Run Training Locally

```bash
cd epic_kitchens
python src/train.py \
    --exp_name test_local \
    --epochs 5 \
    --batch_size 32 \
    --num_workers 4
```

### Run on VSC

```bash
# Sync code
scp -r src/ jobs/ vsc:/data/leuven/380/vsc38064/epic_kitchens/

# Submit job
ssh vsc "cd /data/leuven/380/vsc38064/epic_kitchens && sbatch jobs/exp29_baseline.slurm"

# Monitor
ssh vsc "squeue -u vsc38064 --clusters=wice"
ssh vsc "tail -f /data/leuven/380/vsc38064/epic_kitchens/logs/exp29_*.out"
```

### Training Script Options

```bash
python src/train.py --help

# Key options:
--exp_name          # Experiment name (creates outputs/{exp_name}/)
--epochs            # Number of epochs (default: 30)
--batch_size        # Batch size (default: 64)
--lr                # Learning rate (default: 1e-4)
--backbone          # resnet50, resnet18, efficientnet_b0, efficientnet_b3
--temporal_model    # lstm, transformer, mean
--dropout           # Dropout rate (default: 0.5)
--augmentation      # none, light, medium, heavy
--freeze_backbone   # none, all, early, bn
--early_stopping    # Enable early stopping
--patience          # Early stopping patience (default: 7)
--wandb             # Enable W&B logging
```

---

## Inference

### Test on Custom Video

```python
from src.models import ActionModel
from src.config import Config
import torch

# Load model
cfg = Config()
model = ActionModel(
    num_verb_classes=cfg.NUM_VERB_CLASSES,
    num_noun_classes=cfg.NUM_NOUN_CLASSES,
    backbone='resnet50',
    temporal_model='lstm'
)
model.load_state_dict(torch.load('outputs/exp15_h100/checkpoints/best_model.pth'))
model.eval()

# Inference
frames = preprocess_video('video.mp4', num_frames=16)  # (1, 16, 3, 224, 224)
with torch.no_grad():
    verb_logits, noun_logits = model(frames)
    verb_pred = verb_logits.argmax(1)
    noun_pred = noun_logits.argmax(1)
```

---

## VSC Environment Setup

### First-Time Setup

```bash
# On VSC login node
cd $VSC_DATA
mkdir -p epic_kitchens
cd epic_kitchens

# Create virtual environment
module load Python/3.11.3-GCCcore-12.3.0
python -m venv epic_env
source epic_env/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pandas opencv-python tqdm wandb
```

### Directory Structure on VSC

```
$VSC_DATA/epic_kitchens/
├── epic_env/                    # Virtual environment
├── src/                         # Training code
├── jobs/                        # SLURM scripts
├── outputs/                     # Training outputs
├── logs/                        # SLURM logs
├── EPIC-KITCHENS/              # Dataset
└── extracted_frames/           # Pre-extracted frames
```

---

## W&B Logging

Project: `epic-kitchens-action`
Dashboard: https://wandb.ai/your-username/epic-kitchens-action

Logged metrics:
- train_loss, train_verb, train_noun
- val_loss, val_verb_top1, val_noun_top1, val_action_top1
- val_verb_top5, val_noun_top5

---

## Lessons Learned

1. **Simple architectures work**: ResNet50+LSTM beat fancy attention mechanisms
2. **Don't over-regularize**: The dataset is large enough that heavy regularization hurts
3. **Train the full backbone**: Frozen backbones limit learning capacity
4. **Pre-extract frames**: 10x faster than video decoding during training
5. **Test locally first**: Many bugs discovered before wasting GPU hours
6. **Check directory creation**: Many crashed experiments due to missing dirs
7. **Always enable wandb**: Essential for tracking 30+ experiments
8. **H100 > A100**: H100 partition has better availability and speed

---

## Future Improvements

1. **Multi-crop testing**: TTA with multiple crops
2. **Ensemble models**: Combine LSTM and Transformer predictions
3. **Longer training**: 50+ epochs with lower LR
4. **I3D/SlowFast proper**: Fix implementation issues
5. **Temporal augmentation**: Frame dropping, speed variation
6. **Class weighting**: Address noun class imbalance (300 classes)

---

## References

- EPIC-KITCHENS-100: https://epic-kitchens.github.io/2024
- Challenge: https://codalab.lisn.upsaclay.fr/competitions/617
- Paper: "Rescaling Egocentric Vision" (IJCV 2021)
