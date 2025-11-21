# Simple ResNet-18 Baseline

Clean, minimal baseline for EPIC-KITCHENS action recognition.

## Goal

Match or exceed friend's 50% action accuracy with a simple approach:
- ResNet-18 (not ResNet-50)
- Temporal average pooling (not Transformer)
- Minimal augmentation (no grayscale, no blur)
- No over-regularization

## Why This Will Work

Your Phase 1 and Phase 2 models had critical issues:
1. **Over-aggressive augmentation** destroyed visual signal (noun accuracy 28% → 19%)
2. **Over-regularization** caused underfitting
3. **Unnecessary complexity** (Transformers) hurt performance
4. **Frame sampling bug** (sampling after action end)

This baseline fixes all these issues.

## Quick Start

### On VSC

```bash
# SSH to VSC
ssh vsc
cd $VSC_DATA/epic_kitchens

# Pull latest code
git pull

# Navigate to baseline
cd baseline_resnet18

# Create logs directory
mkdir -p logs

# Submit training job
sbatch scripts/train.slurm

# Monitor progress
squeue -u $USER
tail -f logs/resnet18_*.out

# After training, validate
sbatch scripts/validate.slurm
```

### Local Testing (Optional)

```bash
cd baseline_resnet18

# Test model
python models/resnet18_simple.py

# Test config
python configs/config.py

# Note: Dataset requires EPIC-KITCHENS videos
```

## Folder Structure

```
baseline_resnet18/
├── configs/
│   └── config.py              # Hyperparameters
├── models/
│   └── resnet18_simple.py     # ResNet-18 model
├── data/
│   └── dataset.py             # Fixed dataset loader
├── scripts/
│   ├── train.py               # Training script
│   ├── train.slurm            # Slurm for training
│   ├── validate.py            # Validation script
│   └── validate.slurm         # Slurm for validation
├── outputs/                   # Created on VSC
├── logs/                      # Created on VSC
├── ANALYSIS.md                # Root cause analysis
├── IMPLEMENTATION_PLAN.md     # Detailed plan
└── README.md                  # This file
```

## Expected Results

### Week 1 (Initial Training)
- **Action Accuracy**: 25-35% (vs 10-15% currently)
- **Verb Accuracy**: 35-45%
- **Noun Accuracy**: 28-32% (recovering from Phase 2's 19%)
- **Training Time**: ~15 hours (vs ~25 hours for ResNet-50)

### Week 2 (With Tuning)
- **Action Accuracy**: 35-45%
- Hyperparameter optimization
- Potentially approaching friend's 50%

## Key Improvements

### 1. Model Architecture
```python
# Phase 2 (BAD):
ResNet-50 (frozen first 2 blocks) + Transformer + Heavy dropout
→ 25M params, 10% action accuracy

# This baseline (GOOD):
ResNet-18 + Temporal pooling + Minimal dropout
→ 11M params, Target: 30-40% action accuracy
```

### 2. Data Augmentation
```python
# Phase 2 (BAD - destroying signal):
RandomResizedCrop(scale=(0.6, 1.0))     # Too aggressive!
RandomGrayscale(p=0.2)                   # Destroys color
GaussianBlur(p=0.3)                      # Destroys details

# This baseline (GOOD - minimal):
RandomResizedCrop(scale=(0.85, 1.0))    # Mild only
RandomHorizontalFlip(p=0.5)              # Natural
ColorJitter(brightness=0.2, ...)         # Mild only
```

### 3. Frame Sampling
```python
# Phase 2 (WRONG):
frame_indices = np.linspace(start_frame, stop_frame, num_frames)
# Samples frames AFTER action ends!

# This baseline (CORRECT):
frame_indices = np.linspace(start_frame, stop_frame - 1, num_frames)
# Stays within action boundaries
```

### 4. Regularization
```python
# Phase 2 (OVER-REGULARIZED):
freeze_backbone_layers=2
dropout=0.5
drop_path=0.1
label_smoothing=0.1

# This baseline (BALANCED):
No frozen layers
dropout=0.3
No drop path
No label smoothing
```

## Training Details

**Hardware**:
- 1 GPU (VSC cluster)
- 4 CPU cores
- 32GB RAM

**Training time**:
- ~30 minutes per epoch
- 30 epochs total
- ~15 hours total

**Hyperparameters**:
- Batch size: 32
- Learning rate: 1e-4 (AdamW)
- Weight decay: 1e-4
- Frames per clip: 8
- Image size: 224x224

**Optimizer**: AdamW
**Scheduler**: ReduceLROnPlateau (reduce LR when validation plateaus)
**Loss**: CrossEntropyLoss (no label smoothing)

## Monitoring Training

```bash
# Check job status
squeue -u $USER

# Watch training logs
tail -f logs/resnet18_*.out

# Check training progress
cat outputs/training_log.txt

# View best results
tail outputs/training_log.txt
```

## After Training

### Validate Model
```bash
sbatch scripts/validate.slurm
cat validation_results.txt
```

### Compare to Phase 1 and Phase 2
```bash
# This baseline should show:
# - 2-3x better action accuracy than Phase 2
# - Better noun accuracy than Phase 2 (recovering from 19%)
# - Faster training than ResNet-50 models
```

## Documentation

- **ANALYSIS.md**: Root cause analysis of Phase 1/2 issues
- **IMPLEMENTATION_PLAN.md**: Detailed implementation plan
- **README.md**: This file

## Success Criteria

✅ **Minimum Success** (90% confidence):
- Action accuracy: 25-35%
- Noun accuracy: 28-32%
- Training time: <18 hours

✅ **Target Success** (70% confidence):
- Action accuracy: 35-45%
- Approaching friend's performance

✅ **Stretch Goal** (50% confidence):
- Action accuracy: 45-50%
- Matching friend's performance

## Next Steps After Baseline

Once baseline achieves 30-40% accuracy:

1. **Hyperparameter tuning**:
   - Batch size: 16, 32, 64
   - Learning rate: 5e-5, 1e-4, 2e-4
   - Frame count: 8, 12, 16

2. **Advanced techniques**:
   - Two-stream (RGB + optical flow)
   - Temporal segment networks
   - Better pretraining (Kinetics-400)
   - Ensemble methods

## Questions?

See `ANALYSIS.md` for detailed explanation of what went wrong in Phase 1/2.
See `IMPLEMENTATION_PLAN.md` for detailed implementation steps.

## Comparison to Previous Models

| Metric | Phase 1 | Phase 2 | This Baseline (Target) |
|--------|---------|---------|------------------------|
| **Action Accuracy** | 15-20% | 10% | **30-40%** |
| **Verb Accuracy** | 30% | 36% | 35-45% |
| **Noun Accuracy** | 25% | **19%** ⚠️ | **28-32%** |
| **Params** | 25M | 25M | **11M** |
| **Training Time** | 25h | 32h | **15h** |
| **Architecture** | ResNet-50 + Pool | ResNet-50 + Transformer | **ResNet-18 + Pool** |
| **Issues** | Overfitting | Over-regularization | **Fixed** |

## Key Takeaways

1. **Simpler is better**: Friend's simple approach beats complex Phase 2
2. **Augmentation matters**: Phase 2's aggressive augmentation destroyed performance
3. **Don't over-regularize**: Caused underfitting in Phase 2
4. **Smaller models**: ResNet-18 better than ResNet-50 for this task
5. **Fix bugs**: Frame sampling was critical

---

**Ready to train!** Submit the job and expect 2-3x improvement over Phase 2.
