# Implementation Plan: Simple ResNet-18 Baseline

## Goal

Create a clean, simple baseline that mimics your friend's approach:
- ResNet-18 backbone
- Temporal average pooling
- Minimal augmentation
- **Target**: 30-40% action accuracy (vs current 10-15%)

## Folder Structure

```
baseline_resnet18/
├── models/
│   └── resnet18_simple.py       # ResNet-18 + temporal pooling
├── data/
│   └── dataset.py                # Fixed dataset loader
├── scripts/
│   ├── train.py                  # Clean training script
│   ├── train.slurm               # VSC slurm script
│   └── validate.py               # Validation script
├── configs/
│   └── config.py                 # Hyperparameters
├── outputs/                      # Training outputs (created on VSC)
├── ANALYSIS.md                   # This analysis
└── IMPLEMENTATION_PLAN.md        # This file
```

## Implementation Steps

### Step 1: Configuration (`configs/config.py`)

```python
class Config:
    # Data paths (for VSC)
    DATA_ROOT = "/vsc-hard-mounts/leuven-data/380/vsc38064/EPIC-KITCHENS"
    TRAIN_CSV = f"{DATA_ROOT}/EPIC_100_train.csv"
    VAL_CSV = f"{DATA_ROOT}/EPIC_100_validation.csv"
    VIDEO_DIR = f"{DATA_ROOT}/videos"

    # Model
    NUM_FRAMES = 8              # Match friend's setup
    IMAGE_SIZE = 224
    DROPOUT = 0.3               # Minimal

    # Classes
    NUM_VERB_CLASSES = 97
    NUM_NOUN_CLASSES = 300

    # Training
    BATCH_SIZE = 32             # Standard
    LEARNING_RATE = 1e-4        # Standard for AdamW
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 30

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda"

    # Checkpointing
    SAVE_DIR = "outputs"
    SAVE_EVERY = 5
```

### Step 2: Model (`models/resnet18_simple.py`)

**Key Features**:
- ResNet-18 backbone (11M params)
- Simple temporal average pooling
- Dropout 0.3 only
- No frozen layers
- No complexity

```python
class SimpleActionModel(nn.Module):
    def __init__(self, num_verb_classes=97, num_noun_classes=300, dropout=0.3):
        super().__init__()

        # ResNet-18 (ImageNet pretrained)
        resnet18 = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(resnet18.children())[:-1])

        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Classifiers (512 -> num_classes)
        self.verb_fc = nn.Linear(512, num_verb_classes)
        self.noun_fc = nn.Linear(512, num_noun_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T = x.shape[:2]

        # Extract features per frame
        x = x.view(B*T, *x.shape[2:])
        features = self.backbone(x)  # (B*T, 512, 1, 1)
        features = features.view(B, T, 512)

        # Temporal pooling
        features = features.permute(0, 2, 1)  # (B, 512, T)
        features = self.temporal_pool(features).squeeze(-1)  # (B, 512)

        # Dropout and classify
        features = self.dropout(features)
        verb_out = self.verb_fc(features)
        noun_out = self.noun_fc(features)

        return verb_out, noun_out
```

### Step 3: Dataset (`data/dataset.py`)

**Key Fixes**:
1. Fix frame sampling bug: `stop_frame - 1`
2. Minimal augmentation (no grayscale, no blur)
3. Proper path handling

**Augmentation**:
```python
train_transforms = [
    RandomResizedCrop(224, scale=(0.85, 1.0)),  # Mild crop
    RandomHorizontalFlip(p=0.5),                 # Natural for hands
    ColorJitter(brightness=0.2, contrast=0.2),   # Mild color
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
    RandomErasing(p=0.1, scale=(0.02, 0.1))     # Minimal erasing
]
```

**Frame Sampling** (FIXED):
```python
# CORRECT (not stop_frame)
frame_indices = np.linspace(start_frame, stop_frame - 1, num_frames, dtype=int)
```

### Step 4: Training Script (`scripts/train.py`)

**Key Features**:
- AdamW optimizer (standard for vision)
- ReduceLROnPlateau scheduler
- Mixed precision training (faster)
- Validation every epoch
- Save best model based on action accuracy

```python
# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY
)

# Scheduler (reduce LR on plateau)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3,
    verbose=True
)

# Loss (no label smoothing!)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    train_loss, train_verb_acc, train_noun_acc = train_epoch(...)
    val_loss, val_verb_acc, val_noun_acc = validate(...)

    # Action accuracy (both verb AND noun correct)
    val_action_acc = compute_action_accuracy(...)

    # Save best model
    if val_action_acc > best_action_acc:
        save_checkpoint(...)

    # Reduce LR if needed
    scheduler.step(val_action_acc)
```

### Step 5: Slurm Script (`scripts/train.slurm`)

```bash
#!/bin/bash
#SBATCH --job-name=resnet18_baseline
#SBATCH --output=logs/resnet18_%j.out
#SBATCH --error=logs/resnet18_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=gpu

# Load modules
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Activate environment
source $VSC_DATA/venv/epic_kitchens/bin/activate

# Navigate to directory
cd $VSC_DATA/epic_kitchens/baseline_resnet18

# Create output directory
mkdir -p outputs logs

# Run training
python scripts/train.py \
    --config configs/config.py \
    --output_dir outputs \
    --device cuda \
    --mixed_precision

echo "Training complete!"
```

## Training Workflow

### On Local Machine

1. **Develop and test**:
   ```bash
   cd baseline_resnet18
   python models/resnet18_simple.py  # Test model
   python data/dataset.py            # Test dataset
   ```

2. **Push to git**:
   ```bash
   git add baseline_resnet18/
   git commit -m "Add simple ResNet-18 baseline"
   git push
   ```

### On VSC

3. **Pull latest code**:
   ```bash
   ssh vsc
   cd $VSC_DATA/epic_kitchens
   git pull
   ```

4. **Submit training job**:
   ```bash
   cd baseline_resnet18
   sbatch scripts/train.slurm
   ```

5. **Monitor progress**:
   ```bash
   # Check job status
   squeue -u $USER

   # Watch logs
   tail -f logs/resnet18_*.out

   # Check latest results
   cat outputs/latest_results.txt
   ```

6. **Validation** (after training):
   ```bash
   sbatch scripts/validate.slurm
   ```

## Expected Timeline

### Day 1: Implementation
- ✅ Create folder structure
- ✅ Implement config
- ✅ Implement model
- ✅ Implement dataset
- ✅ Implement training script
- ✅ Create slurm scripts
- ✅ Test locally (if possible)

### Day 1-2: Training
- Submit job to VSC
- Monitor training progress
- First epoch: ~30 minutes
- Full 30 epochs: ~15 hours

### Day 2: Analysis
- Evaluate on validation set
- Compare to Phase 1 and Phase 2
- Analyze results

## Success Criteria

### Week 1 Target (High Confidence)
- **Action Accuracy**: 25-35% (vs 10-15% currently)
- **Verb Accuracy**: 35-45%
- **Noun Accuracy**: 28-32% (recovering from Phase 2's 19%)
- **Training Time**: ~15 hours (vs ~25 hours for ResNet-50)
- **Overfitting Gap**: 8-12% (reasonable)

### Week 2 Target (Medium Confidence)
- **Action Accuracy**: 35-45%
- With hyperparameter tuning:
  - Try batch sizes: 16, 32, 64
  - Try learning rates: 5e-5, 1e-4, 2e-4
  - Try frame counts: 8, 12, 16

### Stretch Goal
- **Action Accuracy**: 45-50% (matching friend)
- May require:
  - Better pretraining (Kinetics-400)
  - Two-stream (RGB + optical flow)
  - Temporal segment networks
  - Ensemble methods

## Comparison Metrics

Track these metrics to compare against Phase 1 and Phase 2:

| Metric | Phase 1 | Phase 2 | Target |
|--------|---------|---------|--------|
| Action Accuracy | 15-20% | 10% | **30-40%** |
| Verb Accuracy | 30% | 36% | 35-45% |
| Noun Accuracy | 25% | 19% | 28-32% |
| Overfitting Gap | 15% | 5% | 8-12% |
| Training Time | 25h | 32h | **15h** |
| Parameters | 25M | 25M | **11M** |

## Debugging Checklist

If results are not as expected:

1. **Check data loading**:
   - Print sample frame indices
   - Visualize augmented images
   - Verify CSV parsing

2. **Check model**:
   - Print output shapes
   - Verify gradients are flowing
   - Check for NaN losses

3. **Check training**:
   - Verify optimizer is updating weights
   - Check learning rate schedule
   - Monitor train vs val gap

4. **Compare to Phase 1**:
   - Same dataset?
   - Same splits?
   - Same preprocessing?

## Questions & Answers

**Q: Why ResNet-18 instead of ResNet-50?**
A: Smaller model (11M vs 25M params), less prone to overfitting, trains 3x faster, friend proved it works.

**Q: Why only 8 frames instead of 16?**
A: Friend used 8 frames successfully. More frames = more computation without proven benefit.

**Q: Why no Transformer?**
A: Added complexity without benefit. Phase 2 Transformer performed worse than Phase 1 simple pooling.

**Q: Why minimal augmentation?**
A: Phase 2 proved aggressive augmentation destroys performance (noun accuracy 28% → 19%).

**Q: Why no frozen layers?**
A: Phase 2 showed frozen layers contribute to underfitting. ResNet-18 is small enough to train end-to-end.

**Q: Why no label smoothing?**
A: Part of Phase 2 over-regularization. Start simple, add only if needed.

## Next Steps After Baseline

Once baseline achieves 30-40% accuracy:

### Immediate Improvements
1. **Hyperparameter tuning**: Grid search on LR, batch size, frames
2. **Better augmentation**: Find optimal balance
3. **Cosine annealing**: Replace ReduceLROnPlateau

### Advanced Improvements
1. **Two-stream**: Add optical flow stream
2. **Temporal segments**: Sample multiple clips per video
3. **Better pretraining**: Use Kinetics-400 pretrained ResNet-18
4. **Ensemble**: Combine multiple models

### Research Directions
1. **Cross-attention**: Phase 3 approach (if it makes sense)
2. **Multi-scale**: Different temporal resolutions
3. **Object detection**: Explicit hand/object modeling

## Files to Create

```
baseline_resnet18/
├── configs/
│   └── config.py                 # Configuration
├── models/
│   └── resnet18_simple.py       # Model
├── data/
│   └── dataset.py                # Dataset
├── scripts/
│   ├── train.py                  # Training
│   ├── train.slurm               # Slurm for training
│   ├── validate.py               # Validation
│   └── validate.slurm            # Slurm for validation
├── ANALYSIS.md                   # Analysis (done)
└── IMPLEMENTATION_PLAN.md        # This file (done)
```

## Ready to Implement!

All analysis is complete. Implementation should take ~2-3 hours, then submit to VSC for overnight training.

Expected results: **2-3x improvement** in action accuracy!
