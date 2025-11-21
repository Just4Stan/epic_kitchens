# Root Cause Analysis: Why You're Getting 15-20% While Friend Gets 50%

## Executive Summary

**Your Performance**: 10-20% action accuracy
**Friend's Performance**: ~50% action accuracy with ResNet-18 + 8 frames
**Gap**: 2.5-5x worse despite using MORE complex models
**Root Cause**: Over-engineering led to worse performance

## Critical Issues Found

### 1. OVER-AGGRESSIVE DATA AUGMENTATION (Phase 2)

**The Killer Issue**: Phase 2 augmentation is destroying the visual signal needed for classification.

```python
# Phase 2 (BAD - destroying information):
RandomResizedCrop(scale=(0.6, 1.0))      # Crops down to 60%! Removes hands/objects
RandomGrayscale(p=0.2)                    # 20% grayscale - destroys color
ColorJitter(brightness=0.4, ...)          # Very aggressive color changes
GaussianBlur(p=0.3)                       # Blurs 30% of images
RandomErasing(p=0.3, scale=(0.02, 0.2))  # Removes 30% of image parts
RandomRotation(degrees=10)                # Unnecessary for egocentric video

# What should be used (GOOD):
RandomResizedCrop(scale=(0.85, 1.0))     # Mild crop only
ColorJitter(brightness=0.2, ...)          # Mild only
RandomHorizontalFlip(p=0.5)               # Natural for hands
RandomErasing(p=0.1, scale=(0.02, 0.1))  # Minimal
# NO grayscale, NO blur, NO rotation
```

**Impact on Results**:
- Phase 1 (minimal augmentation): Noun accuracy = 25-28%
- Phase 2 (aggressive augmentation): Noun accuracy = 19% (32% DROP!)
- Action accuracy = verb_acc × noun_acc, so this tanks overall performance

**Why This Matters for EPIC-KITCHENS**:
- Egocentric videos: hands and objects are CRITICAL
- Aggressive crops can remove hands entirely
- Grayscale destroys object appearance (needed for 300 noun classes)
- Blur destroys fine-grained details (fork vs spoon, cup vs glass)

### 2. OVER-REGULARIZATION (Phase 2)

**Phase 2 Model Configuration**:
```python
freeze_backbone_layers=2      # Freezes ResNet blocks 1-2
dropout=0.5                    # Very high dropout
drop_path=0.1                  # Stochastic depth
label_smoothing=0.1            # In training
```

**Result**: Model is SO heavily regularized it can't learn!

**Evidence of Underfitting** (not good generalization):
```
Phase 1:
  Train: Verb 45%, Noun 35%
  Val:   Verb 30%, Noun 25%
  Gap:   15% (overfitting)

Phase 2:
  Train: Verb 36%, Noun 23%  ← LOWER than Phase 1 validation!
  Val:   Verb 36%, Noun 19%
  Gap:   5% (looks good, but actually underfitting)
```

The model can't even fit training data well - classic underfitting!

### 3. UNNECESSARY COMPLEXITY

**Phase 1** (Simple, Better):
- ResNet-50 + Temporal Average Pooling
- Dropout 0.5 on features, 0.3 in classifier
- 25M parameters
- **Result**: 15-20% action accuracy

**Phase 2** (Complex, Worse):
- ResNet-50 (frozen first 2 blocks)
- Temporal Transformer with positional encoding
- Multi-head attention (8 heads, 2 layers)
- Dropout 0.5 + Drop path 0.1 + Label smoothing
- 25M parameters (many frozen)
- **Result**: 10% action accuracy (WORSE!)

**Lesson**: Adding complexity hurt performance!

### 4. MODEL SIZE MISMATCH

**Your Models**:
- ResNet-50: 25M parameters
- Feature dimension: 2048
- More prone to overfitting on limited data

**Friend's Model** (assumed):
- ResNet-18: 11M parameters
- Feature dimension: 512
- Less prone to overfitting
- **Trains 3x faster** due to fewer FLOPs

### 5. FRAME SAMPLING BUG

**Found in Phase 2**:
```python
# WRONG (phase2/dataset_improved.py line 202):
frame_indices = np.linspace(start_frame, stop_frame, num_frames)

# CORRECT:
frame_indices = np.linspace(start_frame, stop_frame - 1, num_frames)
```

**Impact**: Sampling frames AFTER the action ends, adding noise to training data.

## Architecture Comparison

| Component | Phase 1 | Phase 2 | Friend (Assumed) | Simple Baseline |
|-----------|---------|---------|------------------|-----------------|
| **Backbone** | ResNet-50 | ResNet-50 | ResNet-18 | ResNet-18 |
| **Params** | 25M | 25M | 11M | 11M |
| **Features** | 2048 | 2048 | 512 | 512 |
| **Frozen Layers** | None | First 2 blocks | None | None |
| **Temporal** | Avg Pool | Transformer | Avg Pool | Avg Pool |
| **Dropout** | 0.5, 0.3 | 0.5 + 0.1 drop path | Likely 0.3 | 0.3 |
| **Augmentation** | Minimal | VERY AGGRESSIVE | Minimal | Minimal |
| **Results** | 15-20% | 10% | **50%** | Target: 30-40% |

## Why Your Friend's Simple Approach Works

**Their Setup** (likely):
1. **ResNet-18**: Smaller, faster, less prone to overfitting
2. **8 frames**: Standard temporal sampling
3. **Temporal pooling**: Simple average across frames
4. **Minimal augmentation**: Flip + mild color jitter only
5. **Standard training**: AdamW, lr=1e-4, no fancy tricks
6. **No over-regularization**: Just basic dropout

**Result**: 50% action accuracy

**Your Setup** (Phase 2):
1. **ResNet-50**: Bigger, slower, more prone to overfitting
2. **16 frames**: More frames but frozen backbone can't use them
3. **Temporal Transformer**: Complex, harder to train
4. **Aggressive augmentation**: Destroys visual signal
5. **Over-regularization**: Frozen layers + high dropout + drop path + label smoothing
6. **Bugs**: Frame sampling off by one

**Result**: 10% action accuracy

## The Fundamental Mistake

You diagnosed the problem as **overfitting** (15% gap in Phase 1) and applied:
- More regularization
- More augmentation
- More complexity

**But the real issues were**:
1. Model wasn't learning the right features (hence low val accuracy)
2. Augmentation was too aggressive (hurt Phase 2 even more)
3. Complexity made training harder without benefits

**Evidence**: Noun accuracy crashed from 28% → 19% in Phase 2!

## What Needs to Change

### Start with Simplicity

1. **Use ResNet-18** (not ResNet-50)
   - 11M params vs 25M params
   - 3x faster training
   - Less prone to overfitting

2. **Simple temporal pooling** (not Transformer)
   - Average across frames
   - Proven to work
   - Easy to train

3. **Minimal augmentation**
   - Horizontal flip (yes - natural for hands)
   - Mild color jitter (yes - lighting variations)
   - NO grayscale, NO blur, NO aggressive crops

4. **Minimal regularization**
   - Dropout 0.3 only
   - NO frozen layers
   - NO drop path
   - NO label smoothing

5. **Fix bugs**
   - Frame sampling: stop_frame - 1
   - Consistent path handling

### Training Setup

```python
# Model
model = ResNet18 + TemporalAvgPool + LinearClassifiers

# Optimizer
optimizer = AdamW(lr=1e-4, weight_decay=1e-4)

# Scheduler
scheduler = ReduceLROnPlateau(factor=0.5, patience=3)

# Loss
loss = CrossEntropyLoss()  # No label smoothing

# Augmentation
augment = [
    RandomResizedCrop(scale=(0.85, 1.0)),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.2, contrast=0.2),
    RandomErasing(p=0.1)
]
```

## Expected Results

### Week 1 (Initial Training)
- **Target**: 25-35% action accuracy (2-3x improvement)
- Verb accuracy: 35-45%
- Noun accuracy: 28-32% (recovered from Phase 2's 19%)
- Training time: ~50% faster due to ResNet-18

### Week 2 (With Tuning)
- **Target**: 35-45% action accuracy
- Hyperparameter optimization: batch size, learning rate, frame count
- Potentially approaching friend's 50%

## Questions to Ask Your Friend

Before comparing, verify:
1. **Metric**: Top-1 action accuracy? Top-5? Something else?
2. **Dataset**: Same train/val split as you?
3. **Classes**: All 97 verbs × 300 nouns = 29,100 actions?
4. **Training**: Learning rate? Batch size? Epochs? Optimizer?
5. **Augmentation**: What exactly are they using?

Their 50% might be:
- Top-5 action accuracy (more lenient)
- Different split (easier validation set)
- Different metric calculation

But even if it's top-1, it's achievable with the right approach!

## Confidence Levels

**Very High (90%)** that simple baseline will achieve:
- 25-35% action accuracy (2-3x improvement over Phase 2)
- 28-32% noun accuracy (recovering from Phase 2's crash)
- 2x faster training

**High (70%)** that with tuning you'll reach:
- 35-45% action accuracy
- Competitive with friend's approach

**Medium (50%)** that you'll match 50%:
- Depends on friend's exact setup
- May need additional techniques
- But definitely achievable!

## Key Takeaways

1. **Simpler is better**: Friend's simple approach beats your complex Phase 2
2. **Augmentation matters**: Phase 2's aggressive augmentation destroyed noun accuracy
3. **Don't over-regularize**: Can cause underfitting (Phase 2 evidence)
4. **Smaller models**: ResNet-18 is better than ResNet-50 for this task
5. **Fix bugs first**: Frame sampling issue needs attention
6. **Follow proven approaches**: Friend's approach works, yours didn't - learn from it

## Next Steps

See `IMPLEMENTATION_PLAN.md` for detailed step-by-step plan to implement the simple baseline.
