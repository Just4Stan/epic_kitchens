# Validation Results & Overfitting Analysis

Results from validating models on the true EPIC-KITCHENS validation set.

## Dataset Statistics

**Training set (compressed subset):**
- 272 videos from 28 participants
- 67,217 annotated segments (full dataset)
- Local subset: unknown exact count

**Validation set (compressed subset):**
- 10 out of 138 videos available (7%)
- 1,347 out of 9,668 segments available (14%)
- Missing 128 validation videos

**Note:** Results are from partial validation set only.

## Results

### Baseline ResNet-50

**Checkpoint:** `checkpoint_epoch_10.pth` (epoch 9)

| Metric | Training | Validation | Drop |
|--------|----------|------------|------|
| **Verb Top-1** | 76.0% | 39.35% | **-36.65%** |
| **Noun Top-1** | 78.0% | 33.33% | **-44.67%** |
| **Action (both)** | - | 18.56% | - |
| **Verb Top-5** | - | 80.70% | - |
| **Noun Top-5** | - | 56.50% | - |
| **Loss** | ~2.0 | 6.02 | +4.02 |

**Status:** ⚠️ SEVERE overfitting detected

### LSTM Model

**Checkpoint:** `checkpoint_epoch_10_lstm.pth` (epoch 9)

| Metric | Training | Validation | Drop |
|--------|----------|------------|------|
| **Verb Top-1** | 76.0% | 38.75% | **-37.25%** |
| **Noun Top-1** | 78.0% | 31.25% | **-46.75%** |
| **Action (both)** | - | 18.04% | - |
| **Verb Top-5** | - | 78.40% | - |
| **Noun Top-5** | - | 54.57% | - |
| **Loss** | ~2.0 | 5.97 | +3.97 |

**Status:** ⚠️ SEVERE overfitting detected (worse than baseline!)

## Analysis

### Comparison: Baseline vs LSTM

| Model | Verb Top-1 | Noun Top-1 | Action | Verdict |
|-------|------------|------------|--------|---------|
| Baseline | 39.35% | 33.33% | 18.56% | **Better** |
| LSTM | 38.75% | 31.25% | 18.04% | Worse |
| Difference | -0.60% | -2.08% | -0.52% | - |

**Conclusion:** LSTM provides NO improvement over baseline temporal pooling.

**Possible reasons:**
1. **Insufficient data** - 272 videos not enough for LSTM to learn temporal patterns
2. **Overfitting** - LSTM has more parameters, easier to memorize training data
3. **Short segments** - 8 frames might be too few for LSTM advantage
4. **Simple motions** - Kitchen actions might not require complex temporal modeling

### Overfitting Causes

**1. Limited Training Data**
- Training on compressed subset (272 videos)
- Full EPIC-100 has 700+ videos
- Model seeing same kitchens/people repeatedly

**2. Participant Shift**
- Training: 28 participants
- Validation: Different participants (out of 45 total)
- Model hasn't generalized to new environments

**3. Domain Gap**
- Different kitchens (lighting, layout, objects)
- Different camera angles
- Different recording conditions

**4. Simple Architecture**
- 2D CNN doesn't capture motion well
- Temporal pooling loses sequential information
- No multi-modal learning (only RGB, no optical flow)

### Is This Performance Normal?

**Yes and no:**

**Expected for this setup:**
- Limited training data (272 vs 700 videos)
- Simple baseline model
- Compressed dataset subset

**Compared to literature:**
- EPIC-KITCHENS baselines: ~30-40% verb, ~20-30% noun on full dataset
- Our subset results align with limited data performance
- Top-5 accuracy (80% verb, 56% noun) shows model learned meaningful features

**Not expected:**
- LSTM performing worse than baseline
- Should see at least small improvement with temporal modeling

## Recommendations

### To Reduce Overfitting

**1. Data Augmentation (implemented)**
```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
transforms.RandomRotation(degrees=10)
```

**2. Regularization**
```python
# Dropout in classifier
nn.Dropout(p=0.5)

# L2 weight decay (implemented)
optimizer = AdamW(params, weight_decay=1e-5)

# Early stopping (implemented)
EarlyStopping(patience=5)
```

**3. More Training Data**
- Download full EPIC-KITCHENS dataset
- Use all 700 training videos

**4. Better Architecture**
- 3D CNNs (I3D, R(2+1)D)
- Two-stream networks (RGB + optical flow)
- Pre-trained video models (VideoMAE, TimeSformer)

**5. Transfer Learning**
- Use models pre-trained on Kinetics-400 (video dataset)
- Fine-tune on EPIC-KITCHENS

### Current Best Practices

**For baseline comparison:**
- ✓ Simple 2D ResNet + temporal pooling
- ✓ Pre-trained ImageNet weights
- ✓ Data augmentation
- ✓ Early stopping

**For better performance:**
- Use 3D ConvNets or transformers
- Multi-modal learning (RGB + flow)
- More training data
- Longer training (50+ epochs)

## Validation Methods Tested

### 1. Same-Split Validation (INVALID)
**File:** `validation/validate_checkpoint.py`
**Method:** Uses same random split as training
**Result:** 78.78% verb, 85.70% noun (HIGHER than training!)
**Problem:** Testing on training data
**Verdict:** ✗ Invalid

### 2. Augmented Validation
**File:** `validation/validate_checkpoint_augmented.py`
**Method:** Extreme augmentations (flip, rotate, blur, grayscale)
**Result:** 36.40% verb, 33.00% noun
**Verdict:** ✓ Shows overfitting, but not true validation

### 3. True Validation (OFFICIAL)
**File:** `validation/validate_true.py`
**Method:** Uses EPIC_100_validation.csv
**Result:** 39.35% verb, 33.33% noun (baseline)
**Verdict:** ✓ Correct method

## Command Reference

**Run from `epic_kitchens/` directory:**

```bash
cd epic_kitchens
source ../venv/bin/activate

# Validate baseline model
python validation/validate_true.py \
    --checkpoint outputs/checkpoints/checkpoint_epoch_10.pth \
    --batch_size 32

# Validate LSTM model
python validation/validate_true.py \
    --checkpoint outputs/checkpoints/checkpoint_epoch_10_lstm.pth \
    --batch_size 32

# Check available cameras for real-time testing
python inference/list_cameras.py

# Real-time webcam inference
python inference/realtime_webcam.py \
    --checkpoint outputs/checkpoints/checkpoint_epoch_10.pth \
    --camera 1  # Use iPhone camera
```

## Conclusion

**Model Status:**
- ✓ Model is functioning correctly
- ✓ Learned meaningful visual features (high top-5 accuracy)
- ✗ Severe overfitting due to limited data and participant shift
- ✗ LSTM provides no benefit over baseline

**For Assignment:**
- Results are acceptable given constraints
- Overfitting analysis demonstrates understanding
- Can discuss improvements in report

**For Production:**
- Need full dataset
- Need better architecture (3D CNN or transformer)
- Need multi-modal learning
