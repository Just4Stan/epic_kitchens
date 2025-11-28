# Implementation Plan: Fixing Model Performance

## Summary

Your models are getting 10-15% action accuracy while your friend's simple ResNet-18 approach gets ~50%.

**Root Causes Identified**:
1. Over-aggressive data augmentation destroying object details
2. Over-regularization preventing effective learning
3. Unnecessary model complexity
4. Frame sampling bug (using `stop_frame` instead of `stop_frame - 1`)

## Files Created

### 1. `ANALYSIS_AND_FIXES.md`
Comprehensive analysis document covering:
- All issues identified in your current implementation
- Comparison to your friend's approach
- Detailed explanations of each problem
- Expected performance improvements

### 2. `model_simple_baseline.py`
Simple ResNet-18 baseline model:
- 512-dimensional features (vs 2048 in ResNet-50)
- Simple temporal average pooling
- Minimal dropout (0.3)
- No transformers, no attention, no complexity
- ~11M parameters (vs 25M in ResNet-50)

### 3. `dataset_simple_fixed.py`
Fixed dataset loader:
- **Bug fix**: Uses `stop_frame - 1` (not `stop_frame`)
- Minimal augmentation:
  - Horizontal flip (50%)
  - Small color jitter (brightness=0.2, contrast=0.2)
  - Mild random crop (scale 0.85-1.0, not 0.6-1.0)
  - Small random erasing (10%, not 30%)
- **Removed**: Grayscale, Gaussian blur, aggressive cropping

### 4. `train_simple_baseline.py`
Clean training script:
- Standard cross-entropy loss (no label smoothing)
- AdamW optimizer with moderate weight decay (1e-4)
- ReduceLROnPlateau scheduler
- Mixed precision training
- Clear logging and checkpointing

### 5. `IMPLEMENTATION_PLAN.md` (this file)
Step-by-step implementation guide

## Step-by-Step Instructions

### Step 1: Test the New Model Locally (Optional)

```bash
cd /path/to/epic_kitchens

# Test the model
python model_simple_baseline.py

# Test the dataset
python dataset_simple_fixed.py
```

### Step 2: Train the Simple Baseline

**Option A: Local Training** (if you have GPU)
```bash
python train_simple_baseline.py \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-4 \
    --num_frames 8 \
    --output_dir outputs_simple_baseline
```

**Option B: VSC Cluster Training** (recommended)

Create a SLURM script: `train_simple_baseline.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=simple_baseline
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=simple_baseline_output_%j.txt
#SBATCH --error=simple_baseline_error_%j.txt

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd $VSC_DATA/epic_kitchens
source epic_env/bin/activate

echo "========================================"
echo "Training Simple Baseline Model"
echo "ResNet-18 + Temporal Pooling"
echo "========================================"

python train_simple_baseline.py \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-4 \
    --num_frames 8 \
    --output_dir outputs_simple_baseline

echo "========================================"
echo "Training Complete!"
echo "========================================"
```

Submit to cluster:
```bash
sbatch train_simple_baseline.slurm
```

### Step 3: Monitor Training

**Local**:
Watch the terminal output - you'll see progress bars and epoch summaries

**VSC Cluster**:
```bash
# Check job status
squeue -u $USER

# View output
tail -f simple_baseline_output_*.txt

# View errors
tail -f simple_baseline_error_*.txt
```

### Step 4: Evaluate Results

After training completes, check:

```bash
cd outputs_simple_baseline/logs
cat training_history.json
```

**Expected improvements**:
- **Week 1 Goal**: 25-35% action accuracy (2-3x improvement)
- **Week 2 Goal**: 35-45% action accuracy with tuning

**What to look for**:
1. Noun accuracy should be 25-32% (vs 19% in Phase 2)
2. Verb accuracy should be 35-45% (vs 36% in Phase 2)
3. Action accuracy should be 25-35% (vs 10% in Phase 2)
4. Overfitting gap should be 5-10% (healthy range)

### Step 5: Hyperparameter Tuning (if needed)

If results are good but not matching friend's 50%, try:

**Experiment 1: Different learning rates**
```bash
python train_simple_baseline.py --lr 5e-5 --output_dir outputs_simple_lr5e5
python train_simple_baseline.py --lr 2e-4 --output_dir outputs_simple_lr2e4
```

**Experiment 2: Different batch sizes**
```bash
python train_simple_baseline.py --batch_size 16 --output_dir outputs_simple_bs16
python train_simple_baseline.py --batch_size 64 --output_dir outputs_simple_bs64
```

**Experiment 3: Different frame counts**
```bash
python train_simple_baseline.py --num_frames 16 --output_dir outputs_simple_16frames
```

**Experiment 4: Longer training**
```bash
python train_simple_baseline.py --epochs 50 --output_dir outputs_simple_50epochs
```

## Comparison Script

Create `compare_models.py` to compare all approaches:

```python
import json
from pathlib import Path

models = {
    'Phase 1 Baseline': 'outputs/logs/training_history.json',
    'Phase 2 Improved': 'outputs_improved/logs/training_history.json',
    'Simple Baseline': 'outputs_simple_baseline/logs/training_history.json',
}

print("=" * 80)
print("Model Performance Comparison")
print("=" * 80)
print(f"{'Model':<20} {'Verb Acc':>12} {'Noun Acc':>12} {'Action Acc':>12}")
print("-" * 80)

for name, path in models.items():
    try:
        with open(path, 'r') as f:
            history = json.load(f)

        best_verb = max(history['val_verb_acc'])
        best_noun = max(history['val_noun_acc'])
        best_action = max(history['val_action_acc'])

        print(f"{name:<20} {best_verb:>11.2f}% {best_noun:>11.2f}% {best_action:>11.2f}%")
    except FileNotFoundError:
        print(f"{name:<20} {'Not trained':>12} {'':>12} {'':>12}")

print("=" * 80)
```

Run with:
```bash
python compare_models.py
```

## Questions to Ask Your Friend

Before claiming success, verify with your friend:

1. **What exact metric?**
   - Top-1 action accuracy? (both verb and noun correct)
   - Top-5 action accuracy? (verb or noun in top 5)
   - Something else?

2. **What dataset split?**
   - EPIC-KITCHENS-100 train/val split?
   - A custom subset?
   - Different annotation version?

3. **How many classes?**
   - All 97 verbs and 300 nouns?
   - A subset (e.g., top 20 most common)?

4. **Training details**:
   - Learning rate?
   - Batch size?
   - Number of epochs?
   - Any augmentation?

## Expected Timeline

**Week 1** (Training & Validation):
- Day 1: Test code, submit training job
- Days 2-3: Training runs (~24 hours)
- Day 4: Analyze results
- Days 5-7: Hyperparameter experiments

**Week 2** (Optimization):
- Continue hyperparameter tuning
- Try longer training if needed
- Document best practices

**Week 3** (Analysis & Presentation):
- Compare all approaches
- Update presentation
- Write up findings

## Success Criteria

**Minimum success** (should achieve in Week 1):
- 25%+ action accuracy (2.5x improvement over Phase 2)
- 28%+ noun accuracy (vs 19% in Phase 2)
- 35%+ verb accuracy (maintained from Phase 2)

**Good success** (achievable in Week 2):
- 35%+ action accuracy (3.5x improvement)
- 32%+ noun accuracy
- 40%+ verb accuracy

**Excellent success** (may require additional techniques):
- 45%+ action accuracy (approaching friend's 50%)
- 35%+ noun accuracy
- 45%+ verb accuracy

## Troubleshooting

### If results are still poor (<20% action accuracy):

1. **Check data loading**:
   ```python
   python dataset_simple_fixed.py
   # Verify frames load correctly
   ```

2. **Verify model forward pass**:
   ```python
   python model_simple_baseline.py
   # Check output shapes
   ```

3. **Check for NaN losses**:
   - Look in training logs for "loss: nan"
   - May need to reduce learning rate

4. **Verify labels**:
   ```python
   # Check label distribution
   import pandas as pd
   df = pd.read_csv('EPIC-KITCHENS/epic-kitchens-100-annotations-master/EPIC_100_train.csv')
   print(df['verb_class'].value_counts())
   print(df['noun_class'].value_counts())
   ```

### If training is too slow:

1. **Reduce batch size**: 32 → 16
2. **Reduce workers**: 8 → 4
3. **Use smaller image size**: 224 → 196
4. **Reduce frames**: 8 → 4

### If overfitting (>15% gap):

1. **Increase dropout**: 0.3 → 0.4
2. **Increase weight decay**: 1e-4 → 1e-3
3. **Add more augmentation** (but not as aggressive as Phase 2!)

### If underfitting (<5% gap, low train accuracy):

1. **Reduce dropout**: 0.3 → 0.2
2. **Reduce weight decay**: 1e-4 → 1e-5
3. **Increase learning rate**: 1e-4 → 2e-4
4. **Train longer**: 30 → 50 epochs

## Next Steps After Success

Once you achieve 30-40% action accuracy:

1. **Document findings**:
   - What worked?
   - What didn't work?
   - Key lessons learned

2. **Update presentation**:
   - Add simple baseline results
   - Explain why simpler approach works better
   - Show comparison graphs

3. **Consider advanced techniques** (only if needed):
   - Two-stream (RGB + optical flow)
   - Better pre-training (Kinetics-400)
   - Temporal segment networks
   - Ensemble methods

4. **Write up analysis**:
   - Create comprehensive report
   - Include ablation studies
   - Document best practices for future work

## Commit and Push Changes

Once you've created and tested the files:

```bash
git add ANALYSIS_AND_FIXES.md
git add IMPLEMENTATION_PLAN.md
git add model_simple_baseline.py
git add dataset_simple_fixed.py
git add train_simple_baseline.py
git add train_simple_baseline.slurm  # if you create this

git commit -m "Add simple baseline model to fix performance issues

- Created ResNet-18 simple baseline (vs ResNet-50)
- Fixed frame sampling bug (stop_frame - 1)
- Reduced aggressive augmentation
- Minimal regularization approach
- Expected 2-3x improvement in action accuracy"

git push -u origin claude/inspect-epic-kitchens-models-01PhYgUEsNzJiPjPH6Z6YLxL
```

## Summary

You now have:
1. ✅ Comprehensive analysis of all issues
2. ✅ Fixed dataset loader (bug fix + appropriate augmentation)
3. ✅ Simple ResNet-18 baseline model
4. ✅ Clean training script
5. ✅ Step-by-step implementation plan

**Expected outcome**: 2-3x improvement (from 10-15% to 25-40% action accuracy)

**Next action**: Train the simple baseline and see results!
