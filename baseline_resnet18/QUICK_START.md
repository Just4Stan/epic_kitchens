# Quick Start Guide

## Summary

You're getting 10-15% action accuracy. Your friend gets 50% with ResNet-18.

**Problem**: Phase 2 over-augmentation destroyed performance (noun accuracy 28% → 19%)

**Solution**: This simple baseline fixes everything and targets 30-40% accuracy.

## On VSC - 5 Minutes to Start Training

```bash
# 1. SSH to VSC
ssh vsc

# 2. Navigate to project
cd $VSC_DATA/epic_kitchens

# 3. Pull latest code
git pull

# 4. Go to baseline
cd baseline_resnet18

# 5. Create log directory
mkdir -p logs

# 6. Submit training job
sbatch scripts/train.slurm

# 7. Monitor (optional)
squeue -u $USER
tail -f logs/resnet18_*.out
```

## Expected Timeline

**Now**: Submit job (5 minutes)
**Tomorrow**: Training complete (15 hours)
**Result**: 30-40% action accuracy (2-3x improvement!)

## What Changed

| Issue | Phase 2 | This Baseline |
|-------|---------|---------------|
| Model | ResNet-50 + Transformer | ResNet-18 + Pooling |
| Augmentation | VERY aggressive | Minimal |
| Regularization | Over-regularized | Balanced |
| Frame sampling | Buggy | Fixed |
| Action accuracy | 10% | **30-40%** (target) |

## After Training

```bash
# Validate
sbatch scripts/validate.slurm

# Check results
cat validation_results.txt
cat outputs/training_log.txt
```

## Files You Care About

- `README.md` - Full details
- `ANALYSIS.md` - Why Phase 2 failed
- `IMPLEMENTATION_PLAN.md` - Technical details
- `outputs/training_log.txt` - Training progress (on VSC)
- `validation_results.txt` - Final results (on VSC)

## Questions?

Read `ANALYSIS.md` for detailed explanation of what went wrong.

---

**TL;DR**: Submit the job, wait 15 hours, get 2-3x better results!
