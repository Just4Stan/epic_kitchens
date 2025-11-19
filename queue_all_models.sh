#!/bin/bash
# Queue all 8 model training jobs

cd /data/leuven/380/vsc38064/epic_kitchens

echo "========================================"
echo "Queuing 8 Model Training Jobs"
echo "========================================"

# Model 1: Improved Transformer (default: lr=5e-5, dropout=0.5)
echo "[1/8] Queuing: Improved Transformer (lr=5e-5, dropout=0.5)"
sbatch train_config1.slurm

# Model 2: Improved Transformer (higher LR)
echo "[2/8] Queuing: Improved Transformer (lr=1e-4, dropout=0.5)"
sbatch train_config2.slurm

# Model 3: Improved Transformer (more dropout)
echo "[3/8] Queuing: Improved Transformer (lr=5e-5, dropout=0.6)"
sbatch train_config3.slurm

# Model 4: Improved Transformer (less dropout)
echo "[4/8] Queuing: Improved Transformer (lr=5e-5, dropout=0.4)"
sbatch train_config4.slurm

# Model 5: Two-Stream Cross-Task (default)
echo "[5/8] Queuing: Two-Stream Cross-Task (lr=5e-5, dropout=0.5)"
sbatch train_config5.slurm

# Model 6: Two-Stream Cross-Task (higher LR)
echo "[6/8] Queuing: Two-Stream Cross-Task (lr=1e-4, dropout=0.5)"
sbatch train_config6.slurm

# Model 7: Two-Stream Cross-Task (more regularization)
echo "[7/8] Queuing: Two-Stream Cross-Task (lr=5e-5, dropout=0.6)"
sbatch train_config7.slurm

# Model 8: Improved Transformer with warmup
echo "[8/8] Queuing: Improved Transformer (lr=1e-4, warmup, dropout=0.5)"
sbatch train_config8.slurm

echo ""
echo "========================================"
echo "All 8 jobs queued!"
echo "========================================"
echo ""
echo "Check status with: squeue -u \$USER"
echo "Monitor with: tail -f training_*_output_*.txt"
