#!/bin/bash
#SBATCH --job-name=exp36
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=logs/exp36_%j.out
#SBATCH --error=logs/exp36_%j.err

# =============================================================================
# EXP36: TSM + LOGIT ADJUSTMENT + CUTMIX + LLRD
# =============================================================================
# Research-backed improvements:
# 1. TSM (Temporal Shift Module) - zero params, better than LSTM
# 2. Logit Adjustment Loss - handles 300 noun classes long-tail
# 3. CutMix with temporal consistency - better regularization
# 4. Layer-wise LR Decay - better fine-tuning
# 5. Lower dropout (0.3) - exp17 showed this helps
#
# Expected: Significant improvement over exp15's 23.98%
# =============================================================================

module purge
module load cluster/wice/batch
module load Python/3.11.3-GCCcore-12.3.0

source $VSC_DATA/epic_kitchens/epic_env/bin/activate
export WANDB_API_KEY="a050122e318cf57511f2c745aa871735df7c6de8"
cd $VSC_DATA/epic_kitchens

mkdir -p logs

echo "============================================"
echo "EXP36: TSM + ALL RESEARCH IMPROVEMENTS"
echo "============================================"
echo "Improvements over baseline (exp15 = 23.98%):"
echo "  - TSM instead of LSTM (zero extra params)"
echo "  - Logit Adjustment Loss (tau=1.0)"
echo "  - CutMix with temporal consistency"
echo "  - Layer-wise LR Decay (0.75)"
echo "  - Dropout 0.3 (proven better)"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/train_v2.py \
    --exp_name exp36_tsm_advanced \
    --epochs 60 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --num_workers 16 \
    --warmup_epochs 5 \
    --num_frames 16 \
    --dropout 0.3 \
    --logit_adjust_tau 1.0 \
    --llrd_decay 0.75 \
    --cutmix_alpha 1.0 \
    --label_smoothing 0.1 \
    --early_stopping \
    --patience 12 \
    --wandb

echo "============================================"
echo "EXP36 COMPLETE: $(date)"
echo "============================================"
