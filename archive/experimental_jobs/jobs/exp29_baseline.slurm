#!/bin/bash
#SBATCH --job-name=exp29
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=logs/exp29_%j.out
#SBATCH --error=logs/exp29_%j.err

# =============================================================================
# EXP29: BASELINE WITH UNIFIED TRAINING SCRIPT
# =============================================================================
# ResNet50 + LSTM, medium augmentation, batch_size=64
# Best config from previous experiments (exp15 achieved 23.98%)
# =============================================================================

module purge
module load cluster/wice/batch
module load Python/3.11.3-GCCcore-12.3.0

source $VSC_DATA/epic_kitchens/epic_env/bin/activate
export WANDB_API_KEY="a050122e318cf57511f2c745aa871735df7c6de8"
cd $VSC_DATA/epic_kitchens

# Create logs directory
mkdir -p logs

echo "============================================"
echo "EXP29: BASELINE (ResNet50 + LSTM)"
echo "Start: $(date)"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/train.py \
    --exp_name exp29_baseline \
    --epochs 40 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --num_workers 16 \
    --warmup_epochs 3 \
    --num_frames 16 \
    --backbone resnet50 \
    --temporal_model lstm \
    --dropout 0.5 \
    --freeze_backbone none \
    --label_smoothing 0.1 \
    --augmentation medium \
    --early_stopping \
    --patience 7 \
    --wandb

echo "============================================"
echo "EXP29 COMPLETE: $(date)"
echo "============================================"
