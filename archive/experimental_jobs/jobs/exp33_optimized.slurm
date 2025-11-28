#!/bin/bash
#SBATCH --job-name=exp33
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=logs/exp33_%j.out
#SBATCH --error=logs/exp33_%j.err

# =============================================================================
# EXP33: OPTIMIZED BASED ON ALL FINDINGS
# =============================================================================
# Best config from exp15 (23.98%) + improvements from exp17 (dropout 0.3)
# Changes: dropout 0.3, batch 80, longer training, longer patience
# =============================================================================

module purge
module load cluster/wice/batch
module load Python/3.11.3-GCCcore-12.3.0

source $VSC_DATA/epic_kitchens/epic_env/bin/activate
export WANDB_API_KEY="a050122e318cf57511f2c745aa871735df7c6de8"
cd $VSC_DATA/epic_kitchens

mkdir -p logs

echo "============================================"
echo "EXP33: OPTIMIZED (dropout=0.3, batch=80, 60 epochs)"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/train.py \
    --exp_name exp33_optimized \
    --epochs 60 \
    --batch_size 80 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --num_workers 16 \
    --warmup_epochs 5 \
    --num_frames 16 \
    --backbone resnet50 \
    --temporal_model lstm \
    --dropout 0.3 \
    --freeze_backbone none \
    --label_smoothing 0.1 \
    --augmentation medium \
    --early_stopping \
    --patience 12 \
    --wandb

echo "EXP33 COMPLETE: $(date)"
