#!/bin/bash
#SBATCH --job-name=exp34
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=logs/exp34_%j.out
#SBATCH --error=logs/exp34_%j.err

# =============================================================================
# EXP34: HIGHER LEARNING RATE WITH LONGER WARMUP
# =============================================================================
# Hypothesis: Higher LR (2e-4) with longer warmup may find better optima
# exp9 with lr=2e-4 got 22.49%, but had short warmup
# =============================================================================

module purge
module load cluster/wice/batch
module load Python/3.11.3-GCCcore-12.3.0

source $VSC_DATA/epic_kitchens/epic_env/bin/activate
export WANDB_API_KEY="a050122e318cf57511f2c745aa871735df7c6de8"
cd $VSC_DATA/epic_kitchens

mkdir -p logs

echo "============================================"
echo "EXP34: LR=2e-4, WARMUP=5, DROPOUT=0.3"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/train.py \
    --exp_name exp34_lr2e4 \
    --epochs 50 \
    --batch_size 80 \
    --lr 2e-4 \
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
    --patience 10 \
    --wandb

echo "EXP34 COMPLETE: $(date)"
