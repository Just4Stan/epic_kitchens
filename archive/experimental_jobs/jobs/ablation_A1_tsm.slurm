#!/bin/bash
#SBATCH --job-name=A1_tsm
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=logs/A1_%j.out
#SBATCH --error=logs/A1_%j.err

# =============================================================================
# A1: TSM ONLY (vs baseline LSTM)
# =============================================================================
# Change: Replace LSTM with Temporal Shift Module
# Everything else identical to baseline
# =============================================================================

module purge
module load cluster/wice/batch
module load Python/3.11.3-GCCcore-12.3.0

source $VSC_DATA/epic_kitchens/epic_env/bin/activate
export WANDB_API_KEY="a050122e318cf57511f2c745aa871735df7c6de8"
cd $VSC_DATA/epic_kitchens

mkdir -p logs

echo "============================================"
echo "A1: TSM ONLY (baseline + TSM instead of LSTM)"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/train_v2.py \
    --exp_name A1_tsm_only \
    --epochs 40 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --num_workers 16 \
    --warmup_epochs 3 \
    --num_frames 16 \
    --dropout 0.5 \
    --logit_adjust_tau 0 \
    --llrd_decay 1.0 \
    --cutmix_alpha 0 \
    --label_smoothing 0.1 \
    --early_stopping \
    --patience 7 \
    --wandb

echo "A1 COMPLETE: $(date)"
