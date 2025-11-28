#!/bin/bash
#SBATCH --job-name=A2_logit
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=logs/A2_%j.out
#SBATCH --error=logs/A2_%j.err

# =============================================================================
# A2: LOGIT ADJUSTMENT ONLY (vs baseline)
# =============================================================================
# Change: Add Logit Adjustment Loss (tau=1.0)
# Uses TSM model but tests the loss function improvement
# =============================================================================

module purge
module load cluster/wice/batch
module load Python/3.11.3-GCCcore-12.3.0

source $VSC_DATA/epic_kitchens/epic_env/bin/activate
export WANDB_API_KEY="a050122e318cf57511f2c745aa871735df7c6de8"
cd $VSC_DATA/epic_kitchens

mkdir -p logs

echo "============================================"
echo "A2: LOGIT ADJUSTMENT ONLY (TSM + Logit Adj)"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/train_v2.py \
    --exp_name A2_logit_adj \
    --epochs 40 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --num_workers 16 \
    --warmup_epochs 3 \
    --num_frames 16 \
    --dropout 0.5 \
    --logit_adjust_tau 1.0 \
    --llrd_decay 1.0 \
    --cutmix_alpha 0 \
    --label_smoothing 0.1 \
    --early_stopping \
    --patience 7 \
    --wandb

echo "A2 COMPLETE: $(date)"
