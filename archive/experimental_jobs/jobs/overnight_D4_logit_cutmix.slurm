#!/bin/bash
#SBATCH --job-name=D4_lc
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=logs/D4_%j.out
#SBATCH --error=logs/D4_%j.err

# D4: Logit Adjustment + CutMix combined (best of both)

module purge
module load cluster/wice/batch
module load Python/3.11.3-GCCcore-12.3.0

source $VSC_DATA/epic_kitchens/epic_env/bin/activate
export WANDB_API_KEY="a050122e318cf57511f2c745aa871735df7c6de8"
cd $VSC_DATA/epic_kitchens

mkdir -p logs

echo "D4: Logit Adj + CutMix (combined)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/train.py \
    --exp_name D4_logit_cutmix \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --num_workers 16 \
    --warmup_epochs 3 \
    --num_frames 16 \
    --backbone resnet50 \
    --temporal_model lstm \
    --dropout 0.5 \
    --label_smoothing 0.1 \
    --augmentation medium \
    --logit_adjust_tau 1.0 \
    --cutmix_alpha 1.0 \
    --early_stopping \
    --patience 5 \
    --wandb

echo "D4 COMPLETE: $(date)"
