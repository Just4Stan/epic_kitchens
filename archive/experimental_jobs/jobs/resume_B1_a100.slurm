#!/bin/bash
#SBATCH --job-name=B1_a100
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=logs/B1_resume_%j.out
#SBATCH --error=logs/B1_resume_%j.err

# B1: Resume Logit Adjustment on A100

module purge
module load cluster/wice/batch
module load Python/3.11.3-GCCcore-12.3.0

source $VSC_DATA/epic_kitchens/epic_env/bin/activate
export WANDB_API_KEY="a050122e318cf57511f2c745aa871735df7c6de8"
cd $VSC_DATA/epic_kitchens

mkdir -p logs

echo "B1 RESUME: Logit Adj on A100 from epoch 7"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/train.py \
    --exp_name B1_logit_adj_resume \
    --epochs 40 \
    --start_epoch 6 \
    --resume outputs/B1_logit_adj/checkpoints/best_model.pth \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --num_workers 16 \
    --warmup_epochs 0 \
    --num_frames 16 \
    --backbone resnet50 \
    --temporal_model lstm \
    --dropout 0.5 \
    --label_smoothing 0.1 \
    --augmentation medium \
    --logit_adjust_tau 1.0 \
    --early_stopping \
    --patience 3 \
    --wandb

echo "B1 RESUME COMPLETE: $(date)"
