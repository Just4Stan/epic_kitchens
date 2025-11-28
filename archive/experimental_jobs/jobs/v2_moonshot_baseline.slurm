#!/bin/bash
#SBATCH --job-name=moonshot_bl
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=logs/moonshot_baseline_%j.out
#SBATCH --error=logs/moonshot_baseline_%j.err

# =============================================================================
# MOONSHOT BASELINE: ALL IMPROVEMENTS WITHOUT CLIP
# =============================================================================
# Fallback if CLIP fails - uses ResNet50 backbone with all other improvements:
# - 320x320 resolution
# - CutMix + MixUp + RandomErasing
# - Focal Loss
# - Class-balanced weighting
# =============================================================================

module purge
module load cluster/wice/batch
module load Python/3.11.3-GCCcore-12.3.0

source $VSC_DATA/epic_kitchens/epic_env/bin/activate

export WANDB_API_KEY="a050122e318cf57511f2c745aa871735df7c6de8"
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=18

cd $VSC_DATA/epic_kitchens
mkdir -p logs

echo "============================================"
echo "MOONSHOT BASELINE (ResNet50 + All Improvements)"
echo "Start: $(date)"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

FRAMES_DIR="/scratch/leuven/380/vsc38064/EPIC-KITCHENS/EPIC-KITCHENS"
ANNOTATIONS_DIR="$VSC_DATA/epic_kitchens/EPIC-KITCHENS/epic-kitchens-100-annotations-master"

if [ ! -d "$FRAMES_DIR" ]; then
    FRAMES_DIR="/data/leuven/380/vsc38064/EPIC-KITCHENS/EPIC-KITCHENS"
    if [ ! -d "$FRAMES_DIR" ]; then
        FRAMES_DIR="$VSC_DATA/EPIC-KITCHENS/EPIC-KITCHENS"
    fi
fi

echo "Frames: $FRAMES_DIR"
echo "Annotations: $ANNOTATIONS_DIR"

python src_v2/train.py \
    --exp_name moonshot_baseline \
    --frames_dir "$FRAMES_DIR" \
    --annotations_dir "$ANNOTATIONS_DIR" \
    --model baseline \
    --backbone resnet50 \
    --image_size 320 \
    --num_frames 32 \
    --batch_size 48 \
    --grad_accum 1 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --num_workers 16 \
    --warmup_epochs 5 \
    --epochs 50 \
    --augmentation heavy \
    --cutmix_alpha 1.0 \
    --mixup_alpha 0.2 \
    --cutmix_prob 0.5 \
    --mixup_prob 0.3 \
    --random_erasing 0.25 \
    --focal_gamma 2.0 \
    --label_smoothing 0.1 \
    --verb_weight 0.4 \
    --noun_weight 0.6 \
    --class_weight_power 0.5 \
    --dropout 0.3 \
    --early_stopping \
    --patience 10 \
    --save_every 5 \
    --wandb

echo "============================================"
echo "MOONSHOT BASELINE COMPLETE: $(date)"
echo "============================================"
