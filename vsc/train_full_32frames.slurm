#!/bin/bash
#SBATCH --job-name=full_32f
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=logs/full_32f_%j.out
#SBATCH --error=logs/full_32f_%j.err

# =============================================================================
# OPTIMIZED FULL DATASET TRAINING - 32 FRAMES
# =============================================================================
# Key improvements:
# - 32 frames (2x more temporal context than 16 frames)
# - Gradient accumulation (effective batch 100 = 50 x 2)
# - Memory-optimized for A100 80GB
# =============================================================================

module purge
module load cluster/wice/batch
module load Python/3.11.3-GCCcore-12.3.0

source $VSC_DATA/epic_kitchens/epic_env/bin/activate
export WANDB_API_KEY="a050122e318cf57511f2c745aa871735df7c6de8"

cd $VSC_DATA/epic_kitchens/vsc
mkdir -p logs outputs

echo "============================================"
echo "OPTIMIZED TRAINING - 32 FRAMES"
echo "============================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Paths
FRAMES_DIR="/scratch/leuven/380/vsc38064/EPIC-KITCHENS/EPIC-KITCHENS"
ANNOTATIONS_DIR="$VSC_DATA/epic_kitchens/EPIC-KITCHENS/epic-kitchens-100-annotations-master"

echo ""
echo "Frames: $FRAMES_DIR"
echo "Annotations: $ANNOTATIONS_DIR"
echo ""

python train_full_optimized.py \
    --exp_name full_32frames_v1 \
    --frames_dir "$FRAMES_DIR" \
    --annotations_dir "$ANNOTATIONS_DIR" \
    --output_dir outputs \
    --epochs 50 \
    --batch_size 50 \
    --accumulation_steps 2 \
    --lr 1.5e-4 \
    --weight_decay 1e-4 \
    --num_workers 18 \
    --warmup_epochs 3 \
    --num_frames 32 \
    --backbone resnet50 \
    --temporal_model lstm \
    --dropout 0.5 \
    --label_smoothing 0.1 \
    --cutmix_alpha 1.0 \
    --augmentation medium \
    --early_stopping \
    --patience 10 \
    --wandb

echo ""
echo "============================================"
echo "TRAINING COMPLETE: $(date)"
echo "============================================"
