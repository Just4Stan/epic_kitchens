#!/bin/bash
#SBATCH --job-name=r2plus1d
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=logs/r2plus1d_%j.out
#SBATCH --error=logs/r2plus1d_%j.err

# =============================================================================
# R(2+1)D Networks for EPIC-KITCHENS-100
# =============================================================================
# Configuration:
# - MODEL: R(2+1)D-18 (factorized 3D CNN, similar to SlowFast)
# - FRAMES: 16 (reduced to fit with VideoMAE on same GPU)
# - BATCH SIZE: 32 (balanced for ~60-70GB GPU usage)
# - LR: 0.01 (SGD with momentum, EPIC-KITCHENS baseline)
# - EPOCHS: 80 (standard for EPIC-KITCHENS)
# - Scheduler: Step decay at epochs 20 and 40
# - Expected: 25-35% action accuracy
# =============================================================================

module purge
module load cluster/wice/batch
module load Python/3.11.3-GCCcore-12.3.0

source $VSC_DATA/epic_kitchens/epic_env/bin/activate

export WANDB_API_KEY="a050122e318cf57511f2c745aa871735df7c6de8"
export OMP_NUM_THREADS=18
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $VSC_DATA/epic_kitchens
mkdir -p logs

echo "============================================"
echo "R(2+1)D Training"
echo "============================================"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "============================================"

FRAMES_DIR="/scratch/leuven/380/vsc38064/EPIC-KITCHENS/EPIC-KITCHENS"
ANNOTATIONS_DIR="$VSC_DATA/epic_kitchens/EPIC-KITCHENS/epic-kitchens-100-annotations-master"

if [ ! -d "$FRAMES_DIR" ]; then
    FRAMES_DIR="/data/leuven/380/vsc38064/EPIC-KITCHENS/EPIC-KITCHENS"
    if [ ! -d "$FRAMES_DIR" ]; then
        FRAMES_DIR="$VSC_DATA/EPIC-KITCHENS/EPIC-KITCHENS"
    fi
fi

echo "Using frames from: $FRAMES_DIR"
echo "Using annotations from: $ANNOTATIONS_DIR"
echo "============================================"

python src_video_models/train_slowfast.py \
    --exp_name r2plus1d_kinetics \
    --frames_dir "$FRAMES_DIR" \
    --annotations_dir "$ANNOTATIONS_DIR" \
    --num_frames 16 \
    --alpha 4 \
    --batch_size 32 \
    --epochs 80 \
    --lr 0.01 \
    --weight_decay 0.0005 \
    --dropout 0.7 \
    --verb_weight 0.4 \
    --noun_weight 0.6 \
    --num_workers 16 \
    --save_every 10 \
    --pretrained \
    --wandb

echo "============================================"
echo "R(2+1)D COMPLETE: $(date)"
echo "============================================"
