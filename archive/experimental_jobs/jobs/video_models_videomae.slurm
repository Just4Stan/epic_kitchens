#!/bin/bash
#SBATCH --job-name=videomae
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=logs/videomae_%j.out
#SBATCH --error=logs/videomae_%j.err

# =============================================================================
# VideoMAE Fine-tuning for EPIC-KITCHENS-100
# =============================================================================
# Configuration:
# - MODEL: VideoMAE-Base (pretrained on Kinetics-400)
# - FRAMES: 16 (standard for VideoMAE)
# - BATCH SIZE: 64 (maximize GPU usage, ~35-40GB / 80GB)
# - LR: 1e-4 (fine-tuning rate)
# - EPOCHS: 50
# - Expected: 30-40% action accuracy
# =============================================================================

module purge
module load cluster/wice/batch
module load Python/3.11.3-GCCcore-12.3.0

source $VSC_DATA/epic_kitchens/epic_env/bin/activate

# Install dependencies
pip install transformers av decord --quiet 2>/dev/null || true

export WANDB_API_KEY="a050122e318cf57511f2c745aa871735df7c6de8"
export OMP_NUM_THREADS=18
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $VSC_DATA/epic_kitchens
mkdir -p logs

echo "============================================"
echo "VideoMAE Fine-tuning"
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

python src_video_models/train_videomae.py \
    --exp_name videomae_base_kinetics \
    --frames_dir "$FRAMES_DIR" \
    --annotations_dir "$ANNOTATIONS_DIR" \
    --pretrained_model MCG-NJU/videomae-base-finetuned-kinetics \
    --num_frames 16 \
    --batch_size 64 \
    --epochs 50 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --dropout 0.5 \
    --verb_weight 0.4 \
    --noun_weight 0.6 \
    --num_workers 16 \
    --save_every 5 \
    --wandb

echo "============================================"
echo "VideoMAE COMPLETE: $(date)"
echo "============================================"
