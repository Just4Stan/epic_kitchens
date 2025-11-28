#!/bin/bash
#SBATCH --job-name=moonshot_v3
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=240G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=logs/moonshot_v3_%j.out
#SBATCH --error=logs/moonshot_v3_%j.err

# =============================================================================
# MOONSHOT V3: I/O OPTIMIZED + MAX GPU UTILIZATION
# =============================================================================
# Key improvements over V2:
# - FEWER FRAMES: 8 instead of 16 (50% less I/O, eliminates bottleneck)
# - LARGER BATCH: 800 (fills GPU memory, 6,400 images through CLIP)
# - Same: Trainable CLIP, 5e-4 LR, medium augmentation
#
# Expected performance:
#   - ~4 min/epoch (2x faster than V2)
#   - ~3 hours total for 50 epochs
#   - ~95% GPU utilization (no more power cycling)
#   - GPU memory: ~65-70GB / 80GB
# =============================================================================

module purge
module load cluster/wice/batch
module load Python/3.11.3-GCCcore-12.3.0

# Activate environment
source $VSC_DATA/epic_kitchens/epic_env/bin/activate

# Install CLIP if not available
pip install git+https://github.com/openai/CLIP.git --quiet 2>/dev/null || true

# Set environment
export WANDB_API_KEY="a050122e318cf57511f2c745aa871735df7c6de8"
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=18

cd $VSC_DATA/epic_kitchens

# Create logs directory
mkdir -p logs

echo "============================================"
echo "MOONSHOT V3: I/O OPTIMIZED"
echo "============================================"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "============================================"

# Paths on VSC
FRAMES_DIR="/scratch/leuven/380/vsc38064/EPIC-KITCHENS/EPIC-KITCHENS"
ANNOTATIONS_DIR="$VSC_DATA/epic_kitchens/EPIC-KITCHENS/epic-kitchens-100-annotations-master"

# Check paths exist
if [ ! -d "$FRAMES_DIR" ]; then
    echo "ERROR: Frames directory not found: $FRAMES_DIR"
    echo "Looking for alternative locations..."
    # Try alternative paths
    FRAMES_DIR="/data/leuven/380/vsc38064/EPIC-KITCHENS/EPIC-KITCHENS"
    if [ ! -d "$FRAMES_DIR" ]; then
        FRAMES_DIR="$VSC_DATA/EPIC-KITCHENS/EPIC-KITCHENS"
    fi
fi

echo "Using frames from: $FRAMES_DIR"
echo "Using annotations from: $ANNOTATIONS_DIR"
echo "============================================"

# Run training with I/O optimized settings
python src_v2/train.py \
    --exp_name moonshot_v3 \
    --frames_dir "$FRAMES_DIR" \
    --annotations_dir "$ANNOTATIONS_DIR" \
    --model twostream \
    --image_size 224 \
    --num_frames 8 \
    --batch_size 800 \
    --grad_accum 1 \
    --lr 5e-4 \
    --weight_decay 1e-4 \
    --num_workers 16 \
    --warmup_epochs 3 \
    --epochs 50 \
    --augmentation medium \
    --cutmix_alpha 1.0 \
    --mixup_alpha 0.2 \
    --cutmix_prob 0.5 \
    --mixup_prob 0.3 \
    --random_erasing 0.15 \
    --focal_gamma 2.0 \
    --label_smoothing 0.1 \
    --verb_weight 0.4 \
    --noun_weight 0.6 \
    --class_weight_power 0.5 \
    --dropout 0.3 \
    --use_cosine_noun \
    --cosine_temp 0.05 \
    --use_cross_attention \
    --early_stopping \
    --patience 10 \
    --save_every 5 \
    --wandb

echo "============================================"
echo "MOONSHOT V3 COMPLETE: $(date)"
echo "============================================"
