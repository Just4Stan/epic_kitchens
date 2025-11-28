#!/bin/bash
#SBATCH --job-name=moonshot_v7
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=240G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=logs/moonshot_v7_%j.out
#SBATCH --error=logs/moonshot_v7_%j.err

# =============================================================================
# MOONSHOT V7: FEWER FRAMES = 50% LESS CLIP MEMORY
# =============================================================================
# All critical bugs fixed + aggressive memory optimization:
# - ✅ CLIP now trainable (torch.no_grad removed)
# - ✅ self.freeze_clip instance variable added
# - ✅ Temporal projection uses ALL 1024 dims (not just 512)
# - ✅ CLIP normalization (not ImageNet)
#
# Memory optimizations:
# - FRAMES: 4 (50% reduction = 50% less CLIP memory!)
# - GRADIENT CHECKPOINTING: Enabled for CLIP
# - BATCH SIZE: 250 (1,000 images through CLIP per batch)
# - GRAD ACCUM: 6 (effective batch = 1500)
# - CPU CORES: 36 (better data loading)
# - Memory fragmentation fix (expandable_segments)
# - Expected GPU: ~50GB / 80GB (very conservative)
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
export OMP_NUM_THREADS=36
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $VSC_DATA/epic_kitchens

# Create logs directory
mkdir -p logs

echo "============================================"
echo "MOONSHOT V7: FEWER FRAMES"
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

# Run training with fixed model + aggressive memory optimization
python src_v2/train.py \
    --exp_name moonshot_v7 \
    --frames_dir "$FRAMES_DIR" \
    --annotations_dir "$ANNOTATIONS_DIR" \
    --model twostream \
    --image_size 224 \
    --num_frames 4 \
    --batch_size 250 \
    --grad_accum 6 \
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
echo "MOONSHOT V7 COMPLETE: $(date)"
echo "============================================"
