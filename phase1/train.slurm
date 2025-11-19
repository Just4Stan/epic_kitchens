#!/bin/bash -l
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=epic_kitchens
#SBATCH --output=training_output_%j.txt
#SBATCH --error=training_error_%j.txt

module purge
module load cluster/wice/interactive
module load Python/3.11.3-GCCcore-12.3.0

# Activate RDLAB venv (has PyTorch already)
source $VSC_DATA/RDLAB/venv/bin/activate

# Navigate to project directory
cd $VSC_DATA/epic_kitchens

echo "=" * 70
echo "EPIC-KITCHENS-100 Training Job"
echo "=" * 70
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Check dataset
echo "Checking dataset..."
ls -lh EPIC-KITCHENS/videos_640x360/ | head -10
echo ""

# Run training with optimized hyperparameters
python train.py \
    --data_dir EPIC-KITCHENS \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --num_frames 8 \
    --device cuda \
    --num_workers 8

echo ""
echo "=" * 70
echo "Training complete!"
echo "End time: $(date)"
echo "=" * 70
echo ""

# Show outputs
echo "Generated files:"
ls -lh outputs/checkpoints/
