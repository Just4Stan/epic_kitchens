#!/bin/bash
#SBATCH --job-name=train_improved
#SBATCH --time=12:00:00
#SBATCH --nodes=1#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=training_improved_output_%j.txt
#SBATCH --error=training_improved_error_%j.txt

# Load modules
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Activate environment
cd $VSC_DATA/epic_kitchens
source epic_env/bin/activate

# Print environment info
echo "========================================"
echo "Training Improved Model"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

nvidia-smi
echo ""

# Run training with improved model
python train_improved.py \
    --epochs 30 \
    --batch_size 24 \
    --lr 5e-5 \
    --output_dir outputs_improved

echo ""
echo "========================================"
echo "Training Completed!"
echo "End time: $(date)"
echo "========================================"
