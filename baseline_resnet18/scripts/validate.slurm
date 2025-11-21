#!/bin/bash
#SBATCH --job-name=resnet18_validate
#SBATCH --output=logs/validate_%j.out
#SBATCH --error=logs/validate_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --account=lp_stadius_fpga_ai

# ==============================================================================
# Validate ResNet-18 Baseline Model
# ==============================================================================

echo "========================================================================"
echo "Validation started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================================================"

# Load required modules
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Navigate to project directory
cd $VSC_DATA/epic_kitchens/baseline_resnet18

# Run validation
echo "Running validation..."
python scripts/validate.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --output validation_results.txt

echo ""
echo "========================================================================"
echo "Validation complete at: $(date)"
echo "========================================================================"
