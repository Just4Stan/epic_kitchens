#!/bin/bash
#SBATCH --job-name=resnet18_baseline
#SBATCH --output=logs/resnet18_%j.out
#SBATCH --error=logs/resnet18_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --account=lp_stadius_fpga_ai

# ==============================================================================
# Simple ResNet-18 Baseline Training
# ==============================================================================

echo "========================================================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================================================"

# Load required modules
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Show loaded modules
echo "Loaded modules:"
module list

# Activate virtual environment (if using one)
# source $VSC_DATA/venv/epic_kitchens/bin/activate

# Navigate to project directory
cd $VSC_DATA/epic_kitchens/baseline_resnet18

# Create output directories
mkdir -p outputs logs outputs/checkpoints

# Show Python and PyTorch info
echo ""
echo "========================================================================"
echo "Python Environment"
echo "========================================================================"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0)}')"
fi

# Show disk usage
echo ""
echo "========================================================================"
echo "Disk Usage"
echo "========================================================================"
echo "Project directory: $(pwd)"
df -h $VSC_DATA

# Run training
echo ""
echo "========================================================================"
echo "Starting Training"
echo "========================================================================"
echo "Start time: $(date)"

python scripts/train.py

echo ""
echo "========================================================================"
echo "Training Complete"
echo "========================================================================"
echo "End time: $(date)"

# Show output files
echo ""
echo "Output files:"
ls -lh outputs/

# Show best results
if [ -f "outputs/training_log.txt" ]; then
    echo ""
    echo "Best results:"
    tail -5 outputs/training_log.txt
fi

echo ""
echo "========================================================================"
echo "Job finished at: $(date)"
echo "========================================================================"
