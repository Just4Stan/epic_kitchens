#!/bin/bash
#SBATCH --job-name=model_8
#SBATCH --time=14:00:00
#SBATCH --nodes=1#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=training_config_8_output_%j.txt
#SBATCH --error=training_config_8_error_%j.txt

# Load modules
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Activate environment
cd $VSC_DATA/epic_kitchens
source epic_env/bin/activate

# Print info
echo "========================================"
echo "Training Model 8: MODEL_DESC"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

nvidia-smi
echo ""

# Run training
TRAIN_CMD

echo ""
echo "========================================"
echo "Training Completed!"
echo "End time: $(date)"
echo "========================================"
