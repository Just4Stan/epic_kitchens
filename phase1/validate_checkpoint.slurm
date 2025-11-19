#!/bin/bash
#SBATCH --job-name=epic_validate
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --output=validation_output_%j.txt
#SBATCH --error=validation_error_%j.txt

# Load modules
module purge
module load Python/3.11.5-GCCcore-13.2.0

# Navigate to project directory
cd $VSC_DATA/lp_edu_rdlab/epic_kitchens

# Activate virtual environment
source venv/bin/activate

# Run validation on the latest checkpoint
echo "Starting validation..."
echo "Checkpoint: $1"
python validate_checkpoint.py --checkpoint "$1" --batch_size 32

echo "Validation complete!"
