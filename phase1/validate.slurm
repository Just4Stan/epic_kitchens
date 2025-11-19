#!/bin/bash
#SBATCH --job-name=validate_models
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=validation_output_%j.txt
#SBATCH --error=validation_error_%j.txt

module purge
module load cluster/wice/interactive
module load Python/3.11.3-GCCcore-12.3.0

cd $VSC_DATA/epic_kitchens
source epic_env/bin/activate

echo "========================================"
echo "Validating All 5 Trained Models"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
nvidia-smi

# Check if validation annotations exist
if [ ! -f "EPIC_100_validation.csv" ]; then
    echo "ERROR: Validation CSV not found!"
    exit 1
fi

# Run validation
python validate_models.py \
    --val_csv EPIC_100_validation.csv \
    --val_video_dir EPIC-KITCHENS/videos_640x360_validation \
    --output_file validation_results.json \
    --batch_size 32 \
    --num_workers 8

echo ""
echo "========================================"
echo "Validation Complete!"
echo "End: $(date)"
echo "========================================"
