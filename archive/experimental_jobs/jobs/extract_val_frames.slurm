#!/bin/bash
#SBATCH --job-name=extract_val
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
#SBATCH --partition=batch
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=logs/extract_val_%j.out
#SBATCH --error=logs/extract_val_%j.err

# Extract validation frames for faster validation

module purge
module load cluster/wice/batch
module load Python/3.11.3-GCCcore-12.3.0

source $VSC_DATA/epic_kitchens/epic_env/bin/activate
cd $VSC_DATA/epic_kitchens

mkdir -p logs extracted_frames/val

echo "Extracting validation frames..."
echo "Start: $(date)"

python src/extract_val_frames.py \
    --val_csv EPIC-KITCHENS/epic-kitchens-100-annotations-master/EPIC_100_validation.csv \
    --video_dir EPIC-KITCHENS/videos_640x360_validation \
    --output_dir extracted_frames/val \
    --workers 16

echo "Done: $(date)"
