#!/bin/bash
#SBATCH --job-name=extract_32
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=64G
#SBATCH --partition=batch
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=logs/extract_smart_%j.out
#SBATCH --error=logs/extract_smart_%j.err

# Extract action frames from rgb_frames with blur-aware sampling (32 frames)

module purge
module load cluster/wice/batch
module load Python/3.11.3-GCCcore-12.3.0

source $VSC_DATA/epic_kitchens/epic_env/bin/activate
cd $VSC_DATA/epic_kitchens

mkdir -p logs

# Paths
RGB_ROOT="/scratch/leuven/380/vsc38064/EPIC-KITCHENS/EPIC-KITCHENS"
TRAIN_CSV="EPIC-KITCHENS/epic-kitchens-100-annotations-master/EPIC_100_train.csv"
VAL_CSV="EPIC-KITCHENS/epic-kitchens-100-annotations-master/EPIC_100_validation.csv"
OUTPUT_TRAIN="/scratch/leuven/380/vsc38064/epic_kitchens/extracted_frames_32/train"
OUTPUT_VAL="/scratch/leuven/380/vsc38064/epic_kitchens/extracted_frames_32/val"

echo "Smart Frame Extraction (32 frames, blur-aware)"
echo "=============================================="
echo "Start: $(date)"
echo ""

# Extract training frames
echo "Extracting training frames..."
python src/extract_from_rgb_frames.py \
    --csv "$TRAIN_CSV" \
    --rgb_frames_root "$RGB_ROOT" \
    --output_dir "$OUTPUT_TRAIN" \
    --num_frames 32 \
    --workers 32

echo ""
echo "Extracting validation frames..."
python src/extract_from_rgb_frames.py \
    --csv "$VAL_CSV" \
    --rgb_frames_root "$RGB_ROOT" \
    --output_dir "$OUTPUT_VAL" \
    --num_frames 32 \
    --workers 32

echo ""
echo "Done: $(date)"
