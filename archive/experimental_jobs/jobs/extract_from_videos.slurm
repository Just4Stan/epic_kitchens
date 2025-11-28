#!/bin/bash
#SBATCH --job-name=extract_vid
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=64G
#SBATCH --partition=batch
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=logs/extract_vid_%j.out
#SBATCH --error=logs/extract_vid_%j.err

# Extract action frames from REDUCED dataset videos with blur-aware sampling (32 frames)

module purge
module load cluster/wice/batch
module load Python/3.11.3-GCCcore-12.3.0

source $VSC_DATA/epic_kitchens/epic_env/bin/activate
cd $VSC_DATA/epic_kitchens

mkdir -p logs

# Paths - using reduced dataset videos uploaded to scratch
VIDEO_DIR="/scratch/leuven/380/vsc38064/epic_kitchens/videos_reduced"
TRAIN_CSV="EPIC-KITCHENS/epic-kitchens-100-annotations-master/EPIC_100_train.csv"
VAL_CSV="EPIC-KITCHENS/epic-kitchens-100-annotations-master/EPIC_100_validation.csv"
OUTPUT_TRAIN="/scratch/leuven/380/vsc38064/epic_kitchens/extracted_frames_32_reduced/train"
OUTPUT_VAL="/scratch/leuven/380/vsc38064/epic_kitchens/extracted_frames_32_reduced/val"

echo "Smart Frame Extraction from Videos (32 frames, blur-aware)"
echo "==========================================================="
echo "Start: $(date)"
echo "Video dir: $VIDEO_DIR"
echo ""

# Check videos exist
echo "Checking videos..."
ls -la "$VIDEO_DIR" | head -10
echo ""

# Extract training frames
echo "Extracting training frames..."
python src/extract_frames_smart.py \
    --csv "$TRAIN_CSV" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_TRAIN" \
    --num_frames 32 \
    --workers 32

echo ""
echo "Extracting validation frames..."
python src/extract_frames_smart.py \
    --csv "$VAL_CSV" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_VAL" \
    --num_frames 32 \
    --workers 32

echo ""
echo "Done: $(date)"
