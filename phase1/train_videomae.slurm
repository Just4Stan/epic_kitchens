#!/bin/bash -l
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=epic_videomae_sota
#SBATCH --output=training_videomae_output_%j.txt
#SBATCH --error=training_videomae_error_%j.txt

module purge
module load cluster/wice/interactive
module load Python/3.11.3-GCCcore-12.3.0

source $VSC_DATA/RDLAB/venv/bin/activate
cd $VSC_DATA/epic_kitchens

echo "======================================================================"
echo "EPIC-KITCHENS VideoMAE Training (STATE-OF-THE-ART)"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""
echo "Model: VideoMAE-base (MCG-NJU/videomae-base)"
echo "Pre-trained on: Kinetics-400"
echo "Expected performance: 60-70% verb, 55-65% noun accuracy"
echo ""

# Install required packages
echo "Installing VideoMAE dependencies..."
pip install transformers>=4.35.0 timm>=0.9.0 einops>=0.7.0 --quiet

python train_videomae.py \
    --data_dir EPIC-KITCHENS \
    --epochs 50 \
    --batch_size 16 \
    --lr 5e-5 \
    --num_frames 16 \
    --device cuda \
    --num_workers 8

echo ""
echo "======================================================================"
echo "Training complete!"
echo "End time: $(date)"
echo "======================================================================"
