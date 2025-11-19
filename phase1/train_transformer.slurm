#!/bin/bash -l
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=epic_transformer
#SBATCH --output=training_transformer_output_%j.txt
#SBATCH --error=training_transformer_error_%j.txt

module purge
module load cluster/wice/interactive
module load Python/3.11.3-GCCcore-12.3.0

source $VSC_DATA/RDLAB/venv/bin/activate
cd $VSC_DATA/epic_kitchens

echo "======================================================================"
echo "EPIC-KITCHENS Transformer Model Training"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

python train_transformer.py \
    --data_dir EPIC-KITCHENS \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --num_frames 8 \
    --device cuda \
    --num_workers 8

echo ""
echo "======================================================================"
echo "Training complete!"
echo "End time: $(date)"
echo "======================================================================"
