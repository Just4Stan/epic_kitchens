#!/bin/bash
#SBATCH --job-name=model2_higherlr
#SBATCH --time=40:00:00
#SBATCH --nodes=1#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=training_model2_output_%j.txt
#SBATCH --error=training_model2_error_%j.txt

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd $VSC_DATA/epic_kitchens
source epic_env/bin/activate

echo "========================================"
echo "Model 2: Improved Transformer (lr=1e-4, dropout=0.5)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
nvidia-smi

python train_improved.py \
    --epochs 30 \
    --batch_size 24 \
    --lr 1e-4 \
    --output_dir outputs_model2

echo "End: $(date)"
