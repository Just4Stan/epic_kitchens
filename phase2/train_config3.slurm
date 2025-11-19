#!/bin/bash
#SBATCH --job-name=model3_moredrop
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=training_model3_output_%j.txt
#SBATCH --error=training_model3_error_%j.txt

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd $VSC_DATA/epic_kitchens
source epic_env/bin/activate

echo "========================================"
echo "Model 3: Improved Transformer (lr=5e-5, dropout=0.6)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
nvidia-smi

# Modify model creation to use dropout=0.6
python -c "
import sys
sys.path.insert(0, '.')

# Monkey-patch the model creation
import model_improved
original_get = model_improved.get_improved_model

def get_model_dropout06(config):
    from model_improved import ImprovedActionRecognitionModel
    return ImprovedActionRecognitionModel(
        num_verb_classes=config.NUM_VERB_CLASSES,
        num_noun_classes=config.NUM_NOUN_CLASSES,
        num_frames=config.NUM_FRAMES,
        dropout=0.6,  # Modified
        drop_path=0.1
    )

model_improved.get_improved_model = get_model_dropout06

# Now run training
exec(open('train_improved.py').read())
" --epochs 30 --batch_size 24 --lr 5e-5 --output_dir outputs_model3

echo "End: $(date)"
