# Training Guide - EPIC-KITCHENS Action Recognition

## Quick Start

### Prerequisites
```bash
# Python 3.8+
# PyTorch 2.0+
# CUDA 11.7+ (for GPU training)

# Install dependencies
pip install -r requirements.txt
```

### Project Structure

The project is organized into phases:
- `phase1/`: Architecture comparison (completed)
- `phase2/`: Hyperparameter optimization (in progress)
- `phase3/`: Cross-attention model (ready to train)

## Training on Local Machine

### Phase 1: Architecture Comparison

```bash
# Baseline model
cd phase1
python train.py

# LSTM model
python train_lstm.py --epochs 30 --batch_size 32

# Transformer model
python train_transformer.py --epochs 30

# 3D CNN
python train_3dcnn.py --batch_size 16  # Smaller batch for memory

# EfficientNet + LSTM
python train_efficientnet_lstm.py

# EfficientNet + Transformer
python train_efficientnet_transformer.py
```

### Phase 2: Hyperparameter Optimization

```bash
# Run from root directory
python train_improved.py --help

# Train with specific configuration
python train_improved.py \
  --epochs 30 \
  --batch_size 24 \
  --lr 5e-5 \
  --dropout 0.5 \
  --label_smoothing 0.1 \
  --output_dir outputs_model1
```

### Phase 3: Cross-Attention

```bash
# Run from phase3 directory or root
cd phase3
python train_cross_attention.py \
  --epochs 30 \
  --batch_size 24 \
  --lr 5e-5 \
  --dropout 0.5 \
  --output_dir outputs_cross_attention
```

## Training on VSC Cluster

### Setup Environment

```bash
# SSH to VSC
ssh vsc

# Navigate to project
cd $VSC_DATA/epic_kitchens

# Load modules
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
```

### Submit Training Jobs

#### Phase 1 Models

```bash
cd phase1

# Submit all Phase 1 jobs
sbatch train.slurm
sbatch train_lstm.slurm
sbatch train_transformer.slurm
sbatch train_3dcnn.slurm
sbatch train_efficientnet_lstm.slurm
sbatch train_efficientnet_transformer.slurm
```

#### Phase 2 Configurations

```bash
cd phase2

# Submit specific configuration
sbatch train_config1.slurm

# Submit all configurations
for i in {1..8}; do
  sbatch train_config${i}.slurm
done
```

#### Phase 3 Cross-Attention

```bash
cd phase3

# Create SLURM file first (see template below)
sbatch train_cross_attention.slurm
```

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View job output
tail -f training_model1_output_*.txt

# Check errors
tail -f training_model1_error_*.txt

# Cancel job
scancel <job_id>
```

## Validation

### Validate Single Model

```bash
# On specific checkpoint
python validate_vsc.py \
  --checkpoint outputs/checkpoints/best_model.pth \
  --val_csv EPIC_100_validation.csv \
  --val_video_dir EPIC-KITCHENS/videos_640x360
```

### Validate All Models

```bash
# Batch validation
python validate_models.py \
  --val_csv EPIC_100_validation.csv \
  --val_video_dir EPIC-KITCHENS/videos_640x360 \
  --output_file validation_results.json
```

### Validation Scripts

- `validate_vsc.py`: Single model validation on cluster
- `validate_models.py`: Batch validation of all Phase 1 models
- `validation/validate_true.py`: Validation on true validation set
- `validation/validate_checkpoint.py`: Checkpoint validation with splits

## Configuration

Edit `common/config.py` to modify:

```python
class Config:
    # Paths
    DATA_DIR = Path('EPIC-KITCHENS')
    OUTPUT_DIR = Path('outputs')

    # Model
    NUM_FRAMES = 16
    IMAGE_SIZE = 224

    # Training
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-4

    # Dataset
    NUM_VERB_CLASSES = 97
    NUM_NOUN_CLASSES = 300
```

## SLURM Template

Create custom SLURM files for new experiments:

```bash
#!/bin/bash
#SBATCH --job-name=epic_train
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --output=training_output_%j.txt
#SBATCH --error=training_error_%j.txt

# Load modules
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Run training
cd $VSC_DATA/epic_kitchens
python train_improved.py \
  --epochs 30 \
  --batch_size 24 \
  --lr 5e-5 \
  --dropout 0.5 \
  --output_dir outputs_experiment
```

## Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
--batch_size 16  # or even 8

# Reduce number of frames
# Edit config.py: NUM_FRAMES = 8

# Enable gradient accumulation
# Modify train script to accumulate over 2-4 steps
```

### Slow Training

```python
# Increase data loader workers
# Edit config.py: NUM_WORKERS = 16

# Use mixed precision (already enabled)
# Implemented via torch.cuda.amp

# Freeze more backbone layers
# Edit model: freeze_backbone_layers=3
```

### Overfitting

```python
# Increase dropout
--dropout 0.6

# Increase label smoothing
--label_smoothing 0.15

# Freeze more layers
# Edit model_improved.py: freeze_backbone_layers=3

# Reduce model capacity
# Use EfficientNet-B0 instead of ResNet-50
```

### Import Errors

All imports now use the phase-based structure:

```python
from common.config import Config
from common.dataset import get_dataloaders
from phase1.model import get_model
from phase2.model_improved import get_improved_model
from phase3.model_cross_attention import get_cross_attention_model
```

If you encounter import errors, ensure you're running from the project root:
```bash
cd /path/to/epic_kitchens
python phase1/train.py
```

## Results Tracking

### Training Logs

All training runs save:
- **Checkpoints**: `{output_dir}/checkpoints/`
- **Best Model**: `{output_dir}/checkpoints/best_model.pth`
- **History**: `{output_dir}/training_history.json`
- **Logs**: SLURM output files on cluster

### Visualize Results

```python
import json
import matplotlib.pyplot as plt

# Load history
with open('outputs/training_history.json') as f:
    history = json.load(f)

# Plot training curves
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Val')
plt.legend()
plt.show()
```

## Next Steps

1. **Phase 2 Completion**: Finish remaining hyperparameter configurations
2. **Analysis**: Compare all Phase 2 results in `validation_results.json`
3. **Phase 3**: Train cross-attention model with best hyperparameters
4. **Presentation**: Prepare results summary and visualizations

## Contact & Support

For questions or issues:
- Check `docs/ARCHITECTURE.md` for model details
- Review training logs for error messages
- Consult Phase-specific README files
