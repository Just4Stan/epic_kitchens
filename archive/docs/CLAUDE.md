# EPIC-KITCHENS Action Recognition Project

## Quick Context
- **Task**: Action recognition on EPIC-KITCHENS-100 (97 verbs, 300 nouns)
- **Best Result**: 23.98% action accuracy (exp15: ResNet50 + LSTM)
- **VSC Path**: `/data/leuven/380/vsc38064/epic_kitchens/`

## Project Structure
```
epic_kitchens/
├── src/                    # Main code
│   ├── config.py           # Paths, hyperparams
│   ├── datasets.py         # TrainDataset, ValDataset
│   ├── models.py           # ActionModel (ResNet50+LSTM)
│   └── train.py            # Unified training script
├── jobs/                   # SLURM job scripts
├── outputs/                # Training outputs ({exp_name}/)
├── logs/                   # SLURM logs
├── results/                # Best models, analysis
├── inference/              # Webcam/video inference
├── EPIC-KITCHENS/          # Dataset
└── PROJECT.md              # Full documentation
```

## Common Commands

### VSC Connection
```bash
ssh vsc
squeue -u vsc38064 --clusters=wice          # Check jobs
sbatch jobs/exp29_baseline.slurm            # Submit job
scancel <job_id> --clusters=wice            # Cancel job
tail -f logs/exp29_*.out                    # Watch output
```

### File Transfer
```bash
scp file vsc:/data/leuven/380/vsc38064/epic_kitchens/
rsync -avz src/ vsc:/data/leuven/380/vsc38064/epic_kitchens/src/
```

### Training
```bash
# Local test
python src/train.py --exp_name test --epochs 5 --batch_size 32

# VSC submission
python src/train.py \
    --exp_name exp_name \
    --epochs 40 \
    --batch_size 64 \
    --backbone resnet50 \
    --temporal_model lstm \
    --wandb
```

## SLURM Template
```bash
#!/bin/bash
#SBATCH --job-name=exp_name
#SBATCH --time=08:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge
module load cluster/wice/batch
module load Python/3.11.3-GCCcore-12.3.0

source $VSC_DATA/epic_kitchens/epic_env/bin/activate
export WANDB_API_KEY="a050122e318cf57511f2c745aa871735df7c6de8"
cd $VSC_DATA/epic_kitchens

python src/train.py --exp_name experiment --wandb
```

## Key Findings
- **LSTM > Transformer** for temporal modeling
- **No backbone freezing** gives best results
- **Batch size 64-80** optimal for H100
- **Dropout 0.3-0.5**, label smoothing 0.1
- **Medium augmentation** (RandomResizedCrop 0.6-1.0)

## W&B
- Project: `epic-kitchens-action`
- API Key: `a050122e318cf57511f2c745aa871735df7c6de8`

Current dataset has 16 preextracted frames per action.
