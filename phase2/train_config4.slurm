#!/bin/bash
#SBATCH --job-name=model4_lessdrop
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --output=training_model4_output_%j.txt
#SBATCH --error=training_model4_error_%j.txt

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd $VSC_DATA/epic_kitchens
source epic_env/bin/activate

echo "========================================"
echo "Model 4: Improved Transformer (lr=5e-5, dropout=0.4)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
nvidia-smi

# Create modified training script with dropout=0.4
cat > train_model4.py << 'EOPYTHON'
import sys
import argparse
from config import Config
from dataset_improved import get_improved_dataloaders
from model_improved import ImprovedActionRecognitionModel
import torch

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--output_dir', type=str, default='outputs_model4')
args = parser.parse_args()

# Create config
config = Config()
config.EPOCHS = args.epochs
config.BATCH_SIZE = args.batch_size
config.LEARNING_RATE = args.lr

# Create model with dropout=0.4
model = ImprovedActionRecognitionModel(
    num_verb_classes=config.NUM_VERB_CLASSES,
    num_noun_classes=config.NUM_NOUN_CLASSES,
    num_frames=config.NUM_FRAMES,
    dropout=0.4,
    drop_path=0.1
)

# Import and run training logic from train_improved
from train_improved import train_epoch, validate, LabelSmoothingCrossEntropy, EarlyStopping
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler
from pathlib import Path
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

train_loader, val_loader = get_improved_dataloaders(config)

criterion_verb = LabelSmoothingCrossEntropy(smoothing=0.1)
criterion_noun = LabelSmoothingCrossEntropy(smoothing=0.1)

optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
scaler = GradScaler()
early_stopping = EarlyStopping(patience=7)

checkpoint_dir = Path(args.output_dir) / "checkpoints"
log_dir = Path(args.output_dir) / "logs"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

history = {'train_loss': [], 'train_verb_acc': [], 'train_noun_acc': [],
           'val_loss': [], 'val_verb_acc': [], 'val_noun_acc': [], 'val_action_acc': []}
best_val_acc = 0.0

print(f"Training with dropout=0.4")

for epoch in range(config.EPOCHS):
    train_loss, train_verb_acc, train_noun_acc = train_epoch(
        model, train_loader, optimizer, criterion_verb, criterion_noun,
        device, scaler, epoch, config
    )

    val_metrics = validate(model, val_loader, criterion_verb, criterion_noun, device)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['train_verb_acc'].append(train_verb_acc)
    history['train_noun_acc'].append(train_noun_acc)
    history['val_loss'].append(val_metrics['loss'])
    history['val_verb_acc'].append(val_metrics['verb_acc_top1'])
    history['val_noun_acc'].append(val_metrics['noun_acc_top1'])
    history['val_action_acc'].append(val_metrics['action_acc'])

    print(f"Epoch {epoch+1}: Val Verb={val_metrics['verb_acc_top1']:.2f}%, Noun={val_metrics['noun_acc_top1']:.2f}%")

    if (epoch + 1) % 5 == 0:
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                   'val_metrics': val_metrics}, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth")

    val_acc_avg = (val_metrics['verb_acc_top1'] + val_metrics['noun_acc_top1']) / 2
    if val_acc_avg > best_val_acc:
        best_val_acc = val_acc_avg
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                   'val_metrics': val_metrics}, checkpoint_dir / "best_model.pth")

    early_stopping(val_metrics['loss'])
    if early_stopping.early_stop:
        break

with open(log_dir / "training_history.json", 'w') as f:
    json.dump(history, f, indent=2)

print(f"Best validation: {best_val_acc:.2f}%")
EOPYTHON

python train_model4.py --epochs 30 --batch_size 24 --lr 5e-5 --output_dir outputs_model4

echo "End: $(date)"
