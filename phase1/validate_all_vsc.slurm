#!/bin/bash
#SBATCH --job-name=validate_all
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_ciis_lab
#SBATCH --clusters=wice
#SBATCH --output=validation_all_output_%j.txt
#SBATCH --error=validation_all_error_%j.txt

# Activate environment
cd $VSC_DATA/epic_kitchens
source epic_env/bin/activate

# Create results directory
mkdir -p validation_results

echo "========================================"
echo "Validating All Models"
echo "========================================"
date
echo ""

# Validate Baseline models
echo "[1/6] Validating Baseline (epoch 5)..."
python validate_vsc.py \
    --checkpoint outputs/checkpoints/checkpoint_epoch_5.pth \
    --batch_size 64 \
    --output validation_results/baseline_epoch5.json

echo "[2/6] Validating Baseline (epoch 10)..."
python validate_vsc.py \
    --checkpoint outputs/checkpoints/checkpoint_epoch_10.pth \
    --batch_size 64 \
    --output validation_results/baseline_epoch10.json

echo "[3/6] Validating Baseline (epoch 15)..."
python validate_vsc.py \
    --checkpoint outputs/checkpoints/checkpoint_epoch_15.pth \
    --batch_size 64 \
    --output validation_results/baseline_epoch15.json

# Validate LSTM models
echo "[4/6] Validating LSTM (epoch 5)..."
python validate_vsc.py \
    --checkpoint outputs_lstm/checkpoints/checkpoint_epoch_5.pth \
    --batch_size 64 \
    --output validation_results/lstm_epoch5.json

echo "[5/6] Validating LSTM (epoch 10)..."
python validate_vsc.py \
    --checkpoint outputs_lstm/checkpoints/checkpoint_epoch_10.pth \
    --batch_size 64 \
    --output validation_results/lstm_epoch10.json

# Validate Transformer model
echo "[6/6] Validating Transformer (epoch 15)..."
python validate_vsc.py \
    --checkpoint outputs_transformer/checkpoints/checkpoint_epoch_15.pth \
    --batch_size 64 \
    --output validation_results/transformer_epoch15.json

echo ""
echo "========================================"
echo "Validation Complete!"
echo "========================================"
date

# Create summary
python -c "
import json
from pathlib import Path

results_dir = Path('validation_results')
results = []

for json_file in sorted(results_dir.glob('*.json')):
    with open(json_file) as f:
        data = json.load(f)
        results.append(data)

print('\\n' + '='*80)
print('VALIDATION RESULTS SUMMARY')
print('='*80)
print(f'{'Model':<30} {'Epoch':<8} {'Verb Top-1':<12} {'Noun Top-1':<12} {'Action':<10}')
print('-'*80)

for r in results:
    model_name = Path(r['checkpoint']).stem
    print(f'{model_name:<30} {r['epoch']:<8} {r['verb_acc_top1']:<12.2f} {r['noun_acc_top1']:<12.2f} {r['action_acc']:<10.2f}')

print('='*80)
"
