# Quick Start Guide

Get up and running in 5 minutes.

## Setup

```bash
# 1. Navigate to project
cd /path/to/RDLAB/epic_kitchens

# 2. Activate environment
source ../venv/bin/activate

# 3. Verify dataset
ls EPIC-KITCHENS/videos_640x360/  # Should show participant folders
```

**Important:** All commands below assume you're in the `epic_kitchens/` directory with venv activated.

## Train

```bash
# Baseline model (recommended)
python train.py --epochs 30 --batch_size 16

# LSTM model
python train_lstm.py --epochs 30 --batch_size 16
```

**Training time:** ~4-5 hours on A100 GPU (VSC)

## Validate

```bash
python validation/validate_true.py \
    --checkpoint outputs/checkpoints/checkpoint_epoch_10.pth \
    --batch_size 32
```

**Runtime:** ~2-3 minutes on M3 Pro

**Expected results:**
- Verb: ~39%
- Noun: ~33%
- Action: ~19%

## Real-time Demo

```bash
# List available cameras
python inference/list_cameras.py

# Run webcam inference (use iPhone camera for best results)
python inference/realtime_webcam.py \
    --checkpoint outputs/checkpoints/checkpoint_epoch_10.pth \
    --camera 1

# Press 'q' to quit
```

## Test Custom Video

```bash
python inference/test_custom_video.py \
    --video path/to/your/video.mp4 \
    --checkpoint outputs/checkpoints/checkpoint_epoch_10.pth \
    --start 0 \
    --duration 5
```

## File Structure

```
epic_kitchens/
├── train.py              # Main training script
├── config.py             # Configuration
├── model.py              # Baseline model
├── dataset.py            # Data loader
│
├── validation/           # Validation scripts
│   └── validate_true.py # True validation set
│
├── inference/            # Inference & demos
│   ├── realtime_webcam.py
│   └── test_custom_video.py
│
└── outputs/              # Training outputs
    └── checkpoints/      # Model checkpoints
```

## Common Issues

**"No module named X"**
```bash
source venv/bin/activate
pip install torch torchvision opencv-python pandas numpy tqdm
```

**"File not found"**
```bash
# Check you're in the right directory
pwd  # Should be .../RDLAB/epic_kitchens

# Check dataset exists
ls EPIC-KITCHENS/
```

**"Out of memory"**
```bash
# Reduce batch size
python train.py --batch_size 8
```

**Camera not working**
```bash
# List cameras
python inference/list_cameras.py

# Try different camera ID
python inference/realtime_webcam.py --camera 0
```

## Next Steps

- [Model Architecture Details](MODEL_PIPELINE.md)
- [Validation Results & Analysis](VALIDATION_RESULTS.md)
- [Training on VSC](../README.md#vsc-training)
