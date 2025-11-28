# EPIC-KITCHENS-100 Action Recognition Research Brief

## Executive Summary

This document provides complete context for continuing research on egocentric action recognition for the EPIC-KITCHENS-100 challenge. The goal is to improve model accuracy using state-of-the-art techniques while working within available compute constraints.

---

## 1. Dataset: EPIC-KITCHENS-100

### Overview
- **Type**: Egocentric (first-person) video dataset of kitchen activities
- **Scale**: 100 hours of video, 90K action segments
- **Participants**: 45 unique participants in their own kitchens
- **Challenge**: Multi-label classification (verb + noun prediction)

### Class Distribution
| Component | Classes | Examples |
|-----------|---------|----------|
| Verbs | 97 | take, put, open, close, wash, cut, mix, pour |
| Nouns | 300 | plate, knife, pan, water, onion, cupboard, fridge |
| Actions | ~4000 unique | "take plate", "cut onion", "open fridge" |

### Data Splits
| Split | Segments | Purpose |
|-------|----------|---------|
| Train | 67,217 | Model training |
| Validation | 9,668 | Local evaluation & hyperparameter tuning |
| Test | 9,449 | CodaBench submission (no labels) |

### Key Challenges
1. **Long-tail distribution**: Many rare verb-noun combinations
2. **Fine-grained actions**: Subtle differences between similar actions
3. **Egocentric viewpoint**: Hands occlude objects, camera motion
4. **Temporal variation**: Actions range from <1s to >30s
5. **Multi-label**: Must predict both verb AND noun correctly for "action accuracy"

### Video Properties
- **Resolution**: 1920x1080 (original), 640x360 (reduced)
- **Frame rate**: 60 FPS (original), 50 FPS (some videos)
- **Format**: MP4 (H.264)

---

## 2. Competition: CodaBench Challenge

### Evaluation Metrics
```
Primary: Action Accuracy = (verb correct AND noun correct) / total
Secondary:
  - Top-1 Verb Accuracy
  - Top-1 Noun Accuracy
  - Top-5 Recall (verb, noun, action)
```

### Submission Format
```python
# submission.pt - list of dicts, one per test segment
[
    {
        'narration_id': 'P01_101_0',      # Unique segment ID
        'verb_output': tensor[97],         # Probabilities or logits
        'noun_output': tensor[300]         # Probabilities or logits
    },
    ...
]
# Must contain exactly 9,449 predictions for test set
```

### Leaderboard Context (approximate)
| Rank | Method | Action Acc | Verb Acc | Noun Acc |
|------|--------|------------|----------|----------|
| 1st | Video-Swin + SlowFast ensemble | ~48% | ~70% | ~58% |
| Top 10 | Various transformer methods | ~42-45% | ~65% | ~55% |
| Baseline | TSN (ImageNet) | ~25% | ~55% | ~40% |

---

## 3. Current Implementation

### Model Architecture
```python
class ActionModel(nn.Module):
    """ResNet50 backbone + Bidirectional LSTM temporal model"""

    def __init__(self):
        # Backbone: ResNet50 pretrained on ImageNet
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_dim = 2048
        self.backbone.fc = nn.Identity()  # Remove classifier

        # Temporal: 2-layer Bidirectional LSTM
        self.temporal = nn.LSTM(
            input_size=2048,
            hidden_size=1024,  # 2048 total with bidirectional
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Heads: Independent verb and noun classifiers
        self.verb_head = nn.Linear(2048, 97)
        self.noun_head = nn.Linear(2048, 300)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # Extract per-frame features
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)  # (B*T, 2048)
        features = features.view(B, T, -1)  # (B, T, 2048)

        # Temporal modeling
        temporal_out, _ = self.temporal(features)  # (B, T, 2048)
        pooled = temporal_out.mean(dim=1)  # (B, 2048)

        # Classification
        return self.verb_head(pooled), self.noun_head(pooled)
```

### Training Configuration
```python
# Current best configuration
NUM_FRAMES = 16  # or 32 for longer temporal context
IMAGE_SIZE = 224
BATCH_SIZE = 16  # Per GPU
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30
OPTIMIZER = AdamW
SCHEDULER = CosineAnnealingLR

# Data augmentation
- RandomResizedCrop(224, scale=(0.8, 1.0))
- RandomHorizontalFlip(p=0.5)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
- Normalize(ImageNet mean/std)
```

### Current Results (Local Validation)

| Model | Frames | Verb Acc | Noun Acc | Action Acc |
|-------|--------|----------|----------|------------|
| ResNet50 + LSTM | 16 | 52.3% | 42.1% | 31.8% |
| ResNet50 + LSTM | 32 | 54.1% | 43.5% | 33.2% |
| Ensemble (16+32) | - | 55.2% | 44.8% | 35.1% |

**Note**: These are on a subset of validation (1,347/9,668 segments due to missing videos locally). Full validation requires all 138 unique videos.

### Checkpoint Locations
```
outputs/full_a100_v3/checkpoints/best_model.pth    # 16-frame model
outputs/full_32frames_v1/checkpoints/best_model.pth # 32-frame model
```

---

## 4. Compute Resources

### VSC (Vlaams Supercomputer Centrum)

#### Available Hardware
| Resource | Specs |
|----------|-------|
| GPU | NVIDIA A100 40GB |
| CPU | AMD EPYC 7H12 (64 cores) |
| RAM | 256 GB per node |
| Storage | 500 GB scratch per user |

#### Job Configuration (SLURM)
```bash
#SBATCH --cluster=wice
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
```

#### Practical Constraints
- **Max job time**: 72 hours (typical jobs: 12-24h)
- **GPU memory**: 40 GB allows batch_size=16 with 32 frames
- **Storage**: Pre-extracted frames take ~200GB, use videos directly
- **Queue time**: Can be 1-4 hours during peak times

### Local Machine (Mac)
- Apple Silicon (M-series) with MPS acceleration
- Useful for: Quick experiments, inference, debugging
- Not practical for: Full training (too slow)

---

## 5. What Has Been Tried

### Successful Approaches
1. **Temporal modeling**: LSTM significantly better than mean pooling (+8% action acc)
2. **More frames**: 32 frames better than 16 frames (+1.5% action acc)
3. **Ensemble**: Averaging 16+32 frame models improves results (+2%)
4. **Data augmentation**: Standard augmentation helps generalization

### Approaches That Didn't Work Well
1. **Transformer temporal model**: Similar to LSTM, higher compute cost
2. **3D CNNs (R3D, R(2+1)D)**: Slower training, no accuracy improvement
3. **Very high dropout (>0.5)**: Hurts performance
4. **Learning rate > 1e-3**: Training instability

### Not Yet Explored
- [ ] Video transformers (TimeSformer, Video Swin)
- [ ] SlowFast networks
- [ ] Pre-training on Kinetics-400/700
- [ ] VideoMAE / masked video modeling
- [ ] Multi-scale temporal modeling
- [ ] Loss function improvements (focal loss, class balancing)
- [ ] Test-time augmentation
- [ ] Knowledge distillation
- [ ] Object detection integration (hand-object interaction)

---

## 6. State-of-the-Art Techniques to Explore

### 6.1 Video Transformers

#### TimeSformer
```
Paper: "Is Space-Time Attention All You Need for Video Understanding?"
Key idea: Divided space-time attention for efficiency
Expected improvement: +5-10% over CNN+LSTM baseline
Compute: ~2x current training time
```

#### Video Swin Transformer
```
Paper: "Video Swin Transformer"
Key idea: Shifted window attention for videos
Expected improvement: +8-12% over baseline
Compute: ~3x current training time, needs larger batch
Implementation: Available in timm/mmaction2
```

### 6.2 SlowFast Networks
```
Paper: "SlowFast Networks for Video Recognition"
Key idea: Two pathways - slow (semantic) + fast (motion)
Slow pathway: 4 frames, high channel capacity
Fast pathway: 32 frames, lightweight
Expected improvement: +5-8%
Advantage: Efficient, well-suited for action recognition
```

### 6.3 Pre-training Strategies

#### Kinetics Pre-training
```
Approach: Initialize from Kinetics-400/700 pretrained model
Available: torchvision, pytorchvideo, mmaction2
Expected improvement: +3-5%
Easy to implement, highly recommended
```

#### VideoMAE Pre-training
```
Paper: "VideoMAE: Masked Autoencoders are Data-Efficient Learners"
Key idea: Mask 90% of video patches, reconstruct
Expected improvement: +5-8%
Compute: Expensive pre-training, but weights available
```

### 6.4 Loss Function Improvements

#### Focal Loss for Long-tail
```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
```

#### Class-Balanced Loss
```python
# Weight rare classes higher
class_counts = train_df['verb_class'].value_counts()
class_weights = 1.0 / (class_counts ** 0.5)  # Square root scaling
class_weights = class_weights / class_weights.sum() * len(class_weights)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### 6.5 Multi-Scale Temporal Modeling
```python
class MultiScaleTemporalModel(nn.Module):
    def __init__(self, feature_dim):
        self.short_term = nn.LSTM(feature_dim, feature_dim//2, bidirectional=True)
        self.medium_term = nn.LSTM(feature_dim, feature_dim//2, bidirectional=True)
        self.long_term = nn.LSTM(feature_dim, feature_dim//2, bidirectional=True)
        self.fusion = nn.Linear(feature_dim * 3, feature_dim)

    def forward(self, x):
        # x: (B, T, D)
        # Different temporal scales via pooling
        short = self.short_term(x)[0].mean(1)
        medium = self.medium_term(x[:, ::2, :])[0].mean(1)  # Every 2nd frame
        long = self.long_term(x[:, ::4, :])[0].mean(1)      # Every 4th frame
        return self.fusion(torch.cat([short, medium, long], dim=1))
```

### 6.6 Test-Time Augmentation
```python
def tta_predict(model, video, num_crops=3):
    """Multi-crop TTA for inference"""
    predictions = []

    # Center crop
    predictions.append(model(center_crop(video)))

    # Multiple spatial crops
    for crop in [left_crop, right_crop]:
        predictions.append(model(crop(video)))

    # Temporal variations
    for offset in [0, 2, 4]:
        predictions.append(model(video[:, offset::1, ...]))

    # Average predictions
    verb_avg = torch.stack([p[0] for p in predictions]).mean(0)
    noun_avg = torch.stack([p[1] for p in predictions]).mean(0)
    return verb_avg, noun_avg
```

---

## 7. Recommended Research Priorities

### Priority 1: Quick Wins (1-2 days each)
1. **Kinetics pre-training**: Use torchvision's video_resnet50 pretrained on Kinetics
2. **Class-balanced loss**: Implement focal loss or class weighting
3. **More aggressive augmentation**: MixUp, CutMix for videos
4. **Learning rate warmup**: 5 epochs warmup before cosine decay

### Priority 2: Medium Effort (3-5 days each)
1. **SlowFast network**: Implement dual-pathway architecture
2. **Larger ensemble**: Train 5 models with different seeds, average
3. **Multi-scale temporal**: Combine short/medium/long term features
4. **Better backbone**: EfficientNet-B4 or ConvNeXt instead of ResNet50

### Priority 3: High Effort, High Reward (1-2 weeks)
1. **Video Swin Transformer**: Best single-model performance
2. **VideoMAE fine-tuning**: Use pretrained weights from HuggingFace
3. **Hand-object detection**: Add object detector branch for nouns

---

## 8. Code Structure

```
epic_kitchens/
├── EPIC-KITCHENS/                    # Dataset
│   ├── epic-kitchens-100-annotations-master/
│   │   ├── EPIC_100_train.csv
│   │   ├── EPIC_100_validation.csv
│   │   ├── EPIC_100_verb_classes.csv
│   │   └── EPIC_100_noun_classes.csv
│   └── videos_640x360/               # Video files by participant
│       ├── P01/
│       ├── P02/
│       └── ...
├── outputs/                          # Training outputs
│   ├── full_a100_v3/                # 16-frame model
│   ├── full_32frames_v1/            # 32-frame model
│   └── submissions/                  # CodaBench submissions
├── src/                              # Main source code
│   ├── models.py                     # Model architectures
│   ├── dataset.py                    # Data loading
│   ├── train.py                      # Training script
│   └── generate_submission.py        # Submission generation
├── validation/                       # Validation scripts
│   ├── validate_checkpoint.py
│   └── generate_ensemble_submission.py
└── inference/                        # Inference/demo scripts
    └── webcam_full_model.py
```

---

## 9. Key Papers to Reference

1. **EPIC-KITCHENS Dataset**: Damen et al., "Scaling Egocentric Vision", IJCV 2022
2. **SlowFast**: Feichtenhofer et al., "SlowFast Networks", ICCV 2019
3. **TimeSformer**: Bertasius et al., "Is Space-Time Attention All You Need?", ICML 2021
4. **Video Swin**: Liu et al., "Video Swin Transformer", CVPR 2022
5. **VideoMAE**: Tong et al., "VideoMAE: Masked Autoencoders", NeurIPS 2022
6. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

---

## 10. Success Criteria

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| Action Accuracy | 35% | 42% | 48% |
| Verb Accuracy | 55% | 62% | 68% |
| Noun Accuracy | 45% | 52% | 58% |

**Key insight**: Noun accuracy is the bottleneck. Focus on:
- Better object recognition (noun classes)
- Hand-object interaction features
- Longer temporal context for object state changes

---

## 11. Quick Start Commands

### Training on VSC
```bash
# SSH to VSC
ssh vsc

# Navigate to project
cd /data/leuven/380/vsc38064/epic_kitchens

# Submit training job
sbatch train.slurm

# Monitor job
squeue -u $USER
tail -f slurm-*.out
```

### Local Validation
```bash
# Validate checkpoint
python validation/validate_checkpoint.py \
    --checkpoint outputs/full_a100_v3/checkpoints/best_model.pth

# Generate submission
python validation/generate_ensemble_submission.py
```

### Generate Submission
```bash
# Creates submission.pt and ensemble_submission.zip
python validation/generate_ensemble_submission.py

# Upload ensemble_submission.zip to CodaBench
```

---

## 12. Contact & Resources

- **CodaBench Challenge**: https://codalab.lisn.upsaclay.fr/competitions/[epic-kitchens-100]
- **Dataset Download**: https://epic-kitchens.github.io/
- **VSC Documentation**: https://docs.vscentrum.be/
- **PyTorchVideo**: https://pytorchvideo.org/
- **MMAction2**: https://github.com/open-mmlab/mmaction2

---

*Last updated: November 2024*
*Current best: 35.1% action accuracy (ensemble of 16+32 frame models)*
