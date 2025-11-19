# EPIC-KITCHENS Action Recognition: Multi-Phase Deep Learning Approach

**15-Minute Research Presentation**

**Date**: November 19, 2025

**Repository**: https://github.com/YOUR_USERNAME/epic-kitchens-action-recognition

---

## Slide 1: Title Slide (30 seconds)

### EPIC-KITCHENS Action Recognition
**A Three-Phase Approach to Egocentric Video Understanding**

- Phase 1: Architecture Comparison
- Phase 2: Hyperparameter Optimization
- Phase 3: Cross-Task Attention

**Dataset**: EPIC-KITCHENS-100 [1]
- 67,217 training segments
- 9,668 validation segments
- 97 verb classes, 300 noun classes

---

## Slide 2: Problem Statement (1 minute)

### Egocentric Action Recognition Challenge

**Multi-task Learning Problem**:
- Predict VERB (what action?) + NOUN (which object?)
- Example: "open" + "refrigerator" → Action: "open refrigerator"

**Key Challenges**:
1. **Temporal Understanding**: Actions unfold over time (8-16 frames)
2. **Multi-task Prediction**: Joint verb-noun classification
3. **Severe Overfitting**: Models memorize training data
4. **Class Imbalance**: 97 verbs × 300 nouns = 29,100 possible actions

**Related Work**:
- TSN (Temporal Segment Networks) [2]
- SlowFast Networks [3]
- VideoMAE [4]
- Cross-task attention mechanisms [5]

---

## Slide 3: Dataset Overview (1 minute)

### EPIC-KITCHENS-100 Dataset

**Scale & Diversity**:
- 100 hours of egocentric kitchen activities
- 45 different kitchens
- First-person perspective videos
- Realistic, unscripted actions

**Data Statistics**:
```
Training:    67,217 action segments
Validation:   9,668 action segments
Verb Classes: 97 (e.g., open, close, put, take)
Noun Classes: 300 (e.g., door, tap, knife, plate)
```

**Evaluation Metrics**:
- Top-1 Accuracy (verb, noun, action)
- Top-5 Accuracy
- Action Accuracy = Both verb AND noun correct

**Reference**: Damen et al., "Rescaling Egocentric Vision" (IJCV 2022) [1]

---

## Slide 4: Phase 1 - Architecture Comparison (2 minutes)

### Research Question: Which Architecture Works Best?

**6 Models Implemented & Trained**:

| Model | Architecture | Params | Training |
|-------|-------------|--------|----------|
| Baseline | ResNet-50 + Temporal Pool | 25M | 25 epochs |
| LSTM | ResNet-50 + 2-layer LSTM | 27M | 25 epochs |
| Transformer | ResNet-50 + 6-layer Transformer | 30M | 35 epochs |
| 3D CNN | R(2+1)D [6] | 28M | 20 epochs |
| EfficientNet-LSTM | EfficientNet-B3 + LSTM | 22M | 20 epochs |
| EfficientNet-Trans | EfficientNet-B3 + Transformer | 24M | 20 epochs |

**All models trained on VSC cluster with:**
- NVIDIA GPUs (CUDA 12.1)
- Mixed precision (FP16)
- Batch size: 16-32
- Learning rate: 1e-4

**References**:
- ResNet [7], EfficientNet [8], R(2+1)D [6], Transformers [9]

---

## Slide 5: Phase 1 Results (2 minutes)

### Key Finding: Severe Overfitting

**Performance Comparison**:

| Model | Train Acc | Val Verb | Val Noun | Overfitting Gap |
|-------|-----------|----------|----------|-----------------|
| Baseline | ~45% | ~30% | ~25% | ~15% |
| LSTM | ~48% | ~32% | ~26% | ~16% |
| Transformer | ~50% | ~35% | ~28% | **~15%** |
| 3D CNN | ~47% | ~33% | ~27% | ~14% |

**Observations**:
1. All models overfit significantly
2. Transformer achieved best validation accuracy
3. Gap between train/val is problematic
4. Models memorize training data

**Analysis**:
- Large parameter count (25-30M)
- Limited training data relative to model capacity
- Need for regularization strategies

---

## Slide 6: Phase 2 - Addressing Overfitting (2 minutes)

### Systematic Regularization Approach

**Strategy**: Comprehensive regularization + data augmentation

**Model Improvements**:
1. **Frozen Backbone**: Freeze first 2 ResNet blocks
   - Reduces trainable parameters
   - Prevents low-level feature overfitting

2. **Dropout**: 0.3-0.6 throughout network
   - Random neuron deactivation during training

3. **Drop Path** (Stochastic Depth): 0.05-0.15
   - Randomly skip layers during training [10]

4. **Label Smoothing**: 0.1-0.2
   - Prevents overconfident predictions [11]

**Training Improvements**:
- Lower learning rate (5e-5)
- Cosine annealing schedule
- Gradient clipping (max_norm=1.0)
- AdamW optimizer with weight decay

---

## Slide 7: Phase 2 - Data Augmentation (1.5 minutes)

### Aggressive Augmentation Pipeline

**Spatial Augmentations**:
```python
RandomHorizontalFlip(p=0.5)
RandomRotation(degrees=15)
ColorJitter(brightness=0.4, contrast=0.4,
            saturation=0.4, hue=0.1)
RandomGrayscale(p=0.1)
RandomErasing(p=0.15)  # Cutout
```

**Rationale**:
- Horizontal flip: Hands can be on either side
- Rotation: Camera movement in egocentric videos
- Color jitter: Different lighting conditions
- Random erasing: Occlusion robustness

**Inspiration**:
- AutoAugment [12]
- RandAugment [13]
- Cutout/Random Erasing [14]

---

## Slide 8: Phase 2 - Hyperparameter Search (1.5 minutes)

### 8 Configurations Tested

**Hyperparameter Grid**:

| Config | LR | Batch | Dropout | Label Smooth | Drop Path |
|--------|-----|-------|---------|--------------|-----------|
| 1 | 5e-5 | 24 | 0.5 | 0.1 | 0.1 |
| 2 | 1e-4 | 24 | 0.4 | 0.1 | 0.1 |
| 3 | 5e-5 | 32 | 0.5 | 0.15 | 0.1 |
| 4 | 5e-5 | 24 | 0.6 | 0.1 | 0.15 |
| ... | ... | ... | ... | ... | ... |

**Current Status**:
- 3 out of 8 configurations completed
- Training on VSC cluster (30 epochs each)
- 5 configurations queued

**Best Model So Far**: Config 1
- Balanced regularization
- Conservative learning rate
- Stable training dynamics

---

## Slide 9: Phase 2 Results (2 minutes)

### SUCCESS: Overfitting Dramatically Reduced

**Comparison: Phase 1 vs Phase 2**

| Metric | Phase 1 (Best) | Phase 2 (Config 1) | Improvement |
|--------|----------------|-------------------|-------------|
| Train Verb Acc | ~50% | ~36% | Lower (good!) |
| Val Verb Acc | ~35% | ~36% | +1% |
| Val Noun Acc | ~28% | ~19% | -9% |
| **Overfitting Gap** | **~15%** | **~5%** | **-10% (66% reduction)** |

**Key Achievement**:
- Overfitting reduced from 15% → 5%
- Model generalizes much better
- Validation performance more reliable
- Trade-off: Lower noun accuracy (needs investigation)

**Analysis**:
- Regularization working as intended
- May be over-regularizing nouns
- Need to complete full hyperparameter sweep

---

## Slide 10: Phase 3 - Cross-Task Attention (1.5 minutes)

### Innovation: Modeling Verb-Noun Dependencies

**Motivation**:
- Actions have semantic structure
- Verbs and nouns are NOT independent
- Example: "open" → likely "door", "refrigerator", "drawer"
- NOT likely: "open" + "chair" or "open" + "water"

**Architecture**:
```
Input Frames
    ↓
ResNet-50 Backbone (frozen early layers)
    ↓
Multi-Scale Temporal Modeling (4, 8, 16 frame windows)
    ↓
    ├─ Verb Branch ──→ Verb Features ─┐
    └─ Noun Branch ──→ Noun Features ─┤
                                       │
                    Cross-Attention ←──┘
                    (bidirectional)
                           ↓
                    ├─ Verb Logits
                    └─ Noun Logits
```

**Reference**: Cross-modal attention [15], Co-attention [16]

---

## Slide 11: Phase 3 - Technical Details (1 minute)

### Cross-Attention Mechanism

**Bidirectional Information Flow**:

1. **Verb attends to Noun**:
   - Verb features query noun features
   - Learn: "What objects are relevant for this action?"

2. **Noun attends to Verb**:
   - Noun features query verb features
   - Learn: "What actions are performed on this object?"

**Implementation**:
```python
class CrossAttention(nn.Module):
    def forward(self, verb_feat, noun_feat):
        # Verb queries Noun
        verb_attn = MultiHeadAttention(
            query=verb_feat,
            key=noun_feat,
            value=noun_feat
        )
        # Noun queries Verb
        noun_attn = MultiHeadAttention(
            query=noun_feat,
            key=verb_feat,
            value=verb_feat
        )
        return verb_attn, noun_attn
```

**Status**: Implemented, ready to train after Phase 2 completes

---

## Slide 12: Experimental Setup (1 minute)

### Training Infrastructure

**Hardware**:
- VSC (Vlaams Supercomputer Centrum)
- NVIDIA GPUs with CUDA 12.1
- 16-32 CPU cores per job
- 24-hour time limits

**Software Stack**:
```
PyTorch 2.1.2
Python 3.11
torchvision 0.16
Mixed Precision Training (AMP)
```

**Data Pipeline**:
- Efficient video frame sampling
- On-the-fly augmentation
- 8-16 data loader workers
- Pinned memory for GPU transfer

**Reproducibility**:
- All code on GitHub (phase-based structure)
- SLURM job files included
- Configuration files documented
- Random seeds fixed

---

## Slide 13: Current Results Summary (1.5 minutes)

### Overall Progress

**Phase 1: COMPLETED**
- 6 models trained and validated
- Identified severe overfitting problem
- Transformer performed best

**Phase 2: IN PROGRESS (3/8 done)**
- Successfully reduced overfitting by 66%
- Best model validation accuracy: 27% (combined)
- 5 more configurations to evaluate

**Phase 3: READY**
- Model implemented
- Training script created
- Awaiting Phase 2 completion for hyperparameter selection

**Validation Metrics** (Best Phase 2 Model):
```
Verb Top-1:  36%
Verb Top-5:  78%
Noun Top-1:  19%
Noun Top-5:  38%
Action Acc:  10%
```

---

## Slide 14: Challenges & Insights (1 minute)

### Key Learnings

**Challenges Encountered**:

1. **Data-Model Mismatch**
   - Large models (25M+ params)
   - Limited effective training data
   - Solution: Aggressive regularization

2. **Noun Classification Harder than Verb**
   - 300 noun classes vs 97 verb classes
   - More class imbalance
   - Needs focused attention (Phase 3!)

3. **Computational Constraints**
   - Each config: ~24 hours training
   - 8 configs × 24 hours = 8 days total
   - Parallel execution on cluster

**Technical Insights**:
- Frozen backbones crucial for generalization
- Label smoothing prevents overconfidence
- Data augmentation must match domain (egocentric)

---

## Slide 15: Future Work & Conclusions (1 minute)

### Next Steps

**Immediate (Next 2 Weeks)**:
1. Complete Phase 2 hyperparameter sweep (5 configs)
2. Analyze results, select best configuration
3. Train Phase 3 cross-attention model
4. Compare all approaches

**Future Directions**:
1. **Temporal Modeling**
   - Optical flow integration
   - Two-stream networks [17]
   - Temporal segment networks

2. **Advanced Architectures**
   - Video Transformers [18]
   - SlowFast networks [3]
   - VideoMAE fine-tuning [4]

3. **Data Strategies**
   - Semi-supervised learning
   - Self-supervised pre-training
   - Knowledge distillation

---

## Slide 16: Conclusions (30 seconds)

### Summary

**Achievements**:
- Systematic comparison of 6 architectures
- 66% reduction in overfitting through regularization
- Novel cross-attention architecture for verb-noun interaction

**Key Contributions**:
1. Comprehensive baseline comparisons
2. Effective regularization strategies for egocentric video
3. Cross-task attention mechanism

**Impact**:
- Better generalization to unseen data
- More reliable validation metrics
- Foundation for future improvements

**Code & Data**: github.com/YOUR_USERNAME/epic-kitchens-action-recognition

---

## References

[1] Damen, D. et al. (2022). "Rescaling Egocentric Vision: Collection, Pipeline and Challenges for EPIC-KITCHENS-100." IJCV.

[2] Wang, L. et al. (2016). "Temporal Segment Networks: Towards Good Practices for Deep Action Recognition." ECCV.

[3] Feichtenhofer, C. et al. (2019). "SlowFast Networks for Video Recognition." ICCV.

[4] Tong, Z. et al. (2022). "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training." NeurIPS.

[5] Kazakos, E. et al. (2019). "EPIC-Fusion: Audio-Visual Temporal Binding for Egocentric Action Recognition." ICCV.

[6] Tran, D. et al. (2018). "A Closer Look at Spatiotemporal Convolutions for Action Recognition." CVPR.

[7] He, K. et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.

[8] Tan, M. & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." ICML.

[9] Vaswani, A. et al. (2017). "Attention is All You Need." NeurIPS.

[10] Huang, G. et al. (2016). "Deep Networks with Stochastic Depth." ECCV.

[11] Szegedy, C. et al. (2016). "Rethinking the Inception Architecture for Computer Vision." CVPR.

[12] Cubuk, E.D. et al. (2019). "AutoAugment: Learning Augmentation Strategies from Data." CVPR.

[13] Cubuk, E.D. et al. (2020). "RandAugment: Practical Automated Data Augmentation." CVPR Workshops.

[14] DeVries, T. & Taylor, G.W. (2017). "Improved Regularization of Convolutional Neural Networks with Cutout." arXiv.

[15] Lu, J. et al. (2016). "Hierarchical Question-Image Co-Attention for Visual Question Answering." NeurIPS.

[16] Yu, Z. et al. (2019). "Deep Modular Co-Attention Networks for Visual Question Answering." CVPR.

[17] Simonyan, K. & Zisserman, A. (2014). "Two-Stream Convolutional Networks for Action Recognition in Videos." NeurIPS.

[18] Arnab, A. et al. (2021). "ViViT: A Video Vision Transformer." ICCV.

---

## Appendix: Technical Details

### Model Architectures

**Baseline (ResNet-50 + Temporal Pool)**:
```python
ResNet50(pretrained=ImageNet)
  → Extract features per frame (2048-dim)
  → Average pool across time
  → FC(2048 → 97) for verbs
  → FC(2048 → 300) for nouns
```

**Transformer**:
```python
ResNet50(pretrained)
  → Frame features (T × 2048)
  → Positional encoding
  → 6-layer Transformer Encoder
    - d_model=512, nhead=8
    - FFN dim=2048
  → [CLS] token output
  → Verb/Noun classifiers
```

**Phase 3 Cross-Attention**:
```python
ResNet50(frozen layers 1-2)
  → Multi-scale temporal (windows: 4,8,16)
  → Task branches
    ├─ Verb branch → verb_feat
    └─ Noun branch → noun_feat
  → CrossAttention(verb_feat, noun_feat)
    - Bidirectional attention
    - 8 heads, 512 dim
  → Enhanced features
  → Final classifiers
```

---

## Backup Slides

### Detailed Phase 2 Results

**Config 1 (Best) - Epoch-by-Epoch**:
```
Epoch 1:  Train V=25.3% N=9.0%  | Val V=30.3% N=13.4%
Epoch 2:  Train V=30.8% N=15.0% | Val V=35.7% N=19.1%
Epoch 5:  Train V=35.2% N=18.3% | Val V=36.1% N=19.5%
Epoch 10: Train V=36.8% N=19.8% | Val V=36.4% N=19.2%
```

**Overfitting Gap Progression**:
- Epoch 1: -5% (model underfitting)
- Epoch 5: ~1% (ideal)
- Epoch 10: ~2% (slight overfitting)
- Early stopping prevented further overfitting

### Computing Resources

**Total GPU Hours**:
- Phase 1: 6 models × 25 epochs × 1.5 hrs = ~225 GPU hours
- Phase 2: 3 models × 30 epochs × 2 hrs = ~180 GPU hours
- Total so far: ~405 GPU hours

**Data Storage**:
- Training data: 85 GB
- Validation data: 12 GB
- Model checkpoints: ~18 GB (6 models × 3 GB)
- Excluded from Git (see .gitignore)

---

**END OF PRESENTATION**

**Questions?**
