# Model Architecture & Pipeline

Detailed explanation of how the EPIC-KITCHENS action recognition model works.

## Overview

**Task:** Multi-task video classification - predict verb AND noun from egocentric video clips.

**Input:** 8 RGB frames from a video segment (shape: `[8, 3, 224, 224]`)
**Output:**
- Verb logits (`[97]`) - scores for 97 verb classes
- Noun logits (`[300]`) - scores for 300 noun classes

## Architecture

### Baseline Model

```
Video Segment (8 frames, 224x224x3)
    ↓
┌───────────────────────────────────┐
│  ResNet-50 Feature Extractor      │  Pre-trained on ImageNet
│  (applied to each frame)          │
│                                   │
│  Input:  1 frame [3, 224, 224]   │
│  Output: 2048 features            │
└───────────────────────────────────┘
    ↓
[2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048]  (8 frame features)
    ↓
┌───────────────────────────────────┐
│  Temporal Average Pooling          │
│  Average across 8 frames           │
└───────────────────────────────────┘
    ↓
Single video representation [2048]
    ↓
┌─────────────────┬─────────────────┐
│  Verb Head      │  Noun Head      │
│  FC(2048 → 97)  │  FC(2048 → 300) │
└─────────────────┴─────────────────┘
    ↓              ↓
Verb logits [97]  Noun logits [300]
```

**Parameters:** ~25.8M total
- ResNet-50 backbone: ~23.5M (frozen layers: 0M, trainable: 23.5M)
- Verb classifier: ~200K
- Noun classifier: ~614K

### LSTM Model

```
Video Segment (8 frames)
    ↓
ResNet-50 Feature Extractor (same as baseline)
    ↓
Frame features: [8, 2048]
    ↓
┌───────────────────────────────────┐
│  Bi-directional LSTM              │
│  Hidden size: 512                 │
│  Input:  [8, 2048]                │
│  Output: [8, 1024]  (512×2)       │
└───────────────────────────────────┘
    ↓
Take last hidden state: [1024]
    ↓
┌─────────────────┬─────────────────┐
│  Verb Head      │  Noun Head      │
│  FC(1024 → 97)  │  FC(1024 → 300) │
└─────────────────┴─────────────────┘
```

**Parameters:** ~25.8M total
- ResNet-50: ~23.5M
- LSTM: ~8.4M (2048 → 512, bidirectional)
- Classifiers: ~400K

**Advantage:** Captures temporal dynamics and motion patterns between frames.

### Transformer Model

```
Video Segment (8 frames)
    ↓
ResNet-50 Feature Extractor
    ↓
Frame features: [8, 2048]
    ↓
┌───────────────────────────────────┐
│  Add Positional Encoding          │
│  [8, 2048] + position embeddings  │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│  Multi-Head Self-Attention        │
│  Num heads: 8                     │
│  Num layers: 2                    │
└───────────────────────────────────┘
    ↓
[CLS] token representation: [2048]
    ↓
┌─────────────────┬─────────────────┐
│  Verb Head      │  Noun Head      │
└─────────────────┴─────────────────┘
```

**Parameters:** ~30M total

**Advantage:** Attends to relevant frames for each task.

## Data Pipeline

### 1. Dataset Loading ([dataset.py](../dataset.py))

```python
class EPICKitchensDataset:
    def __getitem__(self, idx):
        # 1. Read annotation
        row = self.annotations.iloc[idx]
        video_path = f"{video_dir}/{row['participant_id']}/{row['video_id']}.MP4"
        start_frame = row['start_frame']
        stop_frame = row['stop_frame']

        # 2. Extract frames uniformly
        frame_indices = linspace(start_frame, stop_frame, num_frames=8)
        frames = []
        for frame_idx in frame_indices:
            cap.set(CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            frames.append(frame)

        # 3. Preprocess each frame
        for frame in frames:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = normalize(frame)  # ImageNet mean/std

        return frames, verb_label, noun_label
```

**Key decisions:**
- **Uniform sampling:** Sample 8 frames evenly across the segment (not consecutive frames)
- **Why 8 frames?** Balance between temporal coverage and memory usage
- **ImageNet normalization:** Pre-trained ResNet expects this format

### 2. Data Augmentation

**Training augmentations:**
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),     # Mirror flip
    transforms.ColorJitter(brightness=0.2,       # Lighting variation
                          contrast=0.2,
                          saturation=0.2),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet stats
                        [0.229, 0.224, 0.225])
])
```

**Validation:** Only resize + normalize (no augmentation)

### 3. Batching

```python
DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
```

**Batch shape:** `[16, 8, 3, 224, 224]`
- 16 video segments
- 8 frames each
- 3 color channels
- 224×224 pixels

## Training

### Loss Function

**Multi-task loss:**
```python
loss = λ_verb * CrossEntropy(verb_logits, verb_labels) +
       λ_noun * CrossEntropy(noun_logits, noun_labels)
```

**Weights:** `λ_verb = 0.5`, `λ_noun = 0.5` (equal weighting)

**Why separate losses?**
- Verbs and nouns are independent tasks
- Different class counts (97 vs 300)
- Allows task-specific tuning

### Optimizer

```python
AdamW(
    params=model.parameters(),
    lr=1e-4,
    weight_decay=1e-5,
    betas=(0.9, 0.999)
)
```

**Learning rate schedule:** Cosine annealing with warmup
```python
CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,      # Restart every 10 epochs
    T_mult=2,    # Double period after restart
    eta_min=1e-6 # Minimum LR
)
```

### Training Loop

```python
for epoch in range(epochs):
    for frames, verb_labels, noun_labels in train_loader:
        # Forward pass
        verb_logits, noun_logits = model(frames)

        # Compute losses
        loss_verb = criterion_verb(verb_logits, verb_labels)
        loss_noun = criterion_noun(noun_logits, noun_labels)
        loss = 0.5 * loss_verb + 0.5 * loss_noun

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        verb_acc = (verb_logits.argmax(1) == verb_labels).float().mean()
        noun_acc = (noun_logits.argmax(1) == noun_labels).float().mean()

    # Validation
    validate(model, val_loader)
    save_checkpoint(model, optimizer, epoch)
```

## Inference

### Single Video Prediction

```python
# 1. Load and preprocess video
frames = extract_frames(video_path, num_frames=8)
frames = preprocess(frames)  # Shape: [8, 3, 224, 224]

# 2. Add batch dimension
frames = frames.unsqueeze(0)  # Shape: [1, 8, 3, 224, 224]

# 3. Forward pass
model.eval()
with torch.no_grad():
    verb_logits, noun_logits = model(frames)

# 4. Get predictions
verb_probs = softmax(verb_logits, dim=1)
noun_probs = softmax(noun_logits, dim=1)

verb_pred = verb_probs.argmax()  # Most likely verb
noun_pred = noun_probs.argmax()  # Most likely noun

# 5. Map to class names
action = f"{verb_map[verb_pred]} {noun_map[noun_pred]}"
```

### Real-time Webcam

```python
frame_buffer = deque(maxlen=8)  # Rolling buffer of last 8 frames

while True:
    # Capture frame
    ret, frame = cap.read()

    # Preprocess and add to buffer
    processed = preprocess(frame)
    frame_buffer.append(processed)

    # Predict every N frames
    if len(frame_buffer) == 8 and frame_count % predict_every == 0:
        frames = torch.stack(list(frame_buffer))
        verb_logits, noun_logits = model(frames.unsqueeze(0))

        # Display prediction
        action = get_action(verb_logits, noun_logits)
        draw_text(frame, action)

    cv2.imshow('Action Recognition', frame)
```

## Key Design Decisions

### Why ResNet-50?
- Pre-trained on ImageNet (good visual features)
- Efficient (25M params vs 138M for ResNet-152)
- Standard baseline for action recognition

### Why temporal average pooling?
- Simple and effective baseline
- No additional parameters
- Proven to work well for action recognition

### Why 8 frames?
- Covers ~1 second of video (EPIC-KITCHENS segments are 1-10 seconds)
- Fits in GPU memory with batch size 16
- Captures temporal context without excessive computation

### Why separate verb/noun heads?
- Tasks are independent (cutting can happen to many objects)
- Different class counts (97 vs 300)
- Allows measuring performance on each task separately

## Performance Bottlenecks

**Training:**
- Video loading: ~40% of time (I/O bound)
- ResNet forward pass: ~50%
- Loss computation + backward: ~10%

**Solutions:**
- Use `num_workers=4` for parallel data loading
- Use mixed precision training (FP16)
- Pre-extract features and train only classifier heads

**Inference:**
- Real-time webcam: ~100ms per prediction (10 FPS) on M3 Pro
- Batch processing: ~500 videos/minute on A100 GPU

## Model Checkpoints

Saved in `outputs/checkpoints/`:
```python
checkpoint = {
    'epoch': 10,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': 2.34,
    'val_loss': 5.67,
    'verb_acc': 76.0,
    'noun_acc': 78.0
}
```

Load checkpoint:
```python
checkpoint = torch.load('checkpoint_epoch_10.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```
