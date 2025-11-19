# Phase 3: Cross-Attention Model

**Status**: Prepared (not yet trained)
**Architecture**: ResNet-50 + Temporal Transformer + Cross-Task Attention
**Goal**: Improve action accuracy by modeling verb-noun dependencies

---

## Key Innovation

### Cross-Task Attention
Unlike Phase 2 (independent verb/noun heads), Phase 3 models **verb-noun interactions**:

```python
# Phase 2: Independent predictions
shared_features → verb_head → verb_logits
                → noun_head → noun_logits

# Phase 3: Cross-attention interaction
shared_features → verb_branch → verb_features ↘
                → noun_branch → noun_features → Cross-Attention → refined_features
                                                                      ↓
                                                                verb_logits, noun_logits
```

**Example**: If model predicts "cut" (verb), cross-attention biases noun prediction towards knife, vegetables, etc.

---

## Architecture Details

### Model: `CrossAttentionActionModel`

**1. Shared Backbone**
- ResNet-50 (first 2 blocks frozen)
- Temporal Transformer (2 layers, 8 heads)
- Output: 2048-d shared features

**2. Task-Specific Branches**
```python
verb_features = verb_branch(shared_features)  # 2048 → 1024
noun_features = noun_branch(shared_features)  # 2048 → 1024
```

**3. Cross-Task Attention**
```python
class CrossTaskAttention:
    # Bidirectional attention
    verb_to_noun: MultiheadAttention(1024-d, 8 heads)
    noun_to_verb: MultiheadAttention(1024-d, 8 heads)

    forward(verb_features, noun_features):
        # Verb attends to noun
        verb_attended = attn(Q=verb, K=noun, V=noun)
        verb_refined = verb + FFN(verb + verb_attended)

        # Noun attends to verb
        noun_attended = attn(Q=noun, K=verb, V=verb)
        noun_refined = noun + FFN(noun + noun_attended)

        return verb_refined, noun_refined
```

**4. Final Classifiers**
- Verb: 1024 → 512 → 97 classes
- Noun: 1024 → 512 → 300 classes

### Parameters
- Total: ~145M (vs 128M in Phase 2)
- Additional: ~17M for cross-attention
- Trainable: ~144M (first 2 ResNet blocks frozen)

---

## Expected Improvements

### Phase 2 (Independent)
- Verb accuracy: 35-45%
- Noun accuracy: 30-40%
- Action accuracy: 20-30%

### Phase 3 (Cross-Attention)
- Verb accuracy: 35-45% (similar)
- Noun accuracy: 30-40% (similar)
- **Action accuracy: 25-35%** (+5% improvement)

**Why**: Cross-attention enforces semantic consistency between verb and noun predictions.

---

## Training Configuration

Copy from Phase 2 best model, then:

```python
# Use best hyperparameters from Phase 2
lr = 5e-5  # (or whatever worked best)
dropout = 0.5
batch_size = 24

# Same training setup
optimizer = AdamW(lr=lr, weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(T_0=10)
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

# Train for 30 epochs
# Expected time: 15-20 hours on A100
```

---

## Files

- `model_cross_attention.py` - Model implementation
- `train_cross_attention.py` - Training script (to be created)
- `README.md` - This file

---

## Next Steps

1. **Wait for Phase 2 to complete** (~2-3 days)
2. **Identify best hyperparameters** from 8 Phase 2 models
3. **Train Phase 3** with those hyperparameters
4. **Compare** Phase 2 vs Phase 3 action accuracy

---

## Theoretical Background

### Why Cross-Attention Works

**Semantic Dependencies**:
- "cut" verb → knife, bread, vegetables (nouns)
- "pour" verb → water, milk, juice (nouns)
- "open" verb → door, fridge, jar (nouns)

**Current Problem (Phase 2)**:
- Model might predict "cut" + "water" (semantically odd)
- Independent heads don't enforce consistency

**Solution (Phase 3)**:
- Verb features attend to noun features
- Noun features attend to verb features
- Learned attention weights enforce semantic correlations

### SOTA References

1. **TBN (Temporal Binding Network)**: Uses compositional rules for verb-noun
2. **Rulformer**: Transformer with explicit compositional attention
3. **EgoVLP**: Vision-language pretraining captures verb-noun semantics

Our approach is most similar to Rulformer but simpler (bidirectional cross-attention).

---

## Comparison Summary

| Aspect | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| **Architecture** | Various (baseline, LSTM, Transformer, 3D-CNN) | Improved Transformer | Transformer + Cross-Attention |
| **Temporal** | LSTM / Transformer | Transformer (2L) | Transformer (2L) |
| **Verb-Noun** | Independent | Independent | **Cross-Attention** |
| **Regularization** | Minimal | Heavy (dropout, label smooth, drop path) | Same as Phase 2 |
| **Validation** | Empty (bug) | Working | Working |
| **Expected Action Acc** | N/A | 20-30% | **25-35%** |
| **Parameters** | 25.8M | 128M | **145M** |
| **Training Time** | 2-3 days | 2-3 days | ~1 day (single model) |
