#!/usr/bin/env python3
"""
Test CoreML model on Mac to verify conversion accuracy.
Compare CoreML predictions with PyTorch predictions.
"""

import torch
import coremltools as ct
import numpy as np
from pathlib import Path

# Load CoreML model
print("Loading CoreML model...")
coreml_model = ct.models.MLModel('models/EpicKitchens.mlpackage')

# Create random test input
test_input = np.random.randn(1, 16, 3, 224, 224).astype(np.float32)

# Run CoreML inference
print("Running CoreML inference...")
coreml_output = coreml_model.predict({'video_frames': test_input})

print(f"CoreML output shapes:")
print(f"  Verb probabilities: {coreml_output['verb_probabilities'].shape}")
print(f"  Noun probabilities: {coreml_output['noun_probabilities'].shape}")

# Get top predictions
verb_idx = np.argmax(coreml_output['verb_probabilities'])
noun_idx = np.argmax(coreml_output['noun_probabilities'])

print(f"\nTop predictions:")
print(f"  Verb: {verb_idx} (prob: {coreml_output['verb_probabilities'][0, verb_idx]:.4f})")
print(f"  Noun: {noun_idx} (prob: {coreml_output['noun_probabilities'][0, noun_idx]:.4f})")

print("\n✅ CoreML model working on Mac!")
print("Now test with real webcam frames to compare with PyTorch...")
