#!/usr/bin/env python3
"""
Convert EPIC-KITCHENS PyTorch model to CoreML for iPhone deployment.

Requirements:
    pip install coremltools

Usage:
    python inference/export_to_coreml.py \
        --checkpoint outputs/full_a100_v3/checkpoints/best_model.pth \
        --output models/EpicKitchens.mlpackage
"""

import torch
import torch.nn as nn
import torchvision.models as models
import coremltools as ct
import argparse
from pathlib import Path
import pandas as pd


class ActionModel(nn.Module):
    """Action recognition model - matches webcam_full_model.py."""

    def __init__(self, num_verb_classes=97, num_noun_classes=300,
                 backbone='resnet50', temporal_model='lstm',
                 dropout=0.5, num_frames=16):
        super().__init__()

        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feature_dim = 2048
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feature_dim = 512
            self.backbone.fc = nn.Identity()

        self.temporal_model = temporal_model
        if temporal_model == 'lstm':
            self.temporal = nn.LSTM(
                self.feature_dim, self.feature_dim // 2,
                num_layers=2, batch_first=True, bidirectional=True, dropout=0.3
            )
            self.temporal_dim = self.feature_dim
        elif temporal_model == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.feature_dim, nhead=8, dim_feedforward=2048,
                dropout=0.1, batch_first=True
            )
            self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.temporal_dim = self.feature_dim
        else:
            self.temporal = None
            self.temporal_dim = self.feature_dim

        self.dropout = nn.Dropout(dropout)
        self.verb_head = nn.Linear(self.temporal_dim, num_verb_classes)
        self.noun_head = nn.Linear(self.temporal_dim, num_noun_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)
        features = features.view(B, T, -1)

        if self.temporal_model == 'lstm':
            temporal_out, _ = self.temporal(features)
            pooled = temporal_out.mean(dim=1)
        elif self.temporal_model == 'transformer':
            temporal_out = self.temporal(features)
            pooled = temporal_out.mean(dim=1)
        else:
            pooled = features.mean(dim=1)

        pooled = self.dropout(pooled)
        verb_logits = self.verb_head(pooled)
        noun_logits = self.noun_head(pooled)

        # Return softmax probabilities for easier iOS usage
        verb_probs = torch.softmax(verb_logits, dim=1)
        noun_probs = torch.softmax(noun_logits, dim=1)

        return verb_probs, noun_probs


def export_to_coreml(checkpoint_path: str, output_path: str, num_frames: int = 16):
    """Convert PyTorch model to CoreML format."""

    print(f"Loading PyTorch model from {checkpoint_path}...")

    # Load model
    model = ActionModel(
        num_verb_classes=97,
        num_noun_classes=300,
        backbone='resnet50',
        temporal_model='lstm',
        dropout=0.0,  # No dropout during inference
        num_frames=num_frames
    )

    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"✅ Model loaded successfully")

    # Create example input (batch_size=1, num_frames, channels=3, height=224, width=224)
    example_input = torch.randn(1, num_frames, 3, 224, 224)

    print(f"Tracing model with input shape: {example_input.shape}")

    # Trace the model
    traced_model = torch.jit.trace(model, example_input)

    print("Converting to CoreML...")

    # Load class names
    epic_dir = Path(__file__).parent.parent
    verb_csv = epic_dir / "EPIC-KITCHENS" / "epic-kitchens-100-annotations-master" / "EPIC_100_verb_classes.csv"
    noun_csv = epic_dir / "EPIC-KITCHENS" / "epic-kitchens-100-annotations-master" / "EPIC_100_noun_classes.csv"

    verb_df = pd.read_csv(verb_csv)
    noun_df = pd.read_csv(noun_csv)
    verb_names = verb_df['key'].tolist()
    noun_names = noun_df['key'].tolist()

    # Convert to CoreML with metadata
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="video_frames",
                shape=example_input.shape,
                dtype=float
            )
        ],
        outputs=[
            ct.TensorType(name="verb_probabilities"),
            ct.TensorType(name="noun_probabilities")
        ],
        compute_units=ct.ComputeUnit.ALL,  # Use Neural Engine + GPU + CPU
        minimum_deployment_target=ct.target.iOS17,  # iPhone 16 supports iOS 18
    )

    # Add metadata
    mlmodel.author = "EPIC-KITCHENS Action Recognition"
    mlmodel.short_description = "Real-time egocentric action recognition (97 verbs × 300 nouns)"
    mlmodel.version = "1.0"
    mlmodel.license = "MIT"

    # Add input/output descriptions
    mlmodel.input_description["video_frames"] = f"Video frames ({num_frames} frames, 224x224 RGB)"
    mlmodel.output_description["verb_probabilities"] = "Probability distribution over 97 verb classes"
    mlmodel.output_description["noun_probabilities"] = "Probability distribution over 300 noun classes"

    # Save the model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving CoreML model to {output_path}...")
    mlmodel.save(str(output_path))

    # Get model size
    if output_path.suffix == '.mlpackage':
        import shutil
        size_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / (1024 * 1024)
    else:
        size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n{'='*60}")
    print("✅ CoreML CONVERSION SUCCESSFUL!")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Input: [{1}, {num_frames}, 3, 224, 224] (batch, frames, RGB, height, width)")
    print(f"Outputs:")
    print(f"  - verb_probabilities: [{1}, 97] (97 verb classes)")
    print(f"  - noun_probabilities: [{1}, 300] (300 noun classes)")
    print(f"Deployment: iOS 17+ (optimized for iPhone 16 Neural Engine)")
    print(f"\nNext Steps:")
    print(f"1. Add {output_path.name} to your Xcode project")
    print(f"2. Use Vision framework for video capture")
    print(f"3. Preprocess frames: resize to 224x224, normalize with ImageNet stats")
    print(f"4. Run inference on Neural Engine for real-time performance")
    print(f"{'='*60}\n")

    # Save class mappings as JSON for iOS app
    import json
    mappings = {
        'verbs': {str(i): name for i, name in enumerate(verb_names)},
        'nouns': {str(i): name for i, name in enumerate(noun_names)}
    }

    mapping_path = output_path.parent / "class_mappings.json"
    with open(mapping_path, 'w') as f:
        json.dump(mappings, f, indent=2)

    print(f"✅ Saved class mappings to {mapping_path}")

    return mlmodel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export PyTorch model to CoreML for iPhone')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to PyTorch checkpoint (.pth file)')
    parser.add_argument('--output', type=str, default='models/EpicKitchens.mlpackage',
                       help='Output path for CoreML model (.mlpackage)')
    parser.add_argument('--num_frames', type=int, default=16,
                       help='Number of frames (16 or 32)')

    args = parser.parse_args()

    export_to_coreml(args.checkpoint, args.output, args.num_frames)
