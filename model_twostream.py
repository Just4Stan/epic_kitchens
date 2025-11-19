"""
Two-Stream Model with Optical Flow
Combines RGB frames + Optical Flow for better motion understanding
"""

import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np


class DropPath(nn.Module):
    """Stochastic Depth for regularization."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class SpatialStream(nn.Module):
    """RGB spatial stream - processes appearance."""
    def __init__(self, pretrained=True, dropout=0.5):
        super().__init__()

        # ResNet-50 for RGB
        resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, num_frames, 3, H, W)
        Returns:
            (batch, 2048)
        """
        batch, num_frames, c, h, w = x.shape

        # Process all frames
        x = x.view(batch * num_frames, c, h, w)
        features = self.backbone(x)  # (batch*num_frames, 2048, 7, 7)
        features = self.pool(features)  # (batch*num_frames, 2048, 1, 1)
        features = features.view(batch, num_frames, 2048)

        # Temporal pooling
        features = features.mean(dim=1)  # (batch, 2048)

        return self.dropout(features)


class TemporalStream(nn.Module):
    """Optical flow temporal stream - processes motion."""
    def __init__(self, pretrained=True, dropout=0.5):
        super().__init__()

        # ResNet-50 for optical flow (2 channels: x, y flow)
        # Modify first conv to accept 2*num_flow_frames channels
        resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)

        # Replace first conv: 3 -> 2 channels (flow x, y) * num_stacked_flows
        # Stack 5 consecutive flows = 10 input channels
        original_conv = resnet.conv1
        self.conv1 = nn.Conv2d(
            10,  # 5 flow frames * 2 (x,y)
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Initialize from pretrained weights (average across input channels)
        if pretrained:
            with torch.no_grad():
                # Average RGB weights to initialize flow conv
                self.conv1.weight[:, :3, :, :] = original_conv.weight / 3.0
                self.conv1.weight[:, 3:6, :, :] = original_conv.weight / 3.0
                self.conv1.weight[:, 6:9, :, :] = original_conv.weight / 3.0
                self.conv1.weight[:, 9:, :, :] = original_conv.weight[:, :1, :, :].repeat(1, 1, 1, 1)

        # Rest of ResNet
        self.backbone = nn.Sequential(
            self.conv1,
            *list(resnet.children())[1:-2]
        )

        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, 10, H, W) - stacked optical flow
        Returns:
            (batch, 2048)
        """
        features = self.backbone(x)  # (batch, 2048, 7, 7)
        features = self.pool(features)  # (batch, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (batch, 2048)

        return self.dropout(features)


class TwoStreamActionRecognition(nn.Module):
    """
    Two-Stream Network:
    - Spatial stream: RGB frames (appearance)
    - Temporal stream: Optical flow (motion)
    - Late fusion for classification
    """
    def __init__(self, num_verb_classes=97, num_noun_classes=300,
                 num_frames=8, pretrained=True, dropout=0.5, drop_path=0.1):
        super().__init__()

        print("="*70)
        print("Creating Two-Stream Model (RGB + Optical Flow)")
        print("="*70)

        # Spatial stream (RGB)
        self.spatial_stream = SpatialStream(pretrained=pretrained, dropout=dropout)

        # Temporal stream (Optical Flow)
        self.temporal_stream = TemporalStream(pretrained=pretrained, dropout=dropout)

        # Stochastic depth
        self.drop_path = DropPath(drop_path)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Classification heads
        self.verb_classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_verb_classes)
        )

        self.noun_classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_noun_classes)
        )

        # Initialize classifiers
        self._initialize_weights()

        print(f"✓ Spatial stream:  ResNet-50 (RGB)")
        print(f"✓ Temporal stream: ResNet-50 (Optical Flow)")
        print(f"✓ Dropout: {dropout}")
        print(f"✓ Drop path: {drop_path}")
        print(f"✓ Fusion: Late fusion (concat)")
        print("="*70)

    def _initialize_weights(self):
        """Initialize classifier weights."""
        for m in [self.fusion, self.verb_classifier, self.noun_classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(self, rgb_frames, flow_frames):
        """
        Args:
            rgb_frames: (batch, num_frames, 3, H, W)
            flow_frames: (batch, 10, H, W) - stacked optical flow
        Returns:
            verb_logits: (batch, num_verb_classes)
            noun_logits: (batch, num_noun_classes)
        """
        # Process spatial stream
        spatial_features = self.spatial_stream(rgb_frames)

        # Process temporal stream
        temporal_features = self.temporal_stream(flow_frames)

        # Apply stochastic depth
        spatial_features = self.drop_path(spatial_features)
        temporal_features = self.drop_path(temporal_features)

        # Fuse features
        combined = torch.cat([spatial_features, temporal_features], dim=1)
        fused_features = self.fusion(combined)

        # Classify
        verb_logits = self.verb_classifier(fused_features)
        noun_logits = self.noun_classifier(fused_features)

        return verb_logits, noun_logits


def compute_optical_flow_farneback(frames):
    """
    Compute optical flow using Farneback method.

    Args:
        frames: List of numpy arrays (H, W, 3) in RGB
    Returns:
        flows: List of (H, W, 2) flow fields
    """
    flows = []

    for i in range(len(frames) - 1):
        # Convert to grayscale
        prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)

        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        flows.append(flow)

    return flows


def get_twostream_model(config):
    """Factory function to create two-stream model."""
    model = TwoStreamActionRecognition(
        num_verb_classes=config.NUM_VERB_CLASSES,
        num_noun_classes=config.NUM_NOUN_CLASSES,
        num_frames=config.NUM_FRAMES,
        pretrained=True,
        dropout=0.5,
        drop_path=0.1
    )
    return model


if __name__ == '__main__':
    # Test the model
    print("\nTesting Two-Stream Model...")

    model = TwoStreamActionRecognition(
        num_verb_classes=97,
        num_noun_classes=300,
        num_frames=8
    )

    # Dummy inputs
    rgb_frames = torch.randn(2, 8, 3, 224, 224)
    flow_frames = torch.randn(2, 10, 224, 224)

    # Forward pass
    verb_logits, noun_logits = model(rgb_frames, flow_frames)

    print(f"\nOutput shapes:")
    print(f"  Verb logits: {verb_logits.shape}")
    print(f"  Noun logits: {noun_logits.shape}")
    print(f"\n✓ Two-Stream model works correctly!")
