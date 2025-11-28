"""
Simple Baseline Model - ResNet-18 + Temporal Pooling
Following the approach that achieves ~50% action accuracy

This model is intentionally simple:
- ResNet-18 (smaller, less prone to overfitting than ResNet-50)
- Temporal average pooling (simple and effective)
- Minimal dropout (0.3 only)
- No complex attention mechanisms
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SimpleActionRecognitionModel(nn.Module):
    """
    Simple baseline model for action recognition.

    Architecture:
    1. ResNet-18 backbone (per-frame features)
    2. Temporal average pooling
    3. Dropout (0.3)
    4. Linear classifiers for verb and noun
    """

    def __init__(self, num_verb_classes=97, num_noun_classes=300,
                 num_frames=8, pretrained=True, dropout=0.3):
        """
        Args:
            num_verb_classes: Number of verb classes (97)
            num_noun_classes: Number of noun classes (300)
            num_frames: Number of frames per clip
            pretrained: Use ImageNet pre-trained weights
            dropout: Dropout rate (default: 0.3)
        """
        super().__init__()

        print(f"=" * 70)
        print(f"Creating Simple Baseline Model (ResNet-18)")
        print(f"=" * 70)

        self.num_frames = num_frames

        # ResNet-18 backbone (much smaller than ResNet-50)
        if pretrained:
            resnet18 = models.resnet18(weights='IMAGENET1K_V1')
            print("Using ImageNet pre-trained ResNet-18")
        else:
            resnet18 = models.resnet18(weights=None)
            print("Using randomly initialized ResNet-18")

        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet18.children())[:-1])
        self.feature_dim = 512  # ResNet-18 output dimension

        # Keep backbone trainable (don't freeze)
        for param in self.backbone.parameters():
            param.requires_grad = True

        # Temporal pooling (simple average)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # Minimal dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Simple verb classifier
        self.verb_classifier = nn.Linear(self.feature_dim, num_verb_classes)

        # Simple noun classifier
        self.noun_classifier = nn.Linear(self.feature_dim, num_noun_classes)

        # Initialize classifier weights
        nn.init.normal_(self.verb_classifier.weight, std=0.01)
        nn.init.constant_(self.verb_classifier.bias, 0)
        nn.init.normal_(self.noun_classifier.weight, std=0.01)
        nn.init.constant_(self.noun_classifier.bias, 0)

        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"Architecture:         ResNet-18 + Temporal Pooling")
        print(f"Feature dimension:    {self.feature_dim}")
        print(f"Dropout:              {dropout}")
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"=" * 70)
        print()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, num_frames, 3, H, W)

        Returns:
            verb_logits: (batch_size, num_verb_classes)
            noun_logits: (batch_size, num_noun_classes)
        """
        batch_size, num_frames, C, H, W = x.shape

        # Reshape to process all frames together
        # (batch_size * num_frames, 3, H, W)
        x = x.view(batch_size * num_frames, C, H, W)

        # Extract features with ResNet-18
        # (batch_size * num_frames, 512, 1, 1)
        features = self.backbone(x)

        # Remove spatial dimensions
        # (batch_size * num_frames, 512)
        features = features.view(batch_size * num_frames, self.feature_dim)

        # Reshape to (batch_size, num_frames, 512)
        features = features.view(batch_size, num_frames, self.feature_dim)

        # Temporal pooling: average across frames
        # (batch_size, 512, num_frames) -> (batch_size, 512)
        features = features.permute(0, 2, 1)  # (batch, 512, num_frames)
        features = self.temporal_pool(features).squeeze(-1)  # (batch, 512)

        # Apply dropout
        features = self.dropout(features)

        # Classify verb and noun
        verb_logits = self.verb_classifier(features)
        noun_logits = self.noun_classifier(features)

        return verb_logits, noun_logits


def get_simple_model(config):
    """
    Create simple baseline model.

    Args:
        config: Configuration object

    Returns:
        model: SimpleActionRecognitionModel
    """
    model = SimpleActionRecognitionModel(
        num_verb_classes=config.NUM_VERB_CLASSES,
        num_noun_classes=config.NUM_NOUN_CLASSES,
        num_frames=config.NUM_FRAMES,
        pretrained=True,
        dropout=0.3
    )
    return model


# Test the model
if __name__ == "__main__":
    print("Testing SimpleActionRecognitionModel...")

    # Create dummy input
    batch_size = 4
    num_frames = 8
    x = torch.randn(batch_size, num_frames, 3, 224, 224)

    print(f"Input shape: {x.shape}")

    # Create model
    model = SimpleActionRecognitionModel(
        num_verb_classes=97,
        num_noun_classes=300,
        num_frames=8,
        pretrained=False,  # Don't download weights for testing
        dropout=0.3
    )

    # Forward pass
    model.eval()
    with torch.no_grad():
        verb_logits, noun_logits = model(x)

    print(f"\nOutput shapes:")
    print(f"  Verb logits: {verb_logits.shape}")  # (4, 97)
    print(f"  Noun logits: {noun_logits.shape}")  # (4, 300)

    print(f"\n✓ Model test passed!")
