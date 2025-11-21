"""
Simple ResNet-18 Baseline Model
Mimicking friend's approach: ResNet-18 + temporal pooling

Architecture:
1. ResNet-18 backbone (ImageNet pretrained)
2. Temporal average pooling
3. Dropout 0.3
4. Linear classifiers for verb and noun

No complexity, no frozen layers, no fancy tricks.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SimpleResNet18(nn.Module):
    """
    Simple action recognition model.

    Following friend's approach:
    - ResNet-18 (11M params vs ResNet-50's 25M)
    - Temporal average pooling (not Transformer)
    - Minimal dropout (0.3)
    - No frozen layers
    """

    def __init__(self, num_verb_classes=97, num_noun_classes=300,
                 num_frames=8, dropout=0.3):
        """
        Args:
            num_verb_classes: Number of verb classes (97)
            num_noun_classes: Number of noun classes (300)
            num_frames: Number of frames per clip (8)
            dropout: Dropout rate (0.3)
        """
        super().__init__()

        self.num_frames = num_frames
        self.dropout_rate = dropout

        # ResNet-18 backbone (ImageNet pretrained)
        print("Loading ResNet-18 (ImageNet pretrained)...")
        resnet18 = models.resnet18(weights='IMAGENET1K_V1')

        # Remove final FC layer and avgpool
        self.backbone = nn.Sequential(*list(resnet18.children())[:-1])

        # Feature dimension (ResNet-18 outputs 512-d features)
        self.feature_dim = 512

        # Keep backbone fully trainable (no frozen layers)
        for param in self.backbone.parameters():
            param.requires_grad = True

        # Temporal pooling (simple average)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # Dropout (minimal regularization)
        self.dropout = nn.Dropout(dropout)

        # Simple linear classifiers (no MLP, keep it simple)
        self.verb_fc = nn.Linear(self.feature_dim, num_verb_classes)
        self.noun_fc = nn.Linear(self.feature_dim, num_noun_classes)

        # Initialize classifier weights
        self._init_classifiers()

        # Print model info
        self._print_model_info(num_verb_classes, num_noun_classes)

    def _init_classifiers(self):
        """Initialize classifier weights with small std."""
        nn.init.normal_(self.verb_fc.weight, std=0.01)
        nn.init.constant_(self.verb_fc.bias, 0)
        nn.init.normal_(self.noun_fc.weight, std=0.01)
        nn.init.constant_(self.noun_fc.bias, 0)

    def _print_model_info(self, num_verb_classes, num_noun_classes):
        """Print model architecture info."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("=" * 70)
        print("Simple ResNet-18 Baseline Model")
        print("=" * 70)
        print(f"Architecture:      ResNet-18 + Temporal Avg Pooling")
        print(f"Feature dimension: {self.feature_dim}")
        print(f"Frames per clip:   {self.num_frames}")
        print(f"Dropout:           {self.dropout_rate}")
        print(f"Verb classes:      {num_verb_classes}")
        print(f"Noun classes:      {num_noun_classes}")
        print(f"Total parameters:  {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"Trainable params:  {trainable_params:,}")
        print("=" * 70)

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

        # Extract per-frame features with ResNet-18
        # (batch_size * num_frames, 512, 1, 1)
        features = self.backbone(x)

        # Remove spatial dimensions
        # (batch_size * num_frames, 512)
        features = features.view(batch_size * num_frames, self.feature_dim)

        # Reshape to (batch_size, num_frames, 512)
        features = features.view(batch_size, num_frames, self.feature_dim)

        # Temporal pooling: average across frames
        # (batch, 512, num_frames) -> (batch, 512, 1) -> (batch, 512)
        features = features.permute(0, 2, 1)  # (batch, 512, num_frames)
        features = self.temporal_pool(features).squeeze(-1)  # (batch, 512)

        # Apply dropout
        features = self.dropout(features)

        # Classify verb and noun
        verb_logits = self.verb_fc(features)  # (batch, 97)
        noun_logits = self.noun_fc(features)  # (batch, 300)

        return verb_logits, noun_logits


def create_model(config):
    """
    Factory function to create model from config.

    Args:
        config: Configuration object

    Returns:
        model: SimpleResNet18 model
    """
    model = SimpleResNet18(
        num_verb_classes=config.NUM_VERB_CLASSES,
        num_noun_classes=config.NUM_NOUN_CLASSES,
        num_frames=config.NUM_FRAMES,
        dropout=config.DROPOUT
    )
    return model


# Test the model
if __name__ == "__main__":
    print("Testing SimpleResNet18 model...\n")

    # Create model
    model = SimpleResNet18(
        num_verb_classes=97,
        num_noun_classes=300,
        num_frames=8,
        dropout=0.3
    )

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    num_frames = 8
    x = torch.randn(batch_size, num_frames, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        verb_logits, noun_logits = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Verb output:  {verb_logits.shape}  (expected: {batch_size}, 97)")
    print(f"Noun output:  {noun_logits.shape}  (expected: {batch_size}, 300)")

    # Verify shapes
    assert verb_logits.shape == (batch_size, 97), "Verb output shape mismatch!"
    assert noun_logits.shape == (batch_size, 300), "Noun output shape mismatch!"

    print("\n✓ All tests passed!")
