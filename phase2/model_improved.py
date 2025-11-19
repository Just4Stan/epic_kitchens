"""
Improved EPIC-KITCHENS Action Recognition Model
Incorporates all best practices:
- Temporal Transformer with multi-head attention
- Aggressive dropout for regularization
- Stochastic depth (drop path)
- Better feature aggregation
"""

import torch
import torch.nn as nn
import torchvision.models as models
import math


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample for regularization."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class TemporalTransformer(nn.Module):
    """Temporal transformer for video understanding."""
    def __init__(self, feature_dim=2048, num_heads=8, num_layers=2, dropout=0.3):
        super().__init__()
        self.feature_dim = feature_dim

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 8, feature_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, num_frames, feature_dim)
        Returns:
            (batch, feature_dim)
        """
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Apply transformer
        x = self.transformer(x)

        # Global average pooling over time
        x = x.mean(dim=1)

        return self.dropout(x)


class ImprovedActionRecognitionModel(nn.Module):
    """
    Improved model with:
    - ResNet-50 backbone (frozen early layers for regularization)
    - Temporal Transformer
    - Multi-level dropout
    - Stochastic depth
    - Better initialization
    """
    def __init__(self, num_verb_classes=97, num_noun_classes=300,
                 num_frames=8, pretrained=True, dropout=0.5,
                 drop_path=0.1, freeze_backbone_layers=2):
        super().__init__()

        print(f"======================================================================")
        print(f"Creating Improved Model")
        print(f"======================================================================")

        # ResNet-50 backbone
        resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)

        # Remove final FC layer and pooling
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.feature_dim = 2048

        # Freeze early layers for regularization
        if freeze_backbone_layers > 0:
            print(f"Freezing first {freeze_backbone_layers} ResNet blocks")
            layers_to_freeze = list(self.backbone.children())[:freeze_backbone_layers+4]
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

        # Adaptive pooling to fixed spatial size
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Stochastic depth (drop path)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Temporal transformer
        self.temporal_transformer = TemporalTransformer(
            feature_dim=self.feature_dim,
            num_heads=8,
            num_layers=2,
            dropout=0.3
        )

        # Feature dropout
        self.feature_dropout = nn.Dropout(dropout * 0.5)

        # Classification heads with dropout
        self.verb_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim // 2, num_verb_classes)
        )

        self.noun_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim // 2, num_noun_classes)
        )

        # Better initialization for classifier heads
        self._init_classifier_weights()

        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Architecture: ResNet-50 + Temporal Transformer + MLP Heads")
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters:    {total_params - trainable_params:,}")
        print(f"Dropout rate:         {dropout}")
        print(f"Drop path rate:       {drop_path}")
        print(f"======================================================================\n")

    def _init_classifier_weights(self):
        """Initialize classifier weights with better strategy."""
        for m in [self.verb_classifier, self.noun_classifier]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (batch, num_frames, channels, height, width)
        Returns:
            verb_logits: (batch, num_verb_classes)
            noun_logits: (batch, num_noun_classes)
        """
        batch_size, num_frames, C, H, W = x.shape

        # Reshape to process all frames at once
        x = x.view(batch_size * num_frames, C, H, W)

        # Extract spatial features with ResNet
        x = self.backbone(x)  # (batch*num_frames, 2048, H', W')

        # Spatial pooling
        x = self.spatial_pool(x)  # (batch*num_frames, 2048, 1, 1)
        x = x.view(batch_size, num_frames, self.feature_dim)  # (batch, num_frames, 2048)

        # Apply stochastic depth
        x = self.drop_path(x)

        # Temporal modeling with transformer
        x = self.temporal_transformer(x)  # (batch, 2048)

        # Feature dropout
        x = self.feature_dropout(x)

        # Classification
        verb_out = self.verb_classifier(x)
        noun_out = self.noun_classifier(x)

        return verb_out, noun_out


def get_improved_model(config):
    """Factory function to create improved model."""
    model = ImprovedActionRecognitionModel(
        num_verb_classes=config.NUM_VERB_CLASSES,
        num_noun_classes=config.NUM_NOUN_CLASSES,
        num_frames=config.NUM_FRAMES,
        pretrained=True,
        dropout=0.5,
        drop_path=0.1,
        freeze_backbone_layers=2  # Freeze conv1, bn1, layer1, layer2
    )
    return model


if __name__ == '__main__':
    # Test model
    from config import Config
    config = Config()

    model = get_improved_model(config)
    model.eval()

    # Test forward pass
    batch_size = 2
    num_frames = 8
    test_input = torch.randn(batch_size, num_frames, 3, 224, 224)

    with torch.no_grad():
        verb_out, noun_out = model(test_input)

    print(f"\nTest Forward Pass:")
    print(f"Input shape:  {test_input.shape}")
    print(f"Verb output:  {verb_out.shape}")
    print(f"Noun output:  {noun_out.shape}")
    print(f"\n✓ Model works correctly!")
