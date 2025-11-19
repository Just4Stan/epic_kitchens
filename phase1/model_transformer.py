"""
Transformer-based Action Recognition Model for EPIC-KITCHENS-100
Uses ResNet-50 for frame features + Transformer for temporal modeling
"""

import torch
import torch.nn as nn
import torchvision.models as models
import math


class ActionRecognitionTransformer(nn.Module):
    """
    ResNet-50 + Transformer for temporal modeling.

    Better than LSTM because:
    - Self-attention learns which frames are important
    - Can attend to any frame (not sequential)
    - Parallelizable on GPU
    - State-of-the-art for many vision tasks
    """

    def __init__(self, num_verb_classes=97, num_noun_classes=300, num_frames=8, pretrained=True):
        super().__init__()

        self.num_frames = num_frames

        # ResNet-50 backbone
        resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, num_frames, self.feature_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.feature_dim) * 0.02)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Verb classification head
        self.verb_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_verb_classes)
        )

        # Noun classification head
        self.noun_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_noun_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_frames, 3, H, W)

        Returns:
            verb_logits: (batch_size, num_verb_classes)
            noun_logits: (batch_size, num_noun_classes)
        """
        batch_size, num_frames, C, H, W = x.shape

        # Extract features per frame
        x = x.view(batch_size * num_frames, C, H, W)
        features = self.feature_extractor(x)
        features = features.view(batch_size * num_frames, self.feature_dim)

        # Reshape: (batch, num_frames, feature_dim)
        features = features.view(batch_size, num_frames, self.feature_dim)

        # Add positional encoding
        features = features + self.positional_encoding

        # Add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        features = torch.cat([cls_tokens, features], dim=1)  # (batch, num_frames+1, feature_dim)

        # Apply Transformer
        features = self.transformer(features)

        # Take [CLS] token output
        features = features[:, 0]  # (batch_size, feature_dim)
        features = self.dropout(features)

        # Classify
        verb_logits = self.verb_classifier(features)
        noun_logits = self.noun_classifier(features)

        return verb_logits, noun_logits


def get_model(config):
    """Create Transformer-based model."""
    print("=" * 70)
    print("Creating Model: ResNet-50 + Transformer")
    print("=" * 70)

    model = ActionRecognitionTransformer(
        num_verb_classes=config.NUM_VERB_CLASSES,
        num_noun_classes=config.NUM_NOUN_CLASSES,
        num_frames=config.NUM_FRAMES,
        pretrained=True
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 70)
    print()

    return model
