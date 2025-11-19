"""
Phase 3: Cross-Attention Model for EPIC-KITCHENS
Implements verb-noun interaction via cross-task attention
"""

import torch
import torch.nn as nn
import torchvision.models as models


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
        return x.div(keep_prob) * random_tensor


class TemporalTransformer(nn.Module):
    """Temporal transformer for video understanding."""
    def __init__(self, feature_dim=2048, num_heads=8, num_layers=2, dropout=0.3, window_size=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.window_size = window_size

        # Positional encoding (adjust based on window size)
        max_len = window_size if window_size else 32
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, feature_dim))

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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, num_frames, feature_dim)
        # Handle variable sequence lengths if window_size is set
        seq_len = min(x.size(1), self.window_size) if self.window_size else x.size(1)
        x = x[:, :seq_len, :]
        x = x + self.positional_encoding[:, :seq_len, :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.dropout(x)


class CrossTaskAttention(nn.Module):
    """
    Cross-attention module for verb-noun interaction.

    Allows verb features to attend to noun features and vice versa,
    modeling dependencies like "cut" verb → knife/vegetable nouns.
    """
    def __init__(self, feature_dim=1024, num_heads=8, dropout=0.3):
        super().__init__()

        # Bidirectional cross-attention
        self.verb_to_noun_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.noun_to_verb_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.norm_verb = nn.LayerNorm(feature_dim)
        self.norm_noun = nn.LayerNorm(feature_dim)

        # Feed-forward networks
        self.ffn_verb = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(dropout)
        )
        self.ffn_noun = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(dropout)
        )

    def forward(self, verb_features, noun_features):
        """
        Args:
            verb_features: (batch, feature_dim)
            noun_features: (batch, feature_dim)
        Returns:
            verb_features_refined: (batch, feature_dim)
            noun_features_refined: (batch, feature_dim)
        """
        # Add sequence dimension for attention
        verb_feat = verb_features.unsqueeze(1)  # (batch, 1, feature_dim)
        noun_feat = noun_features.unsqueeze(1)  # (batch, 1, feature_dim)

        # Verb attends to noun
        verb_attended, _ = self.verb_to_noun_attn(
            query=verb_feat,
            key=noun_feat,
            value=noun_feat
        )
        verb_feat = self.norm_verb(verb_feat + verb_attended)
        verb_feat = verb_feat + self.ffn_verb(verb_feat)

        # Noun attends to verb
        noun_attended, _ = self.noun_to_verb_attn(
            query=noun_feat,
            key=verb_feat,
            value=verb_feat
        )
        noun_feat = self.norm_noun(noun_feat + noun_attended)
        noun_feat = noun_feat + self.ffn_noun(noun_feat)

        return verb_feat.squeeze(1), noun_feat.squeeze(1)


class CrossAttentionActionModel(nn.Module):
    """
    Phase 3: Cross-attention model with verb-noun interaction.

    Architecture:
    1. ResNet-50 backbone (frozen early layers)
    2. Temporal Transformer
    3. Task-specific branches (verb and noun)
    4. Cross-task attention (verb ↔ noun)
    5. Final classifiers
    """
    def __init__(self, num_verb_classes=97, num_noun_classes=300,
                 num_frames=16,
                 pretrained=True, dropout=0.5,
                 drop_path=0.1, freeze_backbone_layers=2):
        super().__init__()
        
        print("=" * 70)
        print("Phase 3: Cross-Attention Model")
        print("=" * 70)

        # ResNet-50 backbone (simpler approach)
        resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.feature_dim = 2048

        # Freeze early layers
        if freeze_backbone_layers > 0:
            print(f"✓ Freezing first {freeze_backbone_layers} ResNet blocks")
            layers_to_freeze = list(self.backbone.children())[:4 + freeze_backbone_layers]
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Multi-scale temporal modeling
        self.temporal_scales = nn.ModuleList([
            TemporalTransformer(feature_dim=2048, window_size=4),   # Local
            TemporalTransformer(feature_dim=2048, window_size=8),   # Medium  
            TemporalTransformer(feature_dim=2048, window_size=16),  # Global
        ])

        # Task-specific feature branches
        self.verb_branch = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        self.noun_branch = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # Cross-task attention module (KEY INNOVATION)
        self.cross_attention = CrossTaskAttention(
            feature_dim=1024,
            num_heads=8,
            dropout=0.3
        )

        # Final classifiers
        self.verb_classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_verb_classes)
        )
        self.noun_classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_noun_classes)
        )

        # Additional backbones and fusion
        self.rgb_backbone = ResNet50()
        self.flow_backbone = ResNet50()  # Train on optical flow
        self.fusion = CrossModalAttention()

        # Initialize weights
        self._init_weights()

        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Architecture: ResNet-50 + Multi-Scale Temporal Transformer + Cross-Attention")
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters:    {total_params - trainable_params:,}")
        print(f"Dropout rate:         {dropout}")
        print(f"Drop path rate:       {drop_path}")
        print("=" * 70)

    def _init_weights(self):
        """Initialize classifier weights."""
        for m in [self.verb_classifier, self.noun_classifier]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (batch, num_frames, 3, H, W)
        Returns:
            verb_logits: (batch, num_verb_classes)
            noun_logits: (batch, num_noun_classes)
        """
        batch_size, num_frames, C, H, W = x.shape

        # Extract spatial features
        x = x.view(batch_size * num_frames, C, H, W)
        x = self.backbone(x)
        x = self.spatial_pool(x)
        x = x.view(batch_size, num_frames, 2048)

        # Stochastic depth
        x = self.drop_path(x)

        # Multi-scale temporal modeling
        temporal_features = []
        for temporal_scale in self.temporal_scales:
            temporal_features.append(temporal_scale(x))
        shared_features = torch.stack(temporal_features).mean(dim=0)  # (batch, 2048)

        # Task-specific branches
        verb_features = self.verb_branch(shared_features)
        noun_features = self.noun_branch(shared_features)

        # Cross-task attention
        verb_features, noun_features = self.cross_attention(
            verb_features, noun_features
        )

        # Final classification
        verb_logits = self.verb_classifier(verb_features)
        noun_logits = self.noun_classifier(noun_features)

        return verb_logits, noun_logits


def get_cross_attention_model(config, dropout=0.5, drop_path=0.1, freeze_backbone_layers=2):
    """Factory function for Phase 3 model."""
    model = CrossAttentionActionModel(
        num_verb_classes=config.NUM_VERB_CLASSES,
        num_noun_classes=config.NUM_NOUN_CLASSES,
        num_frames=config.NUM_FRAMES,
        pretrained=True,
        dropout=dropout,
        drop_path=drop_path,
        freeze_backbone_layers=freeze_backbone_layers
    )
    return model


if __name__ == '__main__':
    # Test model
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from common.config import Config
    config = Config()

    model = get_cross_attention_model(config)
    model.eval()

    # Test forward pass
    batch_size = 2
    num_frames = 16
    test_input = torch.randn(batch_size, num_frames, 3, 224, 224)

    with torch.no_grad():
        verb_out, noun_out = model(test_input)

    print(f"\nTest Forward Pass:")
    print(f"Input shape:  {test_input.shape}")
    print(f"Verb output:  {verb_out.shape}")
    print(f"Noun output:  {noun_out.shape}")
    print(f"\n✓ Phase 3 model works correctly!")
