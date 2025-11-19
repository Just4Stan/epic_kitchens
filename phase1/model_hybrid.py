"""
HYBRID Model: Best of All Worlds
Combines:
- 3D CNN (R(2+1)D) for spatiotemporal features
- LSTM for sequential modeling
- Transformer for attention
- All regularization techniques
"""

import torch
import torch.nn as nn
import torchvision.models.video as video_models
import math


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


class HybridTemporalModule(nn.Module):
    """
    Hybrid temporal processing:
    - LSTM for sequential dependencies
    - Transformer for attention
    - Fusion of both
    """
    def __init__(self, feature_dim=512, hidden_dim=256, num_heads=8, dropout=0.3):
        super().__init__()

        # BiLSTM branch
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Transformer branch
        self.positional_encoding = nn.Parameter(torch.randn(1, 8, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Fusion
        lstm_output_dim = hidden_dim * 2  # Bidirectional
        self.fusion = nn.Sequential(
            nn.Linear(lstm_output_dim + feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, num_frames, feature_dim)
        Returns:
            (batch, feature_dim)
        """
        # LSTM branch
        lstm_out, _ = self.lstm(x)  # (batch, num_frames, hidden_dim*2)
        lstm_features = lstm_out.mean(dim=1)  # Global avg pooling

        # Transformer branch
        x_pos = x + self.positional_encoding[:, :x.size(1), :]
        transformer_out = self.transformer(x_pos)  # (batch, num_frames, feature_dim)
        transformer_features = transformer_out.mean(dim=1)  # Global avg pooling

        # Fuse both
        combined = torch.cat([lstm_features, transformer_features], dim=1)
        fused = self.fusion(combined)

        return fused


class HybridActionRecognitionModel(nn.Module):
    """
    HYBRID Model Architecture:
    1. R(2+1)D 3D CNN backbone (spatiotemporal features)
    2. Hybrid temporal module (LSTM + Transformer)
    3. Multi-task classification heads
    4. Full regularization (dropout, drop path, label smoothing)
    """
    def __init__(self, num_verb_classes=97, num_noun_classes=300,
                 num_frames=8, pretrained=True, dropout=0.5, drop_path=0.1):
        super().__init__()

        print(f"======================================================================")
        print(f"Creating HYBRID Model (3D CNN + LSTM + Transformer)")
        print(f"======================================================================")

        # R(2+1)D 18-layer 3D CNN backbone
        r2plus1d = video_models.r2plus1d_18(weights='DEFAULT' if pretrained else None)

        # Remove final classification layer
        self.backbone_3d = nn.Sequential(*list(r2plus1d.children())[:-1])
        self.feature_dim_3d = 512

        # Adaptive pooling
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # Pool spatial, keep temporal

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Hybrid temporal module (LSTM + Transformer)
        self.temporal_module = HybridTemporalModule(
            feature_dim=self.feature_dim_3d,
            hidden_dim=256,
            num_heads=8,
            dropout=dropout * 0.6
        )

        # Feature dropout
        self.feature_dropout = nn.Dropout(dropout * 0.5)

        # Multi-layer classification heads
        self.verb_classifier = nn.Sequential(
            nn.Linear(self.feature_dim_3d, self.feature_dim_3d // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim_3d // 2, num_verb_classes)
        )

        self.noun_classifier = nn.Sequential(
            nn.Linear(self.feature_dim_3d, self.feature_dim_3d // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim_3d // 2, num_noun_classes)
        )

        # Initialize weights
        self._init_weights()

        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Architecture: R(2+1)D 3D CNN + BiLSTM + Transformer")
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Backbone: R(2+1)D-18 (3D CNN)")
        print(f"Temporal: Hybrid LSTM+Transformer")
        print(f"Dropout: {dropout}, Drop path: {drop_path}")
        print(f"======================================================================\n")

    def _init_weights(self):
        """Better weight initialization."""
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
            verb_logits, noun_logits
        """
        batch_size, num_frames, C, H, W = x.shape

        # Reshape for 3D CNN: (batch, channels, num_frames, height, width)
        x = x.permute(0, 2, 1, 3, 4)

        # 3D CNN backbone - extracts spatiotemporal features
        x = self.backbone_3d(x)  # (batch, 512, T', H', W')

        # Spatial pooling (keep temporal dimension)
        x = self.spatial_pool(x)  # (batch, 512, T', 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (batch, 512, T')

        # Permute for temporal module: (batch, T', 512)
        x = x.permute(0, 2, 1)

        # Apply stochastic depth
        x = self.drop_path(x)

        # Hybrid temporal module (LSTM + Transformer fusion)
        x = self.temporal_module(x)  # (batch, 512)

        # Feature dropout
        x = self.feature_dropout(x)

        # Classification
        verb_out = self.verb_classifier(x)
        noun_out = self.noun_classifier(x)

        return verb_out, noun_out


def get_hybrid_model(config):
    """Factory function to create hybrid model."""
    model = HybridActionRecognitionModel(
        num_verb_classes=config.NUM_VERB_CLASSES,
        num_noun_classes=config.NUM_NOUN_CLASSES,
        num_frames=config.NUM_FRAMES,
        pretrained=True,
        dropout=0.5,
        drop_path=0.1
    )
    return model


if __name__ == '__main__':
    # Test model
    from config import Config
    config = Config()

    model = get_hybrid_model(config)
    model.eval()

    # Test forward pass
    batch_size = 2
    num_frames = 8
    test_input = torch.randn(batch_size, num_frames, 3, 112, 112)  # Note: smaller input for 3D CNN

    with torch.no_grad():
        verb_out, noun_out = model(test_input)

    print(f"\nTest Forward Pass:")
    print(f"Input shape:  {test_input.shape}")
    print(f"Verb output:  {verb_out.shape}")
    print(f"Noun output:  {noun_out.shape}")
    print(f"\n✓ Hybrid model works correctly!")
