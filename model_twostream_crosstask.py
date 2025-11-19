"""
Two-Stream Model with Cross-Task Interaction
- RGB + Optical Flow (motion modeling)
- Verb and Noun interact via cross-attention (better action understanding)
"""

import torch
import torch.nn as nn
import torchvision.models as models
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


class SpatialStream(nn.Module):
    """RGB spatial stream."""
    def __init__(self, pretrained=True, dropout=0.5):
        super().__init__()

        resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (batch, num_frames, 3, H, W) -> (batch, 2048)"""
        batch, num_frames, c, h, w = x.shape
        x = x.view(batch * num_frames, c, h, w)
        features = self.backbone(x)
        features = self.pool(features)
        features = features.view(batch, num_frames, 2048)
        features = features.mean(dim=1)
        return self.dropout(features)


class TemporalStream(nn.Module):
    """Optical flow temporal stream."""
    def __init__(self, pretrained=True, dropout=0.5):
        super().__init__()

        resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)

        # Modify first conv for optical flow (10 channels: 5 flows * 2)
        original_conv = resnet.conv1
        self.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if pretrained:
            with torch.no_grad():
                # Initialize from pretrained RGB weights
                self.conv1.weight[:, :3, :, :] = original_conv.weight / 3.0
                self.conv1.weight[:, 3:6, :, :] = original_conv.weight / 3.0
                self.conv1.weight[:, 6:9, :, :] = original_conv.weight / 3.0
                self.conv1.weight[:, 9:, :, :] = original_conv.weight[:, :1, :, :].repeat(1, 1, 1, 1)

        self.backbone = nn.Sequential(self.conv1, *list(resnet.children())[1:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (batch, 10, H, W) -> (batch, 2048)"""
        features = self.backbone(x)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        return self.dropout(features)


class CrossTaskAttention(nn.Module):
    """
    Cross-task attention between verb and noun.
    Allows verb features to attend to noun features and vice versa.
    """
    def __init__(self, feature_dim=512, num_heads=8, dropout=0.3):
        super().__init__()

        # Multi-head cross-attention: verb queries noun, noun queries verb
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

        # Feed-forward
        self.ff_verb = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(dropout)
        )

        self.ff_noun = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
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
            verb_enhanced: (batch, feature_dim)
            noun_enhanced: (batch, feature_dim)
        """
        # Add sequence dimension for attention
        verb_feat = verb_features.unsqueeze(1)  # (batch, 1, feature_dim)
        noun_feat = noun_features.unsqueeze(1)  # (batch, 1, feature_dim)

        # Cross-attention: verb attends to noun
        verb_attn_out, _ = self.verb_to_noun_attn(
            query=verb_feat,
            key=noun_feat,
            value=noun_feat
        )
        verb_feat = self.norm_verb(verb_feat + verb_attn_out)

        # Cross-attention: noun attends to verb
        noun_attn_out, _ = self.noun_to_verb_attn(
            query=noun_feat,
            key=verb_feat,
            value=verb_feat
        )
        noun_feat = self.norm_noun(noun_feat + noun_attn_out)

        # Feed-forward
        verb_enhanced = verb_feat.squeeze(1) + self.ff_verb(verb_feat.squeeze(1))
        noun_enhanced = noun_feat.squeeze(1) + self.ff_noun(noun_feat.squeeze(1))

        return verb_enhanced, noun_enhanced


class TwoStreamCrossTaskModel(nn.Module):
    """
    Two-Stream Model with Cross-Task Interaction:
    1. Spatial stream (RGB) + Temporal stream (Optical Flow)
    2. Cross-task attention between verb and noun
    3. Better action understanding
    """
    def __init__(self, num_verb_classes=97, num_noun_classes=300,
                 num_frames=8, pretrained=True, dropout=0.5, drop_path=0.1):
        super().__init__()

        print("="*70)
        print("Creating Two-Stream Cross-Task Model")
        print("  - RGB + Optical Flow (motion)")
        print("  - Verb ↔ Noun interaction (cross-attention)")
        print("="*70)

        # Two streams
        self.spatial_stream = SpatialStream(pretrained=pretrained, dropout=dropout)
        self.temporal_stream = TemporalStream(pretrained=pretrained, dropout=dropout)

        # Stochastic depth
        self.drop_path = DropPath(drop_path)

        # Fusion layer (combine RGB + Flow)
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Task-specific feature extraction
        self.verb_feature_extractor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.noun_feature_extractor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Cross-task attention (verb and noun interact!)
        self.cross_task_attention = CrossTaskAttention(
            feature_dim=512,
            num_heads=8,
            dropout=dropout
        )

        # Final classifiers (after cross-task interaction)
        self.verb_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_verb_classes)
        )

        self.noun_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_noun_classes)
        )

        # Initialize weights
        self._initialize_weights()

        print(f"✓ Spatial stream:  ResNet-50 (RGB)")
        print(f"✓ Temporal stream: ResNet-50 (Optical Flow)")
        print(f"✓ Cross-task attention: Verb ↔ Noun")
        print(f"✓ Dropout: {dropout}, Drop path: {drop_path}")
        print("="*70)

    def _initialize_weights(self):
        """Initialize classifier weights."""
        for m in [self.fusion, self.verb_feature_extractor, self.noun_feature_extractor,
                  self.verb_classifier, self.noun_classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(self, rgb_frames, flow_frames):
        """
        Args:
            rgb_frames: (batch, num_frames, 3, H, W)
            flow_frames: (batch, 10, H, W)

        Returns:
            verb_logits: (batch, num_verb_classes)
            noun_logits: (batch, num_noun_classes)
        """
        # Extract features from both streams
        spatial_features = self.spatial_stream(rgb_frames)
        temporal_features = self.temporal_stream(flow_frames)

        # Apply stochastic depth
        spatial_features = self.drop_path(spatial_features)
        temporal_features = self.drop_path(temporal_features)

        # Fuse RGB + Flow features
        combined = torch.cat([spatial_features, temporal_features], dim=1)
        fused_features = self.fusion(combined)  # (batch, 1024)

        # Extract task-specific features (before interaction)
        verb_features = self.verb_feature_extractor(fused_features)  # (batch, 512)
        noun_features = self.noun_feature_extractor(fused_features)  # (batch, 512)

        # Cross-task interaction (KEY INNOVATION!)
        # Verb and noun features interact to understand action better
        verb_enhanced, noun_enhanced = self.cross_task_attention(
            verb_features, noun_features
        )

        # Final classification (after interaction)
        verb_logits = self.verb_classifier(verb_enhanced)
        noun_logits = self.noun_classifier(noun_enhanced)

        return verb_logits, noun_logits


def get_twostream_crosstask_model(config):
    """Factory function."""
    model = TwoStreamCrossTaskModel(
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
    print("\\nTesting Two-Stream Cross-Task Model...")

    model = TwoStreamCrossTaskModel(
        num_verb_classes=97,
        num_noun_classes=300,
        num_frames=8
    )

    # Dummy inputs
    rgb_frames = torch.randn(2, 8, 3, 224, 224)
    flow_frames = torch.randn(2, 10, 224, 224)

    # Forward pass
    verb_logits, noun_logits = model(rgb_frames, flow_frames)

    print(f"\\nOutput shapes:")
    print(f"  Verb logits: {verb_logits.shape}")
    print(f"  Noun logits: {noun_logits.shape}")
    print(f"\\n✓ Cross-Task Two-Stream model works!")
    print(f"\\nKey feature: Verb and noun predictions interact via attention")
    print(f"  → 'cut' verb can attend to 'knife' noun features")
    print(f"  → 'pour' verb can attend to 'liquid' noun features")
    print(f"  → Better cohesive action understanding!")
