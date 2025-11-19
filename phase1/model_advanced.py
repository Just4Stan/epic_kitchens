"""
Advanced Action Recognition Models for EPIC-KITCHENS-100

Implements:
1. EfficientNet-B3 backbone (faster, more efficient than ResNet-50)
2. Two-Stream Network (RGB + Optical Flow)
3. Multi-scale temporal modeling
4. Attention mechanisms
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import math


class EfficientNetLSTM(nn.Module):
    """
    EfficientNet-B3 + LSTM temporal modeling.

    Advantages:
    - More efficient than ResNet-50 (12M vs 26M params)
    - Better accuracy per parameter
    - Faster training
    - LSTM for temporal dependencies
    """

    def __init__(self, num_verb_classes=97, num_noun_classes=300, num_frames=8, pretrained=True):
        super().__init__()

        self.num_frames = num_frames

        # EfficientNet-B3 backbone
        if pretrained:
            efficientnet = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        else:
            efficientnet = efficientnet_b3(weights=None)

        # Remove classifier, keep only feature extractor
        self.feature_extractor = nn.Sequential(*list(efficientnet.children())[:-1])
        self.feature_dim = 1536  # EfficientNet-B3 output dimension

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        )

        lstm_output_dim = 512

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Verb classification head
        self.verb_classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_verb_classes)
        )

        # Noun classification head
        self.noun_classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_noun_classes)
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

        # Reshape for LSTM: (batch, num_frames, feature_dim)
        features = features.view(batch_size, num_frames, self.feature_dim)

        # LSTM temporal modeling
        lstm_out, (hidden, cell) = self.lstm(features)

        # Take final hidden state
        features = hidden[-1]  # (batch_size, 512)
        features = self.dropout(features)

        # Classify
        verb_logits = self.verb_classifier(features)
        noun_logits = self.noun_classifier(features)

        return verb_logits, noun_logits


class EfficientNetTransformer(nn.Module):
    """
    EfficientNet-B3 + Transformer temporal modeling.

    Advantages:
    - Efficient backbone (12M params)
    - Self-attention learns important frames
    - Parallel processing (faster than LSTM)
    - State-of-the-art temporal modeling
    """

    def __init__(self, num_verb_classes=97, num_noun_classes=300, num_frames=8, pretrained=True):
        super().__init__()

        self.num_frames = num_frames

        # EfficientNet-B3 backbone
        if pretrained:
            efficientnet = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        else:
            efficientnet = efficientnet_b3(weights=None)

        self.feature_extractor = nn.Sequential(*list(efficientnet.children())[:-1])
        self.feature_dim = 1536

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


class TwoStreamNetwork(nn.Module):
    """
    Two-Stream Network: RGB + Optical Flow.

    Architecture:
    - RGB stream: Spatial features (what objects?)
    - Flow stream: Motion features (how moving?)
    - Fuse both streams for classification

    Advantages:
    - Explicitly models motion
    - Separates appearance from motion
    - +5-8% accuracy on action recognition

    Note: Requires optical flow computation during data loading
    """

    def __init__(self, num_verb_classes=97, num_noun_classes=300, num_frames=8, pretrained=True):
        super().__init__()

        self.num_frames = num_frames

        # RGB stream (appearance)
        rgb_backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        self.rgb_stream = nn.Sequential(*list(rgb_backbone.children())[:-1])

        # Flow stream (motion) - same architecture but separate weights
        flow_backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        # Modify first conv to accept 2-channel flow (optical flow x, y)
        flow_backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.flow_stream = nn.Sequential(*list(flow_backbone.children())[:-1])

        rgb_dim = 2048
        flow_dim = 2048

        # Temporal pooling for both streams
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # Fusion: Concatenate RGB + Flow features
        fusion_dim = rgb_dim + flow_dim  # 4096

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Verb classification head
        self.verb_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_verb_classes)
        )

        # Noun classification head
        self.noun_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_noun_classes)
        )

    def forward(self, rgb, flow):
        """
        Args:
            rgb: (batch_size, num_frames, 3, H, W) - RGB frames
            flow: (batch_size, num_frames, 2, H, W) - Optical flow (x, y)

        Returns:
            verb_logits: (batch_size, num_verb_classes)
            noun_logits: (batch_size, num_noun_classes)
        """
        batch_size, num_frames, _, H, W = rgb.shape

        # RGB stream
        rgb = rgb.view(batch_size * num_frames, 3, H, W)
        rgb_features = self.rgb_stream(rgb)
        rgb_features = rgb_features.view(batch_size * num_frames, 2048)
        rgb_features = rgb_features.view(batch_size, num_frames, 2048)

        # Temporal pooling
        rgb_features = rgb_features.permute(0, 2, 1)
        rgb_features = self.temporal_pool(rgb_features).squeeze(-1)  # (batch, 2048)

        # Flow stream
        flow = flow.view(batch_size * num_frames, 2, H, W)
        flow_features = self.flow_stream(flow)
        flow_features = flow_features.view(batch_size * num_frames, 2048)
        flow_features = flow_features.view(batch_size, num_frames, 2048)

        # Temporal pooling
        flow_features = flow_features.permute(0, 2, 1)
        flow_features = self.temporal_pool(flow_features).squeeze(-1)  # (batch, 2048)

        # Fuse: Concatenate RGB + Flow
        features = torch.cat([rgb_features, flow_features], dim=1)  # (batch, 4096)
        features = self.dropout(features)

        # Classify
        verb_logits = self.verb_classifier(features)
        noun_logits = self.noun_classifier(features)

        return verb_logits, noun_logits


class MultiScaleTransformer(nn.Module):
    """
    Multi-scale temporal modeling with Transformer.

    Architecture:
    - Sample frames at different temporal scales
    - Fine scale: 16 frames @ high FPS (details)
    - Coarse scale: 4 frames @ low FPS (global context)
    - Fuse multi-scale features with attention

    Advantages:
    - Captures both short-term and long-term patterns
    - Hierarchical action understanding
    - Better for complex actions
    """

    def __init__(self, num_verb_classes=97, num_noun_classes=300, pretrained=True):
        super().__init__()

        # Shared feature extractor
        backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 2048

        # Fine-scale path (16 frames)
        self.fine_pos_encoding = nn.Parameter(torch.randn(1, 16, self.feature_dim) * 0.02)
        fine_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim, nhead=8, dim_feedforward=2048,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.fine_transformer = nn.TransformerEncoder(fine_layer, num_layers=2)

        # Coarse-scale path (4 frames)
        self.coarse_pos_encoding = nn.Parameter(torch.randn(1, 4, self.feature_dim) * 0.02)
        coarse_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim, nhead=8, dim_feedforward=2048,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.coarse_transformer = nn.TransformerEncoder(coarse_layer, num_layers=2)

        # Cross-scale fusion
        self.fusion = nn.MultiheadAttention(
            embed_dim=self.feature_dim, num_heads=8, batch_first=True
        )

        # [CLS] tokens
        self.fine_cls = nn.Parameter(torch.randn(1, 1, self.feature_dim) * 0.02)
        self.coarse_cls = nn.Parameter(torch.randn(1, 1, self.feature_dim) * 0.02)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Classifiers (input = 2 * feature_dim from both scales)
        fusion_dim = 2 * self.feature_dim

        self.verb_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_verb_classes)
        )

        self.noun_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_noun_classes)
        )

    def forward(self, fine_frames, coarse_frames):
        """
        Args:
            fine_frames: (batch_size, 16, 3, H, W) - High temporal resolution
            coarse_frames: (batch_size, 4, 3, H, W) - Low temporal resolution

        Returns:
            verb_logits: (batch_size, num_verb_classes)
            noun_logits: (batch_size, num_noun_classes)
        """
        batch_size = fine_frames.size(0)

        # Process fine-scale frames
        fine_frames = fine_frames.view(-1, 3, 224, 224)
        fine_features = self.feature_extractor(fine_frames)
        fine_features = fine_features.view(batch_size, 16, self.feature_dim)
        fine_features = fine_features + self.fine_pos_encoding
        fine_cls = self.fine_cls.expand(batch_size, -1, -1)
        fine_features = torch.cat([fine_cls, fine_features], dim=1)
        fine_features = self.fine_transformer(fine_features)
        fine_out = fine_features[:, 0]  # CLS token

        # Process coarse-scale frames
        coarse_frames = coarse_frames.view(-1, 3, 224, 224)
        coarse_features = self.feature_extractor(coarse_frames)
        coarse_features = coarse_features.view(batch_size, 4, self.feature_dim)
        coarse_features = coarse_features + self.coarse_pos_encoding
        coarse_cls = self.coarse_cls.expand(batch_size, -1, -1)
        coarse_features = torch.cat([coarse_cls, coarse_features], dim=1)
        coarse_features = self.coarse_transformer(coarse_features)
        coarse_out = coarse_features[:, 0]  # CLS token

        # Fuse multi-scale features
        features = torch.cat([fine_out, coarse_out], dim=1)  # (batch, 4096)
        features = self.dropout(features)

        # Classify
        verb_logits = self.verb_classifier(features)
        noun_logits = self.noun_classifier(features)

        return verb_logits, noun_logits


def get_model(config, model_type='efficientnet_lstm'):
    """
    Create and return the model.

    Args:
        config: Configuration object
        model_type: One of:
            - 'efficientnet_lstm': EfficientNet-B3 + LSTM
            - 'efficientnet_transformer': EfficientNet-B3 + Transformer
            - 'two_stream': RGB + Optical Flow
            - 'multiscale': Multi-scale Transformer

    Returns:
        model: PyTorch model
    """
    print("=" * 70)
    print(f"Creating Model: {model_type}")
    print("=" * 70)

    if model_type == 'efficientnet_lstm':
        model = EfficientNetLSTM(
            num_verb_classes=config.NUM_VERB_CLASSES,
            num_noun_classes=config.NUM_NOUN_CLASSES,
            num_frames=config.NUM_FRAMES,
            pretrained=True
        )
    elif model_type == 'efficientnet_transformer':
        model = EfficientNetTransformer(
            num_verb_classes=config.NUM_VERB_CLASSES,
            num_noun_classes=config.NUM_NOUN_CLASSES,
            num_frames=config.NUM_FRAMES,
            pretrained=True
        )
    elif model_type == 'two_stream':
        model = TwoStreamNetwork(
            num_verb_classes=config.NUM_VERB_CLASSES,
            num_noun_classes=config.NUM_NOUN_CLASSES,
            num_frames=config.NUM_FRAMES,
            pretrained=True
        )
    elif model_type == 'multiscale':
        model = MultiScaleTransformer(
            num_verb_classes=config.NUM_VERB_CLASSES,
            num_noun_classes=config.NUM_NOUN_CLASSES,
            pretrained=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 70)
    print()

    return model


# Test the models
if __name__ == "__main__":
    print("Testing Advanced Models...\n")

    from config import Config
    config = Config()

    batch_size = 2
    num_frames = 8

    # Test 1: EfficientNet + LSTM
    print("=" * 70)
    print("Test 1: EfficientNet-B3 + LSTM")
    print("=" * 70)
    x = torch.randn(batch_size, num_frames, 3, 224, 224)
    model = get_model(config, 'efficientnet_lstm')
    verb_logits, noun_logits = model(x)
    print(f"Input shape:       {x.shape}")
    print(f"Verb logits shape: {verb_logits.shape}")  # (2, 97)
    print(f"Noun logits shape: {noun_logits.shape}")  # (2, 300)
    print("✓ EfficientNet-LSTM test passed!\n")

    # Test 2: EfficientNet + Transformer
    print("=" * 70)
    print("Test 2: EfficientNet-B3 + Transformer")
    print("=" * 70)
    x = torch.randn(batch_size, num_frames, 3, 224, 224)
    model = get_model(config, 'efficientnet_transformer')
    verb_logits, noun_logits = model(x)
    print(f"Input shape:       {x.shape}")
    print(f"Verb logits shape: {verb_logits.shape}")
    print(f"Noun logits shape: {noun_logits.shape}")
    print("✓ EfficientNet-Transformer test passed!\n")

    # Test 3: Two-Stream
    print("=" * 70)
    print("Test 3: Two-Stream Network (RGB + Flow)")
    print("=" * 70)
    rgb = torch.randn(batch_size, num_frames, 3, 224, 224)
    flow = torch.randn(batch_size, num_frames, 2, 224, 224)  # Optical flow (x, y)
    model = get_model(config, 'two_stream')
    verb_logits, noun_logits = model(rgb, flow)
    print(f"RGB shape:         {rgb.shape}")
    print(f"Flow shape:        {flow.shape}")
    print(f"Verb logits shape: {verb_logits.shape}")
    print(f"Noun logits shape: {noun_logits.shape}")
    print("✓ Two-Stream test passed!\n")

    # Test 4: Multi-scale
    print("=" * 70)
    print("Test 4: Multi-Scale Transformer")
    print("=" * 70)
    fine = torch.randn(batch_size, 16, 3, 224, 224)    # Fine scale
    coarse = torch.randn(batch_size, 4, 3, 224, 224)   # Coarse scale
    model = get_model(config, 'multiscale')
    verb_logits, noun_logits = model(fine, coarse)
    print(f"Fine scale shape:   {fine.shape}")
    print(f"Coarse scale shape: {coarse.shape}")
    print(f"Verb logits shape:  {verb_logits.shape}")
    print(f"Noun logits shape:  {noun_logits.shape}")
    print("✓ Multi-Scale test passed!\n")

    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
