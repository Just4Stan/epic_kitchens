"""
Action Recognition Model for EPIC-KITCHENS-100
Two-stream model with verb and noun classification heads
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ActionRecognitionModel(nn.Module):
    """
    Baseline action recognition model.

    Architecture:
    1. ResNet-50 backbone (per-frame feature extraction)
    2. Temporal pooling (average across frames)
    3. Two classification heads:
       - Verb head (97 classes)
       - Noun head (300 classes)
    """

    def __init__(self, num_verb_classes=97, num_noun_classes=300, num_frames=8, pretrained=True):
        """
        Args:
            num_verb_classes: Number of verb classes (97)
            num_noun_classes: Number of noun classes (300)
            num_frames: Number of frames per video segment
            pretrained: Use ImageNet pre-trained weights
        """
        super().__init__()

        self.num_frames = num_frames

        # ResNet-50 backbone (remove final FC layer)
        resnet = models.resnet50(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC
        self.feature_dim = 2048  # ResNet-50 output dimension

        # Freeze early layers (optional - can fine-tune for better results)
        # Uncomment to freeze:
        # for param in list(self.feature_extractor.parameters())[:-20]:
        #     param.requires_grad = False

        # Temporal pooling (average across frames)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Verb classification head
        self.verb_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_verb_classes)
        )

        # Noun classification head
        self.noun_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_noun_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, num_frames, 3, H, W)

        Returns:
            verb_logits: Tensor (batch_size, num_verb_classes)
            noun_logits: Tensor (batch_size, num_noun_classes)
        """
        batch_size, num_frames, C, H, W = x.shape

        # Reshape to process all frames together
        # (batch_size * num_frames, 3, H, W)
        x = x.view(batch_size * num_frames, C, H, W)

        # Extract features with ResNet
        # (batch_size * num_frames, feature_dim, 1, 1)
        features = self.feature_extractor(x)

        # Remove spatial dimensions
        # (batch_size * num_frames, feature_dim)
        features = features.view(batch_size * num_frames, self.feature_dim)

        # Reshape to (batch_size, num_frames, feature_dim)
        features = features.view(batch_size, num_frames, self.feature_dim)

        # Temporal pooling: average across frames
        # (batch_size, feature_dim, num_frames) -> (batch_size, feature_dim, 1)
        features = features.permute(0, 2, 1)  # (batch, feature_dim, num_frames)
        features = self.temporal_pool(features).squeeze(-1)  # (batch, feature_dim)

        # Apply dropout
        features = self.dropout(features)

        # Classify verb and noun
        verb_logits = self.verb_classifier(features)
        noun_logits = self.noun_classifier(features)

        return verb_logits, noun_logits


class ActionRecognitionModel3D(nn.Module):
    """
    3D CNN-based action recognition model (better for temporal modeling).

    Uses ResNet3D-18 for spatiotemporal feature extraction.
    """

    def __init__(self, num_verb_classes=97, num_noun_classes=300, pretrained=True):
        """
        Args:
            num_verb_classes: Number of verb classes (97)
            num_noun_classes: Number of noun classes (300)
            pretrained: Use Kinetics pre-trained weights
        """
        super().__init__()

        # Load pre-trained 3D ResNet (if available)
        # Note: Requires torchvision >= 0.15.0 with video models
        try:
            from torchvision.models.video import r3d_18, R3D_18_Weights
            if pretrained:
                self.backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            else:
                self.backbone = r3d_18(weights=None)

            # Remove final FC layer
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        except ImportError:
            raise ImportError("3D ResNet requires torchvision >= 0.15.0")

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Verb classification head
        self.verb_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_verb_classes)
        )

        # Noun classification head
        self.noun_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_noun_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, num_frames, 3, H, W)

        Returns:
            verb_logits: Tensor (batch_size, num_verb_classes)
            noun_logits: Tensor (batch_size, num_noun_classes)
        """
        batch_size, num_frames, C, H, W = x.shape

        # Reshape for 3D ResNet: (batch_size, 3, num_frames, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # Extract spatiotemporal features
        features = self.backbone(x)  # (batch_size, feature_dim)

        # Apply dropout
        features = self.dropout(features)

        # Classify verb and noun
        verb_logits = self.verb_classifier(features)
        noun_logits = self.noun_classifier(features)

        return verb_logits, noun_logits


def get_model(config, use_3d=False):
    """
    Create and return the model.

    Args:
        config: Configuration object
        use_3d: Use 3D CNN (slower but better temporal modeling)

    Returns:
        model: PyTorch model
    """
    print("=" * 70)
    print("Creating Model")
    print("=" * 70)

    if use_3d:
        print("Using 3D ResNet-18 (spatiotemporal)")
        model = ActionRecognitionModel3D(
            num_verb_classes=config.NUM_VERB_CLASSES,
            num_noun_classes=config.NUM_NOUN_CLASSES,
            pretrained=True
        )
    else:
        print("Using 2D ResNet-50 + Temporal Pooling (baseline)")
        model = ActionRecognitionModel(
            num_verb_classes=config.NUM_VERB_CLASSES,
            num_noun_classes=config.NUM_NOUN_CLASSES,
            num_frames=config.NUM_FRAMES,
            pretrained=True
        )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 70)
    print()

    return model


# Test the model
if __name__ == "__main__":
    print("Testing ActionRecognitionModel...")

    # Create dummy input
    batch_size = 4
    num_frames = 8
    x = torch.randn(batch_size, num_frames, 3, 224, 224)

    # 2D model
    print("\n2D ResNet-50 Model:")
    model = ActionRecognitionModel(num_verb_classes=97, num_noun_classes=300, num_frames=8)
    verb_logits, noun_logits = model(x)

    print(f"Input shape:       {x.shape}")
    print(f"Verb logits shape: {verb_logits.shape}")  # (4, 97)
    print(f"Noun logits shape: {noun_logits.shape}")  # (4, 300)

    # 3D model
    print("\n3D ResNet-18 Model:")
    try:
        model_3d = ActionRecognitionModel3D(num_verb_classes=97, num_noun_classes=300)
        verb_logits, noun_logits = model_3d(x)

        print(f"Input shape:       {x.shape}")
        print(f"Verb logits shape: {verb_logits.shape}")  # (4, 97)
        print(f"Noun logits shape: {noun_logits.shape}")  # (4, 300)
    except ImportError as e:
        print(f"3D model not available: {e}")

    print("\n✓ Model test passed!")
