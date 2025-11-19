"""
LSTM-based Action Recognition Model for EPIC-KITCHENS-100
Uses ResNet-50 for frame features + LSTM for temporal modeling
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ActionRecognitionLSTM(nn.Module):
    """
    ResNet-50 + LSTM for temporal modeling.

    Better than average pooling because:
    - Models temporal dependencies (frame order matters)
    - Can capture long-term patterns
    - Learns which frames are important
    """

    def __init__(self, num_verb_classes=97, num_noun_classes=300, num_frames=8, pretrained=True):
        super().__init__()

        self.num_frames = num_frames

        # ResNet-50 backbone
        resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        feature_dim = 2048

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        )

        # Feature dimension after LSTM
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
        features = features.view(batch_size * num_frames, -1)

        # Reshape for LSTM: (batch, num_frames, feature_dim)
        features = features.view(batch_size, num_frames, -1)

        # LSTM temporal modeling
        lstm_out, (hidden, cell) = self.lstm(features)

        # Take final hidden state
        features = hidden[-1]  # (batch_size, 512)
        features = self.dropout(features)

        # Classify
        verb_logits = self.verb_classifier(features)
        noun_logits = self.noun_classifier(features)

        return verb_logits, noun_logits


def get_model(config):
    """Create LSTM-based model."""
    print("=" * 70)
    print("Creating Model: ResNet-50 + LSTM")
    print("=" * 70)

    model = ActionRecognitionLSTM(
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
