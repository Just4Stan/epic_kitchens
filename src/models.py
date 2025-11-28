"""
EPIC-KITCHENS-100 Action Recognition - Models
==============================================
Neural network architectures for action recognition.
"""

import torch
import torch.nn as nn
from torchvision import models
import copy


# =============================================================================
# TEMPORAL MODELS
# =============================================================================

class TemporalTransformer(nn.Module):
    """Transformer encoder for temporal modeling."""

    def __init__(self, d_model=512, nhead=8, num_layers=2, dropout=0.1, max_frames=16):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, max_frames + 1, d_model))

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.transformer(x)
        return x[:, 0]


class TemporalLSTM(nn.Module):
    """Bidirectional LSTM for temporal modeling."""

    def __init__(self, d_model=512, hidden_dim=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, d_model)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(hidden)


class TemporalMeanPool(nn.Module):
    """Simple mean pooling over time."""

    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x.mean(dim=1))


# =============================================================================
# MAIN MODEL
# =============================================================================

class ActionModel(nn.Module):
    """
    Action Recognition Model

    Architecture:
    - Backbone (ResNet50/EfficientNet) extracts per-frame features
    - Feature projection to 512-dim
    - Temporal model (LSTM/Transformer/MeanPool)
    - Separate verb and noun classification heads
    """

    BACKBONES = {
        'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
        'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
        'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
        'efficientnet_b3': (models.efficientnet_b3, models.EfficientNet_B3_Weights.IMAGENET1K_V1, 1536),
    }

    def __init__(self, num_verb_classes=97, num_noun_classes=300,
                 backbone='resnet50', temporal_model='lstm',
                 dropout=0.5, num_frames=16, freeze_backbone='none'):
        super().__init__()

        # Load backbone
        if backbone not in self.BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone}. Choose from {list(self.BACKBONES.keys())}")

        model_fn, weights, self.feature_dim = self.BACKBONES[backbone]
        base_model = model_fn(weights=weights)

        if 'resnet' in backbone:
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        else:  # efficientnet
            base_model.classifier = nn.Identity()
            self.backbone = base_model

        # Freeze if requested
        self.freeze_backbone = freeze_backbone
        self._apply_freezing()

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Temporal model
        if temporal_model == 'transformer':
            self.temporal = TemporalTransformer(d_model=512, dropout=dropout*0.5, max_frames=num_frames)
        elif temporal_model == 'lstm':
            self.temporal = TemporalLSTM(d_model=512, dropout=dropout*0.5)
        elif temporal_model == 'mean':
            self.temporal = TemporalMeanPool(d_model=512, dropout=dropout*0.5)
        else:
            raise ValueError(f"Unknown temporal model: {temporal_model}")

        # Classification heads
        self.verb_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_verb_classes),
        )

        self.noun_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_noun_classes),
        )

    def _apply_freezing(self):
        """Freeze backbone layers based on strategy."""
        if self.freeze_backbone == 'all':
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif self.freeze_backbone == 'early':
            for layer in list(self.backbone.children())[:7]:
                for param in layer.parameters():
                    param.requires_grad = False
        elif self.freeze_backbone == 'bn':
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone in ['bn', 'all']:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        features = self.backbone(x)
        features = features.view(B * T, -1)
        features = self.feature_proj(features)
        features = features.view(B, T, -1)

        temporal_features = self.temporal(features)

        verb_logits = self.verb_head(temporal_features)
        noun_logits = self.noun_head(temporal_features)

        return verb_logits, noun_logits


# =============================================================================
# LOSSES
# =============================================================================

class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing."""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = nn.functional.log_softmax(pred, dim=1)
        return -(one_hot * log_prob).sum(dim=1).mean()


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_state = None

    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
