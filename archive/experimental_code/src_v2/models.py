"""
EPIC-KITCHENS-100 Action Recognition - V2 Models
=================================================
Two-Stream architecture with CLIP spatial + LSTM temporal.
Includes FocalLoss for long-tail and CosineClassifier for nouns.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    From "Focal Loss for Dense Object Detection" (Lin et al., ICCV 2017)

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Per-class weights (optional)
        smoothing: Label smoothing factor
    """

    def __init__(self, gamma=2.0, alpha=None, smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        n_classes = logits.size(1)

        # Compute probabilities
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, n_classes).float()

        # Apply label smoothing
        if self.smoothing > 0:
            targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        else:
            targets_smooth = targets_one_hot

        # Get probability of true class
        p_t = (probs * targets_one_hot).sum(dim=1)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Cross-entropy with label smoothing
        log_probs = F.log_softmax(logits, dim=1)
        ce_loss = -(targets_smooth * log_probs).sum(dim=1)

        # Apply focal weight
        loss = focal_weight * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        return loss.mean()


class LabelSmoothingLoss(nn.Module):
    """Standard cross-entropy with label smoothing."""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        one_hot = F.one_hot(target, n_classes).float()
        smooth_target = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        log_probs = F.log_softmax(pred, dim=1)
        return -(smooth_target * log_probs).sum(dim=1).mean()


# =============================================================================
# CLASSIFICATION HEADS
# =============================================================================

class CosineClassifier(nn.Module):
    """
    Cosine similarity classifier with temperature scaling.
    Better for handling class imbalance than standard linear classifier.

    Args:
        in_features: Input dimension
        num_classes: Number of output classes
        temperature: Scaling factor (lower = sharper distribution)
    """

    def __init__(self, in_features, num_classes, temperature=0.05):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        self.temperature = temperature
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # L2 normalize features and weights
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity scaled by temperature
        logits = F.linear(x_norm, w_norm) / self.temperature
        return logits


class LinearClassifier(nn.Module):
    """Standard MLP classification head."""

    def __init__(self, in_features, num_classes, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.head(x)


# =============================================================================
# TEMPORAL MODELS
# =============================================================================

class TemporalLSTM(nn.Module):
    """Bidirectional LSTM for temporal modeling."""

    def __init__(self, input_dim, hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.output_dim = hidden_dim * 2  # Bidirectional

    def forward(self, x):
        # x: (B, T, D)
        output, (h_n, _) = self.lstm(x)
        # Use last hidden states from both directions
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return hidden, output


class TemporalTransformer(nn.Module):
    """Transformer encoder for temporal modeling."""

    def __init__(self, input_dim, num_heads=8, num_layers=2, dropout=0.1, max_frames=32):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, max_frames + 1, input_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = input_dim

    def forward(self, x):
        B, T, D = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed[:, :T+1, :]
        x = self.transformer(x)
        return x[:, 0], x[:, 1:]  # CLS token, sequence


# =============================================================================
# CROSS-ATTENTION FUSION
# =============================================================================

class CrossAttention(nn.Module):
    """
    Bidirectional cross-attention for fusing spatial and temporal features.
    Inspired by CAST (Cross-Attention in Space and Time).
    """

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Spatial to Temporal attention
        self.s2t_q = nn.Linear(dim, dim)
        self.s2t_k = nn.Linear(dim, dim)
        self.s2t_v = nn.Linear(dim, dim)

        # Temporal to Spatial attention
        self.t2s_q = nn.Linear(dim, dim)
        self.t2s_k = nn.Linear(dim, dim)
        self.t2s_v = nn.Linear(dim, dim)

        self.proj = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, spatial_feat, temporal_feat):
        """
        Args:
            spatial_feat: (B, D) - CLIP features
            temporal_feat: (B, D) - LSTM features

        Returns:
            fused: (B, D) - Fused features
        """
        B = spatial_feat.size(0)

        # Expand to sequence for attention
        s = spatial_feat.unsqueeze(1)  # (B, 1, D)
        t = temporal_feat.unsqueeze(1)  # (B, 1, D)

        # S2T: Temporal attends to Spatial
        q = self.t2s_q(t)
        k = self.s2t_k(s)
        v = self.s2t_v(s)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        t2s_out = (attn @ v).squeeze(1)
        t_enhanced = self.norm1(temporal_feat + t2s_out)

        # T2S: Spatial attends to Temporal
        q = self.s2t_q(s)
        k = self.t2s_k(t)
        v = self.t2s_v(t)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        s2t_out = (attn @ v).squeeze(1)
        s_enhanced = self.norm2(spatial_feat + s2t_out)

        # Fuse both enhanced features
        fused = self.proj(torch.cat([s_enhanced, t_enhanced], dim=-1))
        return fused


# =============================================================================
# TWO-STREAM MODEL
# =============================================================================

class TwoStreamModel(nn.Module):
    """
    Two-Stream Action Recognition Model (CAST-inspired).

    Architecture:
    - Spatial Stream: CLIP ViT-B/16 (frozen) for spatial understanding
    - Temporal Stream: Bidirectional LSTM for motion/temporal patterns
    - Fusion: Cross-attention between spatial and temporal features
    - Heads: Cosine classifier (nouns), Linear (verbs)

    Only trains: feature projection, LSTM, cross-attention, classification heads
    """

    def __init__(
        self,
        num_verb_classes=97,
        num_noun_classes=300,
        clip_dim=512,  # CLIP ViT-B/16 outputs 512 dims
        proj_dim=512,
        lstm_hidden=512,
        num_frames=32,
        dropout=0.3,
        use_cross_attention=True,
        use_cosine_noun=True,
        cosine_temp=0.05,
        freeze_clip=True
    ):
        super().__init__()
        self.num_frames = num_frames
        self.use_cross_attention = use_cross_attention
        self.use_cosine_noun = use_cosine_noun
        self.freeze_clip = freeze_clip

        # ======================
        # Spatial Stream (CLIP)
        # ======================
        try:
            import clip
            self.clip_model, _ = clip.load("ViT-B/16", device="cpu")
            self.clip_model = self.clip_model.visual
            self.has_clip = True
            print("Loaded CLIP ViT-B/16 for spatial stream")
        except ImportError:
            print("CLIP not available, using ResNet50 for spatial stream")
            self.clip_model = None
            self.has_clip = False
            # Fallback to ResNet50
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])
            clip_dim = 2048

        if freeze_clip and self.has_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Spatial feature projection
        self.spatial_proj = nn.Sequential(
            nn.Linear(clip_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # Temporal feature projection (LSTM outputs 2*hidden_dim due to bidirectionality)
        self.temporal_proj = nn.Sequential(
            nn.Linear(lstm_hidden * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # ======================
        # Temporal Stream (LSTM)
        # ======================
        self.temporal_model = TemporalLSTM(
            input_dim=proj_dim,
            hidden_dim=lstm_hidden,
            num_layers=2,
            dropout=dropout
        )

        # ======================
        # Fusion
        # ======================
        if use_cross_attention:
            self.cross_attention = CrossAttention(
                dim=proj_dim,
                num_heads=8,
                dropout=dropout
            )
            fusion_dim = proj_dim
        else:
            # Simple concatenation
            fusion_dim = proj_dim * 2

        # ======================
        # Classification Heads
        # ======================
        self.verb_head = LinearClassifier(
            fusion_dim, num_verb_classes,
            hidden_dim=proj_dim, dropout=dropout
        )

        if use_cosine_noun:
            # Cosine classifier for nouns (better for long-tail)
            self.noun_proj = nn.Sequential(
                nn.Linear(fusion_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.noun_head = CosineClassifier(
                proj_dim, num_noun_classes,
                temperature=cosine_temp
            )
        else:
            self.noun_proj = nn.Identity()
            self.noun_head = LinearClassifier(
                fusion_dim, num_noun_classes,
                hidden_dim=proj_dim, dropout=dropout
            )

    def extract_spatial_features(self, x):
        """Extract per-frame spatial features using CLIP or ResNet."""
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        if self.has_clip:
            # CLIP expects 224x224 images - resize if needed
            if H != 224 or W != 224:
                x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            # CLIP expects normalized images
            if self.freeze_clip:
                with torch.no_grad():
                    features = self.clip_model(x)
            else:
                # Trainable CLIP (no checkpointing - it causes backward OOM)
                features = self.clip_model(x)
        else:
            # ResNet fallback
            features = self.resnet_backbone(x).squeeze(-1).squeeze(-1)

        features = features.float()  # CLIP outputs float16
        features = self.spatial_proj(features)
        features = features.view(B, T, -1)
        return features

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - Video frames

        Returns:
            verb_logits: (B, num_verbs)
            noun_logits: (B, num_nouns)
        """
        # Extract spatial features for each frame
        spatial_features = self.extract_spatial_features(x)  # (B, T, D)

        # Global spatial feature (mean pool across time)
        spatial_global = spatial_features.mean(dim=1)  # (B, D)

        # Temporal modeling
        temporal_global, _ = self.temporal_model(spatial_features)  # (B, D*2)

        # Project temporal to same dim (use all temporal info, not just first half!)
        temporal_features = self.temporal_proj(temporal_global)  # (B, D)

        # Fusion
        if self.use_cross_attention:
            fused = self.cross_attention(spatial_global, temporal_features)
        else:
            fused = torch.cat([spatial_global, temporal_features], dim=-1)

        # Classification
        verb_logits = self.verb_head(fused)
        noun_features = self.noun_proj(fused)
        noun_logits = self.noun_head(noun_features)

        return verb_logits, noun_logits


# =============================================================================
# BASELINE MODEL (for comparison)
# =============================================================================

class BaselineModel(nn.Module):
    """
    Baseline ResNet50 + LSTM model (same as src/models.py ActionModel).
    Used for comparison with two-stream model.
    """

    def __init__(
        self,
        num_verb_classes=97,
        num_noun_classes=300,
        backbone='resnet50',
        dropout=0.3,
        num_frames=32
    ):
        super().__init__()

        # Backbone
        if backbone == 'resnet50':
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.feature_dim = 2048
        elif backbone == 'resnet18':
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone = nn.Sequential(*list(base.children())[:-1])

        # Feature projection
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # Temporal LSTM
        self.temporal = nn.LSTM(
            512, 512, num_layers=2,
            batch_first=True, bidirectional=True,
            dropout=dropout
        )

        # Classification heads
        self.verb_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_verb_classes)
        )

        self.noun_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_noun_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        features = self.backbone(x).squeeze(-1).squeeze(-1)
        features = self.proj(features)
        features = features.view(B, T, -1)

        temporal_out, (h_n, _) = self.temporal(features)
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)

        verb_logits = self.verb_head(hidden)
        noun_logits = self.noun_head(hidden)

        return verb_logits, noun_logits


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """Early stopping with best model state tracking."""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_state = None

    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_class_weights(class_counts, power=0.5):
    """
    Compute class weights using inverse frequency with power scaling.

    Args:
        class_counts: Tensor of class counts
        power: Scaling power (0.5 = sqrt, 1.0 = inverse)

    Returns:
        weights: Normalized class weights
    """
    # Avoid division by zero
    class_counts = class_counts.float() + 1

    # Inverse frequency with power scaling
    weights = 1.0 / (class_counts ** power)

    # Normalize to sum to num_classes
    weights = weights / weights.sum() * len(weights)

    return weights


def count_parameters(model):
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
