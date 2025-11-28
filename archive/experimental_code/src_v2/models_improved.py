"""
Improved TwoStream Model for Action Recognition
Based on SOTA approaches: SlowFast, TimeSformer, X3D

Key improvements:
1. Multi-scale temporal modeling (slow + fast pathways)
2. Temporal Transformer with learned positional encoding
3. Better CLIP adaptation with lightweight adapters
4. Multi-head temporal pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class TemporalAdapter(nn.Module):
    """Lightweight adapter for CLIP to learn temporal features without full fine-tuning."""

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, dim)
        )
        # Initialize to near-zero so it starts as identity
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)

    def forward(self, x):
        return x + self.adapter(x)


class TemporalTransformer(nn.Module):
    """Transformer for temporal modeling with learned positional encoding."""

    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dim = dim

        # Learned positional encoding (up to 32 frames)
        self.pos_embed = nn.Parameter(torch.randn(1, 32, dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, x):
        """
        Args:
            x: (B, T, D) - frame features
        Returns:
            global_feat: (B, D) - global temporal representation
            frame_feats: (B, T, D) - enhanced frame features
        """
        B, T, D = x.shape

        # Add positional encoding
        x = x + self.pos_embed[:, :T, :]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, D)

        # Transformer
        x = self.transformer(x)

        # Split CLS token and frame features
        global_feat = x[:, 0]  # (B, D)
        frame_feats = x[:, 1:]  # (B, T, D)

        return global_feat, frame_feats


class MultiScaleTemporal(nn.Module):
    """Multi-scale temporal convolutions (inspired by SlowFast)."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()

        # Slow pathway - captures long-term dependencies
        self.slow_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Fast pathway - captures short-term motion
        self.fast_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Lateral connection from fast to slow
        self.lateral = nn.Linear(dim, dim)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            out: (B, T, D) - multi-scale temporal features
        """
        B, T, D = x.shape

        # Transpose for Conv1d: (B, D, T)
        x_t = x.transpose(1, 2)

        # Slow pathway
        slow = self.slow_conv(x_t).transpose(1, 2)  # (B, T, D)

        # Fast pathway
        fast = self.fast_conv(x_t).transpose(1, 2)  # (B, T, D)

        # Lateral connection
        fast = fast + self.lateral(slow)

        # Fusion
        fused = torch.cat([slow, fast], dim=-1)  # (B, T, 2D)
        out = self.fusion(fused)  # (B, T, D)

        return out


class MultiHeadTemporalPooling(nn.Module):
    """Multi-head attention-based temporal pooling."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads

        # Attention weights for each head
        self.attention_weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.Tanh(),
                nn.Linear(dim // 4, 1)
            ) for _ in range(num_heads)
        ])

        # Fusion across heads
        self.fusion = nn.Sequential(
            nn.Linear(dim * num_heads, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, D) - frame features
        Returns:
            pooled: (B, D) - temporally pooled features
        """
        B, T, D = x.shape
        pooled_heads = []

        for attention in self.attention_weights:
            # Compute attention scores
            scores = attention(x).squeeze(-1)  # (B, T)
            weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)

            # Weighted sum
            pooled = (x * weights).sum(dim=1)  # (B, D)
            pooled_heads.append(pooled)

        # Concatenate and fuse
        pooled = torch.cat(pooled_heads, dim=-1)  # (B, D * num_heads)
        pooled = self.fusion(pooled)  # (B, D)

        return pooled


class ImprovedTwoStreamModel(nn.Module):
    """
    Improved Two-Stream Model with:
    - CLIP spatial features + lightweight adapters
    - Multi-scale temporal modeling
    - Temporal Transformer
    - Multi-head temporal pooling
    """

    def __init__(
        self,
        num_verb_classes: int = 97,
        num_noun_classes: int = 300,
        num_frames: int = 8,
        proj_dim: int = 512,
        dropout: float = 0.3,
        use_adapters: bool = True,
        freeze_clip: bool = False,
        use_cosine_noun: bool = True,
        cosine_temp: float = 0.05
    ):
        super().__init__()
        self.num_frames = num_frames
        self.use_adapters = use_adapters
        self.freeze_clip = freeze_clip
        self.use_cosine_noun = use_cosine_noun

        # ======================
        # Spatial Stream (CLIP)
        # ======================
        try:
            import clip
            self.clip_model, _ = clip.load("ViT-B/16", device="cpu")
            self.clip_model = self.clip_model.visual
            clip_dim = 512
            self.has_clip = True
            print(f"✓ CLIP ViT-B/16 loaded ({'frozen' if freeze_clip else 'trainable'})")
        except Exception as e:
            print(f"✗ CLIP failed: {e}. Using ResNet50 fallback.")
            from torchvision.models import resnet50
            self.clip_model = None
            self.resnet_backbone = resnet50(pretrained=True)
            self.resnet_backbone.fc = nn.Identity()
            clip_dim = 2048
            self.has_clip = False

        # Freeze CLIP if requested
        if self.has_clip and freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            print("  → CLIP frozen (no gradients)")

        # Spatial projection
        self.spatial_proj = nn.Sequential(
            nn.Linear(clip_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # Temporal adapters (lightweight, always trainable)
        if use_adapters and self.has_clip:
            self.temporal_adapter = TemporalAdapter(proj_dim, reduction=4)
            print(f"  → Temporal adapters enabled ({sum(p.numel() for p in self.temporal_adapter.parameters()):,} params)")
        else:
            self.temporal_adapter = nn.Identity()

        # ======================
        # Temporal Stream
        # ======================

        # Multi-scale temporal convolutions
        self.multi_scale_temporal = MultiScaleTemporal(proj_dim, dropout)

        # Temporal Transformer
        self.temporal_transformer = TemporalTransformer(
            dim=proj_dim,
            num_heads=8,
            num_layers=3,
            dropout=dropout
        )

        # Multi-head temporal pooling (alternative to CLS token)
        self.temporal_pooling = MultiHeadTemporalPooling(proj_dim, num_heads=4)

        # ======================
        # Fusion
        # ======================
        # Fuse CLS token from transformer + multi-head pooling
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # ======================
        # Classification Heads
        # ======================
        self.verb_head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_verb_classes)
        )

        if use_cosine_noun:
            self.noun_proj = nn.Sequential(
                nn.Linear(proj_dim, proj_dim),
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
            self.noun_head = nn.Sequential(
                nn.Linear(proj_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(proj_dim, num_noun_classes)
            )

        # Print model info
        self._print_model_info()

    def _print_model_info(self):
        """Print model architecture info."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "="*60)
        print("IMPROVED TWO-STREAM MODEL")
        print("="*60)
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters:    {total_params - trainable_params:,}")
        print("="*60)
        print("Architecture:")
        print("  • Spatial: CLIP ViT-B/16 → Projection")
        if self.use_adapters:
            print("  • Adapters: Temporal adapters (lightweight)")
        print("  • Temporal: Multi-scale Conv → Transformer → Pooling")
        print("  • Fusion: CLS token + Multi-head pooling")
        print("  • Heads: Verb (linear) + Noun (cosine)" if self.use_cosine_noun else "  • Heads: Verb + Noun (linear)")
        print("="*60 + "\n")

    def extract_spatial_features(self, x):
        """Extract per-frame spatial features using CLIP."""
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        if self.has_clip:
            # CLIP expects 224x224
            if H != 224 or W != 224:
                x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

            if self.freeze_clip:
                with torch.no_grad():
                    features = self.clip_model(x)
            else:
                features = self.clip_model(x)
        else:
            # ResNet fallback
            features = self.resnet_backbone(x).squeeze(-1).squeeze(-1)

        features = features.float()
        features = self.spatial_proj(features)
        features = features.view(B, T, -1)

        # Apply temporal adapters
        features = self.temporal_adapter(features)

        return features

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - video frames
        Returns:
            verb_logits: (B, num_verb_classes)
            noun_logits: (B, num_noun_classes)
        """
        # Extract spatial features
        spatial_features = self.extract_spatial_features(x)  # (B, T, D)

        # Multi-scale temporal modeling
        temporal_features = self.multi_scale_temporal(spatial_features)  # (B, T, D)

        # Temporal Transformer
        cls_feat, frame_feats = self.temporal_transformer(temporal_features)  # (B, D), (B, T, D)

        # Multi-head temporal pooling
        pooled_feat = self.temporal_pooling(frame_feats)  # (B, D)

        # Fusion
        fused = torch.cat([cls_feat, pooled_feat], dim=-1)  # (B, 2D)
        fused = self.fusion(fused)  # (B, D)

        # Classification
        verb_logits = self.verb_head(fused)
        noun_features = self.noun_proj(fused)
        noun_logits = self.noun_head(noun_features)

        return verb_logits, noun_logits


class CosineClassifier(nn.Module):
    """Cosine similarity classifier with learnable temperature."""

    def __init__(self, in_dim: int, num_classes: int, temperature: float = 0.05):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        self.temperature = temperature
        nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        """
        Args:
            x: (B, D) - input features
        Returns:
            logits: (B, num_classes) - cosine similarity logits
        """
        # L2 normalize
        x = F.normalize(x, p=2, dim=-1)
        weight = F.normalize(self.weight, p=2, dim=-1)

        # Cosine similarity
        logits = F.linear(x, weight) / self.temperature

        return logits


def count_parameters(model):
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
