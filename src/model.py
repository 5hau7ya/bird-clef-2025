"""Framewise sound-event-detection model definition.

``EfficientNetSED`` receives normalized three-channel Mel images shaped
``(B, 3, 224, 512)`` and returns unactivated framewise logits shaped
``(B, num_classes, T)``. For the configured EfficientNet-B0 input geometry,
``T`` is currently 16, but downstream callers must treat it as dynamic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import timm
import torch
from torch import nn


@dataclass(frozen=True)
class SEDConfig:
    backbone: str = "efficientnet_b0"
    num_classes: int = 206
    dropout: float = 0.2
    drop_path_rate: float = 0.0
    pretrained: bool = False
    embedding_dim: int = 512
    gem_p: float = 3.0
    gem_epsilon: float = 1e-6

    @classmethod
    def from_json(cls, path: str | Path = "config/project_config.json") -> "SEDConfig":
        with Path(path).open(encoding="utf-8") as handle:
            values = json.load(handle)["model"]
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in valid})


class GeMFrequencyPooling(nn.Module):
    """Numerically stable, trainable GeM pooling over frequency only.

    Input and output respectively have shapes ``(B, C, F, T)`` and
    ``(B, C, 1, T)``. The time axis is intentionally retained.
    """

    def __init__(self, p: float = 3.0, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.epsilon = epsilon

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("GeMFrequencyPooling expects a tensor shaped (B, C, F, T)")
        exponent = self.p.clamp_min(self.epsilon)
        stable_features = features.clamp_min(self.epsilon)
        return stable_features.pow(exponent).mean(dim=2, keepdim=True).pow(1.0 / exponent)


class EfficientNetSED(nn.Module):
    """EfficientNet-B0 feature extractor with a frequency-pooled SED head."""

    def __init__(self, config: SEDConfig | None = None) -> None:
        super().__init__()
        self.config = config or SEDConfig()
        self.backbone = timm.create_model(
            self.config.backbone,
            pretrained=self.config.pretrained,
            features_only=True,
            out_indices=(4,),
            drop_path_rate=self.config.drop_path_rate,
        )
        self.backbone_channels = self.backbone.feature_info.channels()[-1]
        self.gem = GeMFrequencyPooling(self.config.gem_p, self.config.gem_epsilon)
        self.frame_projection = nn.Sequential(
            nn.Linear(self.backbone_channels, self.config.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
        )
        self.classifier = nn.Conv1d(self.config.embedding_dim, self.config.num_classes, kernel_size=1)

    def extract_backbone_features(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """Extract final CNN features with shape ``(B, C, F, T)``."""
        if spectrograms.ndim != 4 or spectrograms.shape[1] != 3:
            raise ValueError("Expected model input with shape (B, 3, n_mels, time_frames)")
        return self.backbone(spectrograms)[-1]

    def pool_frequency(self, features: torch.Tensor) -> torch.Tensor:
        """Apply GeM across only frequency: ``(B,C,F,T) -> (B,C,1,T)``."""
        return self.gem(features)

    def transform_frames(self, pooled_features: torch.Tensor) -> torch.Tensor:
        """Create `(B, T, 512)` frame embeddings from frequency-pooled features."""
        if pooled_features.ndim != 4 or pooled_features.shape[2] != 1:
            raise ValueError("Expected frequency-pooled features with shape (B, C, 1, T)")
        return self.frame_projection(pooled_features.squeeze(2).transpose(1, 2))

    def forward(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """Return framewise, unactivated multi-label logits shaped ``(B, 206, T)``."""
        features = self.extract_backbone_features(spectrograms)
        pooled = self.pool_frequency(features)
        frame_embeddings = self.transform_frames(pooled)
        return self.classifier(frame_embeddings.transpose(1, 2))
