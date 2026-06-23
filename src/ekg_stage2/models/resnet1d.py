from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _same_padding(kernel_size: int, dilation: int = 1) -> int:
    return dilation * (kernel_size - 1) // 2


class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gate = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(self.pool(x))


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        padding = _same_padding(kernel_size)
        self.body = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            SqueezeExcitation1D(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(x) + self.skip(x))


class ECGResNet1D(nn.Module):
    """Compact residual ECG encoder with independent multi-label logits."""

    def __init__(
        self,
        input_leads: int = 12,
        num_labels: int = 5,
        stem_channels: int = 64,
        stage_channels: Sequence[int] = (64, 128, 256, 384),
        blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
        kernel_size: int = 15,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if len(stage_channels) != len(blocks_per_stage):
            raise ValueError("stage_channels and blocks_per_stage lengths must match")
        self.stem = nn.Sequential(
            nn.Conv1d(input_leads, stem_channels, 25, stride=2, padding=12, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.SiLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        layers: list[nn.Module] = []
        in_channels = stem_channels
        for stage_index, (channels, blocks) in enumerate(
            zip(stage_channels, blocks_per_stage, strict=True)
        ):
            for block_index in range(blocks):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                layers.append(ResidualBlock1D(in_channels, channels, kernel_size, stride))
                in_channels = channels
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_labels),
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.stem(x)))


class StructuredECGResNet1D(ECGResNet1D):
    """Shared ECG encoder with mutually exclusive rhythm and conduction heads."""

    def __init__(
        self,
        input_leads: int = 12,
        stem_channels: int = 64,
        stage_channels: Sequence[int] = (64, 128, 256, 384),
        blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
        kernel_size: int = 15,
        dropout: float = 0.2,
    ) -> None:
        if not stage_channels:
            raise ValueError("At least one encoder stage is required")
        super().__init__(
            input_leads=input_leads,
            num_labels=5,
            stem_channels=stem_channels,
            stage_channels=stage_channels,
            blocks_per_stage=blocks_per_stage,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        feature_channels = stage_channels[-1]
        self.head = nn.Identity()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.rhythm_head = nn.Linear(feature_channels, 4)
        self.conduction_head = nn.Linear(feature_channels, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(self.stem(x))
        pooled = self.dropout(self.pool(features).flatten(1))
        return {
            "rhythm": self.rhythm_head(pooled),
            "conduction": self.conduction_head(pooled),
        }


class RhythmFeatureStructuredECGResNet1D(StructuredECGResNet1D):
    """Structured model with an additive short-record rhythm feature branch."""

    def __init__(self, rhythm_feature_count: int = 10, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.rhythm_feature_head = nn.Sequential(
            nn.Linear(rhythm_feature_count, 32),
            nn.SiLU(),
            nn.Linear(32, 4),
        )
        final = self.rhythm_feature_head[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Unexpected rhythm feature head")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self, x: torch.Tensor, rhythm_features: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        outputs = super().forward(x)
        if rhythm_features is None:
            raise ValueError("Rhythm features are required by this model")
        outputs["rhythm"] = outputs["rhythm"] + self.rhythm_feature_head(rhythm_features)
        return outputs
