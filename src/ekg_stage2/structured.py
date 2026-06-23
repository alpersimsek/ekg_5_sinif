from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from ekg_stage2.constants import LABELS


def encode_structured_targets(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert five binary labels into rhythm and conduction class indices."""
    if labels.ndim != 2 or labels.shape[1] != len(LABELS):
        raise ValueError(f"Expected labels with shape (batch, {len(LABELS)})")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Labels must be binary")

    rhythm_labels = labels[:, :3]
    conduction_labels = labels[:, 3:]
    if torch.any(rhythm_labels.sum(dim=1) > 1):
        raise ValueError("NORMAL, AFIB, and AFL must be mutually exclusive")
    if torch.any(conduction_labels.sum(dim=1) > 1):
        raise ValueError("LBBB and RBBB must be mutually exclusive")

    rhythm = torch.where(
        rhythm_labels.sum(dim=1) > 0,
        rhythm_labels.argmax(dim=1) + 1,
        torch.zeros(len(labels), dtype=torch.long, device=labels.device),
    )
    conduction = torch.where(
        conduction_labels.sum(dim=1) > 0,
        conduction_labels.argmax(dim=1) + 1,
        torch.zeros(len(labels), dtype=torch.long, device=labels.device),
    )
    return rhythm.long(), conduction.long()


def structured_class_weights(
    manifest: pd.DataFrame, power: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return smoothed inverse-frequency weights for both categorical heads."""
    if not 0.0 <= power <= 1.0:
        raise ValueError("Class-weight power must be between zero and one")
    labels = torch.from_numpy(manifest[list(LABELS)].to_numpy(dtype=np.int64))
    rhythm, conduction = encode_structured_targets(labels)
    weights: list[torch.Tensor] = []
    for targets, classes in ((rhythm, 4), (conduction, 3)):
        counts = torch.bincount(targets, minlength=classes).to(torch.float64)
        if torch.any(counts == 0):
            raise ValueError("Every structured class must occur in the training manifest")
        head_weights = (len(targets) / counts).pow(power)
        weights.append((head_weights / head_weights.mean()).to(torch.float32))
    return weights[0], weights[1]


class FocalCrossEntropyLoss(nn.Module):
    """Weighted categorical focal loss with cross-entropy-compatible normalization."""

    def __init__(self, weights: torch.Tensor, gamma: float = 2.0) -> None:
        super().__init__()
        if gamma < 0.0:
            raise ValueError("Focal gamma cannot be negative")
        self.register_buffer("weights", weights)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(logits, targets, weight=self.weights, reduction="none")
        target_probability = logits.softmax(dim=1).gather(1, targets[:, None]).squeeze(1)
        modulated = (1.0 - target_probability).pow(self.gamma) * cross_entropy
        normalization = self.weights[targets].sum().clamp_min(torch.finfo(logits.dtype).eps)
        return modulated.sum() / normalization


class StructuredCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        rhythm_weights: torch.Tensor,
        conduction_weights: torch.Tensor,
        rhythm_fraction: float = 0.6,
        rhythm_loss: str = "cross_entropy",
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= rhythm_fraction <= 1.0:
            raise ValueError("Rhythm loss fraction must be between zero and one")
        if rhythm_loss == "cross_entropy":
            self.rhythm_loss: nn.Module = nn.CrossEntropyLoss(weight=rhythm_weights)
        elif rhythm_loss == "focal":
            self.rhythm_loss = FocalCrossEntropyLoss(rhythm_weights, gamma=focal_gamma)
        else:
            raise ValueError(f"Unknown rhythm loss: {rhythm_loss}")
        self.conduction_loss = nn.CrossEntropyLoss(weight=conduction_weights)
        self.rhythm_fraction = rhythm_fraction

    def forward(self, outputs: Mapping[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        rhythm, conduction = encode_structured_targets(labels)
        return self.rhythm_fraction * self.rhythm_loss(outputs["rhythm"], rhythm) + (
            1.0 - self.rhythm_fraction
        ) * self.conduction_loss(outputs["conduction"], conduction)


def decode_structured_outputs(
    outputs: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return five-label probabilities and constraint-respecting predictions."""
    rhythm = outputs["rhythm"].softmax(dim=1)
    conduction = outputs["conduction"].softmax(dim=1)
    probabilities = torch.cat((rhythm[:, 1:], conduction[:, 1:]), dim=1)
    predictions = torch.zeros_like(probabilities, dtype=torch.int8)

    rhythm_class = rhythm.argmax(dim=1)
    rhythm_active = rhythm_class > 0
    predictions[rhythm_active, rhythm_class[rhythm_active] - 1] = 1

    conduction_class = conduction.argmax(dim=1)
    conduction_active = conduction_class > 0
    predictions[conduction_active, conduction_class[conduction_active] + 2] = 1
    return probabilities, predictions
