from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def optimize_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    minimum: float = 0.05,
    maximum: float = 0.95,
    grid_size: int = 181,
) -> np.ndarray:
    """Optimize each label's F1 using validation predictions only."""
    _validate_shapes(targets, probabilities)
    grid = np.linspace(minimum, maximum, grid_size)
    thresholds = np.empty(targets.shape[1], dtype=np.float32)
    for column in range(targets.shape[1]):
        scores = [
            f1_score(targets[:, column], probabilities[:, column] >= value, zero_division=0)
            for value in grid
        ]
        best_score = max(scores)
        candidates = grid[np.isclose(scores, best_score)]
        thresholds[column] = candidates[np.argmin(np.abs(candidates - 0.5))]
    return thresholds


def apply_thresholds(probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    if probabilities.ndim != 2 or thresholds.shape != (probabilities.shape[1],):
        raise ValueError("Threshold shape must match probability columns")
    return (probabilities >= thresholds[None, :]).astype(np.int8)


def _validate_shapes(targets: np.ndarray, probabilities: np.ndarray) -> None:
    if targets.shape != probabilities.shape or targets.ndim != 2:
        raise ValueError("Targets and probabilities must be matching 2D arrays")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("Targets must be binary")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be in [0, 1]")

