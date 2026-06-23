from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ekg_stage2.constants import LABELS
from ekg_stage2.thresholds import apply_thresholds


def multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None,
    *,
    predictions: np.ndarray | None = None,
) -> dict[str, object]:
    if predictions is None:
        if thresholds is None:
            raise ValueError("Thresholds are required when predictions are not supplied")
        predictions = apply_thresholds(probabilities, thresholds)
    elif predictions.shape != targets.shape or not np.isin(predictions, (0, 1)).all():
        raise ValueError("Predictions must be binary and match target shape")
    result: dict[str, object] = {
        "exact_match_accuracy": float(accuracy_score(targets, predictions)),
        "hamming_loss": float(hamming_loss(targets, predictions)),
        "hamming_accuracy": float(1.0 - hamming_loss(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "weighted_f1": float(
            f1_score(targets, predictions, average="weighted", zero_division=0)
        ),
    }
    per_label: dict[str, object] = {}
    for index, label in enumerate(LABELS):
        truth = targets[:, index]
        predicted = predictions[:, index]
        tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
        per_label[label] = {
            "threshold": None if thresholds is None else float(thresholds[index]),
            "precision": float(precision_score(truth, predicted, zero_division=0)),
            "recall": float(recall_score(truth, predicted, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "f1": float(f1_score(truth, predicted, zero_division=0)),
            "auroc": _safe_score(roc_auc_score, truth, probabilities[:, index]),
            "average_precision": _safe_score(
                average_precision_score, truth, probabilities[:, index]
            ),
            "brier_score": float(brier_score_loss(truth, probabilities[:, index])),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        }
    result["per_label"] = per_label
    return result


def patient_bootstrap_confidence_intervals(
    targets: np.ndarray,
    probabilities: np.ndarray,
    subject_ids: np.ndarray,
    thresholds: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260621,
) -> dict[str, list[float]]:
    """Bootstrap patients, keeping all records from each sampled patient together."""
    if len(targets) != len(probabilities) or len(targets) != len(subject_ids):
        raise ValueError("Targets, probabilities, and subject_ids must have equal lengths")
    unique_subjects = np.unique(subject_ids)
    patient_indices = {
        patient: np.flatnonzero(subject_ids == patient) for patient in unique_subjects
    }
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"macro_f1": [], "exact_match_accuracy": []}
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        indices = np.concatenate([patient_indices[patient] for patient in sampled])
        metrics = multilabel_metrics(targets[indices], probabilities[indices], thresholds)
        for key in values:
            values[key].append(float(metrics[key]))
    alpha = (1.0 - confidence) / 2.0
    return {
        key: [float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))]
        for key, samples in values.items()
    }


def _safe_score(
    metric: Callable[..., float], truth: np.ndarray, scores: np.ndarray
) -> float | None:
    if np.unique(truth).size < 2:
        return None
    return float(metric(truth, scores))
