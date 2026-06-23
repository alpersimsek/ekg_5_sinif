from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import neurokit2 as nk
import numpy as np

RHYTHM_FEATURE_NAMES = (
    "heart_rate_bpm",
    "mean_rr_s",
    "median_rr_s",
    "sdnn_s",
    "rmssd_s",
    "pnn50",
    "rr_cv",
    "minimum_rr_s",
    "maximum_rr_s",
)


def extract_rhythm_features(
    lead_ii: np.ndarray, sampling_rate: int = 500, powerline: int = 60
) -> tuple[np.ndarray, bool]:
    """Extract short-record beat timing features; frequency HRV is intentionally omitted."""
    if lead_ii.ndim != 1 or len(lead_ii) < sampling_rate * 2:
        raise ValueError("Lead II must be a one-dimensional signal at least two seconds long")
    cleaned = nk.ecg_clean(
        lead_ii, sampling_rate=sampling_rate, method="neurokit", powerline=powerline
    )
    _, info = nk.ecg_peaks(
        cleaned, sampling_rate=sampling_rate, method="neurokit", correct_artifacts=True
    )
    peaks = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
    rr = np.diff(peaks).astype(np.float64) / sampling_rate
    physiological = rr[(rr >= 0.30) & (rr <= 2.00)]
    valid_fraction = len(physiological) / max(len(rr), 1)
    if len(physiological) < 3 or valid_fraction < 0.75:
        return np.zeros(len(RHYTHM_FEATURE_NAMES), dtype=np.float32), False

    rr_differences = np.diff(physiological)
    mean_rr = float(physiological.mean())
    features = np.array(
        [
            60.0 / mean_rr,
            mean_rr,
            np.median(physiological),
            np.std(physiological, ddof=1),
            np.sqrt(np.mean(np.square(rr_differences))),
            np.mean(np.abs(rr_differences) > 0.05),
            np.std(physiological, ddof=1) / mean_rr,
            physiological.min(),
            physiological.max(),
        ],
        dtype=np.float32,
    )
    return features, bool(np.isfinite(features).all())


@dataclass(frozen=True)
class RhythmFeatureStats:
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        expected = (len(RHYTHM_FEATURE_NAMES),)
        if self.mean.shape != expected or self.std.shape != expected:
            raise ValueError(f"Rhythm feature statistics must have shape {expected}")
        if (self.std <= 0).any():
            raise ValueError("Rhythm feature standard deviations must be positive")

    def save(self, path: str | Path) -> None:
        np.savez(path, mean=self.mean.astype(np.float32), std=self.std.astype(np.float32))

    @classmethod
    def load(cls, path: str | Path) -> RhythmFeatureStats:
        values = np.load(path)
        return cls(values["mean"], values["std"])

    def transform(self, features: np.ndarray, valid: np.ndarray) -> np.ndarray:
        if features.ndim != 2 or features.shape[1] != len(RHYTHM_FEATURE_NAMES):
            raise ValueError("Unexpected rhythm feature matrix shape")
        normalized = (features - self.mean[None, :]) / self.std[None, :]
        normalized[~valid] = 0.0
        return np.column_stack((normalized, valid.astype(np.float32))).astype(np.float32)
