from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        if self.mean.shape != self.std.shape or self.mean.ndim != 1:
            raise ValueError("Normalization mean/std must be matching 1D arrays")
        if (self.std <= 0).any():
            raise ValueError("Normalization standard deviations must be positive")

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean.astype(np.float32), std=self.std.astype(np.float32))

    @classmethod
    def load(cls, path: str) -> NormalizationStats:
        values = np.load(path)
        return cls(mean=values["mean"], std=values["std"])


class StreamingLeadStatistics:
    """Numerically stable per-lead mean and variance without retaining records."""

    def __init__(self, n_leads: int) -> None:
        self.count = np.zeros(n_leads, dtype=np.int64)
        self.mean = np.zeros(n_leads, dtype=np.float64)
        self.m2 = np.zeros(n_leads, dtype=np.float64)

    def update(self, signal: np.ndarray) -> None:
        if signal.ndim != 2 or signal.shape[0] != len(self.count):
            raise ValueError("Unexpected signal shape")
        for lead in range(signal.shape[0]):
            values = signal[lead].astype(np.float64)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            batch_count = len(values)
            batch_mean = values.mean()
            batch_m2 = np.square(values - batch_mean).sum()
            delta = batch_mean - self.mean[lead]
            total = self.count[lead] + batch_count
            self.mean[lead] += delta * batch_count / total
            self.m2[lead] += batch_m2 + delta**2 * self.count[lead] * batch_count / total
            self.count[lead] = total

    def merge(self, count: np.ndarray, mean: np.ndarray, m2: np.ndarray) -> None:
        """Merge another independent set of per-lead summary statistics."""
        if (
            count.shape != self.count.shape
            or mean.shape != self.mean.shape
            or m2.shape != self.m2.shape
        ):
            raise ValueError("Unexpected summary shape")
        if (count < 0).any():
            raise ValueError("Summary counts cannot be negative")
        active = count > 0
        total = self.count + count
        delta = mean - self.mean
        self.mean[active] += delta[active] * count[active] / total[active]
        self.m2[active] += (
            m2[active]
            + np.square(delta[active]) * self.count[active] * count[active] / total[active]
        )
        self.count = total

    def finalize(self) -> NormalizationStats:
        if (self.count < 2).any():
            raise ValueError("Insufficient values for one or more leads")
        variance = self.m2 / (self.count - 1)
        return NormalizationStats(
            self.mean.astype(np.float32), np.sqrt(variance).astype(np.float32)
        )


def bandpass_filter(
    signal: np.ndarray,
    sampling_rate: float = 500.0,
    low_hz: float = 0.5,
    high_hz: float = 100.0,
    order: int = 3,
) -> np.ndarray:
    if not 0 < low_hz < high_hz < sampling_rate / 2:
        raise ValueError("Bandpass frequencies must satisfy 0 < low < high < Nyquist")
    sos = butter(order, (low_hz, high_hz), btype="bandpass", fs=sampling_rate, output="sos")
    return sosfiltfilt(sos, signal, axis=-1).astype(np.float32)


def preprocess_signal(
    signal: np.ndarray,
    stats: NormalizationStats | None,
    sampling_rate: float = 500.0,
    low_hz: float = 0.5,
    high_hz: float = 100.0,
    filter_order: int = 3,
    clip_millivolts: float = 10.0,
) -> np.ndarray:
    if not np.isfinite(signal).all():
        raise ValueError("Signal contains non-finite values")
    processed = bandpass_filter(signal, sampling_rate, low_hz, high_hz, filter_order)
    processed = np.clip(processed, -clip_millivolts, clip_millivolts)
    if stats is not None:
        processed = (processed - stats.mean[:, None]) / stats.std[:, None]
    return processed.astype(np.float32, copy=False)


@dataclass(frozen=True)
class AugmentationConfig:
    amplitude_scale: float = 0.10
    gaussian_noise_std: float = 0.005
    baseline_wander_std: float = 0.02
    time_mask_fraction: float = 0.02
    max_shift_fraction: float = 0.02
    lead_dropout_probability: float = 0.05


def augment_signal(
    signal: np.ndarray, rng: np.random.Generator, config: AugmentationConfig
) -> np.ndarray:
    """Apply conservative morphology-preserving augmentations to normalized ECG."""
    result = signal.copy()
    scale = rng.uniform(1.0 - config.amplitude_scale, 1.0 + config.amplitude_scale)
    result *= scale
    result += rng.normal(0.0, config.gaussian_noise_std, size=result.shape).astype(np.float32)

    samples = result.shape[1]
    phase = rng.uniform(0, 2 * np.pi)
    cycles = rng.uniform(0.5, 2.0)
    wander = np.sin(np.linspace(phase, phase + 2 * np.pi * cycles, samples))
    result += (wander * rng.normal(0.0, config.baseline_wander_std))[None, :].astype(np.float32)

    mask_length = int(samples * config.time_mask_fraction)
    if mask_length > 0:
        start = int(rng.integers(0, samples - mask_length + 1))
        result[:, start : start + mask_length] = 0.0

    max_shift = int(samples * config.max_shift_fraction)
    if max_shift:
        result = np.roll(result, int(rng.integers(-max_shift, max_shift + 1)), axis=1)
    drop = rng.random(result.shape[0]) < config.lead_dropout_probability
    result[drop] = 0.0
    return result.astype(np.float32, copy=False)
