from __future__ import annotations

import numpy as np
import pytest

from ekg_stage2.data.preprocessing import (
    AugmentationConfig,
    StreamingLeadStatistics,
    augment_signal,
    bandpass_filter,
)
from ekg_stage2.data.wfdb_io import assess_signal_quality
from ekg_stage2.rhythm import extract_rhythm_features


def test_streaming_statistics_match_numpy() -> None:
    rng = np.random.default_rng(1)
    signals = rng.normal(size=(4, 12, 500)).astype(np.float32)
    accumulator = StreamingLeadStatistics(12)
    for signal in signals:
        accumulator.update(signal)
    stats = accumulator.finalize()
    expected = signals.transpose(1, 0, 2).reshape(12, -1)
    np.testing.assert_allclose(stats.mean, expected.mean(axis=1), atol=1e-6)
    np.testing.assert_allclose(stats.std, expected.std(axis=1, ddof=1), atol=1e-6)


def test_streaming_statistics_can_merge_independent_summaries() -> None:
    rng = np.random.default_rng(19)
    signals = rng.normal(size=(4, 12, 100)).astype(np.float32)
    left = StreamingLeadStatistics(12)
    right = StreamingLeadStatistics(12)
    combined = StreamingLeadStatistics(12)
    for signal in signals[:2]:
        left.update(signal)
    for signal in signals[2:]:
        right.update(signal)
    combined.merge(left.count, left.mean, left.m2)
    combined.merge(right.count, right.mean, right.m2)
    stats = combined.finalize()
    expected = signals.transpose(1, 0, 2).reshape(12, -1)
    np.testing.assert_allclose(stats.mean, expected.mean(axis=1), atol=1e-6)
    np.testing.assert_allclose(stats.std, expected.std(axis=1, ddof=1), atol=1e-6)


def test_bandpass_preserves_shape_and_dtype() -> None:
    signal = np.random.default_rng(2).normal(size=(12, 5000)).astype(np.float32)
    filtered = bandpass_filter(signal)
    assert filtered.shape == signal.shape
    assert filtered.dtype == np.float32


def test_augmentation_is_seed_reproducible() -> None:
    signal = np.ones((12, 1000), dtype=np.float32)
    config = AugmentationConfig()
    first = augment_signal(signal, np.random.default_rng(9), config)
    second = augment_signal(signal, np.random.default_rng(9), config)
    np.testing.assert_array_equal(first, second)


def test_quality_detects_flat_lead_and_non_finite() -> None:
    signal = np.ones((12, 5000), dtype=np.float32)
    signal[1, 0] = np.nan
    quality = assess_signal_quality(signal, 500)
    assert not quality.valid
    assert "flat_lead" in quality.reasons
    assert "non_finite" in quality.reasons


def test_rhythm_features_from_regular_peaks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ekg_stage2.rhythm.nk.ecg_clean", lambda signal, **kwargs: signal)
    monkeypatch.setattr(
        "ekg_stage2.rhythm.nk.ecg_peaks",
        lambda signal, **kwargs: ({}, {"ECG_R_Peaks": np.arange(0, 5000, 500)}),
    )
    features, valid = extract_rhythm_features(np.zeros(5000, dtype=np.float32))
    assert valid
    assert features[0] == 60.0
    assert features[1] == 1.0
    assert features[3] == 0.0
