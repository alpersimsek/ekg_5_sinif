from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import wfdb

from ekg_stage2.constants import LEADS


@dataclass(frozen=True)
class RecordQuality:
    valid: bool
    reasons: tuple[str, ...]
    sampling_rate: float
    samples: int
    leads: int
    nan_fraction: float
    minimum_lead_std_mv: float
    maximum_absolute_amplitude_mv: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_record(
    record_path: str | Path,
    expected_leads: Sequence[str] = LEADS,
    expected_sampling_rate: int = 500,
    expected_samples: int = 5000,
) -> np.ndarray:
    """Read physical mV values and return canonical [lead, time] float32 data."""
    record = wfdb.rdrecord(str(record_path), physical=True)
    if int(record.fs) != expected_sampling_rate:
        raise ValueError(f"Expected {expected_sampling_rate} Hz, got {record.fs}")
    if record.sig_len != expected_samples:
        raise ValueError(f"Expected {expected_samples} samples, got {record.sig_len}")
    if record.p_signal is None:
        raise ValueError("WFDB record has no physical signal")

    names = list(record.sig_name)
    missing = set(expected_leads) - set(names)
    duplicated = len(names) != len(set(names))
    if missing or duplicated:
        raise ValueError(f"Invalid lead set; missing={sorted(missing)}, duplicated={duplicated}")
    order = [names.index(lead) for lead in expected_leads]
    signal = np.asarray(record.p_signal[:, order].T, dtype=np.float32)
    return signal


def assess_signal_quality(
    signal: np.ndarray,
    sampling_rate: float,
    flatline_std_mv: float = 0.001,
    extreme_amplitude_mv: float = 20.0,
    max_nan_fraction: float = 0.0,
) -> RecordQuality:
    if signal.ndim != 2:
        raise ValueError(f"Expected [lead, time], got shape {signal.shape}")
    finite = np.isfinite(signal)
    nan_fraction = float(1.0 - finite.mean())
    safe_signal = np.where(finite, signal, np.nan)
    with np.errstate(all="ignore"):
        lead_std = np.nanstd(safe_signal, axis=1)
        maximum = float(np.nanmax(np.abs(safe_signal))) if finite.any() else float("inf")
    minimum_std = float(np.nanmin(lead_std)) if np.isfinite(lead_std).any() else 0.0

    reasons: list[str] = []
    if nan_fraction > max_nan_fraction:
        reasons.append("non_finite")
    if minimum_std < flatline_std_mv:
        reasons.append("flat_lead")
    if maximum > extreme_amplitude_mv:
        reasons.append("extreme_amplitude")
    return RecordQuality(
        valid=not reasons,
        reasons=tuple(reasons),
        sampling_rate=float(sampling_rate),
        samples=int(signal.shape[1]),
        leads=int(signal.shape[0]),
        nan_fraction=nan_fraction,
        minimum_lead_std_mv=minimum_std,
        maximum_absolute_amplitude_mv=maximum,
    )
