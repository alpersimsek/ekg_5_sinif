from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from ekg_stage2.constants import LABELS
from ekg_stage2.data.metadata import metadata_sha256


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


def _subject_table(metadata: pd.DataFrame) -> pd.DataFrame:
    """One row per patient; a label is positive if present on any patient record."""
    return metadata.groupby("subject_id", sort=True, as_index=False)[list(LABELS)].max()


def _iterative_split(
    table: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    x = table[["subject_id"]].to_numpy()
    y = table[list(LABELS)].to_numpy()
    left_indices, right_indices = next(splitter.split(x, y))
    return table.iloc[left_indices].copy(), table.iloc[right_indices].copy()


def build_patient_splits(
    metadata: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 20260621,
) -> SplitResult:
    fractions = np.array([train_fraction, validation_fraction, test_fraction], dtype=float)
    if not np.isclose(fractions.sum(), 1.0) or (fractions <= 0).any():
        raise ValueError("Split fractions must be positive and sum to 1")

    subjects = _subject_table(metadata)
    train_subjects, holdout_subjects = _iterative_split(
        subjects, test_size=validation_fraction + test_fraction, seed=seed
    )
    relative_test = test_fraction / (validation_fraction + test_fraction)
    validation_subjects, test_subjects = _iterative_split(
        holdout_subjects, test_size=relative_test, seed=seed + 1
    )

    ids = {
        "train": set(train_subjects["subject_id"].tolist()),
        "validation": set(validation_subjects["subject_id"].tolist()),
        "test": set(test_subjects["subject_id"].tolist()),
    }
    split = SplitResult(
        train=metadata[metadata["subject_id"].isin(ids["train"])].copy(),
        validation=metadata[metadata["subject_id"].isin(ids["validation"])].copy(),
        test=metadata[metadata["subject_id"].isin(ids["test"])].copy(),
    )
    assert_no_patient_leakage(split)
    if sum(len(frame) for frame in split.as_dict().values()) != len(metadata):
        raise AssertionError("Split did not preserve every metadata row")
    return split


def assert_no_patient_leakage(split: SplitResult) -> None:
    subject_sets = {
        name: set(frame["subject_id"].unique()) for name, frame in split.as_dict().items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = subject_sets[left] & subject_sets[right]
        if overlap:
            raise AssertionError(f"Patient leakage between {left} and {right}: {len(overlap)}")


def split_summary(split: SplitResult) -> dict[str, object]:
    total_records = sum(len(frame) for frame in split.as_dict().values())
    total_subjects = sum(frame["subject_id"].nunique() for frame in split.as_dict().values())
    summary: dict[str, object] = {}
    for name, frame in split.as_dict().items():
        summary[name] = {
            "records": len(frame),
            "record_fraction": len(frame) / total_records,
            "subjects": int(frame["subject_id"].nunique()),
            "subject_fraction": frame["subject_id"].nunique() / total_subjects,
            "label_counts": {label: int(frame[label].sum()) for label in LABELS},
            "label_prevalence": {label: float(frame[label].mean()) for label in LABELS},
        }
    return summary


def write_frozen_manifests(
    split: SplitResult,
    output_dir: str | Path,
    metadata_path: str | Path,
    seed: int,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in split.as_dict().items():
        frame.sort_values(["subject_id", "study_id"]).to_csv(output / f"{name}.csv", index=False)

    summary = split_summary(split)
    (output / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fingerprint = {
        "metadata_sha256": metadata_sha256(metadata_path),
        "seed": seed,
        "labels": list(LABELS),
        "test_study_ids_sha256": _ids_sha256(split.test["study_id"]),
        "test_subject_ids_sha256": _ids_sha256(split.test["subject_id"].drop_duplicates()),
    }
    (output / "manifest_fingerprint.json").write_text(json.dumps(fingerprint, indent=2) + "\n")


def _ids_sha256(values: pd.Series) -> str:
    import hashlib

    payload = "\n".join(str(value) for value in sorted(values.astype(int).tolist()))
    return hashlib.sha256(payload.encode()).hexdigest()

