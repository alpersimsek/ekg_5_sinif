from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ekg_stage2.constants import EXPECTED_METADATA_COLUMNS, LABELS


def load_metadata(path: str | Path, validate_paths: bool = False) -> pd.DataFrame:
    """Load the label registry and enforce invariants needed by downstream code."""
    metadata_path = Path(path).expanduser().resolve()
    df = pd.read_csv(
        metadata_path,
        dtype={
            "subject_id": "int64",
            "study_id": "int64",
            "file_name": "string",
            "path": "string",
            **{label: "int8" for label in LABELS},
        },
    )
    if tuple(df.columns) != EXPECTED_METADATA_COLUMNS:
        raise ValueError(
            "Unexpected metadata columns. "
            f"Expected {EXPECTED_METADATA_COLUMNS}, got {tuple(df.columns)}"
        )
    if df.empty:
        raise ValueError("Metadata is empty")
    if df["study_id"].duplicated().any():
        duplicate = int(df.loc[df["study_id"].duplicated(), "study_id"].iloc[0])
        raise ValueError(f"Duplicate study_id: {duplicate}")
    labels = df[list(LABELS)].to_numpy()
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Label columns must contain only 0/1")
    if (labels.sum(axis=1) == 0).any():
        raise ValueError("Every record must have at least one positive label")
    if df[["subject_id", "study_id", "path"]].isna().any().any():
        raise ValueError("Required metadata fields contain missing values")
    if validate_paths:
        validate_record_paths(df, metadata_path.parent)
    return df


def validate_record_paths(df: pd.DataFrame, data_root: str | Path) -> None:
    root = Path(data_root)
    for relative in df["path"]:
        record = root / str(relative)
        for suffix in (".hea", ".dat"):
            file_path = record.with_suffix(suffix)
            if not file_path.is_file() or file_path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty waveform file: {file_path}")


def metadata_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_metadata(df: pd.DataFrame) -> dict[str, object]:
    label_counts = {label: int(df[label].sum()) for label in LABELS}
    combinations = (
        df[list(LABELS)]
        .astype(str)
        .agg("".join, axis=1)
        .value_counts()
        .sort_values(ascending=False)
    )
    return {
        "records": len(df),
        "subjects": int(df["subject_id"].nunique()),
        "single_label_records": int((df[list(LABELS)].sum(axis=1) == 1).sum()),
        "multi_label_records": int((df[list(LABELS)].sum(axis=1) > 1).sum()),
        "label_counts": label_counts,
        "top_label_combinations": {str(k): int(v) for k, v in combinations.head(20).items()},
    }
