from __future__ import annotations

import numpy as np
import pandas as pd

from ekg_stage2.constants import LABELS
from ekg_stage2.data.split import assert_no_patient_leakage, build_patient_splits


def _synthetic_metadata(subjects: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    for subject in range(subjects):
        labels = np.array(
            [
                rng.random() < 0.80,
                rng.random() < 0.20,
                rng.random() < 0.10,
                rng.random() < 0.12,
                rng.random() < 0.15,
            ],
            dtype=np.int8,
        )
        if not labels.any():
            labels[0] = 1
        for record in range(1 + subject % 3):
            study_id = subject * 10 + record
            rows.append(
                {
                    "subject_id": subject,
                    "study_id": study_id,
                    "file_name": str(study_id),
                    "path": f"files/{study_id}",
                    **dict(zip(LABELS, labels, strict=True)),
                }
            )
    return pd.DataFrame(rows)


def test_patient_split_is_complete_leakage_free_and_deterministic() -> None:
    metadata = _synthetic_metadata()
    first = build_patient_splits(metadata, seed=7)
    second = build_patient_splits(metadata, seed=7)
    assert_no_patient_leakage(first)
    assert sum(len(frame) for frame in first.as_dict().values()) == len(metadata)
    for name in first.as_dict():
        assert first.as_dict()[name]["study_id"].tolist() == second.as_dict()[name][
            "study_id"
        ].tolist()


def test_patient_split_subject_proportions() -> None:
    split = build_patient_splits(_synthetic_metadata(), seed=11)
    fractions = {
        name: frame["subject_id"].nunique() / 200 for name, frame in split.as_dict().items()
    }
    assert abs(fractions["train"] - 0.70) <= 0.025
    assert abs(fractions["validation"] - 0.15) <= 0.025
    assert abs(fractions["test"] - 0.15) <= 0.025
