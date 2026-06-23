from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info

from ekg_stage2.constants import LABELS, LEADS
from ekg_stage2.data.preprocessing import (
    AugmentationConfig,
    NormalizationStats,
    augment_signal,
    preprocess_signal,
)
from ekg_stage2.data.wfdb_io import load_record


class ECGDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        data_root: str | Path,
        stats: NormalizationStats | None,
        training: bool = False,
        augmentation: AugmentationConfig | None = None,
        seed: int = 20260621,
        max_records: int | None = None,
        preprocessing: dict[str, float | int] | None = None,
    ) -> None:
        frame = pd.read_csv(manifest) if isinstance(manifest, (str, Path)) else manifest.copy()
        if max_records is not None:
            frame = frame.iloc[:max_records].copy()
        self.frame = frame.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.stats = stats
        self.training = training
        self.augmentation = augmentation
        self.seed = seed
        self.epoch = 0
        self.preprocessing = preprocessing or {}

    def __len__(self) -> int:
        return len(self.frame)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        signal = load_record(self.data_root / str(row["path"]), expected_leads=LEADS)
        signal = preprocess_signal(signal, self.stats, **self.preprocessing)
        if self.training and self.augmentation is not None:
            worker = get_worker_info()
            worker_id = worker.id if worker else 0
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch, worker_id, index])
            )
            signal = augment_signal(signal, rng, self.augmentation)
        labels = row[list(LABELS)].to_numpy(dtype=np.float32)
        return {
            "signal": torch.from_numpy(signal),
            "labels": torch.from_numpy(labels),
            "study_id": int(row["study_id"]),
            "subject_id": int(row["subject_id"]),
        }


def positive_class_weights(manifest: pd.DataFrame) -> torch.Tensor:
    positives = manifest[list(LABELS)].sum(axis=0).to_numpy(dtype=np.float64)
    negatives = len(manifest) - positives
    if (positives == 0).any():
        raise ValueError("Cannot compute positive class weight for a label with no positives")
    return torch.tensor(negatives / positives, dtype=torch.float32)

