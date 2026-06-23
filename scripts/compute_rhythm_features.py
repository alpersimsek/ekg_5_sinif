#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ekg_stage2.config import ensure_output_directories, load_config
from ekg_stage2.constants import LEADS
from ekg_stage2.data.wfdb_io import load_record
from ekg_stage2.rhythm import RHYTHM_FEATURE_NAMES, RhythmFeatureStats, extract_rhythm_features


def _extract(task: tuple[str, str, int]) -> dict[str, object]:
    root, relative_path, study_id = task
    try:
        signal = load_record(Path(root) / relative_path, expected_leads=LEADS)
        features, valid = extract_rhythm_features(signal[1])
        error = ""
    except Exception as exc:
        features = np.zeros(len(RHYTHM_FEATURE_NAMES), dtype=np.float32)
        valid = False
        error = type(exc).__name__
    return {
        "study_id": study_id,
        **dict(zip(RHYTHM_FEATURE_NAMES, features.tolist(), strict=True)),
        "valid": int(valid),
        "error": error,
    }


def _compute_split(cfg: Any, split: str, workers: int) -> Path:
    manifest = pd.read_csv(Path(cfg.paths.manifests) / f"{split}.csv", usecols=["study_id", "path"])
    output = Path(cfg.paths.stats) / f"rhythm_features_{split}.csv"
    completed: set[int] = set()
    if output.exists():
        completed = set(pd.read_csv(output, usecols=["study_id"])["study_id"].astype(int))
    pending = manifest.loc[~manifest["study_id"].isin(completed)]
    fields = ["study_id", *RHYTHM_FEATURE_NAMES, "valid", "error"]
    with output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if output.stat().st_size == 0:
            writer.writeheader()
        tasks = (
            (str(cfg.paths.data_root), str(row.path), int(row.study_id))
            for row in pending.itertuples(index=False)
        )
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, result in enumerate(pool.map(_extract, tasks, chunksize=64), start=1):
                writer.writerow(result)
                if index % 1000 == 0:
                    handle.flush()
                    print(f"{split}: {len(completed) + index}/{len(manifest)}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "validation", "test"), default=None
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_output_directories(cfg)
    splits = args.splits or ["train", "validation"]
    outputs = {split: _compute_split(cfg, split, args.workers) for split in splits}

    train_path = Path(cfg.paths.stats) / "rhythm_features_train.csv"
    validation_path = Path(cfg.paths.stats) / "rhythm_features_validation.csv"
    if train_path.exists() and validation_path.exists():
        train = pd.read_csv(train_path)
        valid = train["valid"].astype(bool).to_numpy()
        values = train[list(RHYTHM_FEATURE_NAMES)].to_numpy(dtype=np.float64)
        stats = RhythmFeatureStats(
            mean=values[valid].mean(axis=0).astype(np.float32),
            std=values[valid].std(axis=0, ddof=1).astype(np.float32),
        )
        stats.save(Path(cfg.paths.stats) / "rhythm_feature_stats.npz")
        summary = {
            "features": list(RHYTHM_FEATURE_NAMES),
            "train_records": len(train),
            "train_valid": int(valid.sum()),
            "validation_records": len(pd.read_csv(validation_path, usecols=["study_id"])),
            "test_records": int(
                len(pd.read_csv(Path(cfg.paths.manifests) / "test.csv", usecols=["study_id"]))
            )
            if (Path(cfg.paths.manifests) / "test.csv").exists()
            else None,
            "mean": stats.mean.tolist(),
            "std": stats.std.tolist(),
        }
        (Path(cfg.paths.stats) / "rhythm_feature_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary), flush=True)
    else:
        print(f"Completed requested splits: {sorted(outputs)}", flush=True)


if __name__ == "__main__":
    main()
