#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from ekg_stage2.config import ensure_output_directories, load_config
from ekg_stage2.data.preprocessing import StreamingLeadStatistics, preprocess_signal
from ekg_stage2.data.wfdb_io import load_record


def record_statistics(
    task: tuple[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    relative, options = task
    signal = load_record(
        Path(options["data_root"]) / relative,
        expected_leads=tuple(options["leads"]),
        expected_sampling_rate=int(options["sampling_rate"]),
        expected_samples=int(options["samples"]),
    )
    processed = preprocess_signal(
        signal,
        stats=None,
        sampling_rate=float(options["sampling_rate"]),
        low_hz=float(options["low_hz"]),
        high_hz=float(options["high_hz"]),
        filter_order=int(options["filter_order"]),
        clip_millivolts=float(options["clip_millivolts"]),
    ).astype(np.float64)
    count = np.full(processed.shape[0], processed.shape[1], dtype=np.int64)
    mean = processed.mean(axis=1)
    m2 = np.square(processed - mean[:, None]).sum(axis=1)
    return count, mean, m2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_output_directories(cfg)
    manifest = pd.read_csv(Path(cfg.paths.manifests) / "train.csv")
    if args.max_records:
        manifest = manifest.iloc[: args.max_records]

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    options = {
        "data_root": str(cfg.paths.data_root),
        "leads": list(cfg.data.leads),
        "sampling_rate": int(cfg.data.sampling_rate),
        "samples": int(cfg.data.samples),
        "low_hz": float(cfg.data.bandpass_low_hz),
        "high_hz": float(cfg.data.bandpass_high_hz),
        "filter_order": int(cfg.data.filter_order),
        "clip_millivolts": float(cfg.data.clip_millivolts),
    }
    tasks = ((str(relative), options) for relative in manifest["path"])
    accumulator = StreamingLeadStatistics(len(cfg.data.leads))
    if args.workers == 1:
        summaries = map(record_statistics, tasks)
        for summary in tqdm(summaries, total=len(manifest)):
            accumulator.merge(*summary)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            summaries = executor.map(record_statistics, tasks, chunksize=16)
            for summary in tqdm(summaries, total=len(manifest)):
                accumulator.merge(*summary)
    stats = accumulator.finalize()
    output = Path(cfg.paths.stats)
    stats.save(str(output / "normalization.npz"))
    summary = {
        "records": len(manifest),
        "train_manifest": str(Path(cfg.paths.manifests) / "train.csv"),
        "leads": list(cfg.data.leads),
        "mean": stats.mean.tolist(),
        "std": stats.std.tolist(),
    }
    (output / "normalization.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
