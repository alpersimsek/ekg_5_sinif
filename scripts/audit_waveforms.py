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
from ekg_stage2.data.metadata import load_metadata
from ekg_stage2.data.wfdb_io import assess_signal_quality, load_record


def audit_record(task: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, object]:
    row, options = task
    result: dict[str, object] = {
        "study_id": int(row["study_id"]),
        "subject_id": int(row["subject_id"]),
        "path": str(row["path"]),
    }
    try:
        signal = load_record(
            Path(options["data_root"]) / str(row["path"]),
            expected_leads=tuple(options["leads"]),
            expected_sampling_rate=int(options["sampling_rate"]),
            expected_samples=int(options["samples"]),
        )
        result.update(
            assess_signal_quality(
                signal,
                float(options["sampling_rate"]),
                **options["quality"],
            ).to_dict()
        )
    except Exception as error:
        result.update({"valid": False, "reasons": (f"read_error:{error}",)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_output_directories(cfg)
    metadata = load_metadata(cfg.paths.metadata)
    if not args.all:
        rng = np.random.default_rng(int(cfg.seed))
        count = min(args.sample_size, len(metadata))
        metadata = metadata.iloc[np.sort(rng.choice(len(metadata), count, replace=False))]

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    options = {
        "data_root": str(cfg.paths.data_root),
        "leads": list(cfg.data.leads),
        "sampling_rate": int(cfg.data.sampling_rate),
        "samples": int(cfg.data.samples),
        "quality": dict(cfg.quality),
    }
    tasks = ((row, options) for row in metadata.to_dict(orient="records"))
    if args.workers == 1:
        rows = [audit_record(task) for task in tqdm(tasks, total=len(metadata))]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(
                tqdm(
                    executor.map(audit_record, tasks, chunksize=32),
                    total=len(metadata),
                )
            )

    output = Path(cfg.paths.audit)
    frame = pd.DataFrame(rows)
    frame["reasons"] = frame["reasons"].map(lambda value: "|".join(value))
    frame.to_csv(output / "waveform_audit.csv", index=False)
    summary = {
        "records_checked": len(frame),
        "valid_records": int(frame["valid"].sum()),
        "invalid_records": int((~frame["valid"]).sum()),
        "reason_counts": frame["reasons"].value_counts().to_dict(),
    }
    (output / "waveform_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
