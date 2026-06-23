#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ekg_stage2.config import ensure_output_directories, load_config
from ekg_stage2.data.metadata import load_metadata, summarize_metadata
from ekg_stage2.data.split import build_patient_splits, split_summary, write_frozen_manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--quality-audit",
        help="Complete waveform audit CSV; only rows with valid=true are retained",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_output_directories(cfg)
    metadata = load_metadata(cfg.paths.metadata)
    quality_report = None
    if args.quality_audit:
        audit_path = Path(args.quality_audit)
        audit = pd.read_csv(audit_path)
        required = {"study_id", "valid"}
        missing = required - set(audit.columns)
        if missing:
            raise ValueError(f"Quality audit is missing columns: {sorted(missing)}")
        if audit["study_id"].duplicated().any():
            raise ValueError("Quality audit contains duplicate study_id values")
        metadata_ids = set(metadata["study_id"].astype(int))
        audit_ids = set(audit["study_id"].astype(int))
        if audit_ids != metadata_ids:
            raise ValueError(
                "Quality audit must cover the complete metadata exactly; "
                f"missing={len(metadata_ids - audit_ids)}, extra={len(audit_ids - metadata_ids)}"
            )
        valid_ids = set(audit.loc[audit["valid"].astype(bool), "study_id"].astype(int))
        original_records = len(metadata)
        metadata = metadata[metadata["study_id"].isin(valid_ids)].copy()
        quality_report = {
            "audit": str(audit_path.resolve()),
            "records_checked": original_records,
            "records_retained": len(metadata),
            "records_excluded": original_records - len(metadata),
        }
    split = build_patient_splits(
        metadata,
        train_fraction=float(cfg.data.train_fraction),
        validation_fraction=float(cfg.data.validation_fraction),
        test_fraction=float(cfg.data.test_fraction),
        seed=int(cfg.seed),
    )
    write_frozen_manifests(split, cfg.paths.manifests, cfg.paths.metadata, int(cfg.seed))
    report = {
        "dataset": summarize_metadata(metadata),
        "quality_filter": quality_report,
        "splits": split_summary(split),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
