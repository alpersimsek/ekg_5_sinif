#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ekg_stage2.config import ensure_output_directories, load_config
from ekg_stage2.data.metadata import load_metadata
from ekg_stage2.data.wfdb_io import assess_signal_quality, load_record
from ekg_stage2.reproducibility import environment_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_output_directories(cfg)
    metadata = load_metadata(cfg.paths.metadata)
    record_path = Path(cfg.paths.data_root) / str(metadata.iloc[0]["path"])
    signal = load_record(
        record_path,
        expected_leads=tuple(cfg.data.leads),
        expected_sampling_rate=int(cfg.data.sampling_rate),
        expected_samples=int(cfg.data.samples),
    )
    snapshot = environment_snapshot()
    snapshot["sample_record"] = str(record_path)
    snapshot["sample_shape"] = list(signal.shape)
    snapshot["sample_quality"] = assess_signal_quality(
        signal, float(cfg.data.sampling_rate), **dict(cfg.quality)
    ).to_dict()
    if torch.cuda.is_available():
        tensor = torch.randn(128, 128, device="cuda")
        snapshot["cuda_tensor_test"] = float((tensor @ tensor).mean().cpu())
    output = Path(cfg.paths.output_root) / "environment.json"
    output.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(json.dumps(snapshot, indent=2))
    if not torch.cuda.is_available():
        raise SystemExit("Environment checks passed except CUDA is unavailable to PyTorch")


if __name__ == "__main__":
    main()

