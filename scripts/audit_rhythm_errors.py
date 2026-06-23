#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd

from ekg_stage2.config import load_config
from ekg_stage2.data.wfdb_io import load_record


def _plot_examples(frame: pd.DataFrame, data_root: Path, output: Path, title: str) -> None:
    examples = frame.head(12)
    figure, axes = plt.subplots(4, 3, figsize=(15, 10), sharex=True)
    for axis, row in zip(axes.flat, examples.itertuples(index=False), strict=False):
        lead = load_record(data_root / str(row.path))[1]
        cleaned = nk.ecg_clean(lead, sampling_rate=500, method="neurokit", powerline=60)
        _, info = nk.ecg_peaks(cleaned, sampling_rate=500, method="neurokit")
        peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
        time = np.arange(len(cleaned)) / 500
        axis.plot(time, cleaned, color="#17324d", linewidth=0.7)
        axis.scatter(time[peaks], cleaned[peaks], color="#d1495b", s=8)
        axis.set_title(
            f"study {row.study_id} | AFIB p={row.afib_probability:.2f} | "
            f"AFL p={row.afl_probability:.2f}",
            fontsize=8,
        )
    figure.suptitle(title)
    figure.supxlabel("Time (seconds)")
    figure.supylabel("Cleaned lead II (mV)")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--predictions",
        default="outputs/runs/structured_heads_seed20260621/best_validation_predictions.npz",
    )
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    cfg = load_config(args.config)
    artifact = np.load(args.predictions)
    targets = artifact["targets"].astype(int)
    predictions = artifact["predictions"].astype(int)
    probabilities = artifact["probabilities"]
    identity = pd.DataFrame(
        {
            "study_id": artifact["study_ids"].astype(int),
            "subject_id": artifact["subject_ids"].astype(int),
        }
    )
    manifest = pd.read_csv(Path(cfg.paths.manifests) / "validation.csv")
    frame = identity.merge(manifest[["study_id", "path"]], on="study_id", validate="one_to_one")
    frame["true_afib"] = targets[:, 1]
    frame["true_afl"] = targets[:, 2]
    frame["predicted_afib"] = predictions[:, 1]
    frame["predicted_afl"] = predictions[:, 2]
    frame["afib_probability"] = probabilities[:, 1]
    frame["afl_probability"] = probabilities[:, 2]

    false_positive = frame[(frame.true_afl == 0) & (frame.predicted_afl == 1)].sort_values(
        "afl_probability", ascending=False
    )
    false_negative = frame[(frame.true_afl == 1) & (frame.predicted_afl == 0)].sort_values(
        "afl_probability", ascending=True
    )
    output = Path(cfg.paths.audit)
    output.mkdir(parents=True, exist_ok=True)
    false_positive.head(args.limit).to_csv(output / "afl_false_positives.csv", index=False)
    false_negative.head(args.limit).to_csv(output / "afl_false_negatives.csv", index=False)
    _plot_examples(
        false_positive,
        Path(cfg.paths.data_root),
        output / "afl_false_positives.png",
        "AFL false positives",
    )
    _plot_examples(
        false_negative,
        Path(cfg.paths.data_root),
        output / "afl_false_negatives.png",
        "AFL false negatives",
    )
    print(
        {
            "false_positives": len(false_positive),
            "false_negatives": len(false_negative),
            "false_positives_labeled_afib": int(false_positive.true_afib.sum()),
        }
    )


if __name__ == "__main__":
    main()
