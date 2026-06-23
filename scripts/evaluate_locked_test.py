#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ekg_stage2.config import ensure_output_directories, load_config
from ekg_stage2.data.dataset import ECGDataset
from ekg_stage2.data.preprocessing import NormalizationStats
from ekg_stage2.metrics import multilabel_metrics
from ekg_stage2.models import (
    RhythmFeatureStructuredECGResNet1D,
    StructuredECGResNet1D,
)
from ekg_stage2.rhythm import RHYTHM_FEATURE_NAMES, RhythmFeatureStats
from ekg_stage2.structured import decode_structured_outputs


def _rhythm_targets(targets: np.ndarray) -> np.ndarray:
    active = targets[:, :3].sum(axis=1) > 0
    return np.where(active, targets[:, :3].argmax(axis=1) + 1, 0)


def _design_matrix(probabilities: np.ndarray, rhythm_features: np.ndarray) -> np.ndarray:
    none_probability = np.clip(1.0 - probabilities[:, :3].sum(axis=1), 1e-6, 1.0)
    rhythm_probabilities = np.column_stack((none_probability, probabilities[:, :3]))
    rhythm_probabilities = np.clip(rhythm_probabilities, 1e-6, 1.0)
    rhythm_probabilities /= rhythm_probabilities.sum(axis=1, keepdims=True)
    return np.column_stack((np.log(rhythm_probabilities), rhythm_features))


def _decode(classes: np.ndarray, source_predictions: np.ndarray) -> np.ndarray:
    predictions = source_predictions.copy()
    predictions[:, :3] = 0
    active = classes > 0
    predictions[np.flatnonzero(active), classes[active] - 1] = 1
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--checkpoint",
        default="outputs/runs/structured_heads_seed20260621/best.pt",
        help="Frozen source checkpoint to evaluate on the test split",
    )
    parser.add_argument(
        "--calibrator",
        default="outputs/calibration/rhythm_calibrator.joblib",
        help="Optional calibrated rhythm classifier trained on validation only",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_directories(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    test_manifest = pd.read_csv(Path(cfg.paths.manifests) / "test.csv")
    stats = NormalizationStats.load(str(Path(cfg.paths.stats) / "normalization.npz"))
    preprocessing = {
        "sampling_rate": float(cfg.data.sampling_rate),
        "low_hz": float(cfg.data.bandpass_low_hz),
        "high_hz": float(cfg.data.bandpass_high_hz),
        "filter_order": int(cfg.data.filter_order),
        "clip_millivolts": float(cfg.data.clip_millivolts),
    }
    use_rhythm_features = bool(cfg.model.get("rhythm_features", False))
    rhythm_stats = None
    rhythm_feature_path = None
    if use_rhythm_features or Path(args.calibrator).is_file():
        rhythm_stats = RhythmFeatureStats.load(Path(cfg.paths.stats) / "rhythm_feature_stats.npz")
        rhythm_feature_path = Path(cfg.paths.stats) / "rhythm_features_test.csv"
        if not rhythm_feature_path.is_file():
            raise FileNotFoundError(
                "Missing cached test rhythm features. "
                "Run scripts/compute_rhythm_features.py --splits test first."
            )

    dataset = ECGDataset(
        test_manifest,
        cfg.paths.data_root,
        stats,
        preprocessing=preprocessing,
        rhythm_features=rhythm_feature_path if use_rhythm_features else None,
        rhythm_stats=rhythm_stats if use_rhythm_features else None,
    )
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size or int(cfg.training.batch_size),
        num_workers=(
            args.num_workers if args.num_workers is not None else int(cfg.training.num_workers)
        ),
        pin_memory=device.type == "cuda",
        persistent_workers=(
            args.num_workers if args.num_workers is not None else int(cfg.training.num_workers)
        )
        > 0,
    )

    model_options = {
        "stem_channels": int(cfg.model.stem_channels),
        "stage_channels": tuple(cfg.model.stage_channels),
        "blocks_per_stage": tuple(cfg.model.blocks_per_stage),
        "kernel_size": int(cfg.model.kernel_size),
        "dropout": float(cfg.model.dropout),
    }
    model = (
        RhythmFeatureStructuredECGResNet1D(**model_options)
        if use_rhythm_features
        else StructuredECGResNet1D(**model_options)
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError(
            "Checkpoint mismatch: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )

    model.eval()
    collections: dict[str, list[np.ndarray]] = {
        "targets": [],
        "probabilities": [],
        "predictions": [],
        "study_ids": [],
        "subject_ids": [],
    }
    with torch.inference_mode():
        for batch in loader:
            signals = batch["signal"].to(device, non_blocking=True)
            rhythm_features = batch.get("rhythm_features")
            if rhythm_features is not None:
                rhythm_features = rhythm_features.to(device, non_blocking=True)
            outputs = (
                model(signals, rhythm_features=rhythm_features)
                if rhythm_features is not None
                else model(signals)
            )
            probabilities, predictions = decode_structured_outputs(outputs)
            collections["targets"].append(batch["labels"].cpu().numpy())
            collections["probabilities"].append(probabilities.float().cpu().numpy())
            collections["predictions"].append(predictions.cpu().numpy())
            collections["study_ids"].append(np.asarray(batch["study_id"]))
            collections["subject_ids"].append(np.asarray(batch["subject_id"]))

    targets = np.concatenate(collections["targets"])
    source_probabilities = np.concatenate(collections["probabilities"])
    source_predictions = np.concatenate(collections["predictions"])
    study_ids = np.concatenate(collections["study_ids"])
    subject_ids = np.concatenate(collections["subject_ids"])

    source_metrics = multilabel_metrics(
        targets, source_probabilities, None, predictions=source_predictions
    )

    report: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "test_records": int(len(test_manifest)),
        "source_macro_f1": source_metrics["macro_f1"],
        "source_exact_match_accuracy": source_metrics["exact_match_accuracy"],
        "source_per_label": source_metrics["per_label"],
    }

    output = Path(cfg.paths.output_root) / "locked_test"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "source_test_predictions.npz",
        targets=targets,
        probabilities=source_probabilities,
        predictions=source_predictions,
        study_ids=study_ids,
        subject_ids=subject_ids,
    )

    calibrator_path = Path(args.calibrator).resolve()
    if calibrator_path.is_file():
        if rhythm_feature_path is None:
            raise ValueError("Calibration requires cached test rhythm features")
        cached = pd.read_csv(rhythm_feature_path).set_index("study_id").loc[study_ids]
        stats = RhythmFeatureStats.load(Path(cfg.paths.stats) / "rhythm_feature_stats.npz")
        rhythm_features = stats.transform(
            cached[list(RHYTHM_FEATURE_NAMES)].to_numpy(dtype=np.float32),
            cached["valid"].astype(bool).to_numpy(),
        )
        design = _design_matrix(source_probabilities, rhythm_features)
        calibrator = joblib.load(calibrator_path)
        calibrated_classes = calibrator.predict(design)
        calibrated_rhythm_probabilities = calibrator.predict_proba(design)
        calibrated_probabilities = np.column_stack(
            (calibrated_rhythm_probabilities[:, 1:], source_probabilities[:, 3:])
        )
        calibrated_predictions = _decode(calibrated_classes, source_predictions)
        calibrated_metrics = multilabel_metrics(
            targets, calibrated_probabilities, None, predictions=calibrated_predictions
        )
        report.update(
            {
                "calibrator": str(calibrator_path),
                "calibrated_macro_f1": calibrated_metrics["macro_f1"],
                "calibrated_exact_match_accuracy": calibrated_metrics["exact_match_accuracy"],
                "calibrated_per_label": calibrated_metrics["per_label"],
            }
        )
        np.savez_compressed(
            output / "calibrated_test_predictions.npz",
            targets=targets,
            probabilities=calibrated_probabilities,
            predictions=calibrated_predictions,
            study_ids=study_ids,
            subject_ids=subject_ids,
        )

    (output / "locked_test_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
