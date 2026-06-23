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


def _to_heads(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rhythm_none = np.clip(1.0 - probabilities[:, :3].sum(axis=1), 1e-6, 1.0)
    rhythm = np.column_stack((rhythm_none, probabilities[:, :3]))
    rhythm /= rhythm.sum(axis=1, keepdims=True)
    conduction_none = np.clip(1.0 - probabilities[:, 3:].sum(axis=1), 1e-6, 1.0)
    conduction = np.column_stack((conduction_none, probabilities[:, 3:]))
    conduction /= conduction.sum(axis=1, keepdims=True)
    return rhythm, conduction


def _decode_heads(rhythm: np.ndarray, conduction: np.ndarray) -> np.ndarray:
    predictions = np.zeros((len(rhythm), 5), dtype=np.int8)
    rhythm_class = rhythm.argmax(axis=1)
    conduction_class = conduction.argmax(axis=1)
    rhythm_active = rhythm_class > 0
    conduction_active = conduction_class > 0
    predictions[np.flatnonzero(rhythm_active), rhythm_class[rhythm_active] - 1] = 1
    predictions[np.flatnonzero(conduction_active), conduction_class[conduction_active] + 2] = 1
    return predictions


def _design_matrix(probabilities: np.ndarray, rhythm_features: np.ndarray) -> np.ndarray:
    rhythm_probabilities, _ = _to_heads(probabilities)
    return np.column_stack((np.log(np.clip(rhythm_probabilities, 1e-6, 1.0)), rhythm_features))


def _collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    *,
    use_rhythm_features: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                outputs = (
                    model(signals, rhythm_features=rhythm_features)
                    if use_rhythm_features
                    else model(signals)
                )
            probabilities, predictions = decode_structured_outputs(outputs)
            collections["targets"].append(batch["labels"].cpu().numpy())
            collections["probabilities"].append(probabilities.float().cpu().numpy())
            collections["predictions"].append(predictions.cpu().numpy())
            collections["study_ids"].append(np.asarray(batch["study_id"]))
            collections["subject_ids"].append(np.asarray(batch["subject_id"]))
    return (
        np.concatenate(collections["targets"]),
        np.concatenate(collections["probabilities"]),
        np.concatenate(collections["predictions"]),
        np.concatenate(collections["study_ids"]),
        np.concatenate(collections["subject_ids"]),
    )


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
    parser.add_argument(
        "--blend-checkpoint",
        help="Optional second structured checkpoint to blend with the calibrated source model",
    )
    parser.add_argument(
        "--blend-weight",
        type=float,
        default=0.22,
        help="Weight for the blend checkpoint; the calibrated source gets the remaining weight",
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

    (
        targets,
        source_probabilities,
        source_predictions,
        study_ids,
        subject_ids,
    ) = _collect_predictions(
        model, loader, device, bool(cfg.training.amp), use_rhythm_features=use_rhythm_features
    )

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
        calibrator = joblib.load(calibrator_path)
        design = _design_matrix(source_probabilities, rhythm_features)
        calibrated_rhythm_probabilities = calibrator.predict_proba(design)
        calibrated_probabilities = np.column_stack(
            (calibrated_rhythm_probabilities[:, 1:], source_probabilities[:, 3:])
        )
        calibrated_predictions = _decode_heads(
            calibrated_rhythm_probabilities, _to_heads(source_probabilities)[1]
        )
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

        if args.blend_checkpoint:
            blend_checkpoint_path = Path(args.blend_checkpoint).resolve()
            if not blend_checkpoint_path.is_file():
                raise FileNotFoundError(blend_checkpoint_path)
            blend_model = StructuredECGResNet1D(
                stem_channels=int(cfg.model.stem_channels),
                stage_channels=tuple(cfg.model.stage_channels),
                blocks_per_stage=tuple(cfg.model.blocks_per_stage),
                kernel_size=int(cfg.model.kernel_size),
                dropout=float(cfg.model.dropout),
            ).to(device)
            blend_checkpoint = torch.load(
                blend_checkpoint_path, map_location=device, weights_only=False
            )
            blend_load_result = blend_model.load_state_dict(
                blend_checkpoint["model_state_dict"], strict=True
            )
            if blend_load_result.missing_keys or blend_load_result.unexpected_keys:
                raise ValueError(
                    "Checkpoint mismatch: "
                    f"missing={blend_load_result.missing_keys}, "
                    f"unexpected={blend_load_result.unexpected_keys}"
                )
            _, blend_probabilities, _, _, _ = _collect_predictions(
                blend_model,
                loader,
                device,
                bool(cfg.training.amp),
            )
            blend_rhythm, blend_conduction = _to_heads(blend_probabilities)
            source_rhythm, source_conduction = _to_heads(calibrated_probabilities)
            blend_weight = float(args.blend_weight)
            blended_rhythm = blend_weight * blend_rhythm + (1.0 - blend_weight) * source_rhythm
            blended_conduction = (
                blend_weight * blend_conduction + (1.0 - blend_weight) * source_conduction
            )
            blended_predictions = _decode_heads(blended_rhythm, blended_conduction)
            blended_probabilities = np.column_stack(
                (blended_rhythm[:, 1:], blended_conduction[:, 1:])
            )
            blended_metrics = multilabel_metrics(
                targets, blended_probabilities, None, predictions=blended_predictions
            )
            report.update(
                {
                    "blend_checkpoint": str(blend_checkpoint_path),
                    "blend_weight": blend_weight,
                    "blended_macro_f1": blended_metrics["macro_f1"],
                    "blended_exact_match_accuracy": blended_metrics[
                        "exact_match_accuracy"
                    ],
                    "blended_per_label": blended_metrics["per_label"],
                }
            )
            np.savez_compressed(
                output / "blended_test_predictions.npz",
                targets=targets,
                probabilities=blended_probabilities,
                predictions=blended_predictions,
                study_ids=study_ids,
                subject_ids=subject_ids,
            )

    (output / "locked_test_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
