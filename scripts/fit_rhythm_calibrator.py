#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from ekg_stage2.config import load_config
from ekg_stage2.metrics import multilabel_metrics
from ekg_stage2.rhythm import RHYTHM_FEATURE_NAMES, RhythmFeatureStats


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
        "--predictions",
        default="outputs/runs/structured_heads_seed20260621/best_validation_predictions.npz",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--afib-weight", type=float, default=1.5)
    parser.add_argument("--afl-weight", type=float, default=2.0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    artifact = np.load(args.predictions)
    targets = artifact["targets"].astype(np.int8)
    source_probabilities = artifact["probabilities"].astype(np.float64)
    source_predictions = artifact["predictions"].astype(np.int8)
    study_ids = artifact["study_ids"].astype(int)
    subject_ids = artifact["subject_ids"].astype(int)

    cached = pd.read_csv(Path(cfg.paths.stats) / "rhythm_features_validation.csv")
    cached = cached.set_index("study_id").loc[study_ids]
    stats = RhythmFeatureStats.load(Path(cfg.paths.stats) / "rhythm_feature_stats.npz")
    rhythm_features = stats.transform(
        cached[list(RHYTHM_FEATURE_NAMES)].to_numpy(dtype=np.float32),
        cached["valid"].astype(bool).to_numpy(),
    )
    design = _design_matrix(source_probabilities, rhythm_features)
    rhythm_targets = _rhythm_targets(targets)
    class_weight = {0: 1.0, 1: 1.0, 2: args.afib_weight, 3: args.afl_weight}
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=int(cfg.seed)
    )
    oof_classes = np.zeros(len(targets), dtype=np.int64)
    oof_rhythm_probabilities = np.zeros((len(targets), 4), dtype=np.float64)
    fold_rows: list[dict[str, float | int]] = []
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(design, rhythm_targets, groups=subject_ids), start=1
    ):
        model = LogisticRegression(
            C=args.c, class_weight=class_weight, max_iter=500, solver="lbfgs"
        ).fit(design[train_indices], rhythm_targets[train_indices])
        oof_classes[validation_indices] = model.predict(design[validation_indices])
        oof_rhythm_probabilities[validation_indices] = model.predict_proba(
            design[validation_indices]
        )
        fold_predictions = _decode(
            oof_classes[validation_indices], source_predictions[validation_indices]
        )
        fold_metrics = multilabel_metrics(
            targets[validation_indices],
            np.column_stack(
                (
                    oof_rhythm_probabilities[validation_indices, 1:],
                    source_probabilities[validation_indices, 3:],
                )
            ),
            None,
            predictions=fold_predictions,
        )
        fold_rows.append(
            {
                "fold": fold,
                "records": len(validation_indices),
                "macro_f1": float(fold_metrics["macro_f1"]),
                "exact_match_accuracy": float(fold_metrics["exact_match_accuracy"]),
            }
        )

    calibrated_probabilities = np.column_stack(
        (oof_rhythm_probabilities[:, 1:], source_probabilities[:, 3:])
    )
    calibrated_predictions = _decode(oof_classes, source_predictions)
    source_metrics = multilabel_metrics(
        targets, source_probabilities, None, predictions=source_predictions
    )
    calibrated_metrics = multilabel_metrics(
        targets, calibrated_probabilities, None, predictions=calibrated_predictions
    )
    final_model = LogisticRegression(
        C=args.c, class_weight=class_weight, max_iter=500, solver="lbfgs"
    ).fit(design, rhythm_targets)

    output = Path(cfg.paths.output_root) / "calibration"
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, output / "rhythm_calibrator.joblib")
    np.savez_compressed(
        output / "rhythm_calibration_oof_predictions.npz",
        targets=targets,
        probabilities=calibrated_probabilities,
        predictions=calibrated_predictions,
        study_ids=study_ids,
        subject_ids=subject_ids,
    )
    report = {
        "folds": fold_rows,
        "class_weight": class_weight,
        "c": args.c,
        "source_macro_f1": source_metrics["macro_f1"],
        "calibrated_macro_f1": calibrated_metrics["macro_f1"],
        "source_exact_match_accuracy": source_metrics["exact_match_accuracy"],
        "calibrated_exact_match_accuracy": calibrated_metrics["exact_match_accuracy"],
        "source_per_label": source_metrics["per_label"],
        "calibrated_per_label": calibrated_metrics["per_label"],
    }
    (output / "rhythm_calibration_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
