#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ekg_stage2.config import as_plain_dict, ensure_output_directories, load_config
from ekg_stage2.data.dataset import ECGDataset, positive_class_weights
from ekg_stage2.data.preprocessing import AugmentationConfig, NormalizationStats
from ekg_stage2.models import (
    ECGResNet1D,
    RhythmFeatureStructuredECGResNet1D,
    StructuredECGResNet1D,
)
from ekg_stage2.reproducibility import environment_snapshot, seed_everything
from ekg_stage2.rhythm import RhythmFeatureStats
from ekg_stage2.structured import StructuredCrossEntropyLoss, structured_class_weights
from ekg_stage2.training import fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--resume", help="Path to a last.pt checkpoint to resume")
    parser.add_argument(
        "--init-checkpoint", help="Load model weights only and start a new optimizer/run"
    )
    args = parser.parse_args()
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint cannot be used together")
    cfg = load_config(args.config)
    ensure_output_directories(cfg)
    seed_everything(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_manifest = pd.read_csv(Path(cfg.paths.manifests) / "train.csv")
    validation_manifest = pd.read_csv(Path(cfg.paths.manifests) / "validation.csv")
    if args.max_records:
        train_manifest = train_manifest.iloc[: args.max_records].copy()
        validation_manifest = validation_manifest.iloc[: max(32, args.max_records // 4)].copy()
    stats = NormalizationStats.load(str(Path(cfg.paths.stats) / "normalization.npz"))
    preprocessing = {
        "sampling_rate": float(cfg.data.sampling_rate),
        "low_hz": float(cfg.data.bandpass_low_hz),
        "high_hz": float(cfg.data.bandpass_high_hz),
        "filter_order": int(cfg.data.filter_order),
        "clip_millivolts": float(cfg.data.clip_millivolts),
    }
    augmentation = None
    if bool(cfg.augmentation.enabled):
        augmentation = AugmentationConfig(
            **{key: value for key, value in dict(cfg.augmentation).items() if key != "enabled"}
        )
    use_rhythm_features = bool(cfg.model.get("rhythm_features", False))
    rhythm_stats = None
    train_rhythm_features = None
    validation_rhythm_features = None
    if use_rhythm_features:
        rhythm_stats = RhythmFeatureStats.load(Path(cfg.paths.stats) / "rhythm_feature_stats.npz")
        train_rhythm_features = Path(cfg.paths.stats) / "rhythm_features_train.csv"
        validation_rhythm_features = Path(cfg.paths.stats) / "rhythm_features_validation.csv"
    train_dataset = ECGDataset(
        train_manifest,
        cfg.paths.data_root,
        stats,
        training=True,
        augmentation=augmentation,
        seed=int(cfg.seed),
        preprocessing=preprocessing,
        rhythm_features=train_rhythm_features,
        rhythm_stats=rhythm_stats,
    )
    validation_dataset = ECGDataset(
        validation_manifest,
        cfg.paths.data_root,
        stats,
        preprocessing=preprocessing,
        rhythm_features=validation_rhythm_features,
        rhythm_stats=rhythm_stats,
    )
    loader_options = {
        "batch_size": int(cfg.training.batch_size),
        "num_workers": int(cfg.training.num_workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(cfg.training.num_workers) > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    model_options = {
        "stem_channels": int(cfg.model.stem_channels),
        "stage_channels": tuple(cfg.model.stage_channels),
        "blocks_per_stage": tuple(cfg.model.blocks_per_stage),
        "kernel_size": int(cfg.model.kernel_size),
        "dropout": float(cfg.model.dropout),
    }
    head = str(cfg.model.get("head", "independent"))
    if head == "structured":
        model = (
            RhythmFeatureStructuredECGResNet1D(**model_options)
            if use_rhythm_features
            else StructuredECGResNet1D(**model_options)
        ).to(device)
        rhythm_weights, conduction_weights = structured_class_weights(
            train_manifest, power=float(cfg.model.get("class_weight_power", 0.5))
        )
        criterion = StructuredCrossEntropyLoss(
            rhythm_weights.to(device),
            conduction_weights.to(device),
            rhythm_loss=str(cfg.model.get("rhythm_loss", "cross_entropy")),
            focal_gamma=float(cfg.model.get("focal_gamma", 2.0)),
        )
    elif head == "independent":
        model = ECGResNet1D(**model_options).to(device)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=positive_class_weights(train_manifest).to(device)
        )
    else:
        raise ValueError(f"Unknown model head: {head}")
    init_checkpoint = None
    if args.init_checkpoint:
        init_checkpoint = Path(args.init_checkpoint).resolve()
        if not init_checkpoint.is_file():
            raise FileNotFoundError(init_checkpoint)
        checkpoint = torch.load(init_checkpoint, map_location=device, weights_only=False)
        load_result = model.load_state_dict(
            checkpoint["model_state_dict"], strict=not bool(cfg.model.get("partial_init", False))
        )
        allowed_missing_prefix = "rhythm_feature_head."
        if load_result.unexpected_keys or any(
            not key.startswith(allowed_missing_prefix) for key in load_result.missing_keys
        ):
            raise ValueError(
                f"Unexpected checkpoint mismatch: missing={load_result.missing_keys}, "
                f"unexpected={load_result.unexpected_keys}"
            )
    feature_branch_only = bool(cfg.training.get("feature_branch_only", False))
    if feature_branch_only:
        for parameter in model.parameters():
            parameter.requires_grad = False
        feature_head = getattr(model, "rhythm_feature_head", None)
        if feature_head is None:
            raise ValueError("Feature-branch-only training requires rhythm features")
        for parameter in feature_head.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    if args.resume:
        resume_path = Path(args.resume).resolve()
        output = resume_path.parent
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
    else:
        resume_path = None
        run_name = args.run_name or datetime.now(UTC).strftime("resnet_%Y%m%dT%H%M%SZ")
        output = Path(cfg.paths.runs) / run_name
        output.mkdir(parents=True, exist_ok=False)
        run_metadata = {
            "config": as_plain_dict(cfg),
            "environment": environment_snapshot(),
            "device": str(device),
            "train_records": len(train_manifest),
            "validation_records": len(validation_manifest),
            "init_checkpoint": None if init_checkpoint is None else str(init_checkpoint),
        }
        (output / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    fit(
        model,
        train_loader,
        validation_loader,
        optimizer,
        criterion,
        device,
        output,
        epochs=int(cfg.training.epochs),
        patience=int(cfg.training.patience),
        gradient_clip_norm=float(cfg.training.gradient_clip_norm),
        amp=bool(cfg.training.amp),
        threshold_options={
            "minimum": float(cfg.training.threshold_min),
            "maximum": float(cfg.training.threshold_max),
            "grid_size": int(cfg.training.threshold_grid_size),
        },
        resume_path=resume_path,
        scheduler_name=str(cfg.training.get("scheduler", "cosine")),
        scheduler_options={
            "factor": float(cfg.training.get("plateau_factor", 0.3)),
            "patience": int(cfg.training.get("plateau_patience", 2)),
            "min_lr": float(cfg.training.get("minimum_learning_rate", 1e-6)),
        },
        feature_branch_only=feature_branch_only,
    )


if __name__ == "__main__":
    main()
