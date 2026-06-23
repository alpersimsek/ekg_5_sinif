from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ekg_stage2.metrics import multilabel_metrics
from ekg_stage2.structured import decode_structured_outputs
from ekg_stage2.thresholds import optimize_thresholds


@dataclass
class EpochResult:
    loss: float
    targets: np.ndarray
    probabilities: np.ndarray
    study_ids: np.ndarray
    subject_ids: np.ndarray
    predictions: np.ndarray | None = None


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.bad_epochs = 0

    def update(self, value: float) -> bool:
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str,
    epochs: int,
    options: dict[str, float | int] | None = None,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau:
    options = options or {}
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(options.get("factor", 0.3)),
            patience=int(options.get("patience", 2)),
            min_lr=float(options.get("min_lr", 1e-6)),
        )
    raise ValueError(f"Unknown scheduler: {name}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    gradient_clip_norm: float,
    amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_records = 0
    for batch in loader:
        signals = batch["signal"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(signals)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * len(signals)
        total_records += len(signals)
    return total_loss / total_records


@torch.inference_mode()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
) -> EpochResult:
    model.eval()
    total_loss = 0.0
    total_records = 0
    collections: dict[str, list[np.ndarray]] = {
        "targets": [],
        "probabilities": [],
        "study_ids": [],
        "subject_ids": [],
        "predictions": [],
    }
    structured = False
    for batch in loader:
        signals = batch["signal"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(signals)
            loss = criterion(outputs, targets)
        total_loss += float(loss) * len(signals)
        total_records += len(signals)
        collections["targets"].append(targets.cpu().numpy())
        if isinstance(outputs, dict):
            probabilities, predictions = decode_structured_outputs(outputs)
            structured = True
            collections["predictions"].append(predictions.cpu().numpy())
        else:
            probabilities = outputs.sigmoid()
        collections["probabilities"].append(probabilities.float().cpu().numpy())
        collections["study_ids"].append(np.asarray(batch["study_id"]))
        collections["subject_ids"].append(np.asarray(batch["subject_id"]))
    return EpochResult(
        loss=total_loss / total_records,
        targets=np.concatenate(collections["targets"]),
        probabilities=np.concatenate(collections["probabilities"]),
        study_ids=np.concatenate(collections["study_ids"]),
        subject_ids=np.concatenate(collections["subject_ids"]),
        predictions=np.concatenate(collections["predictions"]) if structured else None,
    )


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    output_dir: str | Path,
    epochs: int,
    patience: int,
    gradient_clip_norm: float,
    amp: bool,
    threshold_options: dict[str, float | int],
    resume_path: str | Path | None = None,
    scheduler_name: str = "cosine",
    scheduler_options: dict[str, float | int] | None = None,
) -> list[dict[str, object]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    scheduler = create_scheduler(optimizer, scheduler_name, epochs, scheduler_options)
    stopper = EarlyStopping(patience=patience)
    history: list[dict[str, object]] = []
    start_epoch = 0

    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        stopper.best = float(checkpoint["early_stopping_best"])
        stopper.bad_epochs = int(checkpoint["early_stopping_bad_epochs"])
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"])

    for epoch in range(start_epoch, epochs):
        dataset = train_loader.dataset
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            gradient_clip_norm,
            amp,
        )
        validation = evaluate_epoch(model, validation_loader, criterion, device, amp)
        thresholds = (
            None
            if validation.predictions is not None
            else optimize_thresholds(
                validation.targets, validation.probabilities, **threshold_options
            )
        )
        metrics = multilabel_metrics(
            validation.targets,
            validation.probabilities,
            thresholds,
            predictions=validation.predictions,
        )
        row: dict[str, object] = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation.loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "thresholds": None if thresholds is None else thresholds.tolist(),
            "macro_f1": metrics["macro_f1"],
            "exact_match_accuracy": metrics["exact_match_accuracy"],
        }
        history.append(row)
        improved = stopper.update(float(metrics["macro_f1"]))
        if improved:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "thresholds": thresholds,
                    "decision_rule": (
                        "structured_argmax" if validation.predictions is not None else "thresholds"
                    ),
                    "metrics": metrics,
                },
                output / "best.pt",
            )
            prediction_artifact = {
                "targets": validation.targets,
                "probabilities": validation.probabilities,
                "study_ids": validation.study_ids,
                "subject_ids": validation.subject_ids,
            }
            if thresholds is not None:
                prediction_artifact["thresholds"] = thresholds
            if validation.predictions is not None:
                prediction_artifact["predictions"] = validation.predictions
            np.savez_compressed(output / "best_validation_predictions.npz", **prediction_artifact)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if scheduler_name == "plateau":
            scheduler.step(float(metrics["macro_f1"]))
        else:
            scheduler.step()
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "early_stopping_best": stopper.best,
                "early_stopping_bad_epochs": stopper.bad_epochs,
                "epoch": epoch + 1,
                "history": history,
            },
            output / "last.pt",
        )
        print(json.dumps(row), flush=True)
        if stopper.should_stop:
            break
    return history
