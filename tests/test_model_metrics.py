from __future__ import annotations

import numpy as np
import torch

from ekg_stage2.metrics import multilabel_metrics, patient_bootstrap_confidence_intervals
from ekg_stage2.models import (
    ECGResNet1D,
    RhythmFeatureStructuredECGResNet1D,
    StructuredECGResNet1D,
)
from ekg_stage2.rhythm import RHYTHM_FEATURE_NAMES, RhythmFeatureStats
from ekg_stage2.structured import (
    FocalCrossEntropyLoss,
    StructuredCrossEntropyLoss,
    decode_structured_outputs,
    encode_structured_targets,
)
from ekg_stage2.thresholds import apply_thresholds, optimize_thresholds
from ekg_stage2.training import create_scheduler


def test_model_output_shape_and_backward() -> None:
    model = ECGResNet1D(
        stem_channels=16,
        stage_channels=(16, 32),
        blocks_per_stage=(1, 1),
        kernel_size=7,
    )
    signal = torch.randn(2, 12, 1000)
    output = model(signal)
    assert output.shape == (2, 5)
    output.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_structured_model_output_loss_and_decoding() -> None:
    model = StructuredECGResNet1D(
        stem_channels=16,
        stage_channels=(16, 32),
        blocks_per_stage=(1, 1),
        kernel_size=7,
    )
    signal = torch.randn(3, 12, 1000)
    labels = torch.tensor(
        [[1, 0, 0, 1, 0], [0, 1, 0, 0, 1], [0, 0, 1, 0, 0]], dtype=torch.float32
    )
    outputs = model(signal)
    assert outputs["rhythm"].shape == (3, 4)
    assert outputs["conduction"].shape == (3, 3)
    criterion = StructuredCrossEntropyLoss(torch.ones(4), torch.ones(3))
    criterion(outputs, labels).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    probabilities, predictions = decode_structured_outputs(outputs)
    assert probabilities.shape == predictions.shape == (3, 5)
    assert torch.all(predictions[:, :3].sum(dim=1) <= 1)
    assert torch.all(predictions[:, 3:].sum(dim=1) <= 1)


def test_structured_target_encoding() -> None:
    labels = torch.tensor(
        [[0, 0, 0, 0, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1], [0, 0, 1, 0, 0]]
    )
    rhythm, conduction = encode_structured_targets(labels)
    torch.testing.assert_close(rhythm, torch.tensor([0, 1, 2, 3]))
    torch.testing.assert_close(conduction, torch.tensor([0, 1, 2, 0]))


def test_focal_loss_with_zero_gamma_matches_cross_entropy() -> None:
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.2, 0.3]])
    targets = torch.tensor([0, 2])
    weights = torch.tensor([0.5, 1.0, 2.0])
    focal = FocalCrossEntropyLoss(weights, gamma=0.0)(logits, targets)
    cross_entropy = torch.nn.functional.cross_entropy(logits, targets, weight=weights)
    torch.testing.assert_close(focal, cross_entropy)


def test_zero_initialized_rhythm_feature_branch_preserves_logits() -> None:
    base = StructuredECGResNet1D(
        stem_channels=16, stage_channels=(16, 32), blocks_per_stage=(1, 1), kernel_size=7
    ).eval()
    hybrid = RhythmFeatureStructuredECGResNet1D(
        stem_channels=16, stage_channels=(16, 32), blocks_per_stage=(1, 1), kernel_size=7
    ).eval()
    result = hybrid.load_state_dict(base.state_dict(), strict=False)
    assert result.unexpected_keys == []
    assert all(key.startswith("rhythm_feature_head.") for key in result.missing_keys)
    signal = torch.randn(2, 12, 1000)
    rhythm_features = torch.randn(2, len(RHYTHM_FEATURE_NAMES) + 1)
    with torch.inference_mode():
        base_output = base(signal)
        hybrid_output = hybrid(signal, rhythm_features)
    torch.testing.assert_close(hybrid_output["rhythm"], base_output["rhythm"])
    torch.testing.assert_close(hybrid_output["conduction"], base_output["conduction"])


def test_rhythm_feature_statistics_mask_invalid_rows() -> None:
    count = len(RHYTHM_FEATURE_NAMES)
    stats = RhythmFeatureStats(np.ones(count), np.full(count, 2.0))
    values = np.vstack((np.full(count, 3.0), np.full(count, 9.0)))
    transformed = stats.transform(values, np.array([True, False]))
    np.testing.assert_allclose(transformed[0], np.r_[np.ones(count), 1.0])
    np.testing.assert_allclose(transformed[1], np.zeros(count + 1))


def test_threshold_optimization_and_metrics_on_perfect_predictions() -> None:
    targets = np.array([[1, 0, 0, 1, 0], [0, 1, 0, 0, 1], [0, 0, 1, 0, 0]])
    probabilities = targets * 0.9 + (1 - targets) * 0.1
    thresholds = optimize_thresholds(targets, probabilities)
    predictions = apply_thresholds(probabilities, thresholds)
    np.testing.assert_array_equal(predictions, targets)
    metrics = multilabel_metrics(targets, probabilities, thresholds)
    assert metrics["macro_f1"] == 1.0
    assert metrics["exact_match_accuracy"] == 1.0


def test_patient_bootstrap_keeps_valid_interval() -> None:
    targets = np.tile(np.eye(5, dtype=int), (2, 1))
    probabilities = targets * 0.9 + (1 - targets) * 0.1
    subject_ids = np.repeat(np.arange(5), 2)
    intervals = patient_bootstrap_confidence_intervals(
        targets, probabilities, subject_ids, np.full(5, 0.5), n_bootstrap=20
    )
    assert intervals["exact_match_accuracy"] == [1.0, 1.0]
    assert 0.0 <= intervals["macro_f1"][0] <= intervals["macro_f1"][1] <= 1.0


def test_plateau_scheduler_reduces_learning_rate() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler = create_scheduler(
        optimizer, "plateau", epochs=10, options={"factor": 0.3, "patience": 0}
    )
    scheduler.step(0.8)
    scheduler.step(0.7)
    assert optimizer.param_groups[0]["lr"] == 3e-5
