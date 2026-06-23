# EKG Stage 2

Reproducible, leakage-safe multi-label ECG classification pipeline for the five
labels `NORMAL`, `AFIB`, `AFL`, `LBBB`, and `RBBB`.

## Implemented scope

- strict metadata schema and WFDB path validation;
- deterministic patient-level iterative multi-label train/validation/test split;
- split leakage and distribution reports;
- WFDB loading with canonical lead ordering and configurable signal checks;
- train-only normalization statistics and ECG augmentations;
- streaming PyTorch dataset (no full dataset load into RAM);
- compact 1D ResNet with squeeze-and-excitation;
- weighted BCE training, validation threshold optimization, and early stopping;
- complete multi-label metrics and patient-level bootstrap confidence intervals;
- environment and real-record smoke checks.

## Quick start

Run commands from this directory:

```bash
export PYTHONPATH="$PWD/src"

python scripts/check_environment.py
python scripts/build_manifests.py --config configs/default.yaml
python scripts/audit_waveforms.py --config configs/default.yaml --sample-size 1000
pytest -q
```

Before normalization or training, create a complete quality audit and rebuild
the frozen manifests from valid waveforms only:

```bash
python scripts/audit_waveforms.py --config configs/default.yaml --all --workers 8
python scripts/build_manifests.py --config configs/default.yaml \
  --quality-audit outputs/audit/waveform_audit.csv
```

The test manifest is frozen by `manifest_fingerprint.json`. Re-running the split
with the same metadata and seed is deterministic. Do not use the test manifest
during model or threshold selection.

Training starts only after normalization statistics have been computed from the
training manifest:

```bash
python scripts/compute_normalization.py --config configs/default.yaml --workers 8
python scripts/train.py --config configs/default.yaml
```

Interrupted training can resume from its per-epoch checkpoint:

```bash
python scripts/train.py --config configs/default.yaml \
  --resume outputs/runs/<run-name>/last.pt
```

The structured-head experiment models rhythm (`none/NORMAL/AFIB/AFL`) and
conduction (`none/LBBB/RBBB`) as two mutually exclusive groups:

```bash
python scripts/train.py --config configs/structured.yaml
```

To start a new low-learning-rate run from the best structured weights while
resetting the optimizer and scheduler:

```bash
python scripts/train.py --config configs/structured_finetune.yaml \
  --init-checkpoint outputs/runs/structured_heads_seed20260621/best.pt
```

For a fast end-to-end smoke test, pass `--max-records 256` to the normalization
and training scripts.

## Output layout

All generated artifacts remain beneath `codex_run/outputs/`:

- `manifests/`: frozen split CSVs, summary, and fingerprint;
- `audit/`: waveform inspection results;
- `stats/`: train-only normalization statistics;
- `runs/`: checkpoints, validation predictions, thresholds, and history.

The raw dataset is treated as read-only.
