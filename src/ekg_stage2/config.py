from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    """Load and resolve a YAML configuration, independent of the caller's cwd."""
    config_path = Path(path).expanduser().resolve()
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    return cfg


def ensure_output_directories(cfg: DictConfig) -> None:
    for key in ("output_root", "manifests", "audit", "stats", "runs"):
        Path(cfg.paths[key]).mkdir(parents=True, exist_ok=True)


def as_plain_dict(cfg: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("Configuration root must be a mapping")
    return value

