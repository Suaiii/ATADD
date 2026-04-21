from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ModelConfig:
    name: str
    pretrained_name: str
    freeze_backbone: bool = True
    dropout: float = 0.1


@dataclass
class DataConfig:
    sample_rate: int = 16000
    max_seconds: float = 5.0
    num_classes: int = 2
    audio_column: str = "audio_path"
    label_column: str = "label"


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    main_metric: str = "accuracy"
    early_stop_patience: int = 10
    num_workers: int = 2


@dataclass
class AugmentConfig:
    enable: bool = False
    noise_prob: float = 0.5
    noise_scale: float = 0.003
    gain_prob: float = 0.5
    gain_db: float = 6.0


@dataclass
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    augment: AugmentConfig


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not override:
        return base
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_experiment_config(path: str, override: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    raw = _load_yaml(Path(path))
    raw = _merge(raw, override)
    return ExperimentConfig(
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw.get("data", {})),
        train=TrainConfig(**raw.get("train", {})),
        augment=AugmentConfig(**raw.get("augment", {})),
    )

