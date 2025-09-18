from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    index_path: Path
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    batch_size: int = 16
    num_workers: int = 0
    shuffle: bool = True
    pin_memory: bool = False
    include_duals: bool = True
    preload: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_dir: Path) -> "DataConfig":
        data = dict(data)
        data["index_path"] = _resolve_path(base_dir, data["index_path"])
        return cls(**data)


@dataclass
class ModelConfig:
    name: str
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    dropout: float = 0.0
    activation: str = "relu"
    heads: int = 4
    aggregator: str = "mean"
    typed_message_passing: bool = False
    output_dim: int = 8
    type_embedding_dim: int = 16
    attn_dropout: float = 0.1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(**data)


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: Optional[List[float]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizerConfig":
        return cls(**data)


@dataclass
class SchedulerConfig:
    name: Optional[str] = None
    warmup_steps: int = 0
    decay_steps: int = 0
    min_lr: float = 0.0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SchedulerConfig":
        if data is None:
            return cls()
        return cls(**data)


@dataclass
class LoopConfig:
    epochs: int = 50
    device: str = "cpu"
    gradient_clip_norm: Optional[float] = None
    log_every: int = 50
    eval_every: int = 1
    save_every: int = 1
    seed: Optional[int] = None
    loss: str = "mse"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoopConfig":
        return cls(**data)


@dataclass
class DecoderConfig:
    name: str = "feasibility"
    enforce_nonneg: bool = True
    respect_capacity: bool = True
    dual_keys: Optional[List[str]] = None
    blend_dual_fraction: float = 0.3
    residual_tolerance: float = 1e-3

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "DecoderConfig":
        if data is None:
            return cls()
        return cls(**data)


@dataclass
class MetricsConfig:
    dispatch_error: Dict[str, Any] = field(default_factory=dict)
    cost_gap: Dict[str, Any] = field(default_factory=dict)
    constraint_violation_rate: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "MetricsConfig":
        if data is None:
            return cls()
        return cls(**data)


@dataclass
class TrainingConfig:
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    output_dir: Path = Path("outputs/gnn_runs/default")

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_dir: Path) -> "TrainingConfig":
        data = dict(data)
        output_dir = data.get("output_dir")
        if output_dir is not None:
            data["output_dir"] = _resolve_path(base_dir, output_dir)
        data_cfg = DataConfig.from_dict(data.pop("data"), base_dir)
        model_cfg = ModelConfig.from_dict(data.pop("model"))
        opt_cfg = OptimizerConfig.from_dict(data.pop("optimizer", {}))
        sched_cfg = SchedulerConfig.from_dict(data.pop("scheduler", None))
        loop_cfg = LoopConfig.from_dict(data.pop("loop", {}))
        decoder_cfg = DecoderConfig.from_dict(data.pop("decoder", None))
        metrics_cfg = MetricsConfig.from_dict(data.pop("metrics", None))
        return cls(
            data=data_cfg,
            model=model_cfg,
            optimizer=opt_cfg,
            scheduler=sched_cfg,
            loop=loop_cfg,
            decoder=decoder_cfg,
            metrics=metrics_cfg,
            **data,
        )


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] in {'.', '..'}:
        base = base_dir
    else:
        base = Path.cwd()
    return (base / path).resolve()


def load_training_config(path: Path | str) -> TrainingConfig:
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"Training config at {path} must be a mapping")
    return TrainingConfig.from_dict(raw, cfg_path.parent)
