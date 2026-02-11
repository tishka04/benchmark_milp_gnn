"""
Training configuration for the Hierarchical Temporal Encoder (v3 scenarios).
"""

from dataclasses import dataclass, field
from typing import Tuple
import torch


@dataclass
class TrainingConfig:
    """Training configuration for Hierarchical Temporal Encoder on v3 data."""

    # --------------- paths (Google Drive / Colab) ---------------
    repo_path: str = "/content/drive/MyDrive/benchmark"
    data_dir: str = "/content/drive/MyDrive/benchmark/outputs/graphs/hetero_temporal_v3"
    scenario_dir: str = "/content/drive/MyDrive/benchmark/outputs/scenarios_v3"
    save_dir: str = "/content/drive/MyDrive/benchmark/outputs/encoders/hierarchical_temporal_v3"

    # --------------- data ---------------
    train_split: float = 0.8
    seed: int = 42

    # --------------- model architecture ---------------
    hidden_dim: int = 128
    num_spatial_layers: int = 2
    num_temporal_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 168

    # --------------- training ---------------
    epochs: int = 150
    batch_size: int = 1          # 1 graph at a time (variable N_base)
    lr: float = 1.5e-4
    max_lr: float = 1.8e-4
    weight_decay: float = 1e-5
    grad_clip: float = 0.5
    warmup_epochs: int = 10
    accumulation_steps: int = 8
    early_stopping_patience: int = 10
    save_freq: int = 10

    # --------------- loss ---------------
    lags: Tuple[int, ...] = (1, 4, 8)
    neg_sample_ratio: float = 0.25
    neg_sample_ratio_val: float = 0.10
    max_nodes: int = 160
    temperature: float = 0.10
    temperature_val: float = 0.20

    # --------------- misc ---------------
    num_workers: int = 0

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulation_steps
