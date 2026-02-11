# ==============================================================================
# EBM V3 CONFIGURATION
# ==============================================================================
# Dataclass-based configuration for Graph EBM v3 training pipeline.
# Two-step training: Step A (gold pretrain) + Step B (silver finetune).
# ==============================================================================

from __future__ import annotations

import os
import torch
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class EBMv3Config:
    """Configuration for Graph EBM v3 two-step training."""

    # ── Paths (Colab: /content/drive/MyDrive/benchmark) ──
    base_dir: str = "/content/drive/MyDrive/benchmark"

    @property
    def reports_dir(self) -> str:
        return os.path.join(self.base_dir, "outputs", "scenarios_v3", "reports")

    @property
    def scenarios_dir(self) -> str:
        return os.path.join(self.base_dir, "outputs", "scenarios_v3")

    @property
    def classification_index_path(self) -> str:
        return os.path.join(self.reports_dir, "classification_index.json")

    @property
    def embeddings_dir(self) -> str:
        return os.path.join(
            self.base_dir, "outputs", "encoders",
            "hierarchical_temporal_v3", "embeddings_v3",
        )

    @property
    def output_dir(self) -> str:
        return os.path.join(self.base_dir, "outputs", "ebm_models", "ebm_v3")

    # ── Data dimensions ──
    n_timesteps: int = 24
    n_features: int = 7       # [batt_ch, batt_dis, pump_ch, pump_dis, dr, thermal_su, thermal]
    embed_dim: int = 128      # HTE embedding dimension D

    # ── Model architecture ──
    hidden_dim: int = 128
    gru_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.1
    use_peak_term: bool = True
    peak_tau: float = 0.5
    peak_weight: float = 0.3
    energy_max: float = 50.0    # tanh-bound: energy ∈ [-energy_max, energy_max]

    # ── Sampler (NormalizedTemporalLangevinSampler) ──
    # Training uses fewer, noisier steps for diverse negatives;
    # Inference uses more refined steps.
    langevin_steps: int = 100          # inference steps
    langevin_train_steps_start: int = 10  # curriculum: weak adversary at epoch 0
    langevin_train_steps_end: int = 50    # curriculum: full-strength by mid-training
    langevin_step_size: float = 0.05
    langevin_noise: float = 0.50
    langevin_temp_max: float = 1.0
    langevin_temp_min: float = 0.1
    langevin_init_mode: Literal["soft", "prior", "bernoulli", "oracle"] = "soft"
    langevin_prior_p: float = 0.025
    langevin_prior_strength: float = 0.0
    langevin_normalize_grad: bool = True

    # Mixed negative sampling: anneal from mostly-random to mostly-Langevin
    # langevin_ratio = start + (end - start) * (epoch / total_epochs)
    langevin_ratio_start: float = 1.0   # 100% Langevin from epoch 0
    langevin_ratio_end: float = 1.0     # 100% Langevin throughout
    random_neg_sparsity: float = 0.025  # Bernoulli(p) for random negatives (match data density)
    corruption_flip_rate: float = 0.05  # fraction of bits to flip for corruption negatives

    # ── Training - common ──
    batch_size: int = 32
    num_workers: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    use_amp: bool = True       # Mixed precision on A100
    val_split: float = 0.1
    seed: int = 42

    # ── Step A: Gold pre-training ──
    gold_epochs: int = 50
    gold_lr: float = 2e-5
    gold_patience: int = 10
    gold_cd_steps: int = 1     # Contrastive divergence steps per update

    # ── Step B: Silver fine-tuning ──
    silver_epochs: int = 30
    silver_lr: float = 1e-5
    silver_patience: int = 8
    silver_min_delta: float = 0.01          # minimum ValGap_L improvement for early stopping
    silver_langevin_start: int = 20         # curriculum: gentle start
    silver_langevin_end: int = 35           # curriculum: don't exceed gold's peak (~32)
    silver_lp_eval_every: int = 5           # LP eval frequency (batches)
    silver_lp_scenarios_per_batch: int = 4  # scenarios to LP-evaluate per batch
    silver_preference_margin: float = 0.1   # Margin for preference loss
    silver_lambda_cd: float = 1.0           # Weight of CD loss in silver
    silver_lambda_pref: float = 0.5         # Weight of preference loss in silver

    # ── LP Worker config (silver) ──
    lp_solver: str = "appsi_highs"
    lp_slack_tol: float = 1.0
    lp_deviation_penalty: float = 10000.0
    lp_time_limit: float = 20.0

    # ── Feasibility decoder config ──
    nuclear_must_run_fraction: float = 0.20

    # ── Logging ──
    log_every: int = 10
    save_every_epoch: int = 5

    # ── Device ──
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
