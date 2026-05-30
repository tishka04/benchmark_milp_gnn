"""Shared helpers for the latent UC-JEPA pipeline."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.uc_jepa import UC_FEATURE_NAMES
from src.uc_jepa.models.latent_ebm import LatentEBM
from src.uc_jepa.models.scenario_to_uc_latent import ScenarioToUCLatent
from src.uc_jepa.models.uc_autoencoder import UCAutoencoder


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return out


def parse_feature_vector(
    default_value: float,
    cfg: Optional[Dict[str, Any]],
    n_features: int,
) -> torch.Tensor:
    vec = torch.full((n_features,), float(default_value), dtype=torch.float32)
    if not cfg:
        return vec
    for idx, name in enumerate(UC_FEATURE_NAMES[:n_features]):
        if name in cfg:
            vec[idx] = float(cfg[name])
        elif str(idx) in cfg:
            vec[idx] = float(cfg[str(idx)])
    return vec


def threshold_vector(
    default_threshold: float,
    threshold_cfg: Optional[Dict[str, Any]],
    n_features: int,
) -> torch.Tensor:
    return parse_feature_vector(default_threshold, threshold_cfg, n_features)


def threshold_dict(thresholds: torch.Tensor, n_features: int) -> Dict[str, float]:
    values = thresholds.detach().cpu().tolist()
    return {name: float(values[idx]) for idx, name in enumerate(UC_FEATURE_NAMES[:n_features])}


def apply_thresholds(prob: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    if prob.dim() == 5:
        view = thresholds.view(1, 1, 1, 1, -1)
    elif prob.dim() == 4:
        view = thresholds.view(1, 1, 1, -1)
    elif prob.dim() == 3:
        view = thresholds.view(1, 1, -1)
    else:
        raise ValueError(f"Unsupported prob ndim={prob.dim()}; expected 3, 4 or 5.")
    return (prob >= view).float()


def load_threshold_map(source: Optional[str]) -> Dict[str, float]:
    if not source:
        return {}
    path = Path(source)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        data = json.loads(source)
    return {str(key): float(value) for key, value in dict(data).items()}


def accumulate_confusion(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    pred_bin: torch.Tensor,
    target: torch.Tensor,
    zone_mask: torch.Tensor,
) -> None:
    valid = zone_mask.unsqueeze(-1).unsqueeze(-1).expand_as(target)
    for feat_idx in range(target.shape[-1]):
        tp[feat_idx] += float(
            ((pred_bin[..., feat_idx] == 1) & (target[..., feat_idx] == 1) & (valid[..., feat_idx] == 1)).sum().item(),
        )
        fp[feat_idx] += float(
            ((pred_bin[..., feat_idx] == 1) & (target[..., feat_idx] == 0) & (valid[..., feat_idx] == 1)).sum().item(),
        )
        fn[feat_idx] += float(
            ((pred_bin[..., feat_idx] == 0) & (target[..., feat_idx] == 1) & (valid[..., feat_idx] == 1)).sum().item(),
        )


def f1_from_counts(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
) -> Dict[str, Any]:
    per_f1: List[float] = []
    per_recall: List[float] = []
    per_precision: List[float] = []
    for feat_idx in range(len(tp)):
        precision = tp[feat_idx] / max(1.0, tp[feat_idx] + fp[feat_idx])
        recall = tp[feat_idx] / max(1.0, tp[feat_idx] + fn[feat_idx])
        f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
        per_precision.append(float(precision))
        per_recall.append(float(recall))
        per_f1.append(float(f1))
    return {
        "macro_f1": float(np.mean(per_f1)) if per_f1 else float("nan"),
        "macro_recall": float(np.mean(per_recall)) if per_recall else float("nan"),
        "macro_precision": float(np.mean(per_precision)) if per_precision else float("nan"),
        "per_feature_f1": per_f1,
        "per_feature_recall": per_recall,
        "per_feature_precision": per_precision,
    }


def pairwise_hamming_mean(candidates: torch.Tensor) -> float:
    """Average pairwise Hamming distance for [K, Z, T, F]."""
    if candidates.shape[0] < 2:
        return 0.0
    flat = candidates.reshape(candidates.shape[0], -1)
    dists: List[float] = []
    for i in range(flat.shape[0]):
        for j in range(i + 1, flat.shape[0]):
            dists.append(float((flat[i] != flat[j]).float().mean().item()))
    return float(np.mean(dists)) if dists else 0.0


def load_frozen_ae(ckpt_path: str, device: torch.device) -> UCAutoencoder:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = checkpoint["config"]["model"]
    ae = UCAutoencoder(
        n_features=int(model_cfg["n_features"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        n_tokens=int(model_cfg["n_tokens"]),
        latent_dim=int(model_cfg["latent_dim"]),
        n_heads=int(model_cfg["n_heads"]),
        gru_layers=int(model_cfg["gru_layers"]),
        dropout=float(model_cfg["dropout"]),
        n_zones_max=int(model_cfg["n_zones_max"]),
        n_timesteps_max=int(model_cfg["n_timesteps_max"]),
        scenario_dim=model_cfg.get("scenario_dim"),
        decoder_layers=int(model_cfg.get("decoder_layers", 1)),
    ).to(device)
    ae.load_state_dict(checkpoint["model_state_dict"])
    ae.eval()
    for param in ae.parameters():
        param.requires_grad_(False)
    return ae


def load_frozen_jepa(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[ScenarioToUCLatent, Dict[str, Any]]:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = checkpoint["config"]["model"]
    predictor = ScenarioToUCLatent(
        scenario_dim=int(model_cfg.get("scenario_dim", 128)),
        hidden_dim=int(model_cfg.get("hidden_dim", 384)),
        n_tokens=int(model_cfg["n_tokens"]),
        latent_dim=int(model_cfg["latent_dim"]),
        n_heads=int(model_cfg.get("n_heads", 6)),
        gru_layers=int(model_cfg.get("gru_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        residual_latent=bool(model_cfg.get("residual_latent", False)),
        aux_heads=bool(model_cfg.get("aux_heads", False)),
        derive_startup_from_thermal=bool(model_cfg.get("derive_startup_from_thermal", False)),
        import_head_global=bool(model_cfg.get("import_head_global", False)),
    ).to(device)
    load_result = predictor.load_state_dict(checkpoint["model_state_dict"], strict=False)
    allowed_missing = {
        "time_proj.weight",
        "time_proj.bias",
        "time_norm.weight",
        "time_norm.bias",
    }
    missing = set(load_result.missing_keys)
    unexpected = set(load_result.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected JEPA checkpoint keys: {sorted(unexpected)}")
    if missing - allowed_missing:
        raise RuntimeError(f"Missing JEPA checkpoint keys: {sorted(missing - allowed_missing)}")
    predictor.eval()
    for param in predictor.parameters():
        param.requires_grad_(False)
    return predictor, checkpoint.get("config", {}) or {}


def load_latent_ebm(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[LatentEBM, Dict[str, Any]]:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = checkpoint["config"]["model"]
    prior_mean = checkpoint.get("latent_prior_mean")
    if "n_tokens" not in model_cfg or "latent_dim" not in model_cfg:
        if isinstance(prior_mean, torch.Tensor) and prior_mean.ndim == 2:
            model_cfg = dict(model_cfg)
            model_cfg.setdefault("n_tokens", int(prior_mean.shape[0]))
            model_cfg.setdefault("latent_dim", int(prior_mean.shape[1]))
        else:
            raise KeyError(
                "Latent EBM checkpoint config is missing model.n_tokens/model.latent_dim "
                "and no latent_prior_mean tensor is available to infer them.",
            )
    model = LatentEBM(
        latent_dim=int(model_cfg["latent_dim"]),
        n_tokens=int(model_cfg["n_tokens"]),
        scenario_dim=int(model_cfg.get("scenario_dim", 128)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        token_hidden_dim=model_cfg.get("token_hidden_dim"),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_scenario_context=bool(model_cfg.get("use_scenario_context", True)),
        energy_bound=float(model_cfg.get("energy_bound", 50.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def estimate_latent_prior(
    ae: UCAutoencoder,
    loader,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate per-token latent mean/std from AE-encoded train latents."""
    sum_z: Optional[torch.Tensor] = None
    sum_sq: Optional[torch.Tensor] = None
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            batch = move_batch(batch, device)
            z = ae.encode(batch["u_star"], zone_mask=batch["zone_mask"])  # [B, K, D]
            if sum_z is None:
                sum_z = z.sum(dim=0)
                sum_sq = z.pow(2).sum(dim=0)
            else:
                sum_z = sum_z + z.sum(dim=0)
                sum_sq = sum_sq + z.pow(2).sum(dim=0)
            count += z.shape[0]

    if sum_z is None or sum_sq is None or count <= 0:
        raise RuntimeError("Could not estimate latent prior; loader appears empty.")

    mean = sum_z / count
    var = (sum_sq / count) - mean.pow(2)
    std = var.clamp_min(1e-6).sqrt()
    return mean.detach().cpu(), std.detach().cpu()
