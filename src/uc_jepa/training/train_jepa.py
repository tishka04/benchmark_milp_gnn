"""Train the JEPA predictor ``h_scenario -> z_uc`` with direct UC auxiliary heads.

The target encoder is the *frozen* :class:`UCEncoder` from phase 1; the context
encoder :class:`ScenarioToUCLatent` learns to predict its outputs from the
scenario embedding. The objective combines latent alignment with direct decoded
UC supervision:

    L = w_mse * MSE(z_pred, sg(z_target + noise))
      + w_var * mean_d ReLU(gamma - std(z_pred)_d)          (anti-collapse)
      + w_cov * sum_{i != j} cov(z_pred)_{ij}^2 / D         (decorrelation)
      + w_uc  * BCE_or_Focal(Decoder(z_pred), u_star)
      + w_aux * BCE_or_Focal(aux_logits(h_scenario), u_star)

The variance / covariance terms are applied to ``z_pred`` only (we don't need
to regularize the target since it's frozen). They prevent the trivial solution
where the predictor outputs a constant near the target mean — a well-known
failure mode of pure MSE-on-frozen-target training, while the decoded UC loss
forces the predicted latent to stay useful after the frozen AE decoder.

Usage:
    python -m src.uc_jepa.training.train_jepa --config configs/uc_jepa/jepa.yaml
"""
from __future__ import annotations

from src.demand_response import DR_FORMULATION, require_current_formulation

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from src.uc_jepa import FEAT_IMPORT, UC_FEATURE_NAMES
from src.uc_jepa.data.uc_label_dataset import UCLabelDataset, uc_collate
from src.uc_jepa.models.scenario_to_uc_latent import ScenarioToUCLatent
from src.uc_jepa.models.uc_autoencoder import UCAutoencoder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class JEPATrainConfig:
    data: Dict
    ae: Dict
    model: Dict
    loss: Dict
    optim: Dict
    eval: Dict
    logging: Dict


def _load_config(path: str) -> JEPATrainConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return JEPATrainConfig(
        data=cfg.get("data", {}),
        ae=cfg.get("ae", {}),
        model=cfg.get("model", {}),
        loss=cfg.get("loss", {}),
        optim=cfg.get("optim", {}),
        eval=cfg.get("eval", {}),
        logging=cfg.get("logging", {}),
    )


def _seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _build_dataloaders(data_cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    dataset = UCLabelDataset(
        roots=data_cfg["roots"],
        families=data_cfg.get("families"),
        max_samples=data_cfg.get("max_samples"),
    )
    n = len(dataset)
    val_split = float(data_cfg.get("val_split", 0.1))
    n_val = max(1, int(round(n * val_split)))
    n_train = max(1, n - n_val)
    seed = int(data_cfg.get("seed", 42))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train : n_train + n_val]

    bs = int(data_cfg.get("batch_size", 8))
    nw = int(data_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=bs, shuffle=True,
        num_workers=nw, collate_fn=uc_collate, drop_last=False, pin_memory=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=bs, shuffle=False,
        num_workers=nw, collate_fn=uc_collate, drop_last=False, pin_memory=False,
    )
    print(f"  train={len(train_idx)} val={len(val_idx)} (total={n})")
    return train_loader, val_loader


def _compute_pos_weight(
    loader: DataLoader,
    n_features: int,
    cap: float = 50.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Estimate per-feature ``pos_weight = neg / pos`` from the JEPA train loader."""
    pos = np.zeros(n_features, dtype=np.float64)
    neg = np.zeros(n_features, dtype=np.float64)
    for batch in loader:
        u = batch["u_star"]
        mask = batch["zone_mask"]
        m = mask.unsqueeze(-1).unsqueeze(-1).expand_as(u).numpy()
        u_np = u.numpy()
        valid = m[..., 0]
        n_valid = valid.sum()
        for f in range(n_features):
            p_f = (u_np[..., f] * m[..., f]).sum()
            pos[f] += p_f
            neg[f] += (n_valid - p_f)
    ratio = neg / np.maximum(pos, eps)
    ratio = np.minimum(ratio, cap)
    ratio = np.maximum(ratio, 1.0)
    return torch.tensor(ratio, dtype=torch.float32)


def _parse_feature_vector(
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


def _parse_thresholds(
    default_threshold: float,
    threshold_cfg: Optional[Dict[str, Any]],
    n_features: int,
) -> torch.Tensor:
    return _parse_feature_vector(default_threshold, threshold_cfg, n_features)


def _parse_threshold_search_values(eval_cfg: Dict[str, Any]) -> List[float]:
    values = eval_cfg.get("threshold_search_values")
    if isinstance(values, (list, tuple)) and values:
        grid = [float(v) for v in values]
    else:
        t_min = float(eval_cfg.get("threshold_search_min", 0.20))
        t_max = float(eval_cfg.get("threshold_search_max", 0.80))
        t_step = float(eval_cfg.get("threshold_search_step", 0.05))
        n_steps = max(1, int(round((t_max - t_min) / max(t_step, 1e-6))))
        grid = [t_min + i * t_step for i in range(n_steps + 1)]
        if grid[-1] < t_max - 1e-9:
            grid.append(t_max)
    clipped = sorted({min(0.99, max(0.01, round(v, 4))) for v in grid})
    return clipped


def _threshold_dict(thresholds: torch.Tensor, n_features: Optional[int] = None) -> Dict[str, float]:
    n_features = int(n_features or thresholds.numel())
    vals = thresholds.detach().cpu().tolist()
    return {name: float(vals[i]) for i, name in enumerate(UC_FEATURE_NAMES[:n_features])}


def _apply_thresholds(prob: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    if prob.dim() == 4:
        view = thresholds.view(1, 1, 1, -1)
    elif prob.dim() == 3:
        view = thresholds.view(1, 1, -1)
    else:
        raise ValueError(f"Unsupported prob ndim={prob.dim()}; expected 3 or 4.")
    return (prob >= view).float()


# ---------------------------------------------------------------------------
# AE loading (target encoder, frozen)
# ---------------------------------------------------------------------------

def _load_frozen_ae(ckpt_path: str, device: torch.device) -> UCAutoencoder:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    require_current_formulation(ck, ckpt_path)
    mc = ck["config"]["model"]
    ae = UCAutoencoder(
        n_features=int(mc["n_features"]),
        hidden_dim=int(mc["hidden_dim"]),
        n_tokens=int(mc["n_tokens"]),
        latent_dim=int(mc["latent_dim"]),
        n_heads=int(mc["n_heads"]),
        gru_layers=int(mc["gru_layers"]),
        dropout=float(mc["dropout"]),
        n_zones_max=int(mc["n_zones_max"]),
        n_timesteps_max=int(mc["n_timesteps_max"]),
        scenario_dim=mc.get("scenario_dim"),
        decoder_layers=int(mc.get("decoder_layers", 1)),
    ).to(device)
    ae.load_state_dict(ck["model_state_dict"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def _vicreg_terms(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard VICReg variance and covariance regularizers.

    ``z`` must be 2-D ``[N, D]``. Caller is expected to flatten K-token outputs
    into ``[B*K, D]`` (or pool first).
    """
    N, D = z.shape
    z_centered = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + eps)                     # [D]
    var_loss = F.relu(gamma - std).mean()
    cov = (z_centered.T @ z_centered) / max(N - 1, 1)        # [D, D]
    off = cov.pow(2).sum() - cov.diag().pow(2).sum()
    cov_loss = off / D
    return var_loss, cov_loss


def _masked_mean(x: torch.Tensor, zone_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if zone_mask is None:
        return x.mean()
    mask = zone_mask
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.to(x.dtype).expand_as(x)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_weighted_mean(
    x: torch.Tensor,
    zone_mask: Optional[torch.Tensor],
    feature_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    weight = 1.0
    if feature_weight is not None:
        view = [1] * (x.dim() - 1) + [feature_weight.numel()]
        weight = feature_weight.to(x.device, dtype=x.dtype).view(*view)
        x = x * weight
    if zone_mask is None:
        if isinstance(weight, torch.Tensor):
            return x.sum() / weight.expand_as(x).sum().clamp_min(1.0)
        return x.mean()
    mask = zone_mask
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.to(x.dtype).expand_as(x)
    if isinstance(weight, torch.Tensor):
        denom = (mask * weight.expand_as(x)).sum().clamp_min(1.0)
    else:
        denom = mask.sum().clamp_min(1.0)
    return (x * mask).sum() / denom


def _macro_metrics(
    pred_bin: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    m = mask.unsqueeze(-1).unsqueeze(-1).expand_as(target)
    f1s, recs, precs = [], [], []
    F_ = target.shape[-1]
    for f in range(F_):
        tp = float(((pred_bin[..., f] == 1) & (target[..., f] == 1) & (m[..., f] == 1)).sum().item())
        fp = float(((pred_bin[..., f] == 1) & (target[..., f] == 0) & (m[..., f] == 1)).sum().item())
        fn = float(((pred_bin[..., f] == 0) & (target[..., f] == 1) & (m[..., f] == 1)).sum().item())
        pr = tp / max(1.0, tp + fp)
        rc = tp / max(1.0, tp + fn)
        f1 = 2 * pr * rc / max(1e-9, pr + rc)
        precs.append(pr)
        recs.append(rc)
        f1s.append(f1)
    return {
        "macro_f1": float(np.mean(f1s)),
        "macro_recall": float(np.mean(recs)),
        "macro_precision": float(np.mean(precs)),
    }


def _masked_uc_loss(
    logits: torch.Tensor,
    u_star: torch.Tensor,
    zone_mask: Optional[torch.Tensor],
    pos_weight: Optional[torch.Tensor],
    focal_gamma: float,
    feature_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bce_elem = F.binary_cross_entropy_with_logits(
        logits,
        u_star.float(),
        pos_weight=pos_weight.to(logits.device) if pos_weight is not None else None,
        reduction="none",
    )
    if focal_gamma > 0.0:
        prob = torch.sigmoid(logits)
        p_t = prob * u_star + (1.0 - prob) * (1.0 - u_star)
        bce_elem = ((1.0 - p_t).clamp_min(1e-6).pow(focal_gamma)) * bce_elem
    return _masked_weighted_mean(bce_elem, zone_mask, feature_weight)


def _masked_time_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    pos_weight: Optional[torch.Tensor],
    focal_gamma: float,
    scalar_weight: float = 1.0,
) -> torch.Tensor:
    bce_elem = F.binary_cross_entropy_with_logits(
        logits,
        target.float(),
        pos_weight=pos_weight.to(logits.device) if pos_weight is not None else None,
        reduction="none",
    )
    if focal_gamma > 0.0:
        prob = torch.sigmoid(logits)
        p_t = prob * target + (1.0 - prob) * (1.0 - target)
        bce_elem = ((1.0 - p_t).clamp_min(1e-6).pow(focal_gamma)) * bce_elem
    bce_elem = bce_elem * float(scalar_weight)
    if valid_mask is None:
        return bce_elem.mean()
    mask = valid_mask.to(logits.dtype).unsqueeze(-1).expand_as(bce_elem)
    return (bce_elem * mask).sum() / mask.sum().clamp_min(1.0)


def _temporal_smoothness_loss(
    prob: torch.Tensor,
    zone_mask: Optional[torch.Tensor],
    feature_weights: Optional[torch.Tensor],
) -> torch.Tensor:
    if prob.shape[2] <= 1 or feature_weights is None:
        return prob.new_zeros(())
    if float(feature_weights.sum().item()) <= 0.0:
        return prob.new_zeros(())

    diff = (prob[:, :, 1:, :] - prob[:, :, :-1, :]).abs()
    feat_w = feature_weights.to(prob.device, dtype=prob.dtype).view(1, 1, 1, -1)
    weighted = diff * feat_w

    if zone_mask is not None:
        mask = zone_mask.unsqueeze(-1).unsqueeze(-1).to(prob.dtype).expand_as(diff)
        weighted = weighted * mask
        denom = (mask * feat_w).sum().clamp_min(1.0)
    else:
        denom = feat_w.expand_as(diff).sum().clamp_min(1.0)
    return weighted.sum() / denom


def _evaluate_predictions(
    pred_bin: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    diff = (pred_bin - target.float()).abs()
    uc_hamming = _masked_mean(diff, mask)
    macro = _macro_metrics(pred_bin, target, mask)
    return {
        "uc_hamming": float(uc_hamming.item()),
        "uc_macro_f1": float(macro["macro_f1"]),
        "uc_macro_recall": float(macro["macro_recall"]),
        "uc_macro_precision": float(macro["macro_precision"]),
    }


def _search_best_thresholds(
    prob: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    thresholds: torch.Tensor,
    candidate_values: Sequence[float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    valid = mask.unsqueeze(-1).unsqueeze(-1).expand_as(target).bool()
    tuned = thresholds.detach().cpu().clone()
    candidates = [float(v) for v in candidate_values]

    for f in range(target.shape[-1]):
        p_f = prob[..., f]
        y_f = target[..., f].bool()
        m_f = valid[..., f]
        best_thr = float(tuned[f].item())
        best_f1 = -1.0
        for thr in candidates:
            pred_f = p_f >= thr
            tp = float((pred_f & y_f & m_f).sum().item())
            fp = float((pred_f & (~y_f) & m_f).sum().item())
            fn = float(((~pred_f) & y_f & m_f).sum().item())
            pr = tp / max(1.0, tp + fp)
            rc = tp / max(1.0, tp + fn)
            f1 = 2.0 * pr * rc / max(1e-9, pr + rc)
            if (f1 > best_f1 + 1e-9) or (abs(f1 - best_f1) <= 1e-9 and thr > best_thr):
                best_f1 = f1
                best_thr = thr
        tuned[f] = best_thr

    pred_bin = _apply_thresholds(prob, tuned)
    metrics = _evaluate_predictions(pred_bin, target, mask)
    metrics["feature_thresholds"] = _threshold_dict(tuned, target.shape[-1])
    return tuned, metrics


def _compact_valid_zones(
    x: torch.Tensor,
    target: torch.Tensor,
    zone_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = zone_mask.reshape(-1).bool()
    x_flat = x.reshape(-1, x.shape[-2], x.shape[-1])[valid]
    y_flat = target.reshape(-1, target.shape[-2], target.shape[-1])[valid]
    mask_flat = torch.ones(x_flat.shape[0], dtype=zone_mask.dtype, device=zone_mask.device)
    return x_flat, y_flat, mask_flat


def _extract_import_target(u_star: torch.Tensor) -> torch.Tensor:
    return u_star[:, 0, :, FEAT_IMPORT]


def _zero_loss_like(ref: torch.Tensor) -> torch.Tensor:
    return ref.new_zeros(())


def jepa_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    w_mse: float,
    w_var: float,
    w_cov: float,
    gamma: float,
) -> Dict[str, torch.Tensor]:
    """Returns latent-only JEPA terms; caller can add decoded UC supervision."""
    # z_pred, z_target: [B, K, D]. Stop-grad already enforced via no_grad upstream.
    mse = F.mse_loss(z_pred, z_target)
    z_flat = z_pred.reshape(-1, z_pred.shape[-1])
    var, cov = _vicreg_terms(z_flat, gamma=gamma)
    total = w_mse * mse + w_var * var + w_cov * cov
    with torch.no_grad():
        # Cosine similarity between mean-pooled predicted and target vectors —
        # a scale-invariant readout of how aligned the two latents are.
        p = z_pred.mean(dim=1); t = z_target.mean(dim=1)
        cos = F.cosine_similarity(p, t, dim=-1).mean()
    return {
        "loss_latent": total,
        "mse": mse.detach(),
        "var": var.detach(),
        "cov": cov.detach(),
        "cos": cos,
    }


# ---------------------------------------------------------------------------
# Train / eval pass
# ---------------------------------------------------------------------------

def _move(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def _epoch_pass(
    predictor: ScenarioToUCLatent,
    ae: UCAutoencoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    grad_clip: float,
    epoch: int,
    is_train: bool,
    log_every: int,
    w_mse: float, w_var: float, w_cov: float, gamma: float,
    uc_weight: float,
    aux_weight: float,
    pos_weight: Optional[torch.Tensor],
    focal_gamma: float,
    thresholds: torch.Tensor,
    target_noise_std: float,
    temporal_weight: float,
    temporal_feature_weights: Optional[torch.Tensor],
    decoded_feature_weights: Optional[torch.Tensor],
    aux_feature_weights: Optional[torch.Tensor],
    threshold_search_values: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    if is_train:
        predictor.train()
    else:
        predictor.eval()

    totals = {
        "loss": 0.0,
        "loss_latent": 0.0,
        "uc_loss": 0.0,
        "aux_loss": 0.0,
        "temporal_loss": 0.0,
        "mse": 0.0,
        "var": 0.0,
        "cov": 0.0,
        "cos": 0.0,
        "uc_hamming": 0.0,
        "uc_macro_f1": 0.0,
        "uc_macro_recall": 0.0,
        "uc_macro_precision": 0.0,
        "aux_macro_f1": 0.0,
        "aux_macro_recall": 0.0,
        "aux_macro_precision": 0.0,
    }
    n_seen = 0
    t0 = time.time()
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    thresholds = thresholds.to(device)
    collect_threshold_search = (not is_train) and bool(threshold_search_values)
    prob_cache: List[torch.Tensor] = []
    target_cache: List[torch.Tensor] = []
    mask_cache: List[torch.Tensor] = []
    with ctx:
        for step, batch in enumerate(loader):
            batch = _move(batch, device)
            u = batch["u_star"]
            h = batch["h_scenario"]
            mask = batch["zone_mask"]
            with torch.no_grad():
                z_target = ae.encode(u, zone_mask=mask)
                if target_noise_std > 0.0 and is_train:
                    z_target = z_target + target_noise_std * torch.randn_like(z_target)
            pred_out = predictor(h, zone_mask=mask)
            z_pred = pred_out["z_pred"]
            terms = jepa_loss(z_pred, z_target, w_mse, w_var, w_cov, gamma)
            logits = ae.decode(z_pred, n_zones=u.shape[1], n_timesteps=u.shape[2], h_scenario=h)
            prob = torch.sigmoid(logits)
            uc_loss = _masked_uc_loss(
                logits,
                u,
                mask,
                pos_weight=pos_weight,
                focal_gamma=focal_gamma,
                feature_weight=decoded_feature_weights,
            )
            aux_loss = _zero_loss_like(logits)
            aux_metric_values = {
                "uc_macro_f1": 0.0,
                "uc_macro_recall": 0.0,
                "uc_macro_precision": 0.0,
            }
            aux_logits = pred_out.get("aux_logits")
            if aux_logits is not None:
                aux_zonal_logits = aux_logits[..., :FEAT_IMPORT]
                aux_zonal_target = u[..., :FEAT_IMPORT]
                aux_zonal_pos_weight = pos_weight[:FEAT_IMPORT] if pos_weight is not None else None
                aux_zonal_feature_weights = (
                    aux_feature_weights[:FEAT_IMPORT] if aux_feature_weights is not None else None
                )
                aux_zonal_loss = _masked_uc_loss(
                    aux_zonal_logits,
                    aux_zonal_target,
                    mask,
                    pos_weight=aux_zonal_pos_weight,
                    focal_gamma=focal_gamma,
                    feature_weight=aux_zonal_feature_weights,
                )
                import_loss = _masked_time_bce_loss(
                    pred_out["import_logits_time"],
                    _extract_import_target(u),
                    valid_mask=(mask.sum(dim=1) > 0),
                    pos_weight=(
                        pos_weight[FEAT_IMPORT : FEAT_IMPORT + 1] if pos_weight is not None else None
                    ),
                    focal_gamma=focal_gamma,
                    scalar_weight=(
                        float(aux_feature_weights[FEAT_IMPORT].item())
                        if aux_feature_weights is not None
                        else 1.0
                    ),
                )
                aux_loss = aux_zonal_loss + import_loss
                with torch.no_grad():
                    aux_pred_bin = _apply_thresholds(torch.sigmoid(aux_logits), thresholds)
                    aux_metric_values = _evaluate_predictions(aux_pred_bin, u, mask)
            temporal_loss = _temporal_smoothness_loss(prob, mask, temporal_feature_weights)
            total_loss = (
                terms["loss_latent"]
                + uc_weight * uc_loss
                + aux_weight * aux_loss
                + temporal_weight * temporal_loss
            )

            with torch.no_grad():
                pred_bin = _apply_thresholds(prob, thresholds)
                metric_values = _evaluate_predictions(pred_bin, u, mask)
                if collect_threshold_search:
                    prob_compact, target_compact, mask_compact = _compact_valid_zones(
                        prob.detach().cpu(),
                        u.detach().cpu(),
                        mask.detach().cpu(),
                    )
                    if prob_compact.numel() > 0:
                        prob_cache.append(prob_compact)
                        target_cache.append(target_compact)
                        mask_cache.append(mask_compact)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), grad_clip)
                optimizer.step()

            B = u.shape[0]
            n_seen += B
            totals["loss"] += float(total_loss.item()) * B
            totals["loss_latent"] += float(terms["loss_latent"].item()) * B
            totals["uc_loss"] += float(uc_loss.item()) * B
            totals["aux_loss"] += float(aux_loss.item()) * B
            totals["temporal_loss"] += float(temporal_loss.item()) * B
            totals["mse"] += float(terms["mse"].item()) * B
            totals["var"] += float(terms["var"].item()) * B
            totals["cov"] += float(terms["cov"].item()) * B
            totals["cos"] += float(terms["cos"].item()) * B
            totals["uc_hamming"] += float(metric_values["uc_hamming"]) * B
            totals["uc_macro_f1"] += float(metric_values["uc_macro_f1"]) * B
            totals["uc_macro_recall"] += float(metric_values["uc_macro_recall"]) * B
            totals["uc_macro_precision"] += float(metric_values["uc_macro_precision"]) * B
            totals["aux_macro_f1"] += float(aux_metric_values["uc_macro_f1"]) * B
            totals["aux_macro_recall"] += float(aux_metric_values["uc_macro_recall"]) * B
            totals["aux_macro_precision"] += float(aux_metric_values["uc_macro_precision"]) * B

            if is_train and log_every > 0 and step % log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  [ep{epoch:03d} step{step:04d}] loss={total_loss.item():.4f} "
                      f"latent={terms['loss_latent'].item():.4f} uc={uc_loss.item():.4f} "
                      f"aux={aux_loss.item():.4f} "
                      f"temp={temporal_loss.item():.4f} "
                      f"mse={terms['mse'].item():.4f} var={terms['var'].item():.4f} "
                      f"cov={terms['cov'].item():.4f} cos={terms['cos'].item():.3f} "
                      f"F1={metric_values['uc_macro_f1']:.3f} auxF1={aux_metric_values['uc_macro_f1']:.3f} "
                      f"ham={metric_values['uc_hamming']:.4f} "
                      f"lr={lr:.2e}")

    dt = time.time() - t0
    avg = {k: v / max(1, n_seen) for k, v in totals.items()}
    avg["feature_thresholds"] = _threshold_dict(thresholds, ae.n_features)
    if collect_threshold_search and prob_cache:
        tuned_thresholds, tuned_metrics = _search_best_thresholds(
            torch.cat(prob_cache, dim=0),
            torch.cat(target_cache, dim=0),
            torch.cat(mask_cache, dim=0),
            thresholds.detach().cpu(),
            threshold_search_values or [],
        )
        avg.update(tuned_metrics)
        avg["feature_thresholds"] = tuned_metrics["feature_thresholds"]
    avg["seconds"] = dt
    avg["n"] = n_seen
    return avg


# ---------------------------------------------------------------------------
# Checkpoint + LR schedule
# ---------------------------------------------------------------------------

def _save_ckpt(predictor: ScenarioToUCLatent, optim_: torch.optim.Optimizer,
               epoch: int, metrics: Dict, cfg: JEPATrainConfig, path: Path,
               thresholds: Optional[torch.Tensor] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg_dict = asdict(cfg)
    if thresholds is not None:
        cfg_dict.setdefault("eval", {})
        cfg_dict["eval"]["feature_thresholds"] = _threshold_dict(thresholds)
    torch.save({
        "dr_formulation": DR_FORMULATION,
        "model_state_dict": predictor.state_dict(),
        "optimizer_state_dict": optim_.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": cfg_dict,
    }, path)


def _make_lr_lambda(warmup_epochs: int, total_epochs: int):
    def fn(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / float(warmup_epochs)
        t = max(1, total_epochs - warmup_epochs)
        e = epoch - warmup_epochs
        return 0.5 * (1.0 + math.cos(math.pi * e / t))
    return fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(cfg: JEPATrainConfig, device: Optional[str] = None) -> Dict:
    _seed_all(int(cfg.data.get("seed", 42)))
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {dev}")

    train_loader, val_loader = _build_dataloaders(cfg.data)

    ae = _load_frozen_ae(cfg.ae["checkpoint"], dev)
    print(f"Loaded frozen AE: K={ae.n_tokens} latent_dim={ae.latent_dim} "
          f"params={sum(p.numel() for p in ae.parameters()):,} (frozen)")

    # Ensure predictor latent shape matches AE.
    model_cfg = dict(cfg.model)
    expected_K, expected_D = ae.n_tokens, ae.latent_dim
    if model_cfg.get("n_tokens", expected_K) != expected_K:
        raise ValueError(f"model.n_tokens={model_cfg['n_tokens']} mismatches AE n_tokens={expected_K}")
    if model_cfg.get("latent_dim", expected_D) != expected_D:
        raise ValueError(f"model.latent_dim={model_cfg['latent_dim']} mismatches AE latent_dim={expected_D}")
    model_cfg.setdefault("n_tokens", expected_K)
    model_cfg.setdefault("latent_dim", expected_D)

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
    ).to(dev)
    print(f"ScenarioToUCLatent params: {sum(p.numel() for p in predictor.parameters()):,}")

    w_mse = float(cfg.loss.get("mse", 1.0))
    w_var = float(cfg.loss.get("variance", 1.0))
    w_cov = float(cfg.loss.get("covariance", 0.04))
    gamma = float(cfg.loss.get("variance_target", 1.0))
    uc_weight = float(cfg.loss.get("uc_weight", 1.0))
    aux_weight = float(cfg.loss.get("aux_weight", 0.0))
    focal_gamma = float(cfg.loss.get("focal_gamma", 0.0))
    target_noise_std = float(cfg.loss.get("target_noise_std", 0.0))
    temporal_weight = float(cfg.loss.get("temporal_weight", 0.0))
    print(
        f"Loss weights: mse={w_mse} var={w_var} cov={w_cov} gamma={gamma} "
        f"uc_weight={uc_weight} aux_weight={aux_weight} "
        f"focal_gamma={focal_gamma} target_noise_std={target_noise_std} "
        f"temporal_weight={temporal_weight}"
    )

    pw_cfg = cfg.loss.get("pos_weight", "auto")
    pos_weight: Optional[torch.Tensor] = None
    if pw_cfg in (None, "none", False):
        print("pos_weight: disabled (raw BCE/focal).")
    elif isinstance(pw_cfg, dict):
        pos_weight = _parse_feature_vector(1.0, pw_cfg, ae.n_features)
        print("pos_weight: manual per feature")
    elif isinstance(pw_cfg, (list, tuple)):
        pos_weight = torch.tensor([float(x) for x in pw_cfg], dtype=torch.float32)
        print(f"pos_weight: manual = {pos_weight.tolist()}")
    elif pw_cfg in ("auto", True):
        cap = float(cfg.loss.get("pos_weight_cap", 50.0))
        print(f"pos_weight: computing from train set (cap={cap})...")
        pos_weight = _compute_pos_weight(train_loader, ae.n_features, cap=cap)
        for name, w in zip(UC_FEATURE_NAMES, pos_weight.tolist()):
            print(f"    {name:<24} pos_weight={w:.2f}")
    else:
        raise ValueError(
            f"Unknown loss.pos_weight={pw_cfg!r}; expected 'auto', 'none', a list, or a feature dict."
        )
    if pos_weight is not None:
        if pos_weight.numel() != ae.n_features:
            raise ValueError(f"pos_weight has {pos_weight.numel()} entries; expected {ae.n_features}.")
        for name, w in zip(UC_FEATURE_NAMES, pos_weight.tolist()):
            print(f"    {name:<24} pos_weight={w:.2f}")
        pos_weight = pos_weight.to(dev)

    threshold_default = float(cfg.eval.get("threshold", 0.5))
    thresholds = _parse_thresholds(
        threshold_default,
        cfg.eval.get("feature_thresholds"),
        ae.n_features,
    )
    threshold_search_values = (
        _parse_threshold_search_values(cfg.eval)
        if bool(cfg.eval.get("threshold_search", False))
        else None
    )
    print(
        "Decode thresholds: "
        + ", ".join(f"{name}={thr:.2f}" for name, thr in zip(UC_FEATURE_NAMES, thresholds.tolist()))
    )
    if threshold_search_values:
        print(f"Threshold search enabled on validation: {threshold_search_values}")

    temporal_feature_weights = _parse_feature_vector(
        0.0,
        cfg.loss.get("temporal_feature_weights"),
        ae.n_features,
    )
    decoded_feature_weights = _parse_feature_vector(
        1.0,
        cfg.loss.get("decoded_feature_weights"),
        ae.n_features,
    )
    aux_feature_weights = _parse_feature_vector(
        1.0,
        cfg.loss.get("aux_feature_weights"),
        ae.n_features,
    )
    if temporal_weight > 0.0:
        print(
            "Temporal feature weights: "
            + ", ".join(
                f"{name}={w:.2f}"
                for name, w in zip(UC_FEATURE_NAMES, temporal_feature_weights.tolist())
                if w > 0.0
            )
        )
    print(
        "Decoded feature weights: "
        + ", ".join(f"{name}={w:.2f}" for name, w in zip(UC_FEATURE_NAMES, decoded_feature_weights.tolist()))
    )
    print(
        "Aux feature weights: "
        + ", ".join(f"{name}={w:.2f}" for name, w in zip(UC_FEATURE_NAMES, aux_feature_weights.tolist()))
    )
    temporal_feature_weights = temporal_feature_weights.to(dev)
    decoded_feature_weights = decoded_feature_weights.to(dev)
    aux_feature_weights = aux_feature_weights.to(dev)

    lr = float(cfg.optim.get("lr", 3e-4))
    wd = float(cfg.optim.get("weight_decay", 1e-4))
    epochs = int(cfg.optim.get("epochs", 50))
    patience = int(cfg.optim.get("patience", 8))
    grad_clip = float(cfg.optim.get("grad_clip", 1.0))
    warmup_epochs = int(cfg.optim.get("warmup_epochs", 2))
    best_metric_name = str(cfg.optim.get("best_metric", "uc_macro_f1"))

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_make_lr_lambda(warmup_epochs, epochs),
    )

    out_dir = Path(cfg.logging.get("output_dir", "outputs/uc_jepa/jepa"))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_every = int(cfg.logging.get("save_every", 5))
    log_every = int(cfg.logging.get("log_every", 50))
    best_path = out_dir / "jepa_best.pt"

    # Some metrics are alignment/readout scores (higher better); the rest are losses.
    higher_is_better = best_metric_name in {
        "cos",
        "uc_macro_f1",
        "uc_macro_recall",
        "uc_macro_precision",
        "aux_macro_f1",
        "aux_macro_recall",
        "aux_macro_precision",
    }
    best_val = -float("inf") if higher_is_better else float("inf")
    bad_epochs = 0
    history: List[Dict] = []

    for epoch in range(epochs):
        tr = _epoch_pass(predictor, ae, train_loader, dev, optimizer, grad_clip,
                         epoch, True, log_every, w_mse, w_var, w_cov, gamma,
                         uc_weight, aux_weight, pos_weight, focal_gamma, thresholds, target_noise_std,
                         temporal_weight, temporal_feature_weights, decoded_feature_weights,
                         aux_feature_weights, None)
        va = _epoch_pass(predictor, ae, val_loader, dev, None, 0.0,
                         epoch, False, 0, w_mse, w_var, w_cov, gamma,
                         uc_weight, aux_weight, pos_weight, focal_gamma, thresholds, 0.0,
                         temporal_weight, temporal_feature_weights, decoded_feature_weights,
                         aux_feature_weights, threshold_search_values)
        thresholds = _parse_thresholds(0.5, va.get("feature_thresholds"), ae.n_features)
        scheduler.step()

        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                        "train": tr, "val": va})
        print(f"Epoch {epoch+1}/{epochs} ({tr['seconds']:.1f}s+{va['seconds']:.1f}s) | "
              f"train loss={tr['loss']:.4f} uc={tr['uc_loss']:.4f} aux={tr['aux_loss']:.4f} "
              f"temp={tr['temporal_loss']:.4f} F1={tr['uc_macro_f1']:.3f} auxF1={tr['aux_macro_f1']:.3f} "
              f"mse={tr['mse']:.4f} cos={tr['cos']:.3f} | "
              f"val loss={va['loss']:.4f} uc={va['uc_loss']:.4f} aux={va['aux_loss']:.4f} "
              f"temp={va['temporal_loss']:.4f} F1={va['uc_macro_f1']:.3f} auxF1={va['aux_macro_f1']:.3f} "
              f"ham={va['uc_hamming']:.4f} mse={va['mse']:.4f} "
              f"var={va['var']:.4f} cov={va['cov']:.4f} cos={va['cos']:.3f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")
        print(
            "  val thresholds: "
            + ", ".join(
                f"{name}={thr:.2f}"
                for name, thr in zip(UC_FEATURE_NAMES, thresholds.tolist())
            )
        )

        cur = float(va.get(best_metric_name, va["loss"]))
        improved = (cur > best_val + 1e-6) if higher_is_better else (cur < best_val - 1e-6)
        if improved:
            best_val = cur; bad_epochs = 0
            _save_ckpt(predictor, optimizer, epoch, va, cfg, best_path, thresholds=thresholds)
            print(f"  >> new best val {best_metric_name}={best_val:.4f} -> {best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
                break

        if save_every > 0 and (epoch + 1) % save_every == 0:
            _save_ckpt(predictor, optimizer, epoch, va, cfg,
                       out_dir / f"jepa_epoch_{epoch+1}.pt", thresholds=thresholds)

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best val {best_metric_name}: {best_val:.4f} ({best_path})")
    return {"best_metric": best_metric_name, "best_val": best_val,
            "best_ckpt": str(best_path), "history": history}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the JEPA predictor (Phase 2).")
    p.add_argument("--config", required=True, help="YAML config path.")
    p.add_argument("--device", default=None, help="cuda or cpu (auto if unset).")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    cfg = _load_config(args.config)
    train(cfg, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
