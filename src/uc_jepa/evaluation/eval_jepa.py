"""Evaluate the JEPA predictor by routing its output through the frozen AE decoder.

End-to-end pipeline:

    h_scenario --[JEPA predictor]--> z_pred --[frozen AE decoder]--> u_hat
                                                                       |
    u_star ------------------------------------------------------------+
                                                                       v
                       per-feature precision / recall / F1
                       + macro F1
                       + cosine(z_pred, z_target=AE_encoder(u_star))

Compare F1_jepa against F1_ae (upper bound — same decoder with the *gold*
``z_uc = AE_encoder(u_star)`` instead of the predicted one). The gap tells us
how much information about ``u_star`` is recoverable from ``h_scenario`` alone.

Usage:
    python -m src.uc_jepa.evaluation.eval_jepa \
        --ae outputs/uc_jepa/autoencoder/uc_ae_best.pt \
        --jepa outputs/uc_jepa/jepa/jepa_best.pt \
        --data outputs/uc_jepa/dataset_train \
        --max_samples 400
"""
from __future__ import annotations

from src.demand_response import require_current_formulation

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.uc_jepa import FEAT_IMPORT, UC_FEATURE_NAMES
from src.uc_jepa.data.uc_label_dataset import UCLabelDataset, uc_collate
from src.uc_jepa.evaluation.sanity_check_ae import (
    _aggregate_lp_metrics,
    _deepest_stage_reached,
    _display_stage_name,
    _explicit_flip_counts,
    _extract_stage_flags,
    _infer_scenario_roots,
    _normalize_stage_name,
    _resolve_scenario_root,
    math_isfinite,
)
from src.uc_jepa.models.scenario_to_uc_latent import ScenarioToUCLatent
from src.uc_jepa.models.uc_autoencoder import UCAutoencoder


def _load_ae(path: str, device: torch.device) -> UCAutoencoder:
    ck = torch.load(path, map_location=device, weights_only=False)
    require_current_formulation(ck, "checkpoint")
    mc = ck["config"]["model"]
    ae = UCAutoencoder(
        n_features=int(mc["n_features"]), hidden_dim=int(mc["hidden_dim"]),
        n_tokens=int(mc["n_tokens"]), latent_dim=int(mc["latent_dim"]),
        n_heads=int(mc["n_heads"]), gru_layers=int(mc["gru_layers"]),
        dropout=float(mc["dropout"]),
        n_zones_max=int(mc["n_zones_max"]), n_timesteps_max=int(mc["n_timesteps_max"]),
        scenario_dim=mc.get("scenario_dim"),
        decoder_layers=int(mc.get("decoder_layers", 1)),
    ).to(device)
    ae.load_state_dict(ck["model_state_dict"]); ae.eval()
    for p in ae.parameters(): p.requires_grad_(False)
    return ae


def _load_jepa(path: str, device: torch.device) -> tuple[ScenarioToUCLatent, Dict[str, Any]]:
    ck = torch.load(path, map_location=device, weights_only=False)
    require_current_formulation(ck, "checkpoint")
    mc = ck["config"]["model"]
    pred = ScenarioToUCLatent(
        scenario_dim=int(mc.get("scenario_dim", 128)),
        hidden_dim=int(mc.get("hidden_dim", 384)),
        n_tokens=int(mc["n_tokens"]),
        latent_dim=int(mc["latent_dim"]),
        n_heads=int(mc.get("n_heads", 6)),
        gru_layers=int(mc.get("gru_layers", 1)),
        dropout=float(mc.get("dropout", 0.1)),
        residual_latent=bool(mc.get("residual_latent", False)),
        aux_heads=bool(mc.get("aux_heads", False)),
        derive_startup_from_thermal=bool(mc.get("derive_startup_from_thermal", False)),
        import_head_global=bool(mc.get("import_head_global", False)),
    ).to(device)
    load_res = pred.load_state_dict(ck["model_state_dict"], strict=False)
    allowed_missing = {
        "time_proj.weight",
        "time_proj.bias",
        "time_norm.weight",
        "time_norm.bias",
    }
    missing = set(load_res.missing_keys)
    unexpected = set(load_res.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected JEPA checkpoint keys: {sorted(unexpected)}")
    if missing - allowed_missing:
        raise RuntimeError(f"Missing JEPA checkpoint keys: {sorted(missing - allowed_missing)}")
    pred.eval()
    return pred, ck.get("config", {}) or {}


def _load_threshold_map(source: Optional[str]) -> Dict[str, float]:
    if not source:
        return {}
    path = Path(source)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(source)
    return {str(k): float(v) for k, v in dict(data).items()}


def _threshold_vector(default_threshold: float, threshold_map: Dict[str, float], n_features: int) -> torch.Tensor:
    vec = torch.full((n_features,), float(default_threshold), dtype=torch.float32)
    for idx, name in enumerate(UC_FEATURE_NAMES[:n_features]):
        if name in threshold_map:
            vec[idx] = float(threshold_map[name])
        elif str(idx) in threshold_map:
            vec[idx] = float(threshold_map[str(idx)])
    return vec


def _threshold_dict(thresholds: torch.Tensor, n_features: int) -> Dict[str, float]:
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


def _threshold_grid(t_min: float, t_max: float, t_step: float) -> List[float]:
    n_steps = max(1, int(round((t_max - t_min) / max(t_step, 1e-6))))
    grid = [t_min + i * t_step for i in range(n_steps + 1)]
    if grid[-1] < t_max - 1e-9:
        grid.append(t_max)
    return sorted({min(0.99, max(0.01, round(v, 4))) for v in grid})


def _accumulate(
    TP: np.ndarray, FP: np.ndarray, FN: np.ndarray,
    pred_bin: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> None:
    m = mask.unsqueeze(-1).unsqueeze(-1).expand_as(target)
    F_ = target.shape[-1]
    for f in range(F_):
        TP[f] += float(((pred_bin[..., f] == 1) & (target[..., f] == 1) & (m[..., f] == 1)).sum())
        FP[f] += float(((pred_bin[..., f] == 1) & (target[..., f] == 0) & (m[..., f] == 1)).sum())
        FN[f] += float(((pred_bin[..., f] == 0) & (target[..., f] == 1) & (m[..., f] == 1)).sum())


def _f1_table(TP: np.ndarray, FP: np.ndarray, FN: np.ndarray) -> Dict:
    F_ = len(TP)
    per_f1, per_pr, per_rc = [], [], []
    for f in range(F_):
        pr = TP[f] / max(1.0, TP[f] + FP[f])
        rc = TP[f] / max(1.0, TP[f] + FN[f])
        f1 = 2 * pr * rc / max(1e-9, pr + rc)
        per_pr.append(pr); per_rc.append(rc); per_f1.append(f1)
    return {
        "per_feature_f1": per_f1,
        "per_feature_precision": per_pr,
        "per_feature_recall": per_rc,
        "macro_f1": float(np.mean(per_f1)),
        "macro_precision": float(np.mean(per_pr)),
        "macro_recall": float(np.mean(per_rc)),
    }


def _search_best_thresholds(
    prob: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    thresholds: torch.Tensor,
    candidate_values: Sequence[float],
) -> torch.Tensor:
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
    return tuned


def _compact_valid_zones(
    x: torch.Tensor,
    target: torch.Tensor,
    zone_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = zone_mask.reshape(-1).bool()
    x_flat = x.reshape(-1, x.shape[-2], x.shape[-1])[valid]
    y_flat = target.reshape(-1, target.shape[-2], target.shape[-1])[valid]
    mask_flat = torch.ones(x_flat.shape[0], dtype=zone_mask.dtype, device=zone_mask.device)
    return x_flat, y_flat, mask_flat


def _extract_import_target(u_star: torch.Tensor) -> torch.Tensor:
    return u_star[:, 0, :, FEAT_IMPORT]


def _evaluate_lp_from_decoded(
    dataset: Subset,
    predictor: ScenarioToUCLatent,
    ae: UCAutoencoder,
    device: torch.device,
    thresholds: torch.Tensor,
    scenario_roots: Sequence[str],
    max_lp: int,
    solver: str,
    max_stages: int,
    worker_kwargs: Optional[Dict[str, Any]] = None,
    force_repair_20: bool = False,
) -> Optional[Dict[str, Any]]:
    try:
        from src.milp.lp_worker_two_stage import LPWorkerTwoStage
    except Exception as exc:  # noqa: BLE001
        print(f"LP eval unavailable: {exc}")
        return None

    valid_roots = [root for root in scenario_roots if Path(root).is_dir()]
    if not valid_roots:
        print("LP eval unavailable: no valid scenario roots.")
        return None

    workers: Dict[str, Any] = {}
    worker_kwargs = dict(worker_kwargs or {})
    results: List[Dict[str, Any]] = []

    for local_idx in range(min(len(dataset), max_lp)):
        sample = dataset[local_idx]
        scenario_root = _resolve_scenario_root(sample.family, sample.scenario_id, valid_roots)
        if scenario_root is None:
            continue
        if scenario_root not in workers:
            workers[scenario_root] = LPWorkerTwoStage(
                scenarios_dir=scenario_root,
                solver_name=solver,
                verbose=False,
                **worker_kwargs,
            )
        worker = workers[scenario_root]
        u = sample.u_star.unsqueeze(0).to(device)
        h = sample.h_scenario.unsqueeze(0).to(device)
        mask = torch.ones(1, sample.n_zones, device=device)
        with torch.no_grad():
            pred_out = predictor(h, zone_mask=mask)
            z_pred = pred_out["z_pred"]
            logits = ae.decode(z_pred, n_zones=sample.n_zones, n_timesteps=sample.n_timesteps, h_scenario=h)
            pred_bin = _apply_thresholds(torch.sigmoid(logits), thresholds)[0].cpu()
        pred_7 = pred_bin[:, :, :7]

        r = worker.solve(
            scenario_id=sample.scenario_id,
            decoder_output=pred_7,
            max_stages=max_stages,
            force_repair_20=force_repair_20,
        )
        stage_used = _normalize_stage_name(getattr(r, "stage_used", ""))
        stage_flags = _extract_stage_flags(r)
        stage_label = _display_stage_name(r, stage_used)
        ref = max(abs(sample.objective), 1.0)
        lp_obj = float(getattr(r, "objective_value", float("inf")))
        gap_pct = 100.0 * (lp_obj - sample.objective) / ref if math_isfinite(lp_obj) else float("nan")
        hard_fix_obj = float(getattr(r, "hard_fix_objective", float("inf")) or float("inf"))
        repair_obj = float(getattr(r, "repair_objective", float("inf")) or float("inf"))
        hard_fix_gap_pct = 100.0 * (hard_fix_obj - sample.objective) / ref if math_isfinite(hard_fix_obj) else float("nan")
        repair_gap_pct = 100.0 * (repair_obj - sample.objective) / ref if math_isfinite(repair_obj) else float("nan")
        explicit_flip_info = _explicit_flip_counts(pred_bin, getattr(r, "continuous_vars", {}) or {})
        results.append({
            "scenario_id": sample.scenario_id,
            "family": sample.family,
            "criticality": sample.criticality,
            "milp_objective": sample.objective,
            "lp_objective": lp_obj,
            "lp_gap_pct": gap_pct,
            "lp_status": str(getattr(r, "status", "")),
            "lp_stage_used": stage_used,
            "lp_stage_label": stage_label,
            "lp_stage_reached": _deepest_stage_reached(r),
            "lp_slack": float(getattr(r, "slack_used", 0.0) or 0.0),
            "lp_n_flips": int(getattr(r, "n_flips", 0) or 0),
            "lp_explicit_total_flips": explicit_flip_info["total"],
            "lp_flip_counts_by_family": explicit_flip_info["counts"],
            "lp_hard_fix_success": stage_used == "hard_fix",
            "hard_fix_status": str(getattr(r, "hard_fix_status", "")),
            "hard_fix_termination": str(getattr(r, "hard_fix_termination", "")),
            "hard_fix_has_solution": bool(getattr(r, "hard_fix_has_solution", False)),
            "hard_fix_objective": hard_fix_obj,
            "hard_fix_gap_pct": hard_fix_gap_pct,
            "lp_slack_hard_fix": float(getattr(r, "slack_hard_fix", 0.0) or 0.0),
            "repair_status": str(getattr(r, "repair_status", "")),
            "repair_termination": str(getattr(r, "repair_termination", "")),
            "repair_has_solution": bool(getattr(r, "repair_has_solution", False)),
            "repair_objective": repair_obj,
            "repair_gap_pct": repair_gap_pct,
            "lp_slack_repair_20": float(getattr(r, "slack_repair_20", 0.0) or 0.0),
            "lp_reached_hard_fix": stage_flags["hard_fix"],
            "lp_reached_repair_20": stage_flags["repair_20"],
            "solve_time": float(getattr(r, "solve_time", 0.0) or 0.0),
        })

    return {"summary": _aggregate_lp_metrics(results), "samples": results}


def evaluate(
    ae_path: str,
    jepa_path: str,
    data_root: str,
    max_samples: int = 400,
    batch_size: int = 8,
    threshold: float = 0.5,
    threshold_map: Optional[Dict[str, float]] = None,
    search_thresholds: bool = False,
    threshold_search_min: float = 0.20,
    threshold_search_max: float = 0.80,
    threshold_search_step: float = 0.05,
    write_thresholds_json: Optional[str] = None,
    lp: bool = False,
    scenario_roots: Optional[Sequence[str]] = None,
    max_lp: int = 30,
    lp_max_stages: int = 2,
    lp_solver: str = "appsi_highs",
    force_repair_20: bool = False,
    lp_time_limit_hard_fix: float = 3.0,
    lp_time_limit_repair_20: float = 5.0,
    lp_time_limit_repair_100: float = 5.0,
    lp_time_limit_full_soft: float = 5.0,
    lp_time_limit_round_refix: float = 5.0,
    device: Optional[str] = None,
    seed: int = 42,
) -> Dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {dev}")

    ae = _load_ae(ae_path, dev)
    pred, jepa_cfg = _load_jepa(jepa_path, dev)
    print(f"AE  : K={ae.n_tokens} D={ae.latent_dim}")
    print(f"JEPA: K={pred.n_tokens} D={pred.latent_dim}")
    if (pred.n_tokens != ae.n_tokens) or (pred.latent_dim != ae.latent_dim):
        raise ValueError("Latent shape mismatch between AE and JEPA checkpoints.")
    threshold_map = dict(jepa_cfg.get("eval", {}).get("feature_thresholds", {}) or {}) | dict(threshold_map or {})
    default_threshold = float(jepa_cfg.get("eval", {}).get("threshold", threshold))
    thresholds = _threshold_vector(default_threshold, threshold_map, ae.n_features).to(dev)
    if search_thresholds:
        search_grid = _threshold_grid(threshold_search_min, threshold_search_max, threshold_search_step)
    else:
        search_grid = []

    # Match the train_jepa val split.
    ds = UCLabelDataset(roots=[data_root])
    n = len(ds)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(round(n * 0.1)))
    val_idx = perm[n - n_val : n - n_val + min(max_samples, n_val)]
    val_subset = Subset(ds, val_idx)
    loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=uc_collate)
    print(f"Evaluating on {len(val_idx)} val samples.")
    print(
        "Thresholds: "
        + ", ".join(f"{name}={thr:.2f}" for name, thr in zip(UC_FEATURE_NAMES, thresholds.cpu().tolist()))
    )

    F_ = ae.n_features
    TP_j, FP_j, FN_j = np.zeros(F_), np.zeros(F_), np.zeros(F_)
    TP_aux, FP_aux, FN_aux = np.zeros(F_), np.zeros(F_), np.zeros(F_)
    TP_a, FP_a, FN_a = np.zeros(F_), np.zeros(F_), np.zeros(F_)
    has_aux = False
    cos_sum = 0.0; mse_sum = 0.0; n = 0
    prob_jepa_chunks: List[torch.Tensor] = []
    target_chunks: List[torch.Tensor] = []
    mask_chunks: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            u = batch["u_star"].to(dev); h = batch["h_scenario"].to(dev)
            mask = batch["zone_mask"].to(dev)
            Z, T = u.shape[1], u.shape[2]

            z_target = ae.encode(u, zone_mask=mask)
            pred_out = pred(h, zone_mask=mask)
            z_pred = pred_out["z_pred"]

            cos = F.cosine_similarity(z_pred.reshape(-1, ae.latent_dim),
                                      z_target.reshape(-1, ae.latent_dim), dim=-1).mean()
            mse = F.mse_loss(z_pred, z_target)
            cos_sum += float(cos.item()) * u.shape[0]
            mse_sum += float(mse.item()) * u.shape[0]
            n += u.shape[0]

            # Decode both latents through the same frozen AE decoder.
            logits_j = ae.decode(z_pred, n_zones=Z, n_timesteps=T, h_scenario=h)
            logits_a = ae.decode(z_target, n_zones=Z, n_timesteps=T, h_scenario=h)
            prob_j = torch.sigmoid(logits_j)
            aux_logits = pred_out.get("aux_logits")
            prob_aux = torch.sigmoid(aux_logits) if aux_logits is not None else None
            if search_thresholds:
                prob_compact, target_compact, mask_compact = _compact_valid_zones(
                    prob_j.detach().cpu(),
                    u.detach().cpu(),
                    mask.detach().cpu(),
                )
                if prob_compact.numel() > 0:
                    prob_jepa_chunks.append(prob_compact)
                    target_chunks.append(target_compact)
                    mask_chunks.append(mask_compact)
            pb_j = _apply_thresholds(prob_j, thresholds)
            pb_a = _apply_thresholds(torch.sigmoid(logits_a), thresholds)
            if prob_aux is not None:
                pb_aux = _apply_thresholds(prob_aux, thresholds)
                _accumulate(TP_aux, FP_aux, FN_aux, pb_aux, u, mask)
                has_aux = True

            _accumulate(TP_j, FP_j, FN_j, pb_j, u, mask)
            _accumulate(TP_a, FP_a, FN_a, pb_a, u, mask)

    if search_thresholds and prob_jepa_chunks:
        thresholds = _search_best_thresholds(
            torch.cat(prob_jepa_chunks, dim=0),
            torch.cat(target_chunks, dim=0),
            torch.cat(mask_chunks, dim=0),
            thresholds.detach().cpu(),
            search_grid,
        ).to(dev)
        TP_j, FP_j, FN_j = np.zeros(F_), np.zeros(F_), np.zeros(F_)
        TP_aux, FP_aux, FN_aux = np.zeros(F_), np.zeros(F_), np.zeros(F_)
        TP_a, FP_a, FN_a = np.zeros(F_), np.zeros(F_), np.zeros(F_)
        with torch.no_grad():
            for batch in loader:
                u = batch["u_star"].to(dev); h = batch["h_scenario"].to(dev)
                mask = batch["zone_mask"].to(dev)
                Z, T = u.shape[1], u.shape[2]
                z_target = ae.encode(u, zone_mask=mask)
                pred_out = pred(h, zone_mask=mask)
                z_pred = pred_out["z_pred"]
                logits_j = ae.decode(z_pred, n_zones=Z, n_timesteps=T, h_scenario=h)
                logits_a = ae.decode(z_target, n_zones=Z, n_timesteps=T, h_scenario=h)
                pb_j = _apply_thresholds(torch.sigmoid(logits_j), thresholds)
                pb_a = _apply_thresholds(torch.sigmoid(logits_a), thresholds)
                aux_logits = pred_out.get("aux_logits")
                if aux_logits is not None:
                    pb_aux = _apply_thresholds(torch.sigmoid(aux_logits), thresholds)
                    _accumulate(TP_aux, FP_aux, FN_aux, pb_aux, u, mask)
                _accumulate(TP_j, FP_j, FN_j, pb_j, u, mask)
                _accumulate(TP_a, FP_a, FN_a, pb_a, u, mask)
        tuned_map = _threshold_dict(thresholds, ae.n_features)
        print("\nThreshold search:")
        print("  grid = " + ", ".join(f"{v:.2f}" for v in search_grid))
        print("  best = " + ", ".join(f"{k}={v:.2f}" for k, v in tuned_map.items()))
        if write_thresholds_json:
            out_path = Path(write_thresholds_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(tuned_map, f, indent=2)
            print(f"  wrote thresholds -> {out_path}")

    jepa = _f1_table(TP_j, FP_j, FN_j)
    aux = _f1_table(TP_aux, FP_aux, FN_aux) if has_aux else None
    ae_  = _f1_table(TP_a, FP_a, FN_a)
    cos_avg = cos_sum / max(1, n)
    mse_avg = mse_sum / max(1, n)

    print(f"\nLatent alignment: cos={cos_avg:.4f}  mse={mse_avg:.4f}")
    if aux is not None:
        print(f"\n{'feature':<24}  dec_F1   aux_F1   ae_F1   ratio   dec_R   aux_R")
    else:
        print(f"\n{'feature':<24}  jepa_F1   ae_F1   ratio   jepa_P    ae_P   jepa_R    ae_R")
    for f, name in enumerate(UC_FEATURE_NAMES):
        ratio = jepa["per_feature_f1"][f] / max(ae_["per_feature_f1"][f], 1e-9)
        if aux is not None:
            print(
                f"{name:<24}  {jepa['per_feature_f1'][f]:>6.3f}  {aux['per_feature_f1'][f]:>6.3f}  "
                f"{ae_['per_feature_f1'][f]:>6.3f}  {ratio:>5.2f}   "
                f"{jepa['per_feature_recall'][f]:>6.3f}  {aux['per_feature_recall'][f]:>6.3f}"
            )
        else:
            print(f"{name:<24}  {jepa['per_feature_f1'][f]:>6.3f}  {ae_['per_feature_f1'][f]:>6.3f}  "
                  f"{ratio:>5.2f}   {jepa['per_feature_precision'][f]:>6.3f}  {ae_['per_feature_precision'][f]:>6.3f}  "
                  f"{jepa['per_feature_recall'][f]:>6.3f}  {ae_['per_feature_recall'][f]:>6.3f}")
    print(f"\nmacro F1   jepa={jepa['macro_f1']:.4f}   ae(upper-bound)={ae_['macro_f1']:.4f}   "
          f"gap={ae_['macro_f1'] - jepa['macro_f1']:.4f}")
    if aux is not None:
        print(f"macro F1   aux ={aux['macro_f1']:.4f}")

    lp_result = None
    if lp:
        scenario_roots_list = list(scenario_roots or [])
        if not scenario_roots_list:
            scenario_roots_list = _infer_scenario_roots(ds)
        if scenario_roots_list:
            print(f"\nLP soft-fix probe on up to {max_lp} val samples...")
            lp_result = _evaluate_lp_from_decoded(
                val_subset,
                pred,
                ae,
                dev,
                thresholds,
                scenario_roots=scenario_roots_list,
                max_lp=max_lp,
                solver=lp_solver,
                max_stages=lp_max_stages,
                worker_kwargs={
                    "time_limit_hard_fix": lp_time_limit_hard_fix,
                    "time_limit_repair_20": lp_time_limit_repair_20,
                    "time_limit_repair_100": lp_time_limit_repair_100,
                    "time_limit_full_soft": lp_time_limit_full_soft,
                    "time_limit_round_refix": lp_time_limit_round_refix,
                },
                force_repair_20=force_repair_20,
            )
            if lp_result is not None:
                overall = lp_result["summary"]["overall"]
                print(
                    "LP summary: "
                    f"success={overall['success_rate']:.2%} "
                    f"hard_fix={overall['hard_fix_success_rate']:.2%} "
                    f"soft_fix={overall['soft_fix_success_rate']:.2%} "
                    f"gap={overall['mean_gap_pct']:.2f}%"
                )
        else:
            print("LP soft-fix probe skipped: no scenario roots found.")
    return {
        "jepa": jepa,
        "aux": aux,
        "ae": ae_,
        "cos": cos_avg,
        "mse": mse_avg,
        "n": n,
        "thresholds": _threshold_dict(thresholds, ae.n_features),
        "lp": lp_result,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the JEPA predictor end-to-end.")
    p.add_argument("--ae", required=True, help="AE checkpoint (.pt).")
    p.add_argument("--jepa", required=True, help="JEPA predictor checkpoint (.pt).")
    p.add_argument("--data", required=True, help="Dataset root.")
    p.add_argument("--max_samples", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--thresholds_json", default=None,
                   help="JSON string or JSON file mapping feature name -> threshold.")
    p.add_argument("--search_thresholds", action="store_true",
                   help="Search per-feature thresholds on the validation split to maximize macro F1.")
    p.add_argument("--threshold_search_min", type=float, default=0.20)
    p.add_argument("--threshold_search_max", type=float, default=0.80)
    p.add_argument("--threshold_search_step", type=float, default=0.05)
    p.add_argument("--write_thresholds_json", default=None,
                   help="Optional output JSON path for searched thresholds.")
    p.add_argument("--lp", action="store_true", help="Run LPWorkerTwoStage on decoded JEPA plans.")
    p.add_argument("--scenario_roots", nargs="+", default=None,
                   help="Scenario root directories for LP worker. If omitted, tries outputs/<family>.")
    p.add_argument("--max_lp", type=int, default=30)
    p.add_argument("--lp_max_stages", type=int, default=2)
    p.add_argument("--lp_solver", default="appsi_highs")
    p.add_argument("--force_repair_20", action="store_true",
                   help="Force Stage 2 to run even if hard-fix already meets slack tolerance.")
    p.add_argument("--lp_time_limit_hard_fix", type=float, default=3.0)
    p.add_argument("--lp_time_limit_repair_20", type=float, default=5.0)
    p.add_argument("--lp_time_limit_repair_100", type=float, default=5.0)
    p.add_argument("--lp_time_limit_full_soft", type=float, default=5.0)
    p.add_argument("--lp_time_limit_round_refix", type=float, default=5.0)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    threshold_map = _load_threshold_map(args.thresholds_json)
    evaluate(args.ae, args.jepa, args.data, args.max_samples,
             args.batch_size, args.threshold, threshold_map,
             args.search_thresholds, args.threshold_search_min,
             args.threshold_search_max, args.threshold_search_step,
             args.write_thresholds_json,
             args.lp, args.scenario_roots, args.max_lp, args.lp_max_stages,
             args.lp_solver, args.force_repair_20,
             args.lp_time_limit_hard_fix, args.lp_time_limit_repair_20,
             args.lp_time_limit_repair_100, args.lp_time_limit_full_soft,
             args.lp_time_limit_round_refix,
             args.device, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
