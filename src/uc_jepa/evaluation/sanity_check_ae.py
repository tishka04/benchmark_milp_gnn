"""Sanity-check a trained UC autoencoder (Phase 1 MVP).

Computes reconstruction metrics on the full dataset (or per-family slices) and
optionally runs the existing :class:`src.milp.lp_worker_two_stage.LPWorkerTwoStage`
repair cascade on the reconstructed binaries to gauge how often the decoded UC
plan yields a feasible / cheap LP after hard-fix or soft-fix.

Usage:
    python -m src.uc_jepa.evaluation.sanity_check_ae \
        --checkpoint outputs/uc_jepa/autoencoder/uc_ae_best.pt \
        --data-roots outputs/uc_jepa/dataset \
        --out outputs/uc_jepa/sanity \
        [--lp --scenario-roots <path...> --max-lp 50]

The LP step is **optional** and slow; the first 7 UC channels of the decoder
output (battery±, pumped±, dr, thermal_su, thermal_on) align bit-for-bit with
the F=7 ordering expected by ``LPWorkerTwoStage`` so we slice ``[..., :7]``
before passing to the worker.
"""
from __future__ import annotations

from src.demand_response import require_current_formulation

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.uc_jepa import (
    FEAT_BATT_CHARGE,
    FEAT_BATT_DISCHARGE,
    FEAT_IMPORT,
    FEAT_PUMP_CHARGE,
    FEAT_PUMP_DISCHARGE,
    FEAT_THERMAL,
    FEAT_THERMAL_SU,
    UC_FEATURE_NAMES,
)
from src.uc_jepa.data.uc_label_dataset import UCLabelDataset, uc_collate
from src.uc_jepa.models.uc_autoencoder import UCAutoencoder

_LP_STAGE_ORDER = ["hard_fix", "repair_20", "repair_100", "full_soft", "round_refix"]


def _infer_checkpoint_arch(model_cfg: Dict, state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Backfill architecture knobs when older checkpoint configs omit them."""
    out = dict(model_cfg)
    if "scenario_dim" not in out:
        weight = state_dict.get("decoder.scenario_proj.0.weight")
        if weight is not None and getattr(weight, "ndim", 0) == 2:
            out["scenario_dim"] = int(weight.shape[1])
    if "decoder_layers" not in out:
        layer_ids = set()
        prefix = "decoder.cross_attns."
        for key in state_dict:
            if key.startswith(prefix):
                suffix = key[len(prefix):]
                layer_str = suffix.split(".", 1)[0]
                if layer_str.isdigit():
                    layer_ids.add(int(layer_str))
        if layer_ids:
            out["decoder_layers"] = max(layer_ids) + 1
    return out


def _load_model(ckpt_path: str, device: torch.device) -> UCAutoencoder:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    require_current_formulation(ckpt, "checkpoint")
    cfg = ckpt.get("config", {}) or {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    model_cfg = _infer_checkpoint_arch(model_cfg, ckpt["model_state_dict"])
    sd = model_cfg.get("scenario_dim", None)
    model = UCAutoencoder(
        n_features=int(model_cfg.get("n_features", 8)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        n_tokens=int(model_cfg.get("n_tokens", 16)),
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        gru_layers=int(model_cfg.get("gru_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        n_zones_max=int(model_cfg.get("n_zones_max", 64)),
        n_timesteps_max=int(model_cfg.get("n_timesteps_max", 48)),
        scenario_dim=int(sd) if sd is not None else None,
        decoder_layers=int(model_cfg.get("decoder_layers", 1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _normalize_stage_name(stage: Any) -> str:
    return stage.value if hasattr(stage, "value") else str(stage or "")


def _display_stage_name(lp_result: Any, stage: str) -> str:
    if stage == "repair_20":
        return str(getattr(lp_result, "stage_name_repair_20", "") or stage)
    if stage == "repair_100":
        return str(getattr(lp_result, "stage_name_repair_100", "") or stage)
    return str(stage or "")


def _extract_stage_flags(lp_result: Any) -> Dict[str, bool]:
    final_stage = _normalize_stage_name(getattr(lp_result, "stage_used", ""))
    flags = {}
    for stage in _LP_STAGE_ORDER:
        time_val = float(getattr(lp_result, f"time_{stage}", 0.0) or 0.0)
        slack_val = getattr(lp_result, f"slack_{stage}", 0.0)
        flags[stage] = bool(
            time_val > 0.0
            or final_stage == stage
            or (slack_val is not None and float(slack_val) > 0.0)
        )
    return flags


def _deepest_stage_reached(lp_result: Any) -> str:
    flags = _extract_stage_flags(lp_result)
    for stage in reversed(_LP_STAGE_ORDER):
        if flags[stage]:
            return _display_stage_name(lp_result, stage)
    return _display_stage_name(lp_result, _normalize_stage_name(getattr(lp_result, "stage_used", "")))


def _infer_scenario_roots(dataset: UCLabelDataset) -> List[str]:
    roots: List[str] = []
    seen = set()
    families = sorted({Path(path).parent.name for path in dataset.files})
    for family in families:
        candidate = Path("outputs") / family
        if candidate.is_dir():
            root = str(candidate)
            if root not in seen:
                roots.append(root)
                seen.add(root)
    return roots


def _resolve_scenario_root(
    family: str,
    scenario_id: str,
    scenario_roots: Sequence[str],
) -> Optional[str]:
    if len(scenario_roots) == 1:
        return scenario_roots[0]

    for root in scenario_roots:
        path = Path(root)
        if path.name == family:
            return root

    for root in scenario_roots:
        path = Path(root)
        if (path / f"{scenario_id}.json").exists() or (path / "dispatch_batch" / f"{scenario_id}.json").exists():
            return root
    return None


def _aggregate_lp_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _summary(block: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(block)
        if n == 0:
            return {
                "n": 0,
                "success_rate": 0.0,
                "hard_fix_success_rate": 0.0,
                "soft_fix_success_rate": 0.0,
                "mean_gap_pct": float("nan"),
                "median_gap_pct": float("nan"),
                "mean_binary_flips": float("nan"),
                "median_binary_flips": float("nan"),
                "mean_explicit_binary_flips": float("nan"),
                "median_explicit_binary_flips": float("nan"),
                "repair_zero_explicit_flip_rate": float("nan"),
                "mean_flips_by_family": {},
                "mean_solve_time": float("nan"),
                "stage_distribution": {},
            }

        success_rows = [r for r in block if math_isfinite(r["lp_objective"])]
        repair_rows = [r for r in block if r["lp_stage_used"] not in ("", "hard_fix", "failed")]
        hard_fix_count = sum(r["lp_stage_used"] == "hard_fix" for r in block)
        soft_fix_count = sum(
            r["lp_stage_used"] not in ("", "hard_fix", "failed") and math_isfinite(r["lp_objective"])
            for r in block
        )
        gap_vals = [float(r["lp_gap_pct"]) for r in success_rows if math_isfinite(r["lp_gap_pct"])]
        flip_vals = [float(r["lp_n_flips"]) for r in success_rows]
        explicit_flip_vals = [
            float(r["lp_explicit_total_flips"])
            for r in success_rows
            if "lp_explicit_total_flips" in r and math_isfinite(r["lp_explicit_total_flips"])
        ]
        solve_times = [float(r["solve_time"]) for r in block if math_isfinite(r["solve_time"])]
        stage_counts = Counter((r["lp_stage_used"] or "unknown") for r in block)
        repair_zero_explicit = [
            r for r in repair_rows
            if r.get("lp_explicit_total_flips") is not None and float(r.get("lp_explicit_total_flips", 0)) == 0.0
        ]

        flip_family_names = sorted(
            {
                key
                for r in success_rows
                for key in (r.get("lp_flip_counts_by_family", {}) or {}).keys()
            }
        )
        mean_flips_by_family = {}
        for key in flip_family_names:
            vals = [
                float((r.get("lp_flip_counts_by_family", {}) or {}).get(key, 0.0))
                for r in success_rows
            ]
            mean_flips_by_family[key] = float(np.mean(vals)) if vals else float("nan")

        return {
            "n": n,
            "success_rate": len(success_rows) / n,
            "hard_fix_success_rate": hard_fix_count / n,
            "soft_fix_success_rate": soft_fix_count / n,
            "mean_gap_pct": float(np.mean(gap_vals)) if gap_vals else float("nan"),
            "median_gap_pct": float(np.median(gap_vals)) if gap_vals else float("nan"),
            "mean_binary_flips": float(np.mean(flip_vals)) if flip_vals else float("nan"),
            "median_binary_flips": float(np.median(flip_vals)) if flip_vals else float("nan"),
            "mean_explicit_binary_flips": float(np.mean(explicit_flip_vals)) if explicit_flip_vals else float("nan"),
            "median_explicit_binary_flips": float(np.median(explicit_flip_vals)) if explicit_flip_vals else float("nan"),
            "repair_zero_explicit_flip_rate": (
                len(repair_zero_explicit) / len(repair_rows) if repair_rows else float("nan")
            ),
            "mean_flips_by_family": mean_flips_by_family,
            "mean_solve_time": float(np.mean(solve_times)) if solve_times else float("nan"),
            "stage_distribution": {
                stage: {"count": count, "rate": count / n}
                for stage, count in sorted(
                    stage_counts.items(),
                    key=lambda item: (
                        _LP_STAGE_ORDER.index(item[0]) if item[0] in _LP_STAGE_ORDER else len(_LP_STAGE_ORDER),
                        item[0],
                    ),
                )
            },
        }

    per_family = {
        fam: _summary([r for r in rows if r["family"] == fam])
        for fam in sorted({r["family"] for r in rows})
    }
    return {"overall": _summary(rows), "per_family": per_family}


def _resolve_report_path(scenario_root: str, scenario_id: str) -> Optional[Path]:
    base = Path(scenario_root)
    candidates = [
        base / "reports" / f"{scenario_id}.json",
        base / f"{scenario_id}.json",
        base / "dispatch_batch" / f"{scenario_id}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_report_json(scenario_root: str, scenario_id: str) -> Optional[Dict[str, Any]]:
    path = _resolve_report_path(scenario_root, scenario_id)
    if path is None:
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _zone_names_from_report(report: Dict[str, Any], n_zones: int) -> List[str]:
    detail = report.get("detail", {}) or report.get("pre_dispatch", {})
    zone_names = list(report.get("zones") or [])
    if not zone_names and isinstance(detail, dict):
        for key in ("demand", "thermal_commitment", "thermal", "battery_charge"):
            block = detail.get(key, {})
            if isinstance(block, dict) and block:
                zone_names = sorted(block.keys())
                break
    if len(zone_names) < n_zones and isinstance(detail, dict):
        demand_keys = list((detail.get("demand", {}) or {}).keys())
        for zone in sorted(demand_keys):
            if zone not in zone_names:
                zone_names.append(zone)
            if len(zone_names) >= n_zones:
                break
    if len(zone_names) > n_zones:
        zone_names = zone_names[:n_zones]
    return zone_names


def _detail_zone_matrix(
    detail: Dict[str, Any],
    key: str,
    zone_names: Sequence[str],
    n_timesteps: int,
) -> np.ndarray:
    out = np.zeros((len(zone_names), n_timesteps), dtype=np.float32)
    block = detail.get(key, {})
    if not isinstance(block, dict):
        return out
    for z_idx, zone in enumerate(zone_names):
        series = block.get(zone, [])
        if not isinstance(series, list):
            continue
        for t in range(min(n_timesteps, len(series))):
            try:
                out[z_idx, t] = float(series[t])
            except (TypeError, ValueError):
                out[z_idx, t] = 0.0
    return out


def _safe_masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    mask_bool = np.asarray(mask, dtype=bool)
    denom = int(mask_bool.sum())
    if denom <= 0:
        return float("nan")
    return float(np.asarray(values, dtype=np.float64)[mask_bool].mean())


def _lp_solution_to_uc_tensor(
    continuous_vars: Dict[str, np.ndarray],
    n_zones: int,
    n_timesteps: int,
    power_eps: float = 1e-3,
) -> np.ndarray:
    out = np.zeros((n_zones, n_timesteps, 8), dtype=np.float32)

    def _get_2d(name: str) -> Optional[np.ndarray]:
        arr = continuous_vars.get(name)
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            return None
        return arr[:n_zones, :n_timesteps]

    def _get_1d(name: str) -> Optional[np.ndarray]:
        arr = continuous_vars.get(name)
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 1:
            return None
        return arr[:n_timesteps]

    batt_ch = _get_2d("b_charge")
    if batt_ch is not None:
        out[: batt_ch.shape[0], : batt_ch.shape[1], FEAT_BATT_CHARGE] = (batt_ch > power_eps).astype(np.float32)
    batt_dis = _get_2d("b_discharge")
    if batt_dis is not None:
        out[: batt_dis.shape[0], : batt_dis.shape[1], FEAT_BATT_DISCHARGE] = (batt_dis > power_eps).astype(np.float32)
    pump_ch = _get_2d("pumped_charge")
    if pump_ch is not None:
        out[: pump_ch.shape[0], : pump_ch.shape[1], FEAT_PUMP_CHARGE] = (pump_ch > power_eps).astype(np.float32)
    pump_dis = _get_2d("pumped_discharge")
    if pump_dis is not None:
        out[: pump_dis.shape[0], : pump_dis.shape[1], FEAT_PUMP_DISCHARGE] = (pump_dis > power_eps).astype(np.float32)
    dr_active = _get_2d("dr_active")
    if dr_active is not None:
        out[: dr_active.shape[0], : dr_active.shape[1], 4] = (dr_active >= 0.5).astype(np.float32)
    thermal_su = _get_2d("v_thermal_startup")
    if thermal_su is not None:
        out[: thermal_su.shape[0], : thermal_su.shape[1], FEAT_THERMAL_SU] = (thermal_su >= 0.5).astype(np.float32)
    thermal_on = _get_2d("u_thermal")
    if thermal_on is not None:
        out[: thermal_on.shape[0], : thermal_on.shape[1], FEAT_THERMAL] = (thermal_on >= 0.5).astype(np.float32)
    import_mode = _get_1d("import_mode")
    if import_mode is not None:
        out[:, : import_mode.shape[0], FEAT_IMPORT] = (import_mode[None, :] >= 0.5).astype(np.float32)
    return out


def _explicit_flip_counts(
    pred_bin: torch.Tensor,
    continuous_vars: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    pred_np = np.asarray(pred_bin.detach().cpu().numpy(), dtype=np.float32)
    repaired_np = _lp_solution_to_uc_tensor(
        continuous_vars,
        n_zones=pred_np.shape[0],
        n_timesteps=pred_np.shape[1],
    )
    names = {
        "battery_charge": FEAT_BATT_CHARGE,
        "battery_discharge": FEAT_BATT_DISCHARGE,
        "phs_charge": FEAT_PUMP_CHARGE,
        "phs_discharge": FEAT_PUMP_DISCHARGE,
        "dr_active": 4,
        "thermal_startup": FEAT_THERMAL_SU,
        "thermal_on": FEAT_THERMAL,
        "import_mode": FEAT_IMPORT,
    }
    counts = {}
    total = 0
    for name, feat_idx in names.items():
        diff = np.abs(pred_np[..., feat_idx] - repaired_np[..., feat_idx]) > 0.5
        count = int(diff.sum())
        counts[name] = count
        total += count
    return {"counts": counts, "total": total}


def _compute_stress_metrics(
    pred_bin: torch.Tensor,
    target: torch.Tensor,
    report: Dict[str, Any],
) -> Dict[str, float]:
    pred_np = np.asarray(pred_bin.detach().cpu().numpy(), dtype=np.float32)
    target_np = np.asarray(target.detach().cpu().numpy(), dtype=np.float32)
    z_count, t_count = target_np.shape[:2]
    detail = report.get("detail", {}) or report.get("pre_dispatch", {})
    zone_names = _zone_names_from_report(report, z_count)

    demand = _detail_zone_matrix(detail, "demand", zone_names, t_count)
    solar = _detail_zone_matrix(detail, "solar", zone_names, t_count)
    wind = _detail_zone_matrix(detail, "wind", zone_names, t_count)
    hydro_ror = _detail_zone_matrix(detail, "hydro_ror", zone_names, t_count)
    residual = demand - solar - wind - hydro_ror

    system_demand = demand.sum(axis=0)
    if system_demand.size == 0:
        peak_mask = np.zeros((z_count, t_count), dtype=bool)
    else:
        peak_threshold = float(np.quantile(system_demand, 0.75))
        peak_mask = np.broadcast_to(system_demand[None, :] >= peak_threshold, (z_count, t_count))

    positive_residual = residual[residual > 0]
    if positive_residual.size > 0:
        high_residual_threshold = float(np.quantile(positive_residual, 0.75))
        high_residual_mask = residual >= high_residual_threshold
    else:
        high_residual_threshold = float(np.quantile(residual, 0.75)) if residual.size else 0.0
        high_residual_mask = residual >= high_residual_threshold

    thermal_err = np.abs(pred_np[..., FEAT_THERMAL] - target_np[..., FEAT_THERMAL])
    batt_dis_err = np.abs(pred_np[..., FEAT_BATT_DISCHARGE] - target_np[..., FEAT_BATT_DISCHARGE])
    batt_ch_err = np.abs(pred_np[..., FEAT_BATT_CHARGE] - target_np[..., FEAT_BATT_CHARGE])
    true_on = target_np[..., FEAT_THERMAL] > 0.5
    true_off = target_np[..., FEAT_THERMAL] <= 0.5

    return {
        "hamming_on_peak_hours": _safe_masked_mean(thermal_err, peak_mask),
        "hamming_on_high_residual_demand": _safe_masked_mean(thermal_err, high_residual_mask),
        "hamming_thermal_on_when_true_on": _safe_masked_mean(thermal_err, true_on),
        "hamming_thermal_on_when_true_off": _safe_masked_mean(thermal_err, true_off),
        "battery_discharge_error_when_residual_positive": _safe_masked_mean(batt_dis_err, residual > 0.0),
        "battery_charge_error_when_residual_negative": _safe_masked_mean(batt_ch_err, residual < 0.0),
    }


def evaluate_stress_diagnostics(
    model: UCAutoencoder,
    dataset: UCLabelDataset,
    device: torch.device,
    scenario_roots: Sequence[str],
) -> Optional[Dict[str, Any]]:
    valid_roots = [root for root in scenario_roots if Path(root).is_dir()]
    if not valid_roots:
        return None

    samples: List[Dict[str, Any]] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        scenario_root = _resolve_scenario_root(sample.family, sample.scenario_id, valid_roots)
        if scenario_root is None:
            continue
        report = _load_report_json(scenario_root, sample.scenario_id)
        if report is None:
            continue

        u = sample.u_star.unsqueeze(0).to(device)
        mask = torch.ones(1, sample.n_zones, device=device)
        h = sample.h_scenario.unsqueeze(0).to(device) if model.scenario_dim is not None else None
        with torch.no_grad():
            out = model(u, zone_mask=mask, h_scenario=h)
            pred = (torch.sigmoid(out["logits"]) >= 0.5).float()

        metrics = _compute_stress_metrics(pred[0].cpu(), sample.u_star, report)
        samples.append({
            "scenario_id": sample.scenario_id,
            "family": sample.family,
            "criticality": sample.criticality,
            **metrics,
        })

    if not samples:
        return None

    metric_names = [
        "hamming_on_peak_hours",
        "hamming_on_high_residual_demand",
        "hamming_thermal_on_when_true_on",
        "hamming_thermal_on_when_true_off",
        "battery_discharge_error_when_residual_positive",
        "battery_charge_error_when_residual_negative",
    ]

    def _summary(block: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"n": len(block)}
        for name in metric_names:
            vals = [float(r[name]) for r in block if math_isfinite(r[name])]
            out[f"{name}_mean"] = float(np.mean(vals)) if vals else float("nan")
            out[f"{name}_median"] = float(np.median(vals)) if vals else float("nan")
        return out

    per_family = {
        fam: _summary([r for r in samples if r["family"] == fam])
        for fam in sorted({r["family"] for r in samples})
    }
    return {"summary": {"overall": _summary(samples), "per_family": per_family}, "samples": samples}


def _per_feature_metrics(
    pred_bin: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """pred_bin/target: [B, Z, T, F] in {0,1}. mask: [B, Z]."""
    F_ = pred_bin.shape[-1]
    m = mask.view(*mask.shape, 1, 1).expand_as(pred_bin)
    out: Dict[str, float] = {}
    denom_total = m.sum().clamp_min(1.0).item()
    diff = (pred_bin - target).abs()
    out["hamming_overall"] = float((diff * m).sum().item() / denom_total)
    for f in range(F_):
        mf = m[..., f]
        d = diff[..., f]
        denom = mf.sum().clamp_min(1.0).item()
        name = UC_FEATURE_NAMES[f] if f < len(UC_FEATURE_NAMES) else f"feat_{f}"
        out[f"hamming_{name}"] = float((d * mf).sum().item() / denom)
        # also raw positive rate (target) and predicted positive rate
        out[f"true_rate_{name}"] = float((target[..., f] * mf).sum().item() / denom)
        out[f"pred_rate_{name}"] = float((pred_bin[..., f] * mf).sum().item() / denom)
    return out


def _startup_violation_rate(pred_bin: torch.Tensor, mask: torch.Tensor) -> float:
    """Fraction of (z, t>=1) where startup < ReLU(on[t]-on[t-1]). On-rule violation."""
    on = pred_bin[..., FEAT_THERMAL]            # [B, Z, T]
    su = pred_bin[..., FEAT_THERMAL_SU]
    if on.shape[-1] < 2:
        return 0.0
    target = (on[:, :, 1:] - on[:, :, :-1]).clamp_min(0.0)
    viol = (su[:, :, 1:] < target).float()
    m = mask.view(*mask.shape, 1).expand_as(viol)
    denom = m.sum().clamp_min(1.0).item()
    return float((viol * m).sum().item() / denom)


def _mode_exclusion_violation(pred_bin: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """Battery and pumped charge+discharge can't both be 1 simultaneously."""
    # Indices: 0 batt_ch, 1 batt_dis, 2 pump_ch, 3 pump_dis
    batt_clash = (pred_bin[..., 0] * pred_bin[..., 1])
    pump_clash = (pred_bin[..., 2] * pred_bin[..., 3])
    m = mask.view(*mask.shape, 1).expand_as(batt_clash)
    denom = m.sum().clamp_min(1.0).item()
    return {
        "battery_mode_clash_rate": float((batt_clash * m).sum().item() / denom),
        "pumped_mode_clash_rate": float((pump_clash * m).sum().item() / denom),
    }


def evaluate_dataset(
    model: UCAutoencoder,
    dataset: UCLabelDataset,
    device: torch.device,
    batch_size: int = 8,
) -> Dict:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=uc_collate,
    )
    by_family: Dict[str, Dict] = defaultdict(lambda: {
        "n": 0,
        "bce_sum": 0.0,
        "hamming_sum": 0.0,
        "ham_per_feat_sum": np.zeros(model.n_features, dtype=np.float64),
        "true_rate_sum": np.zeros(model.n_features, dtype=np.float64),
        "pred_rate_sum": np.zeros(model.n_features, dtype=np.float64),
        "startup_viol_sum": 0.0,
        "batt_clash_sum": 0.0,
        "pump_clash_sum": 0.0,
        "criticality_sum": 0.0,
    })
    samples: List[Dict] = []

    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    with torch.no_grad():
        for batch in loader:
            u = batch["u_star"].to(device)
            mask = batch["zone_mask"].to(device)
            h = batch.get("h_scenario") if model.scenario_dim is not None else None
            if h is not None:
                h = h.to(device)
            out = model(u, zone_mask=mask, h_scenario=h)
            logits = out["logits"]
            prob = torch.sigmoid(logits)
            pred_bin = (prob >= 0.5).float()

            m = mask.view(*mask.shape, 1, 1).expand_as(u)
            denom = m.sum(dim=(1, 2, 3)).clamp_min(1.0)               # [B]
            bce_b = (bce(logits, u) * m).sum(dim=(1, 2, 3)) / denom    # [B]
            ham_b = ((pred_bin - u).abs() * m).sum(dim=(1, 2, 3)) / denom

            B = u.shape[0]
            for i in range(B):
                fam = batch["family"][i]
                rec = by_family[fam]
                rec["n"] += 1
                rec["bce_sum"] += float(bce_b[i].item())
                rec["hamming_sum"] += float(ham_b[i].item())
                rec["criticality_sum"] += float(batch["criticality"][i].item())

                # per-feature
                m_i = mask[i].view(-1, 1, 1).expand_as(u[i])
                d_i = (pred_bin[i] - u[i]).abs()
                for f in range(model.n_features):
                    den = m_i[..., f].sum().clamp_min(1.0).item()
                    rec["ham_per_feat_sum"][f] += float((d_i[..., f] * m_i[..., f]).sum().item() / den)
                    rec["true_rate_sum"][f] += float((u[i, ..., f] * m_i[..., f]).sum().item() / den)
                    rec["pred_rate_sum"][f] += float((pred_bin[i, ..., f] * m_i[..., f]).sum().item() / den)

                # constraint violations (per-sample averages)
                pb_i = pred_bin[i:i+1]
                mk_i = mask[i:i+1]
                rec["startup_viol_sum"] += _startup_violation_rate(pb_i, mk_i)
                clash = _mode_exclusion_violation(pb_i, mk_i)
                rec["batt_clash_sum"] += clash["battery_mode_clash_rate"]
                rec["pump_clash_sum"] += clash["pumped_mode_clash_rate"]

                samples.append({
                    "scenario_id": batch["scenario_id"][i],
                    "family": fam,
                    "criticality": float(batch["criticality"][i].item()),
                    "milp_objective": float(batch["milp_objective"][i].item()),
                    "bce": float(bce_b[i].item()),
                    "hamming": float(ham_b[i].item()),
                })

    summary: Dict[str, Dict] = {}
    for fam, rec in by_family.items():
        n = max(1, rec["n"])
        summary[fam] = {
            "n": rec["n"],
            "bce_mean": rec["bce_sum"] / n,
            "hamming_mean": rec["hamming_sum"] / n,
            "criticality_mean": rec["criticality_sum"] / n,
            "startup_viol_mean": rec["startup_viol_sum"] / n,
            "battery_mode_clash_mean": rec["batt_clash_sum"] / n,
            "pumped_mode_clash_mean": rec["pump_clash_sum"] / n,
        }
        for f in range(model.n_features):
            name = UC_FEATURE_NAMES[f] if f < len(UC_FEATURE_NAMES) else f"feat_{f}"
            summary[fam][f"hamming_{name}"] = float(rec["ham_per_feat_sum"][f] / n)
            summary[fam][f"true_rate_{name}"] = float(rec["true_rate_sum"][f] / n)
            summary[fam][f"pred_rate_{name}"] = float(rec["pred_rate_sum"][f] / n)

    return {"per_family": summary, "samples": samples}


def run_lp_check(
    model: UCAutoencoder,
    dataset: UCLabelDataset,
    device: torch.device,
    scenario_roots: Sequence[str],
    max_lp: int = 50,
    solver: str = "appsi_highs",
    max_stages: int = 5,
    worker_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[Dict]:
    """Run AE -> LPWorkerTwoStage and summarize repair behaviour."""
    try:
        from src.milp.lp_worker_two_stage import LPWorkerTwoStage
    except Exception as exc:  # noqa: BLE001
        print(f"  LP check unavailable: {exc}")
        return None

    valid_roots = [root for root in scenario_roots if Path(root).is_dir()]
    if not valid_roots:
        print("  LP check unavailable: no valid scenario roots were found.")
        return None

    workers: Dict[str, Any] = {}
    worker_kwargs = dict(worker_kwargs or {})

    results: List[Dict] = []
    n_done = 0
    for idx in range(len(dataset)):
        if n_done >= max_lp:
            break
        sample = dataset[idx]
        scenario_root = _resolve_scenario_root(sample.family, sample.scenario_id, valid_roots)
        if scenario_root is None:
            print(f"  LP skipped for {sample.scenario_id}: no scenario root for family={sample.family}")
            continue
        if scenario_root not in workers:
            workers[scenario_root] = LPWorkerTwoStage(
                scenarios_dir=scenario_root,
                solver_name=solver,
                verbose=False,
                **worker_kwargs,
            )
        worker = workers[scenario_root]
        u = sample.u_star.unsqueeze(0).to(device)            # [1, Z, T, F=8]
        mask = torch.ones(1, sample.n_zones, device=device)
        h = sample.h_scenario.unsqueeze(0).to(device) if model.scenario_dim is not None else None
        with torch.no_grad():
            out = model(u, zone_mask=mask, h_scenario=h)
            pred = (torch.sigmoid(out["logits"]) >= 0.5).float()
        pred_full = pred[0].cpu()
        # LP worker expects [Z, T, F=7] — strip the extra import_mode column.
        pred_7 = pred_full[:, :, :7]

        t0 = time.time()
        try:
            r = worker.solve(
                scenario_id=sample.scenario_id,
                decoder_output=pred_7,
                max_stages=max_stages,
            )
            dt = time.time() - t0
            stage_used = _normalize_stage_name(getattr(r, "stage_used", ""))
            stage_label = _display_stage_name(r, stage_used)
            lp_obj = float(getattr(r, "objective_value", float("inf")))
            ref = max(abs(sample.objective), 1.0)
            gap_pct = 100.0 * (lp_obj - sample.objective) / ref if math_isfinite(lp_obj) else float("nan")
            stage_flags = _extract_stage_flags(r)
            hard_fix_obj = float(getattr(r, "hard_fix_objective", float("inf")) or float("inf"))
            repair_obj = float(getattr(r, "repair_objective", float("inf")) or float("inf"))
            hard_fix_gap_pct = (
                100.0 * (hard_fix_obj - sample.objective) / ref if math_isfinite(hard_fix_obj) else float("nan")
            )
            repair_gap_pct = (
                100.0 * (repair_obj - sample.objective) / ref if math_isfinite(repair_obj) else float("nan")
            )
            explicit_flip_info = _explicit_flip_counts(pred_full, getattr(r, "continuous_vars", {}) or {})
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
                "lp_slack_repair_100": float(getattr(r, "slack_repair_100", 0.0) or 0.0),
                "lp_slack_full_soft": float(getattr(r, "slack_full_soft", 0.0) or 0.0),
                "lp_slack_round_refix": float(getattr(r, "slack_round_refix", 0.0) or 0.0),
                "lp_reached_hard_fix": stage_flags["hard_fix"],
                "lp_reached_repair_20": stage_flags["repair_20"],
                "lp_reached_repair_100": stage_flags["repair_100"],
                "lp_reached_full_soft": stage_flags["full_soft"],
                "lp_reached_round_refix": stage_flags["round_refix"],
                "solve_time": dt,
                "scenario_root": scenario_root,
            })
            n_done += 1
            if n_done % 5 == 0:
                print(f"  LP {n_done}/{max_lp} done")
        except Exception as exc:  # noqa: BLE001
            print(f"  LP failed for {sample.scenario_id}: {exc}")

    return {"summary": _aggregate_lp_metrics(results), "samples": results}


def math_isfinite(x: float) -> bool:
    try:
        return float(x) == float(x) and abs(float(x)) != float("inf")
    except Exception:
        return False


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity-check a UC autoencoder checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-roots", nargs="+", required=True)
    p.add_argument("--families", nargs="*", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--out", required=True, help="Output directory for JSON summaries.")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    # Optional LP step
    p.add_argument("--lp", action="store_true", help="Run the LPWorkerTwoStage cascade on a subset.")
    p.add_argument("--scenarios-dir", default=None,
                   help="Single scenario root directory consumed by LPWorkerTwoStage.")
    p.add_argument("--scenario-roots", nargs="+", default=None,
                   help="One or more scenario root directories. If omitted, tries outputs/<family>.")
    p.add_argument("--max-lp", type=int, default=30)
    p.add_argument("--lp-max-stages", type=int, default=5,
                   help="How many LP worker stages to allow (1-5). Default: 5.")
    p.add_argument("--lp-solver", default="appsi_highs")
    p.add_argument("--lp-time-limit-hard-fix", type=float, default=30.0)
    p.add_argument("--lp-time-limit-repair-20", type=float, default=30.0)
    p.add_argument("--lp-time-limit-repair-100", type=float, default=45.0)
    p.add_argument("--lp-time-limit-full-soft", type=float, default=60.0)
    p.add_argument("--lp-time-limit-round-refix", type=float, default=60.0)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint}")
    model = _load_model(args.checkpoint, device)

    print(f"Loading dataset from {args.data_roots} (families={args.families})")
    dataset = UCLabelDataset(
        roots=args.data_roots,
        families=args.families,
        max_samples=args.max_samples,
    )
    print(f"  {len(dataset)} samples")
    scenario_roots = list(args.scenario_roots or [])
    if args.scenarios_dir:
        scenario_roots.append(args.scenarios_dir)
    if not scenario_roots:
        scenario_roots = _infer_scenario_roots(dataset)
        if scenario_roots:
            print(f"Auto-detected scenario roots: {scenario_roots}")

    print("Computing reconstruction metrics...")
    recon = evaluate_dataset(model, dataset, device, batch_size=args.batch_size)

    summary_path = out_dir / "reconstruction_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(recon["per_family"], f, indent=2)
    print(f"  per-family metrics -> {summary_path}")

    samples_path = out_dir / "reconstruction_per_sample.json"
    with open(samples_path, "w") as f:
        json.dump(recon["samples"], f, indent=2)
    print(f"  per-sample metrics -> {samples_path}")

    print("\n=== Per-family reconstruction summary ===")
    for fam, m in recon["per_family"].items():
        print(
            f"  [{fam}] n={m['n']} bce={m['bce_mean']:.4f} ham={m['hamming_mean']:.4f} "
            f"startup_viol={m['startup_viol_mean']:.3f} "
            f"batt_clash={m['battery_mode_clash_mean']:.3f} "
            f"pump_clash={m['pumped_mode_clash_mean']:.3f}"
        )

    stress = evaluate_stress_diagnostics(model, dataset, device, scenario_roots)
    if stress is not None:
        stress_summary_path = out_dir / "stress_metrics.json"
        with open(stress_summary_path, "w") as f:
            json.dump(stress["summary"], f, indent=2)
        print(f"  stress metrics -> {stress_summary_path}")
        stress_samples_path = out_dir / "stress_per_sample.json"
        with open(stress_samples_path, "w") as f:
            json.dump(stress["samples"], f, indent=2)
        print(f"  stress samples -> {stress_samples_path}")
        overall = stress["summary"]["overall"]
        print(
            "  [stress overall] "
            f"peak_on={overall['hamming_on_peak_hours_mean']:.4f} "
            f"high_resid_on={overall['hamming_on_high_residual_demand_mean']:.4f} "
            f"true_on={overall['hamming_thermal_on_when_true_on_mean']:.4f} "
            f"true_off={overall['hamming_thermal_on_when_true_off_mean']:.4f}"
        )

    if args.lp:
        if not scenario_roots:
            print("ERROR: --lp requires --scenario-roots / --scenarios-dir, or auto-detectable outputs/<family> roots.")
            return 1
        print(f"\nRunning LP worker check on up to {args.max_lp} scenarios...")
        lp = run_lp_check(
            model, dataset, device,
            scenario_roots=scenario_roots,
            max_lp=args.max_lp,
            solver=args.lp_solver,
            max_stages=args.lp_max_stages,
            worker_kwargs={
                "time_limit_hard_fix": args.lp_time_limit_hard_fix,
                "time_limit_repair_20": args.lp_time_limit_repair_20,
                "time_limit_repair_100": args.lp_time_limit_repair_100,
                "time_limit_full_soft": args.lp_time_limit_full_soft,
                "time_limit_round_refix": args.lp_time_limit_round_refix,
            },
        )
        if lp is not None:
            lp_path = out_dir / "lp_check.json"
            with open(lp_path, "w") as f:
                json.dump(lp, f, indent=2)
            print(f"  LP results -> {lp_path}")
            lp_summary_path = out_dir / "lp_summary.json"
            with open(lp_summary_path, "w") as f:
                json.dump(lp["summary"], f, indent=2)
            print(f"  LP summary -> {lp_summary_path}")
            lp_samples_path = out_dir / "lp_per_sample.json"
            with open(lp_samples_path, "w") as f:
                json.dump(lp["samples"], f, indent=2)
            print(f"  LP samples -> {lp_samples_path}")

            overall = lp["summary"]["overall"]
            print(
                "  [overall] "
                f"n={overall['n']} success={overall['success_rate']:.2%} "
                f"hard_fix={overall['hard_fix_success_rate']:.2%} "
                f"soft_fix={overall['soft_fix_success_rate']:.2%} "
                f"gap={overall['mean_gap_pct']:.2f}% flips={overall['mean_binary_flips']:.2f} "
                f"explicit_flips={overall['mean_explicit_binary_flips']:.2f}"
            )
            if math_isfinite(overall["repair_zero_explicit_flip_rate"]):
                print(
                    "  [overall] "
                    f"repair_zero_explicit_flip_rate={overall['repair_zero_explicit_flip_rate']:.2%}"
                )
            print("  Stage distribution:")
            for stage, stats in overall["stage_distribution"].items():
                print(f"    {stage}: {stats['count']} ({stats['rate']:.2%})")

            for fam, m in lp["summary"]["per_family"].items():
                print(
                    f"  [{fam}] n={m['n']} success={m['success_rate']:.2%} "
                    f"hard_fix={m['hard_fix_success_rate']:.2%} gap={m['mean_gap_pct']:.2f}% "
                    f"time={m['mean_solve_time']:.2f}s"
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
