from __future__ import annotations

from src.demand_response import require_current_formulation

import argparse
import copy
import json
import math
import pickle
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.eval import (
    build_candidate_pool_frame,
    build_physical_complexity_frame,
    compute_best_of_k_curve,
    compute_eval_metrics,
    compute_k_sampling_diagnostics,
    compute_solution_diversity_frame,
    fit_physical_feature_robustness,
    fit_scaling_law_models,
    merge_physical_complexity_features,
)
from src.eval.pipeline_runner import (
    PipelineResult,
    PipelineRunner,
    _binary_stats,
    _pairwise_hamming_stats,
    backfill_projection_diagnostics,
)
from src.uc_jepa.data.uc_label_dataset import UCLabelDataset
from src.uc_jepa.latent_utils import (
    apply_thresholds,
    load_frozen_ae,
    load_frozen_jepa,
    load_latent_ebm,
    load_threshold_map,
    threshold_dict,
    threshold_vector,
)
from src.uc_jepa.sampling.latent_langevin import LatentLangevinConfig, LatentLangevinSampler
from src.uc_jepa.workers.milp_warm_worker import MILPWarmWorker


_STAGE_ORDER = ["hard_fix", "repair_20", "repair_100", "full_soft", "round_refix"]


def _default_from_checkpoint(
    latent_ckpt: Dict[str, Any],
    key: str,
    fallback: Optional[str] = None,
) -> Optional[str]:
    value = latent_ckpt.get(key)
    if value:
        return str(value)
    config_value = (latent_ckpt.get("config", {}) or {}).get(key)
    if config_value:
        return str(config_value)
    return fallback


def _family_alias(name: str) -> str:
    text = str(name or "")
    suffix = "_criticality_scenarios"
    return text[: -len(suffix)] if text.endswith(suffix) else text


def _infer_scenario_roots_from_families(families: Iterable[str]) -> List[str]:
    roots: List[str] = []
    seen = set()
    for family in sorted({str(fam) for fam in families if fam}):
        candidate = Path("outputs") / family
        if candidate.is_dir():
            root = str(candidate)
            if root not in seen:
                roots.append(root)
                seen.add(root)
    return roots


def _parse_best_of_k_values(text: str) -> List[int]:
    values: List[int] = []
    for raw in str(text or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        values.append(int(raw))
    return sorted({value for value in values if value > 0})


def _select_sample_index(
    policy: str,
    energies: Sequence[float],
    objectives: Sequence[float],
) -> int:
    if not energies:
        return 0

    if policy == "first":
        return 0

    if policy == "oracle_lp":
        valid = [
            idx for idx, value in enumerate(objectives)
            if math.isfinite(float(value))
        ]
        if valid:
            return min(valid, key=lambda idx: float(objectives[idx]))

    if policy == "first_feasible":
        for idx, value in enumerate(objectives):
            if math.isfinite(float(value)):
                return idx

    finite_energy = [
        idx for idx, value in enumerate(energies)
        if math.isfinite(float(value))
    ]
    if finite_energy:
        return min(finite_energy, key=lambda idx: float(energies[idx]))
    return 0


def _apply_selected_record(result: PipelineResult, record: Dict[str, Any]) -> None:
    result.dr_metrics = dict(record.get("dr_metrics", {}))
    result.lp_status = str(record.get("status", ""))
    result.lp_stage_used = str(record.get("stage_label") or record.get("stage_used") or "")
    result.lp_stage_reached = str(record.get("stage_reached") or result.lp_stage_used)
    result.lp_objective = float(record.get("lp_objective", float("nan")))
    result.lp_slack = float(record.get("slack_mwh", 0.0) or 0.0)
    result.lp_n_flips = int(record.get("n_flips", 0) or 0)

    stage_norm = str(record.get("stage_used") or "")
    if stage_norm in _STAGE_ORDER:
        idx = _STAGE_ORDER.index(stage_norm)
        result.lp_reached_hard_fix = idx >= 0
        result.lp_reached_repair_20 = idx >= 1
        result.lp_reached_repair_100 = idx >= 2
        result.lp_reached_full_soft = idx >= 3
        result.lp_reached_round_refix = idx >= 4


def _populate_repair_stats(result: PipelineResult, records: Sequence[Dict[str, Any]]) -> None:
    repair_rows = [
        row for row in records
        if str(row.get("stage_used") or "").startswith("repair_")
    ]
    if not repair_rows:
        return

    flips = np.asarray([float(row.get("n_flips", 0) or 0) for row in repair_rows], dtype=float)
    times = np.asarray([float(row.get("solve_time", 0.0) or 0.0) for row in repair_rows], dtype=float)

    result.all_repair_flips = [int(value) for value in flips.tolist()]
    result.all_repair_times = [float(value) for value in times.tolist()]
    result.mean_repair_flips = float(np.mean(flips))
    result.median_repair_flips = float(np.median(flips))
    result.min_repair_flips = float(np.min(flips))
    result.max_repair_flips = float(np.max(flips))
    result.mean_repair_time = float(np.mean(times))
    result.median_repair_time = float(np.median(times))


def _complete_candidate_result(
    result, candidates, records, energies, sampling_times, decoder_times,
    activation_stats, selection_policy, budget_exhausted, started,
):
    """Build the same result for a completed run or an incremental snapshot."""
    result.n_candidates_projected = len(records)
    result.n_samples = len(records)
    result.all_sample_sampling_times = list(sampling_times)
    result.all_sample_decoder_times = list(decoder_times)
    result.all_sample_ebm_energies = list(energies)
    result.all_binary_active_fractions = list(activation_stats)
    hamming_stats = _pairwise_hamming_stats(candidates)
    result.pairwise_hamming_distances = hamming_stats["distances"]
    result.pairwise_hamming_matrix = hamming_stats["matrix"]
    result.sample_mean_pairwise_hamming = hamming_stats["mean"]
    result.sample_max_pairwise_hamming = hamming_stats["max"]
    result.sample_unique_ratio = hamming_stats["unique_ratio"]
    finite_energies = [v for v in energies if math.isfinite(v)]
    if finite_energies:
        result.sample_energy_mean = float(np.mean(finite_energies))
        result.sample_energy_std = float(np.std(finite_energies))
    lp_times = [float(r.get("solve_time", 0.0) or 0.0) for r in records]
    result.time_lp_solve = float(sum(lp_times))
    result.all_sample_lp_solve_times = lp_times
    result.all_sample_total_times = [
        float(sampling_times[i] + decoder_times[i] + lp_times[i]) for i in range(len(records))
    ]
    result.all_objectives_raw = [float(r.get("lp_objective", float("inf"))) for r in records]
    result.all_objectives = list(result.all_objectives_raw)
    result.all_stages = [str(r.get("stage_label") or r.get("stage_used") or "") for r in records]
    result.all_stages_reached = [
        str(r.get("stage_reached") or r.get("stage_label") or "") for r in records
    ]
    result.all_sample_slacks = [float(r.get("slack_mwh", float("nan"))) for r in records]
    result.all_sample_statuses = [str(r.get("status", "")) for r in records]
    if records:
        best_idx = _select_sample_index(selection_policy, energies, result.all_objectives_raw)
        result.best_sample_idx = int(best_idx)
        _apply_selected_record(result, records[best_idx])
        PipelineRunner._assign_selected_sample_times(result, best_idx)
        result.success = math.isfinite(result.lp_objective)
        if str(records[best_idx].get("stage_used") or "").startswith("repair_"):
            result.selected_repair_flips = float(records[best_idx].get("n_flips", 0) or 0)
            result.selected_repair_time = float(records[best_idx].get("solve_time", 0.0) or 0.0)
    else:
        result.success = False
        result.lp_status = "time_budget_exhausted" if budget_exhausted else "no_candidates_evaluated"
        result.error_message = (
            "Global time budget exhausted before any LP candidate completed."
            if budget_exhausted else "No LP candidates were evaluated."
        )
    _populate_repair_stats(result, records)
    backfill_projection_diagnostics([result])
    result.time_total = time.perf_counter() - started
    if budget_exhausted and not result.error_message:
        result.error_message = (
            f"Global time budget exhausted after evaluating {len(records)} candidate(s) "
            f"out of {result.n_candidates_generated} generated."
        )
    return result


def _build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    def _finite_metrics(sub_df: pd.DataFrame) -> pd.DataFrame:
        work = sub_df.replace([np.inf, -np.inf], np.nan)
        return work.dropna(subset=["speedup", "cost_gap_pct"])

    def _row(name: str, sub_df: pd.DataFrame) -> Dict[str, Any]:
        valid = _finite_metrics(sub_df)
        stage_col = "pipeline_stage_reached" if "pipeline_stage_reached" in sub_df.columns else "pipeline_stage"
        hard_fix_rate = (
            ((sub_df[stage_col] == "hard_fix") & sub_df["success"].astype(bool)).mean() * 100.0
            if len(sub_df) else float("nan")
        )
        return {
            "Family": name,
            "N": int(len(sub_df)),
            "Criticality Mean": float(sub_df["criticality_index"].mean()) if len(sub_df) else float("nan"),
            "Pipeline Success %": float(sub_df["success"].mean() * 100.0) if len(sub_df) else float("nan"),
            "Pipeline Speedup Median x": float(valid["speedup"].median()) if len(valid) else float("nan"),
            "Pipeline Speedup Mean x": float(valid["speedup"].mean()) if len(valid) else float("nan"),
            "Pipeline Gap P50 %": float(valid["cost_gap_pct"].median()) if len(valid) else float("nan"),
            "Pipeline Gap P90 %": float(np.percentile(valid["cost_gap_pct"], 90)) if len(valid) else float("nan"),
            "Pipeline Gap P99 %": float(np.percentile(valid["cost_gap_pct"], 99)) if len(valid) else float("nan"),
            "Pipeline Time Mean s": float(sub_df["pipeline_solve_time"].mean()) if len(sub_df) else float("nan"),
            "Hard Fix %": float(hard_fix_rate),
            "MILP Time Mean s": float(sub_df["milp_solve_time"].mean()) if len(sub_df) else float("nan"),
        }

    family_rows = []
    for family in sorted(df["family"].dropna().unique().tolist()):
        family_rows.append(_row(str(family).capitalize(), df[df["family"] == family]))
    family_rows.append(_row("ALL", df))
    return pd.DataFrame(family_rows)


def _summarize_direct_baseline(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return {"overall": {}, "per_family": {}}

    def _summary(sub_df: pd.DataFrame) -> Dict[str, Any]:
        work = sub_df.replace([np.inf, -np.inf], np.nan)
        valid = work.dropna(subset=["gap_pct"])
        return {
            "n": int(len(sub_df)),
            "success_rate_pct": float(sub_df["success"].mean() * 100.0),
            "median_gap_pct": float(valid["gap_pct"].median()) if len(valid) else float("nan"),
            "p90_gap_pct": float(np.percentile(valid["gap_pct"], 90)) if len(valid) else float("nan"),
            "mean_total_time_s": float(sub_df["total_time_s"].mean()),
            "median_total_time_s": float(sub_df["total_time_s"].median()),
        }

    per_family = {
        family: _summary(sub_df)
        for family, sub_df in frame.groupby("family", sort=True)
    }
    return {
        "overall": _summary(frame),
        "per_family": per_family,
    }


def _safe_path_token(text: Any) -> str:
    token = str(text or "unknown").strip()
    for char in ("\\", "/", ":", "*", "?", "\"", "<", ">", "|"):
        token = token.replace(char, "_")
    return token or "unknown"


def _write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _persist_scenario_result(out_path: Path, row: Dict[str, Any]) -> None:
    family = _safe_path_token(row.get("family", "unknown"))
    scenario_id = _safe_path_token(row.get("scenario_id", "scenario"))
    scenario_path = out_path / "scenario_results" / family / f"{scenario_id}.json"
    _write_json(scenario_path, row)


def _persist_family_results(
    out_path: Path,
    family: str,
    rows: Sequence[Dict[str, Any]],
    direct_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    family_token = _safe_path_token(family)
    family_dir = out_path / "family_results"
    payload = list(rows)
    _write_pickle(family_dir / f"pipeline_eval_family_{family_token}.pkl", payload)
    _write_json(family_dir / f"pipeline_eval_family_{family_token}.json", payload)
    _write_json(
        family_dir / f"pipeline_eval_family_{family_token}_summary.json",
        {
            "family": str(family),
            "n_scenarios": int(len(payload)),
        },
    )
    if direct_rows:
        direct_frame = pd.DataFrame(list(direct_rows))
        direct_frame.to_csv(family_dir / f"direct_baseline_family_{family_token}.csv", index=False)
        _write_json(
            family_dir / f"direct_baseline_family_{family_token}_summary.json",
            _summarize_direct_baseline(direct_rows),
        )


def _augment_result_row_with_reference_metrics(row: Dict[str, Any], milp_objective: float) -> Dict[str, Any]:
    out = dict(row)
    out["milp_objective"] = float(milp_objective)
    lp_objective = out.get("lp_objective", float("nan"))
    try:
        lp_objective_f = float(lp_objective)
    except (TypeError, ValueError):
        lp_objective_f = float("nan")
    denom = max(abs(float(milp_objective)), 1.0)
    if math.isfinite(lp_objective_f):
        gap_pct = 100.0 * (lp_objective_f - float(milp_objective)) / denom
        out["milp_gap_pct"] = float(gap_pct)
        out["milp_abs_gap_pct"] = float(abs(gap_pct))
    else:
        out["milp_gap_pct"] = float("nan")
        out["milp_abs_gap_pct"] = float("nan")
    return out


def _load_existing_scenario_rows(out_path: Path) -> List[Dict[str, Any]]:
    scenario_dir = out_path / "scenario_results"
    if not scenario_dir.is_dir():
        return []

    rows: List[Dict[str, Any]] = []
    for path in sorted(scenario_dir.glob("*/*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                row = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            require_current_formulation(row, str(path))
            rows.append(row)
    return rows


def _build_family_dirs(scenario_roots: Sequence[str]) -> Dict[str, Path]:
    family_dirs: Dict[str, Path] = {}
    for root in scenario_roots:
        path = Path(root)
        if not path.is_dir():
            continue
        family_dirs[_family_alias(path.name)] = path
    return family_dirs


def _safe_scaling_summary(
    df: pd.DataFrame,
    feature_col: str,
) -> Dict[str, Any]:
    work = df.copy().replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=[feature_col, "speedup", "cost_gap_pct"])
    work = work[work["speedup"] > 0]
    if len(work) < 2:
        return {}
    return fit_scaling_law_models(work, feature_col=feature_col)


def _analyze_results(
    pipeline_results: List[Dict[str, Any]],
    scenario_roots: Sequence[str],
    output_dir: str,
    best_of_k_values: Sequence[int],
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    backfill_projection_diagnostics(pipeline_results)

    family_dirs = _build_family_dirs(scenario_roots)
    milp_reports_dirs = {
        family: path / "reports"
        for family, path in family_dirs.items()
        if (path / "reports").is_dir()
    }

    eval_output = compute_eval_metrics(
        pipeline_results=pipeline_results,
        milp_reports_dirs=milp_reports_dirs,
        families=sorted(family_dirs.keys()),
    )
    eval_metrics_payload = {
        key: value
        for key, value in eval_output.items()
        if key != "dataframe"
    }
    with (out_path / "eval_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(eval_metrics_payload, handle, indent=2)

    comparison_df = eval_output["dataframe"]
    comparison_df.to_csv(out_path / "full_comparison.csv", index=False)

    summary_table = _build_summary_table(comparison_df)
    summary_table.to_csv(out_path / "summary_table.csv", index=False)

    candidate_pool_df = build_candidate_pool_frame(pipeline_results, comparison_df)
    candidate_pool_df.to_csv(out_path / "candidate_pool_samples.csv", index=False)

    diversity_df = compute_solution_diversity_frame(pipeline_results, comparison_df)
    diversity_df.to_csv(out_path / "solution_diversity.csv", index=False)

    bestofk_summary, bestofk_scenario = compute_best_of_k_curve(
        pipeline_results,
        comparison_df,
        k_values=best_of_k_values or [1],
    )
    if len(bestofk_summary):
        bestofk_summary.to_csv(out_path / "best_of_k_summary.csv", index=False)
        bestofk_scenario.to_csv(out_path / "best_of_k_per_scenario.csv", index=False)

    oracle_summary, oracle_scenario = compute_best_of_k_curve(
        pipeline_results,
        comparison_df,
        k_values=best_of_k_values or [1],
        objective_field="all_objectives_raw",
    )
    if len(oracle_summary):
        oracle_summary.to_csv(out_path / "oracle_best_of_k_summary.csv", index=False)
        oracle_scenario.to_csv(out_path / "oracle_best_of_k_per_scenario.csv", index=False)

    kdiag_summary, kdiag_scenario = compute_k_sampling_diagnostics(
        pipeline_results,
        comparison_df,
        k_values=best_of_k_values or [1],
    )
    if len(kdiag_summary):
        kdiag_summary.to_csv(out_path / "k_sampling_diagnostics_summary.csv", index=False)
        kdiag_scenario.to_csv(out_path / "k_sampling_diagnostics_per_scenario.csv", index=False)

    physical_df = build_physical_complexity_frame(
        family_dirs=family_dirs,
        cache_path=out_path / "physical_complexity_features.csv",
        refresh=False,
    )
    merged_df = merge_physical_complexity_features(comparison_df, physical_df)
    merged_df.to_csv(out_path / "full_comparison_with_physical_features.csv", index=False)

    robustness_df = fit_physical_feature_robustness(merged_df)
    robustness_df.to_csv(out_path / "physical_feature_robustness.csv", index=False)

    scaling_criticality = _safe_scaling_summary(merged_df, feature_col="criticality_index")
    scaling_zones = _safe_scaling_summary(merged_df, feature_col="n_zones")
    hypothesis_summary = {
        "criticality_scaling": scaling_criticality.get("summary", {}) if scaling_criticality else {},
        "zone_scaling": scaling_zones.get("summary", {}) if scaling_zones else {},
    }
    with (out_path / "hypothesis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(hypothesis_summary, handle, indent=2)

    return {
        "eval_output": eval_output,
        "summary_table": summary_table,
        "candidate_pool_df": candidate_pool_df,
        "diversity_df": diversity_df,
        "bestofk_summary": bestofk_summary,
        "bestofk_scenario": bestofk_scenario,
        "kdiag_summary": kdiag_summary,
        "kdiag_scenario": kdiag_scenario,
        "merged_df": merged_df,
        "robustness_df": robustness_df,
        "hypothesis_summary": hypothesis_summary,
    }


def evaluate(
    latent_ebm_path: str,
    data_roots: Sequence[str],
    output_dir: str,
    ae_path: Optional[str] = None,
    jepa_path: Optional[str] = None,
    scenario_roots: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
    max_samples_per_family: Optional[int] = None,
    k: int = 5,
    device: Optional[str] = None,
    threshold: Optional[float] = None,
    threshold_map: Optional[Dict[str, float]] = None,
    num_steps: Optional[int] = None,
    step_size: Optional[float] = None,
    noise_scale: Optional[float] = None,
    drift_weight: Optional[float] = None,
    selection_policy: str = "energy",
    lp_solver: str = "appsi_highs",
    lp_max_stages: int = 3,
    force_repair_20: bool = False,
    lp_time_limit_hard_fix: float = 300.0,
    lp_time_limit_repair_20: float = 300.0,
    lp_time_limit_repair_100: float = 300.0,
    lp_time_limit_full_soft: float = 900.0,
    lp_time_limit_round_refix: float = 900.0,
    global_time_budget_s: Optional[float] = None,
    disable_stage_time_limits: bool = False,
    enforce_hard_lp_timeout: bool = False,
    hard_timeout_grace_s: float = 2.0,
    resume_skip_existing: bool = False,
    skip_direct_baseline: bool = False,
    analyze: bool = True,
    best_of_k_values: Sequence[int] = (1, 2, 3, 5, 10),
    direct_only: bool = False,
    deadline_monotonic: Optional[float] = None,
    result_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    if direct_only:
        if skip_direct_baseline:
            raise ValueError("direct_only requires the JEPA baseline")
        k = 0
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if direct_only:
        # Read calibration metadata without constructing or running the EBM.
        latent_ckpt = torch.load(latent_ebm_path, map_location="cpu", weights_only=False)
        require_current_formulation(latent_ckpt, latent_ebm_path)
        latent_ebm = None
    else:
        latent_ebm, latent_ckpt = load_latent_ebm(latent_ebm_path, dev)
    ae_path = ae_path or _default_from_checkpoint(latent_ckpt, "ae_checkpoint")
    jepa_path = jepa_path or _default_from_checkpoint(latent_ckpt, "jepa_checkpoint")
    if not ae_path or not jepa_path:
        raise ValueError("AE and JEPA checkpoints must be provided explicitly or via the latent EBM checkpoint.")

    ae = load_frozen_ae(ae_path, dev)
    predictor, jepa_cfg = load_frozen_jepa(jepa_path, dev)

    datasets: List[UCLabelDataset] = []
    if max_samples_per_family is not None:
        for root in data_roots:
            datasets.append(UCLabelDataset(roots=[root], max_samples=max_samples_per_family))
    else:
        datasets.append(UCLabelDataset(roots=data_roots, max_samples=max_samples))

    all_families = [sample_family for dataset in datasets for sample_family in {Path(path).parent.name for path in dataset.files}]
    resolved_roots = list(scenario_roots or _infer_scenario_roots_from_families(all_families))
    if not resolved_roots:
        raise ValueError("Scenario roots are required for LP pipeline evaluation.")

    worker_time_limits = {
        "time_limit_hard_fix": lp_time_limit_hard_fix,
        "time_limit_repair_20": lp_time_limit_repair_20,
        "time_limit_repair_100": lp_time_limit_repair_100,
        "time_limit_full_soft": lp_time_limit_full_soft,
        "time_limit_round_refix": lp_time_limit_round_refix,
    }
    if disable_stage_time_limits:
        worker_time_limits = {
            key: 1.0e9
            for key in worker_time_limits
        }

    lp_worker = MILPWarmWorker(
        scenario_roots=resolved_roots,
        solver_name=lp_solver,
        max_stages=lp_max_stages,
        worker_kwargs=worker_time_limits,
        enforce_hard_timeout=enforce_hard_lp_timeout,
        hard_timeout_grace_s=hard_timeout_grace_s,
        verbose=False,
    )

    latent_config = (latent_ckpt.get("config", {}) or {}).get("sampler", {}) or {}
    sampler_cfg = LatentLangevinConfig(
        num_steps=int(num_steps if num_steps is not None else latent_config.get("num_steps", 30)),
        step_size=float(step_size if step_size is not None else latent_config.get("step_size", 0.05)),
        noise_scale=float(noise_scale if noise_scale is not None else latent_config.get("noise_scale", 0.01)),
        temperature=float(latent_config.get("temperature", 1.0)),
        clip_grad=float(latent_config.get("clip_grad", 0.0)),
        normalize_grad=bool(latent_config.get("normalize_grad", True)),
        init_mode=str(latent_config.get("init_mode", "anchor_noise")),
        init_noise_std=float(latent_config.get("init_noise_std", 0.25)),
        drift_weight=float(drift_weight if drift_weight is not None else latent_config.get("drift_weight", 0.10)),
        prior_mix_prob=float(latent_config.get("prior_mix_prob", 0.25)),
        anneal_schedule=str(latent_config.get("anneal_schedule", "cosine")),
        latent_clip=latent_config.get("latent_clip", 6.0),
    )
    sampler = LatentLangevinSampler(sampler_cfg)

    prior_mean = latent_ckpt.get("latent_prior_mean")
    prior_std = latent_ckpt.get("latent_prior_std")
    prior_mean = prior_mean.to(dev) if isinstance(prior_mean, torch.Tensor) else None
    prior_std = prior_std.to(dev) if isinstance(prior_std, torch.Tensor) else None

    ckpt_thresholds = dict(latent_ckpt.get("thresholds", {}) or {})
    jepa_thresholds = dict((jepa_cfg.get("eval", {}) or {}).get("feature_thresholds", {}) or {})
    merged_thresholds = jepa_thresholds | ckpt_thresholds | dict(threshold_map or {})
    default_threshold = float(
        threshold
        if threshold is not None else
        (latent_ckpt.get("config", {}) or {}).get("eval", {}).get(
            "threshold",
            (jepa_cfg.get("eval", {}) or {}).get("threshold", 0.5),
        ),
    )
    thresholds = threshold_vector(default_threshold, merged_thresholds, ae.n_features).to(dev)

    total_samples = sum(len(dataset) for dataset in datasets)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    existing_rows = _load_existing_scenario_rows(out_path) if resume_skip_existing else []
    completed_keys = {
        (_family_alias(str(row.get("family", ""))), str(row.get("scenario_id", "")))
        for row in existing_rows
    }

    print(
        f"Loaded latent pipeline stack | samples={total_samples} "
        f"k={k} selection={selection_policy} device={dev}",
    )
    if resume_skip_existing:
        print(f"Resume mode: found {len(completed_keys)} completed scenario result(s) in {out_path / 'scenario_results'}")
    if global_time_budget_s is not None and float(global_time_budget_s) > 0.0:
        print(
            f"Global per-scenario time budget: {float(global_time_budget_s):.1f}s "
            f"(stage limits {'disabled' if disable_stage_time_limits else 'enabled'})",
        )
    if enforce_hard_lp_timeout:
        print(
            f"LP hard-timeout mode: enabled (subprocess kill, grace={float(hard_timeout_grace_s):.1f}s)",
        )
    print(
        "Thresholds: "
        + ", ".join(
            f"{name}={thr:.2f}"
            for name, thr in threshold_dict(thresholds, ae.n_features).items()
        ),
    )

    results: List[PipelineResult] = []
    result_rows: List[Dict[str, Any]] = list(existing_rows)
    direct_rows: List[Dict[str, Any]] = []
    family_result_rows: Dict[str, List[Dict[str, Any]]] = {}
    family_direct_rows: Dict[str, List[Dict[str, Any]]] = {}

    for row in existing_rows:
        family_alias = _family_alias(str(row.get("family", "")))
        family_result_rows.setdefault(family_alias, []).append(row)

    remaining_total = total_samples - len(completed_keys) if resume_skip_existing else total_samples
    remaining_total = max(0, remaining_total)

    progress = tqdm(total=remaining_total, desc="Latent Pipeline Eval", unit="scenario")
    try:
        for dataset in datasets:
            family_desc = sorted({Path(path).parent.name for path in dataset.files})
            dataset_family_aliases = sorted({_family_alias(name) for name in family_desc})
            if family_desc:
                progress.set_postfix_str(",".join(_family_alias(name) for name in family_desc))
            for idx in range(len(dataset)):
                sample = dataset[idx]
                family_alias = _family_alias(sample.family)
                scenario_key = (family_alias, str(sample.scenario_id))
                if resume_skip_existing and scenario_key in completed_keys:
                    continue
                result = PipelineResult(
                    scenario_id=sample.scenario_id,
                    family=family_alias,
                )
                result.n_zones = int(sample.n_zones)
                result.n_timesteps = int(sample.n_timesteps)
                result.criticality_index = float(sample.criticality)

                u_star = sample.u_star.unsqueeze(0).to(dev)
                h_scenario = sample.h_scenario.unsqueeze(0).to(dev)
                zone_mask = torch.ones(1, sample.n_zones, device=dev)

                t_total_start = time.perf_counter()
                scenario_deadline = (
                    t_total_start + float(global_time_budget_s)
                    if global_time_budget_s is not None and float(global_time_budget_s) > 0.0
                    else None
                )
                if deadline_monotonic is not None:
                    # Map the parent's absolute deadline to this loop's clock.
                    external_deadline = time.perf_counter() + deadline_monotonic - time.monotonic()
                    scenario_deadline = min(scenario_deadline, external_deadline) if scenario_deadline else external_deadline
                budget_exhausted = False

                def _budget_hit() -> bool:
                    nonlocal budget_exhausted
                    if scenario_deadline is None:
                        return False
                    if time.perf_counter() >= scenario_deadline:
                        budget_exhausted = True
                        return True
                    return False

                with torch.no_grad():
                    t0 = time.perf_counter()
                    z_anchor = predictor(h_scenario, zone_mask=zone_mask)["z_pred"]
                    result.time_embedding = time.perf_counter() - t0

                    if not skip_direct_baseline:
                        direct_energy_tensor = (
                            latent_ebm(z_anchor, z_anchor, h_scenario=h_scenario, zone_mask=zone_mask).squeeze(0)
                            if latent_ebm is not None else torch.tensor(float("nan"), device=dev)
                        )
                        t0 = time.perf_counter()
                        direct_logits = ae.decode(
                            z_anchor, n_zones=sample.n_zones,
                            n_timesteps=sample.n_timesteps, h_scenario=h_scenario,
                        )
                        direct_bin = apply_thresholds(torch.sigmoid(direct_logits), thresholds).squeeze(0).cpu()
                        direct_decode_time = time.perf_counter() - t0

                sample_latents: List[torch.Tensor] = []
                sample_energies: List[float] = []
                sample_sampling_times: List[float] = []
                for _ in range(int(k)):
                    if _budget_hit():
                        break
                    t0 = time.perf_counter()
                    z_sample, energy_sample, _ = sampler.sample(
                        ebm=latent_ebm,
                        z_anchor=z_anchor,
                        h_scenario=h_scenario,
                        zone_mask=zone_mask,
                        n_samples=1,
                        prior_mean=prior_mean,
                        prior_std=prior_std,
                        num_steps=num_steps,
                    )
                    sample_sampling_times.append(time.perf_counter() - t0)
                    sample_latents.append(z_sample[:, 0].detach())
                    sample_energies.append(float(energy_sample[0, 0].item()))

                result.time_ebm_sampling = float(sum(sample_sampling_times))
                result.n_candidates_generated = len(sample_latents)

                decoded_candidates: List[torch.Tensor] = []
                decoder_times: List[float] = []
                activation_stats: List[Dict[str, float]] = []
                for z_sample in sample_latents:
                    if _budget_hit():
                        break
                    with torch.no_grad():
                        t0 = time.perf_counter()
                        logits = ae.decode(
                            z_sample,
                            n_zones=sample.n_zones,
                            n_timesteps=sample.n_timesteps,
                            h_scenario=h_scenario,
                        )
                        u_bin = apply_thresholds(torch.sigmoid(logits), thresholds).squeeze(0).cpu()
                        decoder_times.append(time.perf_counter() - t0)
                    decoded_candidates.append(u_bin)
                    activation_stats.append(_binary_stats(u_bin))

                result.time_decoder = float(sum(decoder_times))
                projected_candidates: List[torch.Tensor] = []
                projected_energies: List[float] = []
                projected_sampling_times: List[float] = []
                projected_decoder_times: List[float] = []
                projected_activation_stats: List[Dict[str, float]] = []
                pool_records: List[Dict[str, Any]] = []
                lp_times: List[float] = []

                for cand_idx, u_bin in enumerate(decoded_candidates):
                    if _budget_hit():
                        break
                    remaining_candidate_budget_s = (
                        max(0.0, scenario_deadline - time.perf_counter())
                        if scenario_deadline is not None else None
                    )
                    if remaining_candidate_budget_s is not None and remaining_candidate_budget_s <= 1e-6:
                        budget_exhausted = True
                        break
                    single_records = lp_worker.evaluate_candidates(
                        sample=sample,
                        candidates=u_bin.unsqueeze(0),
                        energies=[sample_energies[cand_idx]],
                        force_repair_20=force_repair_20,
                        candidate_time_budget_s=remaining_candidate_budget_s,
                    )
                    if not single_records:
                        continue
                    row = dict(single_records[0])
                    if result_callback is not None and row.get("status") == "error":
                        raise RuntimeError(f"Candidate repair failed: {row}")
                    row["candidate_idx"] = int(cand_idx)
                    row["family"] = family_alias
                    pool_records.append(row)
                    lp_times.append(float(row.get("solve_time", 0.0) or 0.0))
                    projected_candidates.append(u_bin)
                    projected_energies.append(float(sample_energies[cand_idx]))
                    projected_sampling_times.append(float(sample_sampling_times[cand_idx]))
                    projected_decoder_times.append(float(decoder_times[cand_idx]))
                    projected_activation_stats.append(dict(activation_stats[cand_idx]))
                    if result_callback is not None and not _budget_hit():
                        snapshot = _complete_candidate_result(
                            copy.deepcopy(result), projected_candidates, pool_records,
                            projected_energies, projected_sampling_times, projected_decoder_times,
                            projected_activation_stats, selection_policy, False, t_total_start,
                        )
                        result_callback("pipeline", _augment_result_row_with_reference_metrics(
                            asdict(snapshot), float(sample.objective),
                        ))

                _complete_candidate_result(
                    result, projected_candidates, pool_records, projected_energies,
                    projected_sampling_times, projected_decoder_times, projected_activation_stats,
                    selection_policy, budget_exhausted, t_total_start,
                )
                results.append(result)
                result_row = _augment_result_row_with_reference_metrics(
                    asdict(result),
                    milp_objective=float(sample.objective),
                )
                if result_callback is not None and not direct_only and not _budget_hit():
                    result_callback("pipeline", result_row)
                result_rows.append(result_row)
                family_result_rows.setdefault(family_alias, []).append(result_row)
                _persist_scenario_result(out_path, result_row)

                if not skip_direct_baseline and not _budget_hit():
                    direct_remaining_budget_s = (
                        max(0.0, scenario_deadline - time.perf_counter())
                        if scenario_deadline is not None else None
                    )
                    if direct_remaining_budget_s is not None and direct_remaining_budget_s <= 1e-6:
                        budget_exhausted = True
                        progress.update(1)
                        continue
                    direct_lp_records = lp_worker.evaluate_candidates(
                        sample=sample,
                        candidates=direct_bin.unsqueeze(0),
                        energies=[float(direct_energy_tensor.item())],
                        force_repair_20=force_repair_20,
                        candidate_time_budget_s=direct_remaining_budget_s,
                    )
                    if direct_lp_records:
                        direct_record = dict(direct_lp_records[0])
                        if result_callback is not None and direct_record.get("status") == "error":
                            raise RuntimeError(f"JEPA repair failed: {direct_record}")
                        direct_record["family"] = family_alias
                        direct_record["criticality_index"] = float(sample.criticality)
                        direct_record["milp_objective"] = float(sample.objective)
                        direct_record["direct_energy"] = float(direct_energy_tensor.item())
                        direct_record["decode_time_s"] = float(direct_decode_time)
                        direct_record["embedding_time_s"] = float(result.time_embedding)
                        direct_record["total_time_s"] = float(result.time_embedding + direct_decode_time + float(direct_record["solve_time"]))
                        direct_record["success"] = bool(math.isfinite(float(direct_record.get("lp_objective", float("inf")))))
                        if result_callback is not None and not _budget_hit():
                            result_callback("jepa", direct_record)
                        direct_rows.append(direct_record)
                        family_direct_rows.setdefault(family_alias, []).append(direct_record)
                progress.update(1)
            for family_alias in dataset_family_aliases:
                _persist_family_results(
                    out_path,
                    family_alias,
                    family_result_rows.get(family_alias, []),
                    family_direct_rows.get(family_alias, []),
                )
    finally:
        progress.close()

    results_pkl = out_path / "pipeline_eval_all_families.pkl"
    _write_pickle(results_pkl, result_rows)
    print(f"Saved {len(result_rows)} latent pipeline results to {results_pkl}")

    _write_json(out_path / "pipeline_eval_all_families.json", result_rows)

    for family_alias, rows in sorted(family_result_rows.items()):
        _persist_family_results(
            out_path,
            family_alias,
            rows,
            family_direct_rows.get(family_alias, []),
        )

    if direct_rows:
        pd.DataFrame(direct_rows).to_csv(out_path / "direct_baseline.csv", index=False)
        _write_json(out_path / "direct_baseline_summary.json", _summarize_direct_baseline(direct_rows))

    analysis_payload: Dict[str, Any] = {}
    if analyze:
        analysis_payload = _analyze_results(
            pipeline_results=result_rows,
            scenario_roots=resolved_roots,
            output_dir=output_dir,
            best_of_k_values=best_of_k_values,
        )
        print(f"Wrote notebook-style analysis artifacts to {out_path}")

    return {
        "results": results,
        "results_pkl": str(results_pkl),
        "scenario_roots": resolved_roots,
        "analysis": analysis_payload,
    }


def analyze_existing(
    results_pkl: str,
    output_dir: Optional[str] = None,
    scenario_roots: Optional[Sequence[str]] = None,
    best_of_k_values: Sequence[int] = (1, 2, 3, 5, 10),
) -> Dict[str, Any]:
    path = Path(results_pkl)
    with path.open("rb") as handle:
        pipeline_results = pickle.load(handle)
    for row in pipeline_results:
        require_current_formulation(row, 'saved pipeline result')
    if not isinstance(pipeline_results, list):
        raise TypeError(f"Expected list in {path}, got {type(pipeline_results).__name__}")

    families = [str(row.get("family", "")) for row in pipeline_results if isinstance(row, dict)]
    resolved_roots = list(scenario_roots or _infer_scenario_roots_from_families(f"{family}_criticality_scenarios" if family in {"low", "medium", "high"} else family for family in families))
    if not resolved_roots:
        raise ValueError("Scenario roots are required to analyze cached results.")

    out_dir = output_dir or str(path.parent)
    return _analyze_results(
        pipeline_results=pipeline_results,
        scenario_roots=resolved_roots,
        output_dir=out_dir,
        best_of_k_values=best_of_k_values,
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a notebook-compatible latent EBM + latent SGLD pipeline evaluation.",
    )
    parser.add_argument("--latent-ebm", default=None, help="Latent EBM checkpoint path.")
    parser.add_argument("--results-pkl", default=None, help="Existing pipeline results pickle for analysis-only mode.")
    parser.add_argument("--data-root", action="append", default=None, help="UC label dataset root(s).")
    parser.add_argument("--scenario-root", action="append", default=None, help="Scenario JSON root(s) with reports/ subdirs.")
    parser.add_argument("--ae", default=None, help="Optional AE checkpoint override.")
    parser.add_argument("--jepa", default=None, help="Optional JEPA checkpoint override.")
    parser.add_argument("--output-dir", required=True, help="Directory for pkl/csv/json artifacts.")
    parser.add_argument("--max-samples", type=int, default=None, help="Global dataset cap across all roots.")
    parser.add_argument("--max-samples-per-family", type=int, default=None, help="Cap per dataset root.")
    parser.add_argument("--k", type=int, default=5, help="Number of latent chains per scenario.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--thresholds-json", default=None, help="JSON string or JSON file mapping feature name -> threshold.")
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--drift-weight", type=float, default=None)
    parser.add_argument(
        "--selection-policy",
        default="energy",
        choices=["energy", "oracle_lp", "first", "first_feasible"],
        help="Policy used to define the selected scenario-level pipeline candidate.",
    )
    parser.add_argument("--lp-solver", default="appsi_highs")
    parser.add_argument("--lp-max-stages", type=int, default=3)
    parser.add_argument("--force-repair-20", action="store_true")
    parser.add_argument("--lp-time-limit-hard-fix", type=float, default=300.0)
    parser.add_argument("--lp-time-limit-repair-20", type=float, default=300.0)
    parser.add_argument("--lp-time-limit-repair-100", type=float, default=300.0)
    parser.add_argument("--lp-time-limit-full-soft", type=float, default=900.0)
    parser.add_argument("--lp-time-limit-round-refix", type=float, default=900.0)
    parser.add_argument("--global-time-budget-s", type=float, default=None, help="Optional global wall-clock budget per scenario.")
    parser.add_argument("--disable-stage-time-limits", action="store_true", help="Replace LP stage time limits with a very large value and rely on the global budget loop instead.")
    parser.add_argument("--enforce-hard-lp-timeout", action="store_true", help="Run each LP candidate solve in a subprocess and terminate it when the remaining candidate budget is exceeded.")
    parser.add_argument("--hard-timeout-grace-s", type=float, default=2.0, help="Extra join/cleanup grace in seconds after terminating a hard-timed-out LP subprocess.")
    parser.add_argument("--resume-skip-existing", action="store_true", help="Resume from an existing output-dir by skipping scenarios already present in scenario_results/.")
    parser.add_argument("--skip-direct-baseline", action="store_true", help="Skip the extra direct JEPA+LP baseline solve.")
    parser.add_argument("--skip-eval", action="store_true", help="Only analyze an existing results pickle.")
    parser.add_argument("--no-analysis", action="store_true", help="Skip notebook-style CSV/JSON post-processing.")
    parser.add_argument(
        "--best-of-k-values",
        default="1,2,3,5,10",
        help="Comma-separated prefix K values for best-of-K and diagnostics tables.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    best_of_k_values = _parse_best_of_k_values(args.best_of_k_values)
    threshold_map = load_threshold_map(args.thresholds_json)

    if args.skip_eval:
        if not args.results_pkl:
            raise ValueError("--results-pkl is required with --skip-eval")
        analyze_existing(
            results_pkl=args.results_pkl,
            output_dir=args.output_dir,
            scenario_roots=args.scenario_root,
            best_of_k_values=best_of_k_values,
        )
        return

    if not args.latent_ebm:
        raise ValueError("--latent-ebm is required unless --skip-eval is used")
    if not args.data_root:
        raise ValueError("--data-root is required unless --skip-eval is used")

    evaluate(
        latent_ebm_path=args.latent_ebm,
        data_roots=args.data_root,
        output_dir=args.output_dir,
        ae_path=args.ae,
        jepa_path=args.jepa,
        scenario_roots=args.scenario_root,
        max_samples=args.max_samples,
        max_samples_per_family=args.max_samples_per_family,
        k=args.k,
        device=args.device,
        threshold=args.threshold,
        threshold_map=threshold_map,
        num_steps=args.num_steps,
        step_size=args.step_size,
        noise_scale=args.noise_scale,
        drift_weight=args.drift_weight,
        selection_policy=args.selection_policy,
        lp_solver=args.lp_solver,
        lp_max_stages=args.lp_max_stages,
        force_repair_20=args.force_repair_20,
        lp_time_limit_hard_fix=args.lp_time_limit_hard_fix,
        lp_time_limit_repair_20=args.lp_time_limit_repair_20,
        lp_time_limit_repair_100=args.lp_time_limit_repair_100,
        lp_time_limit_full_soft=args.lp_time_limit_full_soft,
        lp_time_limit_round_refix=args.lp_time_limit_round_refix,
        global_time_budget_s=args.global_time_budget_s,
        disable_stage_time_limits=args.disable_stage_time_limits,
        enforce_hard_lp_timeout=args.enforce_hard_lp_timeout,
        hard_timeout_grace_s=args.hard_timeout_grace_s,
        resume_skip_existing=args.resume_skip_existing,
        skip_direct_baseline=args.skip_direct_baseline,
        analyze=not args.no_analysis,
        best_of_k_values=best_of_k_values,
    )


if __name__ == "__main__":
    main()
