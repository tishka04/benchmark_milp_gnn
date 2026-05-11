"""
Comparative smoke test for silver EBM checkpoints on a fixed small scenario set.

This script evaluates two checkpoints on the same scenarios and candidate-pool
protocol, then compares:
  - Best-of-K gap (LP-best vs energy-best within the sampled pool)
  - Slack of the LP-best candidate
  - Hamming diversity of the sampled binary pool
  - Stage reached / used by the LP-best candidate
  - Spearman correlation between energy and LP score

Usage:
    python scripts/compare_silver_checkpoints_smoketest.py
    python scripts/compare_silver_checkpoints_smoketest.py --n-scenarios 15 --k 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ebm.config_v3 import EBMv3Config
from src.ebm.dataset_v3 import ScenarioReportDataset, temporal_collate_fn
from src.ebm.model_v3 import TrajectoryZonalEBM
from src.ebm.sampler_v3 import NormalizedTemporalLangevinSampler
from src.ebm.train_v3 import (
    _relative_score_gap,
    _sample_lp_candidates_v2,
    _spearman_correlation,
)
from src.eval.pipeline_runner import _pairwise_hamming_stats, _stage_rank
from src.milp.lp_worker_two_stage import LPWorkerTwoStage


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_output_dir = repo_root / "outputs" / "ebm_models" / "ebm_v3" / "smoke_tests"
    parser = argparse.ArgumentParser(
        description="Compare silver checkpoints on the same fixed scenario subset.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=repo_root,
        help=f"Repo root / benchmark base dir. Default: {repo_root}",
    )
    parser.add_argument(
        "--old-checkpoint",
        type=Path,
        default=repo_root / "outputs" / "ebm_models" / "ebm_v3" / "silver_best.pt",
        help="Reference checkpoint path.",
    )
    parser.add_argument(
        "--new-checkpoint",
        type=Path,
        default=repo_root / "outputs" / "ebm_models" / "ebm_v3" / "silver_v2_best.pt",
        help="New checkpoint path.",
    )
    parser.add_argument(
        "--tier",
        choices=["gold", "silver"],
        default="silver",
        help="Dataset tier to draw scenarios from. Default: silver",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "all"],
        default="val",
        help="Dataset split used for the smoke test. Default: val",
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=15,
        help="Number of scenarios to evaluate. Default: 15",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Candidates sampled per scenario. Default: 8",
    )
    parser.add_argument(
        "--max-stages",
        type=int,
        default=4,
        help="LP max_stages for candidate evaluation. Default: 4",
    )
    parser.add_argument(
        "--sampler-steps",
        type=int,
        default=0,
        help="Sampler steps used at inference. Default: 0 -> use silver_langevin_end",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed for subset selection and sampling. Default: 42",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device override: auto, cpu, or cuda. Default: auto",
    )
    parser.add_argument(
        "--slack-weight",
        type=float,
        default=None,
        help="Override silver_pref_slack_weight for LP score J.",
    )
    parser.add_argument(
        "--repair-weight",
        type=float,
        default=None,
        help="Override silver_pref_repair_weight for LP score J.",
    )
    parser.add_argument(
        "--repair-metric",
        choices=["decoder_deviation", "rounded_flips"],
        default=None,
        help="Override silver_pref_repair_metric for LP score J.",
    )
    parser.add_argument(
        "--lp-time-scale",
        type=float,
        default=1.0,
        help="Multiply LP-worker stage time limits by this factor. Default: 1.0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Where to save CSV/JSON outputs. Default: {default_output_dir}",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix tag for output filenames.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> str:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Requested --device cuda, but torch.cuda.is_available() is False. "
                "Your current environment appears to have a CPU-only PyTorch build."
            )
        return "cuda"
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_eval_config(args: argparse.Namespace) -> EBMv3Config:
    config = EBMv3Config(base_dir=str(args.base_dir))
    config.device = _resolve_device(args.device)
    config.use_amp = False
    config.num_workers = 0
    config.batch_size = 1
    config.silver_lp_candidates_per_scenario = int(args.k)
    config.silver_val_candidates_per_scenario = int(args.k)
    config.silver_lp_max_stages = int(args.max_stages)
    config.silver_val_max_stages = int(args.max_stages)
    config.silver_lp_scenarios_per_batch = 1
    config.silver_val_ranking_scenarios = 0
    if args.slack_weight is not None:
        config.silver_pref_slack_weight = float(args.slack_weight)
    if args.repair_weight is not None:
        config.silver_pref_repair_weight = float(args.repair_weight)
    if args.repair_metric is not None:
        config.silver_pref_repair_metric = str(args.repair_metric)
    return config


def _load_dataset(config: EBMv3Config, tier: str) -> ScenarioReportDataset:
    from src.ebm.dataset_v3 import load_classification_index

    index = load_classification_index(config.classification_index_path)
    scenario_files = index.get(tier, [])
    return ScenarioReportDataset(
        reports_dir=config.reports_dir,
        embeddings_dir=config.embeddings_dir,
        scenario_files=scenario_files,
        n_timesteps=config.n_timesteps,
        embed_dim=config.embed_dim,
    )


def _select_indices(
    dataset: ScenarioReportDataset,
    split: str,
    val_split: float,
    seed: int,
    n_scenarios: int,
) -> List[int]:
    all_indices = list(range(len(dataset)))
    if split == "all":
        return all_indices[:n_scenarios]

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    subset_indices = list(train_subset.indices if split == "train" else val_subset.indices)
    return subset_indices[:n_scenarios]


def _load_checkpoint_model(checkpoint_path: Path, config: EBMv3Config) -> TrajectoryZonalEBM:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model = TrajectoryZonalEBM(
        embed_dim=config.embed_dim,
        n_features=config.n_features,
        hidden_dim=config.hidden_dim,
        gru_layers=config.gru_layers,
        bidirectional=config.bidirectional,
        dropout=config.dropout,
        use_peak_term=config.use_peak_term,
        peak_tau=config.peak_tau,
        peak_weight=config.peak_weight,
        energy_max=config.energy_max,
    )
    model.load_state_dict(state_dict)
    model = model.to(config.device)
    model.eval()
    return model


def _make_sampler(model: TrajectoryZonalEBM, config: EBMv3Config, steps: int) -> NormalizedTemporalLangevinSampler:
    return NormalizedTemporalLangevinSampler(
        model=model,
        n_features=config.n_features,
        num_steps=steps,
        step_size=config.langevin_step_size,
        noise_scale=config.langevin_noise,
        temp_max=config.langevin_temp_max,
        temp_min=config.langevin_temp_min,
        init_mode=config.langevin_init_mode,
        prior_p=config.langevin_prior_p,
        prior_strength=config.langevin_prior_strength,
        normalize_grad=config.langevin_normalize_grad,
        device=config.device,
        mode="infer",
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[int, float, int]:
    score = float(candidate.get("score", float("inf")))
    slack = float(candidate.get("slack", float("inf")))
    stage_name = str(candidate.get("stage_used", ""))
    if not math.isfinite(score):
        score = float("inf")
    if not math.isfinite(slack):
        slack = float("inf")
    return (
        0 if math.isfinite(score) else 1,
        score,
        -_stage_rank(stage_name),
    )


def _evaluate_one_checkpoint(
    label: str,
    checkpoint_path: Path,
    dataset: ScenarioReportDataset,
    selected_indices: Sequence[int],
    config: EBMv3Config,
    sampler_steps: int,
    base_seed: int,
    lp_time_scale: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    model = _load_checkpoint_model(checkpoint_path, config)
    sampler = _make_sampler(model, config, sampler_steps)
    sampler.set_mode("infer")
    lp_worker = LPWorkerTwoStage(
        scenarios_dir=config.scenarios_dir,
        solver_name=config.lp_solver,
        slack_tol_mwh=config.lp_slack_tol,
        deviation_penalty=config.lp_deviation_penalty,
        time_limit_hard_fix=20.0 * lp_time_scale,
        time_limit_repair_20=15.0 * lp_time_scale,
        time_limit_repair_100=120.0 * lp_time_scale,
        time_limit_full_soft=900.0 * lp_time_scale,
        time_limit_round_refix=900.0 * lp_time_scale,
        verbose=False,
    )

    rows: List[Dict[str, Any]] = []
    best_stage_counts: Counter[str] = Counter()

    for scenario_offset, dataset_idx in enumerate(selected_indices):
        item = dataset[dataset_idx]
        batch = temporal_collate_fn([item])
        scenario_id = batch["scenario_ids"][0]
        _seed_everything(base_seed + scenario_offset)

        u_pos = batch["u_zt"].to(config.device)
        h_zt = batch["h_zt"].to(config.device)
        zone_mask = batch["zone_mask"].to(config.device)
        n_zones = int(batch["n_zones"][0].item())

        candidates = _sample_lp_candidates_v2(
            model=model,
            sampler=sampler,
            h_i=h_zt,
            mask_i=zone_mask,
            u_pos_i=u_pos,
            scenario_id=scenario_id,
            config=config,
            lp_worker=lp_worker,
            n_candidates=config.silver_lp_candidates_per_scenario,
            max_stages=config.silver_lp_max_stages,
        )

        hamming_stats = _pairwise_hamming_stats([cand["u_bin"][0, :n_zones] for cand in candidates])
        finite_candidates = [
            cand for cand in candidates
            if math.isfinite(float(cand.get("score", float("inf"))))
            and math.isfinite(float(cand.get("energy", float("inf"))))
        ]
        ordered = sorted(finite_candidates, key=lambda cand: float(cand["score"]))
        best_candidate = ordered[0] if ordered else min(candidates, key=_candidate_sort_key)
        energy_best = min(ordered, key=lambda cand: float(cand["energy"])) if ordered else None
        spearman = (
            _spearman_correlation(
                [float(cand["energy"]) for cand in ordered],
                [float(cand["score"]) for cand in ordered],
            )
            if len(ordered) >= 2 else float("nan")
        )
        best_of_k_gap = (
            _relative_score_gap(float(ordered[0]["score"]), float(energy_best["score"]))
            if energy_best is not None and ordered else float("nan")
        )

        best_stage = str(best_candidate.get("stage_used", ""))
        best_stage_counts[best_stage] += 1

        row = {
            "checkpoint_label": label,
            "checkpoint_path": str(checkpoint_path),
            "scenario_id": scenario_id,
            "dataset_index": int(dataset_idx),
            "mip_objective": float(item.get("objective", float("nan"))),
            "n_candidates": len(candidates),
            "n_finite_candidates": len(ordered),
            "pct_non_finite_candidates": 1.0 - (len(ordered) / max(len(candidates), 1)),
            "best_of_k_gap": float(best_of_k_gap),
            "best_lp_score": float(best_candidate.get("score", float("nan"))),
            "best_lp_cost": float(best_candidate.get("cost", float("nan"))),
            "best_slack": float(best_candidate.get("slack", float("nan"))),
            "best_stage_reached": best_stage,
            "best_energy": float(best_candidate.get("energy", float("nan"))),
            "best_decoder_deviation": float(best_candidate.get("decoder_deviation", float("nan"))),
            "best_rounded_flips": float(best_candidate.get("rounded_flips", float("nan"))),
            "hamming_mean": float(hamming_stats["mean"]),
            "hamming_max": float(hamming_stats["max"]),
            "unique_ratio": float(hamming_stats["unique_ratio"]),
            "spearman_energy_lp_score": float(spearman),
        }
        if energy_best is not None:
            row["energy_best_score"] = float(energy_best["score"])
            row["energy_best_slack"] = float(energy_best["slack"])
            row["energy_best_stage_reached"] = str(energy_best.get("stage_used", ""))
        rows.append(row)

    details_df = pd.DataFrame(rows)
    summary = {
        "checkpoint_label": label,
        "checkpoint_path": str(checkpoint_path),
        "n_scenarios": int(len(details_df)),
        "scenario_ids": details_df["scenario_id"].tolist(),
        "mean_best_of_k_gap": float(details_df["best_of_k_gap"].dropna().mean()) if "best_of_k_gap" in details_df else float("nan"),
        "mean_best_slack": float(details_df["best_slack"].dropna().mean()) if "best_slack" in details_df else float("nan"),
        "mean_hamming_diversity": float(details_df["hamming_mean"].dropna().mean()) if "hamming_mean" in details_df else float("nan"),
        "mean_spearman_energy_lp_score": float(details_df["spearman_energy_lp_score"].dropna().mean()) if "spearman_energy_lp_score" in details_df else float("nan"),
        "mean_pct_non_finite_candidates": float(details_df["pct_non_finite_candidates"].dropna().mean()) if "pct_non_finite_candidates" in details_df else float("nan"),
        "n_rankable_scenarios": int(details_df["spearman_energy_lp_score"].notna().sum()) if "spearman_energy_lp_score" in details_df else 0,
        "best_stage_distribution": dict(best_stage_counts),
    }
    return details_df, summary


def _make_output_stem(args: argparse.Namespace) -> str:
    tag = f"_{args.tag}" if args.tag else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"silver_checkpoint_smoketest_{args.tier}_{args.split}_{args.n_scenarios}s_{args.k}k_{timestamp}{tag}"


def _print_summary_table(summary_df: pd.DataFrame) -> None:
    display_cols = [
        "checkpoint_label",
        "mean_best_of_k_gap",
        "mean_best_slack",
        "mean_hamming_diversity",
        "mean_spearman_energy_lp_score",
        "mean_pct_non_finite_candidates",
        "n_rankable_scenarios",
    ]
    print("\nSummary")
    print(summary_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nBest Stage Distribution")
    for row in summary_df.itertuples():
        print(f"  {row.checkpoint_label}: {row.best_stage_distribution}")


def main() -> int:
    args = parse_args()
    config = _build_eval_config(args)
    sampler_steps = int(args.sampler_steps) if args.sampler_steps > 0 else int(config.silver_langevin_end)
    dataset = _load_dataset(config, args.tier)
    selected_indices = _select_indices(dataset, args.split, config.val_split, args.seed, args.n_scenarios)
    if not selected_indices:
        raise ValueError("No scenarios selected for the smoke test.")

    scenario_ids = [dataset.valid_scenarios[idx]["scenario_id"] for idx in selected_indices]
    print(f"Selected {len(selected_indices)} scenarios from {args.tier}/{args.split}:")
    print("  " + ", ".join(scenario_ids))
    print(
        "Eval protocol: "
        f"K={args.k}, max_stages={args.max_stages}, sampler_steps={sampler_steps}, "
        f"lp_time_scale={args.lp_time_scale:.3g}, "
        f"J = cost + {config.silver_pref_slack_weight:.3g} * slack + "
        f"{config.silver_pref_repair_weight:.3g} * {config.silver_pref_repair_metric}"
    )

    checkpoints = [
        ("silver_best", args.old_checkpoint),
        ("silver_v2_best", args.new_checkpoint),
    ]

    all_details: List[pd.DataFrame] = []
    summaries: List[Dict[str, Any]] = []
    for label, checkpoint_path in checkpoints:
        print(f"\nEvaluating {label}: {checkpoint_path}")
        details_df, summary = _evaluate_one_checkpoint(
            label=label,
            checkpoint_path=checkpoint_path,
            dataset=dataset,
            selected_indices=selected_indices,
            config=deepcopy(config),
            sampler_steps=sampler_steps,
            base_seed=args.seed,
            lp_time_scale=float(args.lp_time_scale),
        )
        all_details.append(details_df)
        summaries.append(summary)

    details_df = pd.concat(all_details, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    _print_summary_table(summary_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = _make_output_stem(args)
    details_path = args.output_dir / f"{stem}_details.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"
    selection_path = args.output_dir / f"{stem}_selection.json"

    details_df.to_csv(details_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    with selection_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario_ids": scenario_ids,
                "selected_indices": list(map(int, selected_indices)),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            },
            f,
            indent=2,
        )

    print(f"\nSaved details to {details_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved scenario selection to {selection_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
