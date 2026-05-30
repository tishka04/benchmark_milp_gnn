"""Evaluate direct JEPA decoding vs latent EBM + latent Langevin sampling."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from src.uc_jepa.data.uc_label_dataset import UCLabelDataset
from src.uc_jepa.latent_utils import (
    accumulate_confusion,
    apply_thresholds,
    f1_from_counts,
    load_frozen_ae,
    load_frozen_jepa,
    load_latent_ebm,
    load_threshold_map,
    pairwise_hamming_mean,
    threshold_dict,
    threshold_vector,
)
from src.uc_jepa.sampling.latent_langevin import LatentLangevinConfig, LatentLangevinSampler
from src.uc_jepa.workers.milp_warm_worker import MILPWarmWorker


def _default_from_checkpoint(latent_ckpt: Dict[str, Any], key: str, fallback: Optional[str] = None) -> Optional[str]:
    value = latent_ckpt.get(key)
    if value:
        return str(value)
    config_value = (latent_ckpt.get("config", {}) or {}).get(key)
    if config_value:
        return str(config_value)
    return fallback


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


def evaluate(
    latent_ebm_path: str,
    data_roots: Sequence[str],
    ae_path: Optional[str] = None,
    jepa_path: Optional[str] = None,
    max_samples: int = 100,
    k: int = 5,
    device: Optional[str] = None,
    threshold: Optional[float] = None,
    threshold_map: Optional[Dict[str, float]] = None,
    num_steps: Optional[int] = None,
    step_size: Optional[float] = None,
    noise_scale: Optional[float] = None,
    drift_weight: Optional[float] = None,
    lp: bool = False,
    scenario_roots: Optional[Sequence[str]] = None,
    lp_solver: str = "appsi_highs",
    lp_max_stages: int = 2,
    force_repair_20: bool = False,
    lp_time_limit_hard_fix: float = 3.0,
    lp_time_limit_repair_20: float = 5.0,
    lp_time_limit_repair_100: float = 5.0,
    lp_time_limit_full_soft: float = 5.0,
    lp_time_limit_round_refix: float = 5.0,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    latent_ebm, latent_ckpt = load_latent_ebm(latent_ebm_path, dev)
    ae_path = ae_path or _default_from_checkpoint(latent_ckpt, "ae_checkpoint")
    jepa_path = jepa_path or _default_from_checkpoint(latent_ckpt, "jepa_checkpoint")
    if not ae_path or not jepa_path:
        raise ValueError("AE and JEPA checkpoints must be provided either explicitly or via the latent EBM checkpoint.")

    ae = load_frozen_ae(ae_path, dev)
    predictor, jepa_cfg = load_frozen_jepa(jepa_path, dev)
    dataset = UCLabelDataset(roots=data_roots, max_samples=max_samples)
    print(
        f"Loaded latent EBM + frozen AE/JEPA | "
        f"dataset={len(dataset)} samples | k={k} device={dev}",
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
    print(
        "Thresholds: "
        + ", ".join(
            f"{name}={thr:.2f}"
            for name, thr in threshold_dict(thresholds, ae.n_features).items()
        ),
    )

    direct_tp = np.zeros(ae.n_features)
    direct_fp = np.zeros(ae.n_features)
    direct_fn = np.zeros(ae.n_features)
    latent_tp = np.zeros(ae.n_features)
    latent_fp = np.zeros(ae.n_features)
    latent_fn = np.zeros(ae.n_features)

    lp_worker = None
    if lp:
        roots = list(scenario_roots or [])
        if not roots:
            roots = _infer_scenario_roots(dataset)
        lp_worker = MILPWarmWorker(
            scenario_roots=roots,
            solver_name=lp_solver,
            max_stages=lp_max_stages,
            worker_kwargs={
                "time_limit_hard_fix": lp_time_limit_hard_fix,
                "time_limit_repair_20": lp_time_limit_repair_20,
                "time_limit_repair_100": lp_time_limit_repair_100,
                "time_limit_full_soft": lp_time_limit_full_soft,
                "time_limit_round_refix": lp_time_limit_round_refix,
            },
            verbose=False,
        )

    direct_lp_rows: List[Dict[str, Any]] = []
    latent_best_lp_rows: List[Dict[str, Any]] = []
    latent_pool_rows: List[Dict[str, Any]] = []
    per_sample_rows: List[Dict[str, Any]] = []
    diversity_values: List[float] = []
    energy_std_values: List[float] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        u_star = sample.u_star.unsqueeze(0).to(dev)
        h_scenario = sample.h_scenario.unsqueeze(0).to(dev)
        zone_mask = torch.ones(1, sample.n_zones, device=dev)

        with torch.no_grad():
            z_anchor = predictor(h_scenario, zone_mask=zone_mask)["z_pred"]
            direct_energy = latent_ebm(
                z_anchor,
                z_anchor,
                h_scenario=h_scenario,
                zone_mask=zone_mask,
            ).squeeze(0)
            direct_logits = ae.decode(
                z_anchor,
                n_zones=sample.n_zones,
                n_timesteps=sample.n_timesteps,
                h_scenario=h_scenario,
            )
            direct_bin = apply_thresholds(torch.sigmoid(direct_logits), thresholds)

        z_samples, sample_energies, _ = sampler.sample(
            ebm=latent_ebm,
            z_anchor=z_anchor,
            h_scenario=h_scenario,
            zone_mask=zone_mask,
            n_samples=k,
            prior_mean=prior_mean,
            prior_std=prior_std,
            num_steps=num_steps,
        )
        with torch.no_grad():
            flat_z = z_samples.reshape(z_samples.shape[0] * z_samples.shape[1], z_samples.shape[2], z_samples.shape[3])
            h_expanded = h_scenario.unsqueeze(1).expand(-1, k, -1, -1, -1).reshape(
                flat_z.shape[0],
                h_scenario.shape[1],
                h_scenario.shape[2],
                h_scenario.shape[3],
            )
            sample_logits = ae.decode(
                flat_z,
                n_zones=sample.n_zones,
                n_timesteps=sample.n_timesteps,
                h_scenario=h_expanded,
            ).reshape(1, k, sample.n_zones, sample.n_timesteps, ae.n_features)
            sample_bin = apply_thresholds(torch.sigmoid(sample_logits), thresholds)
            best_bin = sample_bin[:, sample_energies.argmin(dim=1)[0]].squeeze(0)
            best_energy = float(sample_energies.min().item())

        accumulate_confusion(direct_tp, direct_fp, direct_fn, direct_bin, u_star, zone_mask)
        accumulate_confusion(latent_tp, latent_fp, latent_fn, best_bin.unsqueeze(0), u_star, zone_mask)

        diversity = pairwise_hamming_mean(sample_bin.squeeze(0).cpu())
        diversity_values.append(diversity)
        energy_std_values.append(float(sample_energies.std(unbiased=False).item()))

        sample_row: Dict[str, Any] = {
            "scenario_id": sample.scenario_id,
            "family": sample.family,
            "criticality": float(sample.criticality),
            "milp_objective": float(sample.objective),
            "direct_energy": float(direct_energy.item()),
            "latent_best_energy": best_energy,
            "candidate_diversity": diversity,
            "candidate_energy_std": float(sample_energies.std(unbiased=False).item()),
        }

        if lp_worker is not None:
            direct_records = lp_worker.evaluate_candidates(
                sample=sample,
                candidates=direct_bin.squeeze(0).unsqueeze(0).cpu(),
                energies=[float(direct_energy.item())],
                force_repair_20=force_repair_20,
            )
            pool_records = lp_worker.evaluate_candidates(
                sample=sample,
                candidates=sample_bin.squeeze(0).cpu(),
                energies=sample_energies.squeeze(0).detach().cpu().tolist(),
                force_repair_20=force_repair_20,
            )
            best_record = lp_worker.select_best(pool_records)
            if direct_records:
                direct_lp_rows.extend(direct_records)
                sample_row["direct_lp"] = direct_records[0]
            latent_pool_rows.extend(pool_records)
            if best_record is not None:
                latent_best_lp_rows.append(best_record)
                sample_row["latent_best_lp"] = best_record

        per_sample_rows.append(sample_row)

    summary: Dict[str, Any] = {
        "n_samples": len(dataset),
        "k": int(k),
        "thresholds": threshold_dict(thresholds, ae.n_features),
        "direct_decode": f1_from_counts(direct_tp, direct_fp, direct_fn),
        "latent_best_energy_decode": f1_from_counts(latent_tp, latent_fp, latent_fn),
        "mean_candidate_diversity": float(np.mean(diversity_values)) if diversity_values else float("nan"),
        "mean_candidate_energy_std": float(np.mean(energy_std_values)) if energy_std_values else float("nan"),
    }

    if lp_worker is not None:
        direct_summary = lp_worker.summarize(direct_lp_rows)
        latent_summary = lp_worker.summarize(latent_best_lp_rows)
        pool_summary = lp_worker.summarize(latent_pool_rows)
        win_rate = float(
            np.mean(
                [
                    (
                        float(sample_row["latent_best_lp"]["lp_objective"]) <
                        float(sample_row["direct_lp"]["lp_objective"])
                    )
                    for sample_row in per_sample_rows
                    if "direct_lp" in sample_row and "latent_best_lp" in sample_row
                ],
            ),
        ) if per_sample_rows else float("nan")
        improvement = [
            float(sample_row["direct_lp"]["gap_pct"]) - float(sample_row["latent_best_lp"]["gap_pct"])
            for sample_row in per_sample_rows
            if "direct_lp" in sample_row and "latent_best_lp" in sample_row
            and np.isfinite(float(sample_row["direct_lp"]["gap_pct"]))
            and np.isfinite(float(sample_row["latent_best_lp"]["gap_pct"]))
        ]
        summary["lp"] = {
            "direct": direct_summary,
            "latent_best_of_k": latent_summary,
            "latent_candidate_pool": pool_summary,
            "latent_beats_direct_rate": win_rate,
            "mean_gap_improvement_pct_points": float(np.mean(improvement)) if improvement else float("nan"),
        }

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with (out_path / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        with (out_path / "per_sample.json").open("w", encoding="utf-8") as handle:
            json.dump(per_sample_rows, handle, indent=2)
        if latent_pool_rows:
            keys = sorted({key for row in latent_pool_rows for key in row.keys()})
            with (out_path / "candidate_pool.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                for row in latent_pool_rows:
                    writer.writerow(row)
        print(f"Wrote evaluation artifacts to {out_path}")

    print(
        f"Decode macro F1 | direct={summary['direct_decode']['macro_f1']:.4f} "
        f"latent_best_energy={summary['latent_best_energy_decode']['macro_f1']:.4f}",
    )
    if "lp" in summary:
        print(
            f"LP mean gap pct | direct={summary['lp']['direct']['mean_gap_pct']:.4f} "
            f"latent_best_of_k={summary['lp']['latent_best_of_k']['mean_gap_pct']:.4f}",
        )
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the latent UC-JEPA EBM.")
    parser.add_argument("--latent-ebm", required=True, help="Latent EBM checkpoint path.")
    parser.add_argument("--ae", default=None, help="Optional AE checkpoint override.")
    parser.add_argument("--jepa", default=None, help="Optional JEPA checkpoint override.")
    parser.add_argument("--data-root", action="append", required=True, help="Dataset root(s) with *_uc.npz files.")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--k", type=int, default=5, help="Number of latent chains per scenario.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--thresholds-json", default=None, help="JSON string or JSON file mapping feature name -> threshold.")
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--drift-weight", type=float, default=None)
    parser.add_argument("--lp", action="store_true", help="Run LPWorkerTwoStage on decoded candidates.")
    parser.add_argument("--scenario-root", action="append", default=None, help="Scenario JSON root(s) for LP evaluation.")
    parser.add_argument("--lp-solver", default="appsi_highs")
    parser.add_argument("--lp-max-stages", type=int, default=2)
    parser.add_argument("--force-repair-20", action="store_true")
    parser.add_argument("--lp-time-limit-hard-fix", type=float, default=3.0)
    parser.add_argument("--lp-time-limit-repair-20", type=float, default=5.0)
    parser.add_argument("--lp-time-limit-repair-100", type=float, default=5.0)
    parser.add_argument("--lp-time-limit-full-soft", type=float, default=5.0)
    parser.add_argument("--lp-time-limit-round-refix", type=float, default=5.0)
    parser.add_argument("--output-dir", default=None, help="Optional directory for summary/per-sample outputs.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    threshold_map = load_threshold_map(args.thresholds_json)
    evaluate(
        latent_ebm_path=args.latent_ebm,
        ae_path=args.ae,
        jepa_path=args.jepa,
        data_roots=args.data_root,
        max_samples=args.max_samples,
        k=args.k,
        device=args.device,
        threshold=args.threshold,
        threshold_map=threshold_map,
        num_steps=args.num_steps,
        step_size=args.step_size,
        noise_scale=args.noise_scale,
        drift_weight=args.drift_weight,
        lp=args.lp,
        scenario_roots=args.scenario_root,
        lp_solver=args.lp_solver,
        lp_max_stages=args.lp_max_stages,
        force_repair_20=args.force_repair_20,
        lp_time_limit_hard_fix=args.lp_time_limit_hard_fix,
        lp_time_limit_repair_20=args.lp_time_limit_repair_20,
        lp_time_limit_repair_100=args.lp_time_limit_repair_100,
        lp_time_limit_full_soft=args.lp_time_limit_full_soft,
        lp_time_limit_round_refix=args.lp_time_limit_round_refix,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
