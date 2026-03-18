"""
Generate an 'extreme_criticality_scenarios' family (100 scenarios).

Design rationale (from evaluation_pipeline_fine_tuned.ipynb criticality decomposition):
  The MaxTimeLimit scenarios in the high-criticality family are driven by:

  PRIMARY DRIVERS (problem size + demand):
    - B1_n_zones, B3_binary_est  → large topology + many assets
    - A5_demand_scale, A6_peak_to_firm  → high load, low margin

  SECONDARY DRIVERS (coupling + network structure):
    - B8_soc_tightness   = battery_e_to_p_hours / battery_final_soc_tolerance
                           → high E/P ratio + very tight SOC tolerance
    - B10_network_hetero  = std(zones_per_region) / mean(zones_per_region)
                           → wide spread in zones_per_region across regions
    - A11_trade_reliance  = trade_factor(policy) * min(neighbor_nations/4, 1)
                           → "allow" policy + many neighbor nations
    - A12_inv_congestion  = 1 - intertie_density
                           → LOW intertie density → congestion risk

  This script pushes ALL 8 drivers beyond the current 'high' family to
  generate scenarios that should almost certainly hit MaxTimeLimit.

Usage:
    python -m scripts.generate_extreme_scenarios [--pool-size 800] [--count 100]
"""
from __future__ import annotations

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator.generator_v1 import (
    set_seed, load_space, sample_graph, sample_assets, sample_econ, sample_tech,
    sample_exogenous, sample_operation_costs, sample_unit_capacities,
    sample_transport_capacities, sample_mip_gap,
    ScenarioConfig, GraphSpec, AssetSpec, EconPolicy, TechScalers,
    OperationCosts, UnitCapacities, TransportCapacities, ExogenousSpec,
    estimate_milp_size, estimate_solve_time_hours, passes_budget_guard,
    compute_flexibility_metrics, compute_difficulty_indicators, build_meta,
)
from dataclasses import asdict
import uuid

from src.analysis.criticality_index import compute_criticality


# ============================================================
# Extreme criticality space definition
# ============================================================

def make_extreme_criticality_space(base_space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Push ALL identified MaxTimeLimit criticality drivers to extreme values.

    vs make_high_criticality_space changes:
      PRIMARY (size + demand):
        regions       [8,14]  → [12,18]
        zones/region  [6,16]  → [10,18]
        thermal/zone  [1,2]   → [2,3]
        demand_scale  [1.3,1.6] → [1.4,1.7]

      B8 (SOC tightness = e2p / soc_tolerance):
        battery_e_to_p_hours        [4,6]   (high E/P ratio)
        battery_final_soc_tolerance [0.03,0.08]  (very tight, was ~0.1)

      B10 (network heterogeneity = std(zprs)/mean(zprs)):
        zones_per_region [4,20]  (wide spread → high std/mean ratio)

      A11 (trade reliance = factor * neighbors/4):
        cross_border_policy "allow"  (factor=0.25, highest)
        neighbor_nations [6,8]       (saturate min(n/4,1)=1.0)

      A12 (inv congestion = 1 - intertie_density):
        intertie_density [0.15,0.35] (LOW density → high congestion)
    """
    space = json.loads(json.dumps(base_space))  # deep copy

    # -- Global --
    space["global"]["target_scenarios"] = 800
    space["global"]["seed"] = 400
    space["global"]["horizon_hours"] = 24
    space["global"]["dt_minutes"] = 60

    # -- PRIMARY DRIVER #1: Very large topology (B1, B3) --
    space["structure"]["regions"] = [12, 18]
    space["structure"]["zones_per_region"] = [4, 20]    # wide spread for B10
    space["structure"]["intertie_density"] = [0.15, 0.35]  # LOW density for A12
    space["structure"]["neighbor_nations"] = [6, 8]     # many neighbors for A11

    # -- PRIMARY DRIVER #1 cont: Many assets per zone (B3) --
    space["assets"]["thermal_per_zone"] = [2, 3]
    space["assets"]["nuclear_per_region"] = [1, 2]
    space["assets"]["solar_per_zone"] = [2, 4]
    space["assets"]["wind_per_zone"] = [2, 3]
    space["assets"]["battery_per_zone"] = [2, 3]
    space["assets"]["dr_per_zone"] = [2, 4]

    # -- PRIMARY DRIVER #2: Extreme demand stress (A5, A6) --
    space["exogenous"]["weather_profiles"] = ["stormy_winter", "calm_winter"]
    space["exogenous"]["demand_profiles"] = ["cold_snap", "heatwave"]
    space["exogenous"]["demand_scale_factor"] = [1.4, 1.7]
    space["exogenous"]["inflow_factor"] = [0.3, 0.7]

    # -- SECONDARY DRIVER B8: SOC tightness = e2p / tolerance --
    #    High E/P ratio + very tight SOC tolerance → high B8
    if "techno_params_scalers" in space:
        space["techno_params_scalers"]["battery_e_to_p_hours"] = [4.0, 6.0]
        space["techno_params_scalers"]["battery_final_soc_tolerance"] = [0.03, 0.08]
        space["techno_params_scalers"]["battery_roundtrip_eff"] = [0.80, 0.88]
        space["techno_params_scalers"]["battery_self_discharge_per_hour"] = [0.001, 0.002]
        space["techno_params_scalers"]["pumped_final_level_tolerance"] = [0.03, 0.08]
        # Also tighter ramps
        space["techno_params_scalers"]["thermal_ramp_pct"] = [0.5, 0.8]
        space["techno_params_scalers"]["dr_max_shed_share"] = [0.15, 0.30]
        space["techno_params_scalers"]["dr_duration_hours"] = [3, 6]
        space["techno_params_scalers"]["dr_num_blocks"] = [3, 4]

    # -- SECONDARY DRIVER A11: Trade reliance --
    #    "allow" policy (factor=0.25) + many neighbors (6-8) → saturates
    space["economics_policy"]["cross_border_policy"] = ["allow"]
    space["economics_policy"]["co2_price_eur_per_t"] = [100, 200]

    # -- Higher operational costs (amplify startup intensity B6) --
    if "operation_costs" in space:
        space["operation_costs"]["thermal_startup_cost_eur"] = [5000, 8000]
        space["operation_costs"]["nuclear_startup_cost_eur"] = [15000, 40000]
        space["operation_costs"]["demand_response_cost_eur_per_mwh"] = [120, 200]
        space["operation_costs"]["value_of_lost_load_eur_per_mwh"] = [10000, 30000]

    # -- Relaxed budget guard for very large MILPs --
    space["budget_guard"]["max_vars_per_scenario"] = 800000
    space["budget_guard"]["max_cons_per_scenario"] = 1500000
    space["budget_guard"]["max_binary_vars_per_scenario"] = 150000
    space["budget_guard"]["reject_if_est_cpu_hours_gt"] = 12

    return space


# ============================================================
# Generation pipeline (reused from generate_eval_families.py)
# ============================================================

def generate_candidate_pool(
    space: Dict[str, Any],
    pool_size: int = 800,
) -> List[Tuple[Dict, float]]:
    """Generate candidate scenarios and compute their criticality index."""
    set_seed(space["global"]["seed"])
    candidates = []
    attempts = 0
    max_attempts = pool_size * 40
    mixed_horizons = space["global"].get("mixed_horizons", None)

    while len(candidates) < pool_size and attempts < max_attempts:
        attempts += 1
        try:
            if mixed_horizons:
                horizon_h = mixed_horizons[attempts % len(mixed_horizons)]
            else:
                horizon_h = space["global"]["horizon_hours"]

            graph = sample_graph(space)
            assets = sample_assets(space, graph)
            econ = sample_econ(space)
            tech = sample_tech(space)
            costs = sample_operation_costs(space)
            exo = sample_exogenous(space, graph)
            unit_caps = sample_unit_capacities(space)
            transport_caps = sample_transport_capacities(space)
            mip_gap = sample_mip_gap(space)

            cfg = ScenarioConfig(
                id=str(uuid.uuid4()),
                horizon_hours=horizon_h,
                dt_minutes=space["global"]["dt_minutes"],
                graph=graph,
                assets=assets,
                econ_policy=econ,
                tech=tech,
                costs=costs,
                exogenous=exo,
                unit_capacities=unit_caps,
                transport_capacities=transport_caps,
                mip_gap_target_pct=mip_gap,
            )

            vars_total, cons_total, n_binary = estimate_milp_size(cfg)
            est_hours = estimate_solve_time_hours(vars_total, cons_total, n_binary, space)
            if not passes_budget_guard(vars_total, cons_total, n_binary, est_hours, space):
                continue

            meta = build_meta(cfg, vars_total, cons_total, est_hours)
            flexibility_metrics = compute_flexibility_metrics(cfg)
            difficulty_indicators = compute_difficulty_indicators(
                cfg, vars_total, cons_total, est_hours
            )

            payload = {
                "id": cfg.id,
                "horizon_hours": cfg.horizon_hours,
                "dt_minutes": cfg.dt_minutes,
                "graph": asdict(cfg.graph),
                "assets": asdict(cfg.assets),
                "econ_policy": asdict(cfg.econ_policy),
                "tech": asdict(cfg.tech),
                "operation_costs": asdict(cfg.costs),
                "exogenous": asdict(cfg.exogenous),
                "mip_gap_target_pct": cfg.mip_gap_target_pct,
                "estimates": meta["estimates"],
                "meta": {k: v for k, v in meta.items() if k != "estimates"},
                "flexibility_metrics": flexibility_metrics,
                "difficulty_indicators": difficulty_indicators,
            }

            crit_result = compute_criticality(payload, alpha=0.5)
            candidates.append((payload, crit_result.criticality_index))

            if len(candidates) % 100 == 0:
                print(f"    {len(candidates)}/{pool_size} candidates generated "
                      f"({attempts} attempts)")

        except Exception:
            continue

    print(f"  Generated {len(candidates)} candidates from {attempts} attempts")
    return candidates


def save_family(
    selected: List[Tuple[Dict, float]],
    output_dir: Path,
    family_name: str,
) -> Dict[str, Any]:
    """Save selected scenarios and return manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)

    crit_values = []
    for i, (payload, crit_idx) in enumerate(selected):
        payload["criticality_index"] = round(crit_idx, 4)
        scenario_path = output_dir / f"scenario_{i+1:05d}.json"
        with open(scenario_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        crit_values.append(crit_idx)

    manifest = {
        "family": family_name,
        "count": len(selected),
        "criticality_stats": {
            "mean": round(float(np.mean(crit_values)), 4),
            "std": round(float(np.std(crit_values)), 4),
            "min": round(float(np.min(crit_values)), 4),
            "max": round(float(np.max(crit_values)), 4),
            "median": round(float(np.median(crit_values)), 4),
            "p10": round(float(np.percentile(crit_values, 10)), 4),
            "p90": round(float(np.percentile(crit_values, 90)), 4),
        },
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate extreme-criticality evaluation scenarios."
    )
    parser.add_argument(
        "--pool-size", type=int, default=800,
        help="Candidate pool size (default: 800)"
    )
    parser.add_argument(
        "--count", type=int, default=100,
        help="Number of scenarios to select (default: 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=400,
        help="Random seed (default: 400)"
    )
    args = parser.parse_args()

    base_space_path = PROJECT_ROOT / "config" / "scenario_space.yaml"
    if not base_space_path.exists():
        base_space_path = PROJECT_ROOT / "config" / "scenario_space_hard.yaml"
    base_space = load_space(str(base_space_path))

    print("=" * 70)
    print("GENERATING EXTREME CRITICALITY SCENARIOS")
    print("=" * 70)
    print(f"  Pool size: {args.pool_size}")
    print(f"  Target count: {args.count}")
    print(f"  Seed: {args.seed}")
    print()
    print("  Targeted criticality drivers:")
    print("    PRIMARY:   B1 (zones), B3 (binaries), A5 (demand), A6 (peak/firm)")
    print("    SECONDARY: B8 (SOC tightness), B10 (net heterogeneity),")
    print("               A11 (trade reliance), A12 (inv congestion)")

    # Build extreme space
    space = make_extreme_criticality_space(base_space)
    space["global"]["seed"] = args.seed

    # Generate candidate pool
    print(f"\n  Step 1: Generating candidate pool...")
    candidates = generate_candidate_pool(space, pool_size=args.pool_size)

    if not candidates:
        print("  ERROR: No candidates generated. Check scenario space config.")
        sys.exit(1)

    crit_vals = [c for _, c in candidates]
    print(f"  Pool criticality: mean={np.mean(crit_vals):.3f}, "
          f"min={np.min(crit_vals):.3f}, max={np.max(crit_vals):.3f}, "
          f"std={np.std(crit_vals):.3f}")

    # Select top-N by criticality (hardest scenarios)
    print(f"\n  Step 2: Selecting top {args.count} by criticality...")
    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    selected = sorted_candidates[:args.count]

    sel_crit = [c for _, c in selected]
    print(f"  Selected criticality: mean={np.mean(sel_crit):.3f}, "
          f"min={np.min(sel_crit):.3f}, max={np.max(sel_crit):.3f}")

    # Estimated MILP size
    n_vars = [p["estimates"]["vars_total"] for p, _ in selected]
    n_bins = [p["difficulty_indicators"]["n_binary_variables"] for p, _ in selected]
    print(f"  MILP size: vars={np.mean(n_vars):.0f}+/-{np.std(n_vars):.0f}, "
          f"binaries={np.mean(n_bins):.0f}+/-{np.std(n_bins):.0f}")

    # Secondary driver diagnostics
    print(f"\n  Secondary driver diagnostics (selected scenarios):")
    b8_vals, b10_vals, a11_vals, a12_vals = [], [], [], []
    for p, _ in selected:
        tech = p.get("tech", {})
        graph = p.get("graph", {})
        econ = p.get("econ_policy", {})
        # B8
        e2p = tech.get("battery_e_to_p_hours", 4.0)
        soc_tol = tech.get("battery_final_soc_tolerance", 0.1)
        b8_vals.append(e2p / max(soc_tol, 0.01))
        # B10
        zprs = graph.get("zones_per_region", [1])
        if len(zprs) > 1:
            b10_vals.append(float(np.std(zprs) / max(np.mean(zprs), 1)))
        else:
            b10_vals.append(0.0)
        # A11
        nn = graph.get("neighbor_nations", 1)
        cb = econ.get("cross_border_policy", "allow")
        tf = {"allow": 0.25, "cap": 0.15, "block": 0.02}.get(cb, 0.15)
        a11_vals.append(tf * min(nn / 4.0, 1.0))
        # A12
        a12_vals.append(1 - graph.get("intertie_density", 0.4))

    print(f"    B8  SOC tightness:     mean={np.mean(b8_vals):.1f} "
          f"[{np.min(b8_vals):.1f}-{np.max(b8_vals):.1f}]")
    print(f"    B10 Net heterogeneity: mean={np.mean(b10_vals):.3f} "
          f"[{np.min(b10_vals):.3f}-{np.max(b10_vals):.3f}]")
    print(f"    A11 Trade reliance:    mean={np.mean(a11_vals):.3f} "
          f"[{np.min(a11_vals):.3f}-{np.max(a11_vals):.3f}]")
    print(f"    A12 Inv congestion:    mean={np.mean(a12_vals):.3f} "
          f"[{np.min(a12_vals):.3f}-{np.max(a12_vals):.3f}]")

    # Save
    output_dir = PROJECT_ROOT / "outputs" / "extreme_criticality_scenarios"
    print(f"\n  Step 3: Saving to {output_dir}...")
    manifest = save_family(selected, output_dir, "extreme_criticality_scenarios")

    stats = manifest["criticality_stats"]
    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Scenarios: {manifest['count']}")
    print(f"  Output: {output_dir}")
    print(f"  Criticality: mean={stats['mean']:.3f} "
          f"[{stats['min']:.3f}-{stats['max']:.3f}]")
    print(f"  P10={stats['p10']:.3f}, P90={stats['p90']:.3f}")
    print(f"\nNext steps:")
    print(f"  1. Solve with MILP:")
    print(f"     python -m scripts.solve_eval_families "
          f"--families extreme_criticality_scenarios --time-limit 1200")
    print(f"  2. Evaluate with pipeline:")
    print(f"     Run notebooks/evaluation_extreme.ipynb on Colab")


if __name__ == "__main__":
    main()
