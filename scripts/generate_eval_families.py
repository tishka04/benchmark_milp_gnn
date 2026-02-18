"""
Generate 3 families of evaluation scenarios: low, medium, high criticality.

Each family has 100 scenarios. Uses the existing generator_v2 pipeline with
modified scenario_space configs to bias generation toward desired criticality ranges,
then filters/selects using the criticality index.

Usage:
    python -m scripts.generate_eval_families
"""
from __future__ import annotations

import json
import sys
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
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
    rand_int, rand_float, rand_choice,
)
from dataclasses import asdict
import uuid
import yaml

from src.analysis.criticality_index import (
    compute_criticality, compute_stress_metrics, compute_hardness_metrics,
    CriticalityResult,
)


# ============================================================
# Scenario space configs for each criticality level
# ============================================================

def make_low_criticality_space(base_space: Dict[str, Any]) -> Dict[str, Any]:
    """Bias parameters toward easy/low-criticality scenarios.
    
    MILP-hardness analysis shows solve difficulty is driven by:
      1. Problem size (zones, assets, binary vars)  -- d ~ 1.0
      2. Demand stress (demand_scale, peak_to_valley) -- d ~ 1.3
    Cost/tech params have negligible effect (d < 0.1).
    
    Low targets: ~20-40 zones, demand_scale 0.7-0.95, MILP solves < 10s.
    """
    space = json.loads(json.dumps(base_space))  # deep copy
    space["global"]["target_scenarios"] = 500  # Generate pool
    space["global"]["seed"] = 100
    # Small topology → few binary vars (driver #1)
    space["structure"]["regions"] = [2, 5]
    space["structure"]["zones_per_region"] = [2, 8]
    space["structure"]["intertie_density"] = [0.3, 0.6]
    space["structure"]["neighbor_nations"] = [1, 3]
    # Few assets per zone → small MILP
    space["assets"]["thermal_per_zone"] = [0, 1]
    space["assets"]["nuclear_per_region"] = [0, 1]
    space["assets"]["battery_per_zone"] = [1, 2]
    space["assets"]["dr_per_zone"] = [0, 2]
    # Low demand stress (driver #2)
    space["exogenous"]["weather_profiles"] = ["calm_winter", "overcast_summer", "mixed"]
    space["exogenous"]["demand_profiles"] = ["wkend_flat", "shoulder", "wkday_peak"]
    space["exogenous"]["demand_scale_factor"] = [0.7, 0.95]
    space["exogenous"]["inflow_factor"] = [1.0, 1.7]
    # Moderate economics (these don't affect MILP hardness)
    space["economics_policy"]["co2_price_eur_per_t"] = [35, 100]
    space["economics_policy"]["cross_border_policy"] = ["allow", "cap"]
    # Small budget guard
    space["budget_guard"]["max_vars_per_scenario"] = 80000
    space["budget_guard"]["max_binary_vars_per_scenario"] = 10000
    return space


def make_medium_criticality_space(base_space: Dict[str, Any]) -> Dict[str, Any]:
    """Bias parameters toward medium-criticality scenarios.
    
    Bridges between low and high families. Targets the "hard optimal"
    regime from v3 analysis: scenarios that solve in 50-600s.
    
    Medium targets: ~50-80 zones, demand_scale 1.05-1.25, MILP 10-200s.
    """
    space = json.loads(json.dumps(base_space))
    space["global"]["target_scenarios"] = 500
    space["global"]["seed"] = 200
    # Medium topology → moderate binary count (driver #1)
    space["structure"]["regions"] = [5, 10]
    space["structure"]["zones_per_region"] = [4, 12]
    space["structure"]["intertie_density"] = [0.2, 0.5]
    space["structure"]["neighbor_nations"] = [3, 6]
    # Moderate assets per zone
    space["assets"]["thermal_per_zone"] = [1, 2]
    space["assets"]["nuclear_per_region"] = [0, 2]
    space["assets"]["battery_per_zone"] = [1, 2]
    space["assets"]["dr_per_zone"] = [1, 3]
    # Moderate demand stress (driver #2)
    space["exogenous"]["weather_profiles"] = ["stormy_winter", "sunny_summer", "overcast_summer", "mixed"]
    space["exogenous"]["demand_profiles"] = ["wkday_peak", "cold_snap", "heatwave"]
    space["exogenous"]["demand_scale_factor"] = [1.05, 1.25]
    space["exogenous"]["inflow_factor"] = [0.5, 1.1]
    # Moderate economics (low discriminative power)
    space["economics_policy"]["co2_price_eur_per_t"] = [50, 150]
    space["economics_policy"]["cross_border_policy"] = ["allow", "cap"]
    # Medium budget guard
    space["budget_guard"]["max_vars_per_scenario"] = 200000
    space["budget_guard"]["max_binary_vars_per_scenario"] = 30000
    return space


def make_high_criticality_space(base_space: Dict[str, Any]) -> Dict[str, Any]:
    """Bias parameters toward MILP-hard scenarios that hit MaxTimeLimit.
    
    Based on empirical analysis of 612 MaxTimeLimit vs 4388 optimal scenarios
    in scenarios_v3. The two dominant hardness drivers are:
    
      1. Problem size (Cohen's d ~ 1.0-1.1):
         - TL avg: ~98 zones, ~1038 assets, ~12193 binary vars
         - Opt avg: ~68 zones, ~728 assets, ~8500 binary vars
      2. Demand stress (Cohen's d ~ 1.3):
         - TL avg: demand_scale=1.34, peak_to_valley=2.41
         - Opt avg: demand_scale=1.07, peak_to_valley=1.92
    
    Cost/tech parameters have near-zero discriminative power (d < 0.1).
    
    High targets: ~80-120 zones, demand_scale 1.3-1.6, MILP >> 600s.
    """
    space = json.loads(json.dumps(base_space))
    space["global"]["target_scenarios"] = 500
    space["global"]["seed"] = 300
    space["global"]["horizon_hours"] = 24
    space["global"]["dt_minutes"] = 60
    # DRIVER #1: Large topology → many binary variables
    # TL scenarios avg ~9 regions, ~98 zones
    space["structure"]["regions"] = [8, 14]
    space["structure"]["zones_per_region"] = [6, 16]
    space["structure"]["intertie_density"] = [0.3, 0.6]
    space["structure"]["neighbor_nations"] = [3, 7]
    # DRIVER #1 cont: Many assets per zone → large MILP
    # TL scenarios avg ~1.0 thermal/zone, ~5.5 VRE/zone, ~3.5 storage+DR/zone
    space["assets"]["thermal_per_zone"] = [1, 2]
    space["assets"]["nuclear_per_region"] = [0, 2]
    space["assets"]["solar_per_zone"] = [2, 4]
    space["assets"]["wind_per_zone"] = [2, 4]
    space["assets"]["battery_per_zone"] = [1, 2]
    space["assets"]["dr_per_zone"] = [1, 3]
    # DRIVER #2: High demand stress
    # TL scenarios avg demand_scale=1.34, peak_valley=2.41
    space["exogenous"]["weather_profiles"] = ["stormy_winter", "mixed"]
    space["exogenous"]["demand_profiles"] = ["cold_snap", "heatwave", "wkday_peak"]
    space["exogenous"]["demand_scale_factor"] = [1.3, 1.6]
    space["exogenous"]["inflow_factor"] = [0.4, 0.9]
    # Economics/costs: keep at moderate defaults (d < 0.1, no effect on hardness)
    space["economics_policy"]["co2_price_eur_per_t"] = [50, 150]
    space["economics_policy"]["cross_border_policy"] = ["allow"]
    # Budget guard: allow large MILPs matching TL regime
    space["budget_guard"]["max_vars_per_scenario"] = 500000
    space["budget_guard"]["max_cons_per_scenario"] = 1000000
    space["budget_guard"]["max_binary_vars_per_scenario"] = 100000
    space["budget_guard"]["reject_if_est_cpu_hours_gt"] = 8
    return space


def generate_candidate_pool(space: Dict[str, Any], pool_size: int = 500) -> List[Tuple[Dict, float]]:
    """
    Generate a pool of scenario candidates and compute their criticality index.
    
    If space["global"]["mixed_horizons"] is set, rotates through those horizon
    values for each candidate (e.g., [24, 48, 96]).
    
    Returns list of (scenario_payload, criticality_index) tuples.
    """
    set_seed(space["global"]["seed"])
    candidates = []
    attempts = 0
    max_attempts = pool_size * 30
    mixed_horizons = space["global"].get("mixed_horizons", None)

    while len(candidates) < pool_size and attempts < max_attempts:
        attempts += 1
        try:
            # Pick horizon: rotate through mixed list or use fixed value
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
            difficulty_indicators = compute_difficulty_indicators(cfg, vars_total, cons_total, est_hours)

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

            # Compute criticality index
            crit_result = compute_criticality(payload, alpha=0.5)
            candidates.append((payload, crit_result.criticality_index))

        except Exception as e:
            continue

    print(f"  Generated {len(candidates)} candidates from {attempts} attempts")
    return candidates


def select_scenarios_by_criticality(
    candidates: List[Tuple[Dict, float]],
    target_count: int,
    crit_range: Tuple[float, float],
    fallback_expand: float = 0.1,
) -> List[Tuple[Dict, float]]:
    """
    Select target_count scenarios within the given criticality range.
    If not enough, expand the range progressively.
    """
    lo, hi = crit_range
    in_range = [(p, c) for p, c in candidates if lo <= c <= hi]
    
    # Expand range if needed
    expansion = 0
    while len(in_range) < target_count and expansion < 5:
        expansion += 1
        lo_new = max(0, lo - fallback_expand * expansion)
        hi_new = min(1, hi + fallback_expand * expansion)
        in_range = [(p, c) for p, c in candidates if lo_new <= c <= hi_new]
        print(f"    Expanded range to [{lo_new:.2f}, {hi_new:.2f}]: {len(in_range)} candidates")
    
    # Sort by distance from range center and pick top N
    center = (crit_range[0] + crit_range[1]) / 2
    in_range.sort(key=lambda x: abs(x[1] - center))
    selected = in_range[:target_count]
    
    if len(selected) < target_count:
        print(f"    WARNING: Only {len(selected)} scenarios available (target: {target_count})")
    
    return selected


def save_family(
    selected: List[Tuple[Dict, float]],
    output_dir: Path,
    family_name: str,
) -> Dict[str, Any]:
    """Save selected scenarios to output directory and return manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    crit_values = []
    for i, (payload, crit_idx) in enumerate(selected):
        # Add criticality index to payload
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


def main():
    base_space_path = PROJECT_ROOT / "config" / "scenario_space.yaml"
    base_space = load_space(str(base_space_path))
    
    outputs_dir = PROJECT_ROOT / "outputs"
    
    families = {
        "low_criticality_scenarios": {
            "space_fn": make_low_criticality_space,
            "pool_size": 500,
            "select_mode": "bottom",  # take the 100 lowest-criticality from pool
        },
        "medium_criticality_scenarios": {
            "space_fn": make_medium_criticality_space,
            "pool_size": 500,
            "select_mode": "middle",  # take the 100 closest-to-median from pool
        },
        "high_criticality_scenarios": {
            "space_fn": make_high_criticality_space,
            "pool_size": 500,
            "select_mode": "top",  # take the 100 highest-criticality from pool
        },
    }
    
    all_manifests = {}
    
    for family_name, config in families.items():
        print(f"\n{'='*60}")
        print(f"Generating {family_name}")
        print(f"{'='*60}")
        
        family_dir = outputs_dir / family_name
        
        # Create biased space
        space = config["space_fn"](base_space)
        
        # Generate candidate pool
        print(f"  Generating candidate pool ({config['pool_size']} targets)...")
        candidates = generate_candidate_pool(space, pool_size=config["pool_size"])
        
        if candidates:
            crit_vals = [c for _, c in candidates]
            print(f"  Pool criticality: mean={np.mean(crit_vals):.3f}, "
                  f"min={np.min(crit_vals):.3f}, max={np.max(crit_vals):.3f}")
        
        # Select 100 based on select_mode
        mode = config["select_mode"]
        print(f"  Selecting 100 scenarios (mode={mode})...")
        sorted_candidates = sorted(candidates, key=lambda x: x[1])
        if mode == "bottom":
            selected = sorted_candidates[:100]
        elif mode == "top":
            selected = sorted_candidates[-100:]
        else:  # middle
            n = len(sorted_candidates)
            mid = n // 2
            start = max(0, mid - 50)
            selected = sorted_candidates[start:start + 100]
        
        # Save
        manifest = save_family(selected, family_dir, family_name)
        all_manifests[family_name] = manifest
        
        print(f"  Saved {manifest['count']} scenarios to {family_dir}")
        stats = manifest["criticality_stats"]
        print(f"  Criticality: mean={stats['mean']:.3f} [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"  P10={stats['p10']:.3f}, P90={stats['p90']:.3f}")
    
    # Save global manifest
    global_manifest = {
        "total_scenarios": sum(m["count"] for m in all_manifests.values()),
        "families": all_manifests,
    }
    with open(outputs_dir / "eval_families_manifest.json", "w", encoding="utf-8") as f:
        json.dump(global_manifest, f, indent=2)
    
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    for name, m in all_manifests.items():
        s = m["criticality_stats"]
        print(f"  {name}: {m['count']} scenarios, "
              f"crit={s['mean']:.3f} [{s['min']:.3f}-{s['max']:.3f}]")
    print(f"\nTotal: {global_manifest['total_scenarios']} scenarios")


if __name__ == "__main__":
    main()
