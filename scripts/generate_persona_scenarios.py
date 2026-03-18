"""
Generate 3 persona-based extreme-criticality scenario families (100 each).

Each persona biases BOTH the scenario space (parameter ranges) AND the
criticality scoring (custom weights + alpha) toward its specific concerns.

Personas:
  1. VRE/Battery Developer  → high VRE penetration, storage stress, SOC coupling
  2. Network Operator        → large network, congestion, trade, demand stress
  3. Mathematician           → maximum combinatorial hardness (all B-metrics)

Usage:
    python -m scripts.generate_persona_scenarios [--pool-size 800] [--count 100]
    python -m scripts.generate_persona_scenarios --personas vre_battery
    python -m scripts.generate_persona_scenarios --personas network_operator mathematician
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
    ScenarioConfig,
    estimate_milp_size, estimate_solve_time_hours, passes_budget_guard,
    compute_flexibility_metrics, compute_difficulty_indicators, build_meta,
)
from dataclasses import asdict
import uuid

from src.analysis.criticality_index import compute_criticality


# ============================================================
# Helper: normalize weights so they sum to 1.0
# ============================================================

def _norm_weights(w: Dict[str, float]) -> Dict[str, float]:
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


# ============================================================
# Persona definitions
# ============================================================

PERSONAS: Dict[str, Dict[str, Any]] = {}


# ── Persona 1: VRE / Battery Developer ──────────────────────
# Cares about: high renewable penetration, storage adequacy stress,
#   SOC coupling tightness, residual volatility from VRE intermittency.
# Primary drivers: A1, A2, A4, A7, A8, B8
# Secondary: A6 (peak/firm - low firm = more VRE needed at peak),
#            B3 (large storage fleet → many continuous+binary vars)

PERSONAS["vre_battery"] = {
    "label": "VRE / Battery Developer",
    "dir_name": "persona_vre_battery_scenarios",
    "seed": 501,
    "alpha": 0.55,  # slightly stress-heavy (most drivers are stress)
    "stress_weights": _norm_weights({
        "A1_vre_penetration":       3.0,
        "A2_residual_volatility":   2.5,
        "A3_peak_to_valley":        0.5,
        "A4_short_term_variability":2.5,
        "A5_demand_scale":          0.5,
        "A6_peak_to_firm":          2.0,
        "A7_inv_storage_power":     3.0,
        "A8_inv_storage_energy":    3.0,
        "A9_inv_thermal_flex":      0.5,
        "A10_inv_dr_headroom":      0.5,
        "A11_trade_reliance":       0.5,
        "A12_inv_congestion":       0.5,
    }),
    "hardness_weights": _norm_weights({
        "B1_n_zones":           0.5,
        "B2_horizon":           0.5,
        "B3_binary_est":        1.5,
        "B4_thermal_density":   0.3,
        "B5_min_gen_tight":     0.3,
        "B6_startup_intensity": 0.5,
        "B7_ramp_tightness":    0.5,
        "B8_soc_tightness":     3.0,
        "B9_interconn_density": 0.5,
        "B10_network_hetero":   0.5,
    }),
}


# ── Persona 2: Network Operator ─────────────────────────────
# Cares about: large meshed network, congestion, cross-border trade,
#   demand stress across zones, thermal dispatch complexity.
# Primary: A2, A3, A5, A9, A11, A12, B1, B6, B9, B10
# Secondary: A6 (peak vs firm capacity), B3 (binary count from size),
#            B4 (thermal density per zone)

PERSONAS["network_operator"] = {
    "label": "Network Operator",
    "dir_name": "persona_network_operator_scenarios",
    "seed": 502,
    "alpha": 0.55,  # slightly stress-heavy (congestion/trade are stress metrics)
    "stress_weights": _norm_weights({
        "A1_vre_penetration":       0.5,
        "A2_residual_volatility":   2.5,
        "A3_peak_to_valley":        2.5,
        "A4_short_term_variability":0.5,
        "A5_demand_scale":          3.0,
        "A6_peak_to_firm":          2.0,
        "A7_inv_storage_power":     0.5,
        "A8_inv_storage_energy":    0.5,
        "A9_inv_thermal_flex":      2.5,
        "A10_inv_dr_headroom":      0.5,
        "A11_trade_reliance":       3.0,
        "A12_inv_congestion":       3.0,
    }),
    "hardness_weights": _norm_weights({
        "B1_n_zones":           1.5,
        "B2_horizon":           0.5,
        "B3_binary_est":        1.0,
        "B4_thermal_density":   1.5,
        "B5_min_gen_tight":     0.5,
        "B6_startup_intensity": 2.5,
        "B7_ramp_tightness":    0.5,
        "B8_soc_tightness":     0.5,
        "B9_interconn_density": 3.0,
        "B10_network_hetero":   3.0,
    }),
}


# ── Persona 3: Mathematician (combinatorial solver) ─────────
# Cares about: maximum MILP difficulty regardless of physical meaning.
#   All hardness metrics (B1-B10) pushed hard. Stress only matters
#   inasmuch as it creates active constraints (demand stress, tight flex).
# Primary: B1, B2, B3, B4, B5, B6, B7, B8, B9, B10
# Secondary stress: A5 (forces more constraints active), A9 (tight flex)

PERSONAS["mathematician"] = {
    "label": "Mathematician (Combinatorial)",
    "dir_name": "persona_mathematician_scenarios",
    "seed": 503,
    "alpha": 0.20,  # hardness-heavy: 80% hardness, 20% stress
    "stress_weights": _norm_weights({
        "A1_vre_penetration":       0.5,
        "A2_residual_volatility":   0.5,
        "A3_peak_to_valley":        0.5,
        "A4_short_term_variability":0.5,
        "A5_demand_scale":          2.5,
        "A6_peak_to_firm":          1.5,
        "A7_inv_storage_power":     0.5,
        "A8_inv_storage_energy":    0.5,
        "A9_inv_thermal_flex":      2.0,
        "A10_inv_dr_headroom":      1.0,
        "A11_trade_reliance":       0.5,
        "A12_inv_congestion":       0.5,
    }),
    "hardness_weights": _norm_weights({
        "B1_n_zones":           1.5,
        "B2_horizon":           1.0,
        "B3_binary_est":        2.0,
        "B4_thermal_density":   3.0,
        "B5_min_gen_tight":     2.5,
        "B6_startup_intensity": 3.0,
        "B7_ramp_tightness":    3.0,
        "B8_soc_tightness":     2.5,
        "B9_interconn_density": 1.5,
        "B10_network_hetero":   1.5,
    }),
}


# ============================================================
# Scenario space configs per persona
# ============================================================

def make_vre_battery_space(base_space: Dict[str, Any]) -> Dict[str, Any]:
    """
    VRE/Battery Developer: high VRE penetration, limited firm capacity,
    many storage units with tight SOC constraints.

    Zone count capped at ~60-120 (pipeline sweet spot).
    Criticality pushed through VRE/storage stress, NOT raw topology size.

    Levers:
      A1 (VRE pen.)    ← solar [3,5], wind [3,5], few thermal [0,1]
      A2 (resid. vol.) ← stormy/variable weather
      A4 (ST var.)     ← stormy weather, high noise
      A7 (inv stor P)  ← moderate battery power vs high demand
      A8 (inv stor E)  ← moderate battery energy vs high demand
      A6 (peak/firm)   ← few thermal+nuclear = low firm capacity
      B8 (SOC tight.)  ← high E/P [5,7], very tight tolerance [0.02,0.06]
    """
    space = json.loads(json.dumps(base_space))
    space["global"]["target_scenarios"] = 800
    space["global"]["seed"] = 501
    space["global"]["horizon_hours"] = 24
    space["global"]["dt_minutes"] = 60

    # CAPPED topology: ~60-120 zones (pipeline sweet spot)
    space["structure"]["regions"] = [6, 10]
    space["structure"]["zones_per_region"] = [6, 14]
    space["structure"]["intertie_density"] = [0.3, 0.6]
    space["structure"]["neighbor_nations"] = [2, 5]

    # MASSIVE VRE fleet → pushes A1, A2, A4
    space["assets"]["solar_per_zone"] = [3, 5]
    space["assets"]["wind_per_zone"] = [3, 5]
    # Limited firm generation → pushes A6, A7, A8
    space["assets"]["thermal_per_zone"] = [0, 1]
    space["assets"]["nuclear_per_region"] = [0, 1]
    # Large storage fleet → pushes B8, and partially A7/A8
    space["assets"]["battery_per_zone"] = [2, 4]
    space["assets"]["dr_per_zone"] = [1, 2]

    # Variable/stormy weather → residual volatility, ST variability
    space["exogenous"]["weather_profiles"] = ["stormy_winter", "overcast_summer", "mixed"]
    space["exogenous"]["demand_profiles"] = ["wkday_peak", "cold_snap", "heatwave"]
    space["exogenous"]["demand_scale_factor"] = [1.1, 1.4]
    space["exogenous"]["inflow_factor"] = [0.5, 1.0]

    # Very tight SOC constraints → B8 = e2p / tolerance
    if "techno_params_scalers" in space:
        space["techno_params_scalers"]["battery_e_to_p_hours"] = [5.0, 7.0]
        space["techno_params_scalers"]["battery_final_soc_tolerance"] = [0.02, 0.06]
        space["techno_params_scalers"]["battery_roundtrip_eff"] = [0.82, 0.90]
        space["techno_params_scalers"]["battery_self_discharge_per_hour"] = [0.001, 0.003]
        space["techno_params_scalers"]["battery_initial_soc_fraction"] = [0.4, 0.6]

    space["economics_policy"]["co2_price_eur_per_t"] = [50, 150]
    space["economics_policy"]["cross_border_policy"] = ["allow", "cap"]

    # Budget guard matching high-crit regime
    space["budget_guard"]["max_vars_per_scenario"] = 500000
    space["budget_guard"]["max_cons_per_scenario"] = 1000000
    space["budget_guard"]["max_binary_vars_per_scenario"] = 100000
    space["budget_guard"]["reject_if_est_cpu_hours_gt"] = 8
    return space


def make_network_operator_space(base_space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Network Operator: congestion-prone meshed network, high cross-border
    trade, demand stress, tight thermal dispatch.

    Zone count capped at ~60-120 (pipeline sweet spot).
    Criticality pushed through congestion, trade, demand stress, NOT raw size.

    Levers:
      A12 (inv cong.)    ← LOW intertie density [0.15,0.35]
      A11 (trade rel.)   ← "allow" policy + many neighbors [5,8]
      B10 (net hetero.)  ← wide zones_per_region spread [4,16]
      A5  (demand scale) ← [1.3,1.6] high demand
      A2  (resid. vol.)  ← stormy weather
      A3  (peak/valley)  ← cold_snap/heatwave profiles
      A9  (inv therm flex)← few thermal with tight ramps
      B6  (startup int.) ← high startup costs
    """
    space = json.loads(json.dumps(base_space))
    space["global"]["target_scenarios"] = 800
    space["global"]["seed"] = 502
    space["global"]["horizon_hours"] = 24
    space["global"]["dt_minutes"] = 60

    # CAPPED topology: ~60-120 zones, but heterogeneous
    space["structure"]["regions"] = [8, 14]
    space["structure"]["zones_per_region"] = [4, 16]   # wide spread → B10
    space["structure"]["intertie_density"] = [0.15, 0.35]  # LOW → A12
    space["structure"]["neighbor_nations"] = [5, 8]    # many → A11

    # Moderate assets (not the focus, but realistic)
    space["assets"]["thermal_per_zone"] = [1, 2]
    space["assets"]["nuclear_per_region"] = [0, 2]
    space["assets"]["solar_per_zone"] = [2, 4]
    space["assets"]["wind_per_zone"] = [2, 4]
    space["assets"]["battery_per_zone"] = [1, 2]
    space["assets"]["dr_per_zone"] = [1, 3]

    # High demand stress across network → A2, A3, A5
    space["exogenous"]["weather_profiles"] = ["stormy_winter", "calm_winter", "mixed"]
    space["exogenous"]["demand_profiles"] = ["cold_snap", "heatwave", "wkday_peak"]
    space["exogenous"]["demand_scale_factor"] = [1.3, 1.6]
    space["exogenous"]["inflow_factor"] = [0.4, 0.9]

    # Trade-friendly policy (maximizes A11)
    space["economics_policy"]["cross_border_policy"] = ["allow"]
    space["economics_policy"]["co2_price_eur_per_t"] = [80, 180]

    # High startup costs → B6
    if "operation_costs" in space:
        space["operation_costs"]["thermal_startup_cost_eur"] = [6000, 10000]
        space["operation_costs"]["nuclear_startup_cost_eur"] = [20000, 50000]

    # Tight thermal ramps → A9
    if "techno_params_scalers" in space:
        space["techno_params_scalers"]["thermal_ramp_pct"] = [0.4, 0.7]

    # Budget guard matching high-crit regime
    space["budget_guard"]["max_vars_per_scenario"] = 500000
    space["budget_guard"]["max_cons_per_scenario"] = 1000000
    space["budget_guard"]["max_binary_vars_per_scenario"] = 100000
    space["budget_guard"]["reject_if_est_cpu_hours_gt"] = 8
    return space


def make_mathematician_space(base_space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mathematician: maximum combinatorial hardness at moderate zone count.

    Zone count capped at ~80-120 (pipeline sweet spot).
    Criticality pushed through coupling density: many thermal/battery
    per zone, tight ramps, tight SOC, high startups — all of which
    increase branching difficulty without inflating LP size linearly.

    Levers:
      B4  (therm dens.)  ← thermal [2, 3] per zone (many UC decisions/zone)
      B6  (startup int.) ← very high startup costs [8k,12k]
      B7  (ramp tight.)  ← tight ramps [0.35,0.6]
      B8  (SOC tight.)   ← tight [0.02,0.05], high E/P [5,7]
      B3  (binaries)     ← high thermal+battery density per zone
      B9  (interconn.)   ← dense intertie [0.5,0.8]
      B10 (net hetero.)  ← wide zones spread [4,16]
      A5  (demand)       ← high [1.3,1.7] (activates more constraints)
      A9  (inv therm)    ← tight ramps
    """
    space = json.loads(json.dumps(base_space))
    space["global"]["target_scenarios"] = 800
    space["global"]["seed"] = 503
    space["global"]["horizon_hours"] = 24
    space["global"]["dt_minutes"] = 60

    # CAPPED topology: ~80-120 zones, but high coupling density
    space["structure"]["regions"] = [8, 14]
    space["structure"]["zones_per_region"] = [4, 16]   # wide spread → B10
    space["structure"]["intertie_density"] = [0.5, 0.8]  # dense → B9
    space["structure"]["neighbor_nations"] = [4, 7]

    # HIGH UC-bearing asset density per zone → B3, B4, B6
    # (many binary decisions per zone, not many zones)
    space["assets"]["thermal_per_zone"] = [2, 3]
    space["assets"]["nuclear_per_region"] = [1, 2]
    space["assets"]["solar_per_zone"] = [2, 4]
    space["assets"]["wind_per_zone"] = [1, 3]
    space["assets"]["battery_per_zone"] = [2, 3]
    space["assets"]["dr_per_zone"] = [2, 4]

    # Extreme demand (activates constraints)
    space["exogenous"]["weather_profiles"] = ["stormy_winter", "calm_winter"]
    space["exogenous"]["demand_profiles"] = ["cold_snap", "heatwave"]
    space["exogenous"]["demand_scale_factor"] = [1.3, 1.7]
    space["exogenous"]["inflow_factor"] = [0.3, 0.7]

    # Very high startup costs → B6
    if "operation_costs" in space:
        space["operation_costs"]["thermal_startup_cost_eur"] = [8000, 12000]
        space["operation_costs"]["nuclear_startup_cost_eur"] = [25000, 50000]
        space["operation_costs"]["demand_response_cost_eur_per_mwh"] = [150, 250]
        space["operation_costs"]["value_of_lost_load_eur_per_mwh"] = [15000, 40000]

    # Tight everything → B7, B8, B5
    if "techno_params_scalers" in space:
        space["techno_params_scalers"]["thermal_ramp_pct"] = [0.35, 0.6]
        space["techno_params_scalers"]["battery_e_to_p_hours"] = [5.0, 7.0]
        space["techno_params_scalers"]["battery_final_soc_tolerance"] = [0.02, 0.05]
        space["techno_params_scalers"]["battery_roundtrip_eff"] = [0.78, 0.86]
        space["techno_params_scalers"]["dr_max_shed_share"] = [0.15, 0.30]
        space["techno_params_scalers"]["dr_duration_hours"] = [3, 6]
        space["techno_params_scalers"]["dr_num_blocks"] = [3, 4]
        space["techno_params_scalers"]["dr_max_events"] = [3, 6]

    space["economics_policy"]["co2_price_eur_per_t"] = [120, 220]
    space["economics_policy"]["cross_border_policy"] = ["cap", "block"]

    # Budget guard matching high-crit regime
    space["budget_guard"]["max_vars_per_scenario"] = 500000
    space["budget_guard"]["max_cons_per_scenario"] = 1000000
    space["budget_guard"]["max_binary_vars_per_scenario"] = 100000
    space["budget_guard"]["reject_if_est_cpu_hours_gt"] = 8
    return space


SPACE_BUILDERS = {
    "vre_battery": make_vre_battery_space,
    "network_operator": make_network_operator_space,
    "mathematician": make_mathematician_space,
}


# ============================================================
# Generation pipeline
# ============================================================

def generate_candidate_pool(
    space: Dict[str, Any],
    pool_size: int = 800,
) -> List[Tuple[Dict, float, float]]:
    """
    Generate candidates. Returns (payload, default_crit, scenario_uuid) tuples.
    The persona-specific scoring is applied at selection time.
    """
    set_seed(space["global"]["seed"])
    candidates = []
    attempts = 0
    max_attempts = pool_size * 40

    while len(candidates) < pool_size and attempts < max_attempts:
        attempts += 1
        try:
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
            est_hours = estimate_solve_time_hours(
                vars_total, cons_total, n_binary, space
            )
            if not passes_budget_guard(
                vars_total, cons_total, n_binary, est_hours, space
            ):
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

            candidates.append(payload)

            if len(candidates) % 100 == 0:
                print(f"    {len(candidates)}/{pool_size} candidates "
                      f"({attempts} attempts)")

        except Exception:
            continue

    print(f"  Generated {len(candidates)} candidates from {attempts} attempts")
    return candidates


def score_candidates(
    candidates: List[Dict],
    alpha: float,
    stress_weights: Dict[str, float],
    hardness_weights: Dict[str, float],
) -> List[Tuple[Dict, float]]:
    """Score all candidates with persona-specific weights."""
    scored = []
    for payload in candidates:
        result = compute_criticality(
            payload,
            alpha=alpha,
            stress_weights=stress_weights,
            hardness_weights=hardness_weights,
        )
        scored.append((payload, result.criticality_index))
    return scored


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
        description="Generate persona-based extreme-criticality scenarios."
    )
    parser.add_argument(
        "--pool-size", type=int, default=800,
        help="Candidate pool size per persona (default: 800)"
    )
    parser.add_argument(
        "--count", type=int, default=100,
        help="Scenarios to select per persona (default: 100)"
    )
    parser.add_argument(
        "--personas", nargs="*", default=None,
        help="Specific personas to generate (default: all 3). "
             "Options: vre_battery, network_operator, mathematician"
    )
    args = parser.parse_args()

    base_space_path = PROJECT_ROOT / "config" / "scenario_space.yaml"
    if not base_space_path.exists():
        base_space_path = PROJECT_ROOT / "config" / "scenario_space_hard.yaml"
    base_space = load_space(str(base_space_path))

    persona_keys = args.personas or list(PERSONAS.keys())
    invalid = [p for p in persona_keys if p not in PERSONAS]
    if invalid:
        print(f"ERROR: Unknown personas: {invalid}")
        print(f"Available: {list(PERSONAS.keys())}")
        sys.exit(1)

    outputs_dir = PROJECT_ROOT / "outputs"

    print("=" * 70)
    print("GENERATING PERSONA-BASED EXTREME SCENARIOS")
    print("=" * 70)
    print(f"  Personas:   {persona_keys}")
    print(f"  Pool size:  {args.pool_size}")
    print(f"  Target:     {args.count} per persona")

    all_manifests = {}

    for pkey in persona_keys:
        persona = PERSONAS[pkey]
        space_fn = SPACE_BUILDERS[pkey]
        family_dir = outputs_dir / persona["dir_name"]

        print(f"\n{'='*70}")
        print(f"PERSONA: {persona['label']}")
        print(f"  alpha={persona['alpha']}")
        top_stress = sorted(persona["stress_weights"].items(),
                            key=lambda x: x[1], reverse=True)[:3]
        top_hard = sorted(persona["hardness_weights"].items(),
                          key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top stress weights: "
              + ", ".join(f"{k}={v:.2f}" for k, v in top_stress))
        print(f"  Top hardness weights: "
              + ", ".join(f"{k}={v:.2f}" for k, v in top_hard))
        print(f"{'='*70}")

        # Build biased space
        space = space_fn(base_space)

        # Generate candidate pool
        print(f"\n  Step 1: Generating candidate pool ({args.pool_size})...")
        candidates = generate_candidate_pool(space, pool_size=args.pool_size)

        if not candidates:
            print(f"  ERROR: No candidates for {pkey}. Skipping.")
            continue

        # Score with persona weights
        print(f"  Step 2: Scoring with persona weights...")
        scored = score_candidates(
            candidates,
            alpha=persona["alpha"],
            stress_weights=persona["stress_weights"],
            hardness_weights=persona["hardness_weights"],
        )

        crit_vals = [c for _, c in scored]
        print(f"  Pool persona-score: mean={np.mean(crit_vals):.3f}, "
              f"min={np.min(crit_vals):.3f}, max={np.max(crit_vals):.3f}")

        # Select top-N
        print(f"  Step 3: Selecting top {args.count}...")
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = scored[:args.count]

        sel_crit = [c for _, c in selected]
        print(f"  Selected: mean={np.mean(sel_crit):.3f}, "
              f"min={np.min(sel_crit):.3f}, max={np.max(sel_crit):.3f}")

        # MILP size diagnostics
        n_vars = [p["estimates"]["vars_total"] for p, _ in selected]
        n_bins = [p["difficulty_indicators"]["n_binary_variables"]
                  for p, _ in selected]
        print(f"  MILP size: vars={np.mean(n_vars):.0f}+/-{np.std(n_vars):.0f}, "
              f"binaries={np.mean(n_bins):.0f}+/-{np.std(n_bins):.0f}")

        # Save
        print(f"  Step 4: Saving to {family_dir}...")
        manifest = save_family(selected, family_dir, persona["dir_name"])
        all_manifests[pkey] = manifest

        stats = manifest["criticality_stats"]
        print(f"  Saved {manifest['count']} scenarios")
        print(f"  Persona-crit: mean={stats['mean']:.3f} "
              f"[{stats['min']:.3f}-{stats['max']:.3f}]")

    # Global summary
    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")
    for pkey, m in all_manifests.items():
        s = m["criticality_stats"]
        print(f"  {PERSONAS[pkey]['label']:35s}: {m['count']} scenarios, "
              f"crit={s['mean']:.3f} [{s['min']:.3f}-{s['max']:.3f}]")
    total = sum(m["count"] for m in all_manifests.values())
    print(f"\nTotal: {total} scenarios across {len(all_manifests)} personas")

    # Save persona manifest
    persona_manifest = {
        "total_scenarios": total,
        "personas": {
            pkey: {
                "label": PERSONAS[pkey]["label"],
                "dir_name": PERSONAS[pkey]["dir_name"],
                "alpha": PERSONAS[pkey]["alpha"],
                "manifest": all_manifests[pkey],
            }
            for pkey in all_manifests
        },
    }
    with open(outputs_dir / "persona_families_manifest.json", "w",
              encoding="utf-8") as f:
        json.dump(persona_manifest, f, indent=2)

    print(f"\nNext steps:")
    dirs = " ".join(PERSONAS[p]["dir_name"] for p in all_manifests)
    print(f"  1. Solve: python -m scripts.solve_eval_families "
          f"--families {dirs} --time-limit 1200")
    print(f"  2. Evaluate: Run notebooks/evaluation_personas.ipynb on Colab")


if __name__ == "__main__":
    main()
