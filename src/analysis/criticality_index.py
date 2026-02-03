"""
Criticality Index Calculator for MILP Flexibility Scenarios.

Implements the 22-component criticality indicator from criticality_index.tex:
- Physical Stress Index (A1-A12): 12 metrics
- Combinatorial Hardness Index (B1-B10): 10 metrics

Final form: Crit(s) = α·Stress(s) + (1-α)·Hard(s)
"""
from __future__ import annotations

import json
import argparse
import pathlib
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


# ============================================================
# Default weights (equal weighting within categories)
# ============================================================

DEFAULT_STRESS_WEIGHTS = {
    "A1_vre_penetration": 1.0 / 12,
    "A2_residual_volatility": 1.0 / 12,
    "A3_peak_to_valley": 1.0 / 12,
    "A4_short_term_variability": 1.0 / 12,
    "A5_demand_scale": 1.0 / 12,
    "A6_peak_to_firm": 1.0 / 12,
    "A7_inv_storage_power": 1.0 / 12,
    "A8_inv_storage_energy": 1.0 / 12,
    "A9_inv_thermal_flex": 1.0 / 12,
    "A10_inv_dr_headroom": 1.0 / 12,
    "A11_trade_reliance": 1.0 / 12,
    "A12_inv_congestion": 1.0 / 12,
}

DEFAULT_HARDNESS_WEIGHTS = {
    "B1_n_zones": 1.0 / 10,
    "B2_horizon": 1.0 / 10,
    "B3_binary_est": 1.0 / 10,
    "B4_thermal_density": 1.0 / 10,
    "B5_min_gen_tight": 1.0 / 10,
    "B6_startup_intensity": 1.0 / 10,
    "B7_ramp_tightness": 1.0 / 10,
    "B8_soc_tightness": 1.0 / 10,
    "B9_interconn_density": 1.0 / 10,
    "B10_network_hetero": 1.0 / 10,
}

# Normalization bounds (generator-bound scaling for stability)
NORM_BOUNDS = {
    # Stress metrics
    "A1_vre_penetration": (0.0, 1.5),
    "A2_residual_volatility": (0.0, 0.5),
    "A3_peak_to_valley": (1.0, 10.0),
    "A4_short_term_variability": (0.0, 0.3),
    "A5_demand_scale": (0.5, 2.0),
    "A6_peak_to_firm": (0.3, 1.5),
    "A7_inv_storage_power": (-1.0, 1.0),
    "A8_inv_storage_energy": (-1.0, 1.0),
    "A9_inv_thermal_flex": (-1.0, 1.0),
    "A10_inv_dr_headroom": (0.5, 1.0),
    "A11_trade_reliance": (0.0, 0.5),
    "A12_inv_congestion": (-1.0, 1.0),
    # Hardness metrics
    "B1_n_zones": (4, 400),
    "B2_horizon": (12, 168),
    "B3_binary_est": (500, 50000),
    "B4_thermal_density": (0.0, 3.0),
    "B5_min_gen_tight": (0.1, 0.6),
    "B6_startup_intensity": (0.0, 0.5),
    "B7_ramp_tightness": (0.5, 10.0),
    "B8_soc_tightness": (0.1, 100.0),
    "B9_interconn_density": (0.1, 2.0),
    "B10_network_hetero": (0.0, 1.0),
}


# ============================================================
# Data structures
# ============================================================

@dataclass
class StressMetrics:
    """Physical stress index components (A1-A12)."""
    A1_vre_penetration: float = 0.0
    A2_residual_volatility: float = 0.0
    A3_peak_to_valley: float = 0.0
    A4_short_term_variability: float = 0.0
    A5_demand_scale: float = 0.0
    A6_peak_to_firm: float = 0.0
    A7_inv_storage_power: float = 0.0
    A8_inv_storage_energy: float = 0.0
    A9_inv_thermal_flex: float = 0.0
    A10_inv_dr_headroom: float = 0.0
    A11_trade_reliance: float = 0.0
    A12_inv_congestion: float = 0.0


@dataclass
class HardnessMetrics:
    """Combinatorial hardness index components (B1-B10)."""
    B1_n_zones: float = 0.0
    B2_horizon: float = 0.0
    B3_binary_est: float = 0.0
    B4_thermal_density: float = 0.0
    B5_min_gen_tight: float = 0.0
    B6_startup_intensity: float = 0.0
    B7_ramp_tightness: float = 0.0
    B8_soc_tightness: float = 0.0
    B9_interconn_density: float = 0.0
    B10_network_hetero: float = 0.0


@dataclass
class CriticalityResult:
    """Complete criticality analysis result."""
    scenario_id: str
    stress_index: float
    hardness_index: float
    criticality_index: float
    stress_metrics: StressMetrics
    hardness_metrics: HardnessMetrics
    stress_normalized: Dict[str, float] = field(default_factory=dict)
    hardness_normalized: Dict[str, float] = field(default_factory=dict)


# ============================================================
# Metric computation functions
# ============================================================

def normalize_metric(value: float, bounds: Tuple[float, float]) -> float:
    """Normalize metric to [0, 1] using generator-bound scaling."""
    lo, hi = bounds
    if hi <= lo:
        return 0.5
    normalized = (value - lo) / (hi - lo)
    return max(0.0, min(1.0, normalized))


def compute_stress_metrics(scenario: Dict[str, Any]) -> StressMetrics:
    """
    Compute all 12 physical stress metrics from scenario data.
    
    Since we don't have time-series data, we estimate from scenario parameters.
    """
    meta = scenario.get("meta", {})
    assets = meta.get("assets", scenario.get("assets", {}))
    tech = scenario.get("tech", {})
    flex = scenario.get("flexibility_metrics", {})
    diff = scenario.get("difficulty_indicators", {})
    exo = scenario.get("exogenous", {})
    econ = scenario.get("econ_policy", {})
    
    n_zones = meta.get("zones", sum(scenario.get("graph", {}).get("zones_per_region", [1])))
    
    # A1: VRE penetration ratio
    vre_pen = diff.get("vre_penetration_pct", 30.0) / 100.0
    
    # A2: Net demand volatility (from difficulty indicators)
    residual_vol = diff.get("net_demand_volatility", 0.15)
    
    # A3: Peak-to-valley ratio
    ptv = diff.get("peak_to_valley_ratio", 1.8)
    
    # A4: Short-term variability (estimate from weather profile)
    weather_var_map = {
        "calm_winter": 0.08,
        "stormy_winter": 0.22,
        "sunny_summer": 0.18,
        "overcast_summer": 0.06,
        "mixed": 0.14,
    }
    weather_profile = exo.get("weather_profile", "mixed")
    st_var = weather_var_map.get(weather_profile, 0.12)
    
    # A5: Demand scale factor
    demand_scale = exo.get("demand_scale_factor", 1.0)
    
    # A6: Peak load to firm capacity ratio
    # Estimate firm capacity from thermal + nuclear
    n_thermal = assets.get("thermal", 0)
    n_nuclear = assets.get("nuclear", 0)
    avg_thermal_mw = 300.0
    avg_nuclear_mw = 1000.0
    firm_capacity = n_thermal * avg_thermal_mw + n_nuclear * avg_nuclear_mw
    
    # Peak demand estimate
    avg_zone_demand = 500.0
    peak_demand = n_zones * avg_zone_demand * demand_scale * 1.3  # peak factor
    peak_to_firm = peak_demand / max(firm_capacity, 1.0)
    
    # A7: Inverse storage power adequacy
    storage_power = flex.get("total_storage_power_mw", 0.0)
    residual_peak = peak_demand * (1 - vre_pen * 0.3)  # rough residual
    inv_stor_p = 1 - storage_power / max(residual_peak, 1.0)
    inv_stor_p = max(-1.0, min(1.0, inv_stor_p))
    
    # A8: Inverse storage energy adequacy
    storage_energy = flex.get("total_storage_capacity_mwh", 0.0)
    # Positive residual demand over horizon (rough estimate)
    T = scenario.get("horizon_hours", 24)
    pos_residual_sum = residual_peak * T * 0.4  # rough estimate
    inv_stor_e = 1 - storage_energy / max(pos_residual_sum, 1.0)
    inv_stor_e = max(-1.0, min(1.0, inv_stor_e))
    
    # A9: Inverse thermal flexibility ratio
    thermal_ramp_pct = tech.get("thermal_ramp_pct", 0.5)
    thermal_flex = n_thermal * avg_thermal_mw * thermal_ramp_pct
    inv_th_flex = 1 - thermal_flex / max(residual_peak, 1.0)
    inv_th_flex = max(-1.0, min(1.0, inv_th_flex))
    
    # A10: Inverse DR headroom
    dr_capacity = flex.get("total_dr_capacity_mw", 0.0)
    total_demand = n_zones * avg_zone_demand * demand_scale * T
    dr_headroom = dr_capacity * T / max(total_demand, 1.0)
    inv_dr = 1 - dr_headroom
    inv_dr = max(0.0, min(1.0, inv_dr))
    
    # A11: Trade reliance (estimate from cross-border policy and neighbor count)
    neighbor_nations = scenario.get("graph", {}).get("neighbor_nations", 1)
    cross_border = econ.get("cross_border_policy", "allow")
    trade_factor = {"allow": 0.25, "cap": 0.15, "block": 0.02}.get(cross_border, 0.15)
    trade_reliance = trade_factor * min(neighbor_nations / 4.0, 1.0)
    
    # A12: Inverse congestion exposure proxy
    intertie_density = scenario.get("graph", {}).get("intertie_density", 0.4)
    # Lower density = more congestion risk
    inv_cong = 1 - intertie_density
    
    return StressMetrics(
        A1_vre_penetration=vre_pen,
        A2_residual_volatility=residual_vol,
        A3_peak_to_valley=ptv,
        A4_short_term_variability=st_var,
        A5_demand_scale=demand_scale,
        A6_peak_to_firm=peak_to_firm,
        A7_inv_storage_power=inv_stor_p,
        A8_inv_storage_energy=inv_stor_e,
        A9_inv_thermal_flex=inv_th_flex,
        A10_inv_dr_headroom=inv_dr,
        A11_trade_reliance=trade_reliance,
        A12_inv_congestion=inv_cong,
    )


def compute_hardness_metrics(scenario: Dict[str, Any]) -> HardnessMetrics:
    """
    Compute all 10 combinatorial hardness metrics from scenario data.
    """
    meta = scenario.get("meta", {})
    assets = meta.get("assets", scenario.get("assets", {}))
    tech = scenario.get("tech", {})
    costs = scenario.get("operation_costs", {})
    diff = scenario.get("difficulty_indicators", {})
    estimates = scenario.get("estimates", {})
    graph = scenario.get("graph", {})
    
    n_zones = meta.get("zones", sum(graph.get("zones_per_region", [1])))
    T = scenario.get("horizon_hours", 24)
    
    # B1: Number of zones
    n_zones_metric = float(n_zones)
    
    # B2: Horizon length (timesteps)
    dt_minutes = scenario.get("dt_minutes", 60)
    n_timesteps = int(T * 60 / dt_minutes)
    
    # B3: Estimated binary variables
    n_binary = diff.get("n_binary_variables", estimates.get("vars_total", 10000) * 0.1)
    
    # B4: Thermal unit density
    n_thermal = assets.get("thermal", 0)
    thermal_density = n_thermal / max(n_zones, 1)
    
    # B5: Min-gen tightness (P_min / P_max for thermal)
    # Estimate: typical min-gen ratio is 30-50% for thermal
    min_gen_tight = 0.35  # Default estimate
    
    # B6: Startup cost intensity
    thermal_startup = costs.get("thermal_startup_cost_eur", 5000.0)
    nuclear_startup = costs.get("nuclear_startup_cost_eur", 30000.0)
    thermal_var = costs.get("thermal_fuel_eur_per_mwh", 60.0)
    
    n_nuclear = assets.get("nuclear", 0)
    total_startup = n_thermal * thermal_startup + n_nuclear * nuclear_startup
    # Variable cost over horizon (rough estimate)
    avg_thermal_mw = 300.0
    avg_nuclear_mw = 1000.0
    capacity_factor = 0.5
    total_var_cost = (n_thermal * avg_thermal_mw + n_nuclear * avg_nuclear_mw) * T * capacity_factor * thermal_var
    startup_intensity = total_startup / max(total_var_cost, 1.0)
    startup_intensity = min(startup_intensity, 0.5)
    
    # B7: Ramping tightness index
    thermal_ramp_pct = tech.get("thermal_ramp_pct", 0.5)
    total_thermal_cap = n_thermal * avg_thermal_mw
    total_ramp = total_thermal_cap * thermal_ramp_pct
    ramp_tightness = total_thermal_cap / max(total_ramp, 1.0)
    
    # B8: Storage SOC tightness proxy
    battery_e2p = tech.get("battery_e_to_p_hours", 4.0)
    soc_tolerance = tech.get("battery_final_soc_tolerance", 0.1)
    soc_tightness = battery_e2p / max(soc_tolerance, 0.01)
    
    # B9: Interconnection density
    intertie_density = graph.get("intertie_density", 0.4)
    n_regions = graph.get("regions", 1)
    # Estimate number of lines
    n_lines_est = intertie_density * n_regions * (n_regions - 1) / 2
    interconn_density = n_lines_est / max(n_zones, 1)
    
    # B10: Network heterogeneity index
    zones_per_region = graph.get("zones_per_region", [1])
    if len(zones_per_region) > 1:
        net_hetero = float(np.std(zones_per_region) / max(np.mean(zones_per_region), 1))
    else:
        net_hetero = 0.0
    
    return HardnessMetrics(
        B1_n_zones=n_zones_metric,
        B2_horizon=float(n_timesteps),
        B3_binary_est=float(n_binary),
        B4_thermal_density=thermal_density,
        B5_min_gen_tight=min_gen_tight,
        B6_startup_intensity=startup_intensity,
        B7_ramp_tightness=ramp_tightness,
        B8_soc_tightness=soc_tightness,
        B9_interconn_density=interconn_density,
        B10_network_hetero=net_hetero,
    )


def compute_criticality(
    scenario: Dict[str, Any],
    alpha: float = 0.5,
    stress_weights: Optional[Dict[str, float]] = None,
    hardness_weights: Optional[Dict[str, float]] = None,
) -> CriticalityResult:
    """
    Compute the complete criticality index for a scenario.
    
    Args:
        scenario: Scenario JSON data
        alpha: Weight for stress vs hardness (default 0.5)
        stress_weights: Custom weights for stress metrics
        hardness_weights: Custom weights for hardness metrics
        
    Returns:
        CriticalityResult with all metrics and indices
    """
    if stress_weights is None:
        stress_weights = DEFAULT_STRESS_WEIGHTS
    if hardness_weights is None:
        hardness_weights = DEFAULT_HARDNESS_WEIGHTS
    
    # Compute raw metrics
    stress = compute_stress_metrics(scenario)
    hardness = compute_hardness_metrics(scenario)
    
    # Normalize and aggregate stress
    stress_dict = asdict(stress)
    stress_normalized = {}
    stress_index = 0.0
    for key, value in stress_dict.items():
        bounds = NORM_BOUNDS.get(key, (0.0, 1.0))
        norm_val = normalize_metric(value, bounds)
        stress_normalized[key] = norm_val
        weight = stress_weights.get(key, 1.0 / len(stress_dict))
        stress_index += weight * norm_val
    
    # Normalize and aggregate hardness
    hardness_dict = asdict(hardness)
    hardness_normalized = {}
    hardness_index = 0.0
    for key, value in hardness_dict.items():
        bounds = NORM_BOUNDS.get(key, (0.0, 1.0))
        norm_val = normalize_metric(value, bounds)
        hardness_normalized[key] = norm_val
        weight = hardness_weights.get(key, 1.0 / len(hardness_dict))
        hardness_index += weight * norm_val
    
    # Combined criticality index
    criticality_index = alpha * stress_index + (1 - alpha) * hardness_index
    
    return CriticalityResult(
        scenario_id=scenario.get("id", "unknown"),
        stress_index=round(stress_index, 4),
        hardness_index=round(hardness_index, 4),
        criticality_index=round(criticality_index, 4),
        stress_metrics=stress,
        hardness_metrics=hardness,
        stress_normalized=stress_normalized,
        hardness_normalized=hardness_normalized,
    )


def process_scenario_file(scenario_path: pathlib.Path, alpha: float) -> Dict[str, Any]:
    """Process a single scenario file and return criticality data."""
    with open(scenario_path, "r", encoding="utf-8") as f:
        scenario = json.load(f)
    
    result = compute_criticality(scenario, alpha=alpha)
    
    return {
        "file": scenario_path.name,
        "scenario_id": result.scenario_id,
        "stress_index": result.stress_index,
        "hardness_index": result.hardness_index,
        "criticality_index": result.criticality_index,
        "stress_metrics": asdict(result.stress_metrics),
        "hardness_metrics": asdict(result.hardness_metrics),
        "stress_normalized": result.stress_normalized,
        "hardness_normalized": result.hardness_normalized,
    }


def collect_raw_metrics(scenario_path: pathlib.Path) -> Dict[str, Dict[str, float]]:
    """Collect raw metrics from a scenario file without normalization."""
    with open(scenario_path, "r", encoding="utf-8") as f:
        scenario = json.load(f)
    
    stress = compute_stress_metrics(scenario)
    hardness = compute_hardness_metrics(scenario)
    
    return {
        "file": scenario_path.name,
        "stress": asdict(stress),
        "hardness": asdict(hardness),
    }


def compute_data_driven_bounds(
    raw_metrics: List[Dict[str, Dict[str, float]]],
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute normalization bounds from actual data distribution.
    
    Uses percentiles to avoid outlier sensitivity while ensuring
    the full range of scenarios is represented.
    """
    bounds = {}
    
    # Collect all stress metric values
    stress_keys = list(raw_metrics[0]["stress"].keys())
    for key in stress_keys:
        values = [r["stress"][key] for r in raw_metrics]
        lo = float(np.percentile(values, percentile_low))
        hi = float(np.percentile(values, percentile_high))
        # Ensure some spread
        if hi <= lo:
            hi = lo + 0.01
        bounds[key] = (lo, hi)
    
    # Collect all hardness metric values
    hardness_keys = list(raw_metrics[0]["hardness"].keys())
    for key in hardness_keys:
        values = [r["hardness"][key] for r in raw_metrics]
        lo = float(np.percentile(values, percentile_low))
        hi = float(np.percentile(values, percentile_high))
        if hi <= lo:
            hi = lo + 0.01
        bounds[key] = (lo, hi)
    
    return bounds


def normalize_and_aggregate(
    raw_metrics: Dict[str, Dict[str, float]],
    bounds: Dict[str, Tuple[float, float]],
    alpha: float,
    stress_weights: Dict[str, float],
    hardness_weights: Dict[str, float],
) -> Dict[str, Any]:
    """Normalize metrics using provided bounds and compute indices."""
    stress_dict = raw_metrics["stress"]
    hardness_dict = raw_metrics["hardness"]
    
    stress_normalized = {}
    stress_index = 0.0
    for key, value in stress_dict.items():
        norm_val = normalize_metric(value, bounds.get(key, (0.0, 1.0)))
        stress_normalized[key] = norm_val
        stress_index += stress_weights.get(key, 1.0 / len(stress_dict)) * norm_val
    
    hardness_normalized = {}
    hardness_index = 0.0
    for key, value in hardness_dict.items():
        norm_val = normalize_metric(value, bounds.get(key, (0.0, 1.0)))
        hardness_normalized[key] = norm_val
        hardness_index += hardness_weights.get(key, 1.0 / len(hardness_dict)) * norm_val
    
    criticality_index = alpha * stress_index + (1 - alpha) * hardness_index
    
    return {
        "file": raw_metrics["file"],
        "stress_index": round(stress_index, 4),
        "hardness_index": round(hardness_index, 4),
        "criticality_index": round(criticality_index, 4),
        "stress_metrics": stress_dict,
        "hardness_metrics": hardness_dict,
        "stress_normalized": stress_normalized,
        "hardness_normalized": hardness_normalized,
    }


def run_batch_analysis(
    scenarios_dir: str,
    output_path: Optional[str] = None,
    alpha: float = 0.5,
    n_workers: int = 4,
    use_data_driven_bounds: bool = True,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> Dict[str, Any]:
    """
    Run criticality analysis on all scenarios in a directory.
    
    Args:
        scenarios_dir: Path to directory containing scenario JSON files
        output_path: Optional path to save results JSON
        alpha: Weight for stress vs hardness
        n_workers: Number of parallel workers
        use_data_driven_bounds: If True, compute bounds from data (recommended)
        percentile_low: Lower percentile for bound calculation (default 2%)
        percentile_high: Upper percentile for bound calculation (default 98%)
        
    Returns:
        Summary statistics and per-scenario results
    """
    scenarios_path = pathlib.Path(scenarios_dir)
    scenario_files = sorted(scenarios_path.glob("scenario_*.json"))
    
    if not scenario_files:
        print(f"No scenario files found in {scenarios_dir}")
        return {"error": "No scenario files found"}
    
    print(f"Analyzing {len(scenario_files)} scenarios with α={alpha}...")
    
    if use_data_driven_bounds:
        print("Pass 1: Collecting raw metrics for data-driven normalization...")
        raw_metrics_list: List[Dict] = []
        for i, scenario_file in enumerate(scenario_files):
            try:
                raw = collect_raw_metrics(scenario_file)
                raw_metrics_list.append(raw)
                if (i + 1) % 1000 == 0:
                    print(f"  Collected {i + 1}/{len(scenario_files)}...")
            except Exception as e:
                print(f"Error collecting {scenario_file}: {e}")
        
        print(f"Pass 2: Computing bounds from {len(raw_metrics_list)} scenarios...")
        bounds = compute_data_driven_bounds(
            raw_metrics_list, 
            percentile_low=percentile_low,
            percentile_high=percentile_high
        )
        
        print("Pass 3: Normalizing and aggregating...")
        results = []
        for i, raw in enumerate(raw_metrics_list):
            result = normalize_and_aggregate(
                raw, bounds, alpha,
                DEFAULT_STRESS_WEIGHTS, DEFAULT_HARDNESS_WEIGHTS
            )
            results.append(result)
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(raw_metrics_list)}...")
    else:
        print("Using static normalization bounds...")
        results: List[Dict[str, Any]] = []
        for i, scenario_file in enumerate(scenario_files):
            try:
                result = process_scenario_file(scenario_file, alpha)
                results.append(result)
                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1}/{len(scenario_files)} scenarios...")
            except Exception as e:
                print(f"Error processing {scenario_file}: {e}")
    
    # Compute summary statistics
    if results:
        stress_vals = [r["stress_index"] for r in results]
        hardness_vals = [r["hardness_index"] for r in results]
        crit_vals = [r["criticality_index"] for r in results]
        
        summary = {
            "n_scenarios": len(results),
            "alpha": alpha,
            "stress_index": {
                "mean": round(float(np.mean(stress_vals)), 4),
                "std": round(float(np.std(stress_vals)), 4),
                "min": round(float(np.min(stress_vals)), 4),
                "max": round(float(np.max(stress_vals)), 4),
                "median": round(float(np.median(stress_vals)), 4),
            },
            "hardness_index": {
                "mean": round(float(np.mean(hardness_vals)), 4),
                "std": round(float(np.std(hardness_vals)), 4),
                "min": round(float(np.min(hardness_vals)), 4),
                "max": round(float(np.max(hardness_vals)), 4),
                "median": round(float(np.median(hardness_vals)), 4),
            },
            "criticality_index": {
                "mean": round(float(np.mean(crit_vals)), 4),
                "std": round(float(np.std(crit_vals)), 4),
                "min": round(float(np.min(crit_vals)), 4),
                "max": round(float(np.max(crit_vals)), 4),
                "median": round(float(np.median(crit_vals)), 4),
            },
            "criticality_distribution": {
                "low_0_25": sum(1 for v in crit_vals if v < 0.25),
                "medium_25_50": sum(1 for v in crit_vals if 0.25 <= v < 0.50),
                "high_50_75": sum(1 for v in crit_vals if 0.50 <= v < 0.75),
                "critical_75_100": sum(1 for v in crit_vals if v >= 0.75),
            },
        }
        
        # Sort results by criticality
        results_sorted = sorted(results, key=lambda x: x["criticality_index"], reverse=True)
        
        output = {
            "summary": summary,
            "top_10_critical": results_sorted[:10],
            "bottom_10_critical": results_sorted[-10:],
            "all_results": results_sorted,
        }
    else:
        output = {"error": "No results computed", "summary": {}, "all_results": []}
    
    # Save results
    if output_path:
        output_file = pathlib.Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {output_path}")
    
    # Print summary
    if "summary" in output and output["summary"]:
        s = output["summary"]
        print("\n" + "=" * 60)
        print("CRITICALITY INDEX SUMMARY")
        print("=" * 60)
        print(f"Scenarios analyzed: {s['n_scenarios']}")
        print(f"Alpha (stress weight): {s['alpha']}")
        print()
        print("Stress Index:")
        print(f"  Mean: {s['stress_index']['mean']:.4f} ± {s['stress_index']['std']:.4f}")
        print(f"  Range: [{s['stress_index']['min']:.4f}, {s['stress_index']['max']:.4f}]")
        print()
        print("Hardness Index:")
        print(f"  Mean: {s['hardness_index']['mean']:.4f} ± {s['hardness_index']['std']:.4f}")
        print(f"  Range: [{s['hardness_index']['min']:.4f}, {s['hardness_index']['max']:.4f}]")
        print()
        print("Criticality Index:")
        print(f"  Mean: {s['criticality_index']['mean']:.4f} ± {s['criticality_index']['std']:.4f}")
        print(f"  Range: [{s['criticality_index']['min']:.4f}, {s['criticality_index']['max']:.4f}]")
        print()
        print("Distribution:")
        dist = s["criticality_distribution"]
        print(f"  Low (0.00-0.25):      {dist['low_0_25']:5d} ({100*dist['low_0_25']/s['n_scenarios']:.1f}%)")
        print(f"  Medium (0.25-0.50):   {dist['medium_25_50']:5d} ({100*dist['medium_25_50']/s['n_scenarios']:.1f}%)")
        print(f"  High (0.50-0.75):     {dist['high_50_75']:5d} ({100*dist['high_50_75']/s['n_scenarios']:.1f}%)")
        print(f"  Critical (0.75-1.00): {dist['critical_75_100']:5d} ({100*dist['critical_75_100']/s['n_scenarios']:.1f}%)")
        print("=" * 60)
    
    return output


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute criticality index for MILP flexibility scenarios"
    )
    parser.add_argument(
        "scenarios_dir",
        help="Path to directory containing scenario JSON files"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path for results JSON (default: <scenarios_dir>/criticality_results.json)"
    )
    parser.add_argument(
        "-a", "--alpha",
        type=float,
        default=0.5,
        help="Weight for stress index vs hardness index (default: 0.5)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--static-bounds",
        action="store_true",
        help="Use static normalization bounds instead of data-driven (not recommended)"
    )
    parser.add_argument(
        "--percentile-low",
        type=float,
        default=2.0,
        help="Lower percentile for data-driven bounds (default: 2)"
    )
    parser.add_argument(
        "--percentile-high",
        type=float,
        default=98.0,
        help="Upper percentile for data-driven bounds (default: 98)"
    )
    
    args = parser.parse_args()
    
    output_path = args.output
    if output_path is None:
        output_path = str(pathlib.Path(args.scenarios_dir) / "criticality_results.json")
    
    run_batch_analysis(
        scenarios_dir=args.scenarios_dir,
        output_path=output_path,
        alpha=args.alpha,
        n_workers=args.workers,
        use_data_driven_bounds=not args.static_bounds,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
    )


if __name__ == "__main__":
    main()
