"""
Generator V2: Optimized diversity scenario generation.

Implements improvements from max_scenarios_diversity.tex:
- P1: Fixed min-distance acceptance logic
- P3: Enriched diversity vector (stress, flexibility, hardness)
- P4: Regional weather heterogeneity preservation
- P5: Pool + greedy k-center selection
- P6: Stratification/quotas for balanced sampling
- P8: Latin Hypercube Sampling for continuous parameters
"""
from __future__ import annotations

import json
import random
import uuid
import pathlib
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter

import numpy as np
import yaml

# Import shared utilities and dataclasses from v1
from .generator_v1 import (
    DEFAULT_ZONE_PROFILE_VARIANTS,
    set_seed,
    rand_int,
    rand_float,
    rand_choice,
    one_hot,
    resolve_range,
    normalize_weights,
    load_space,
    GraphSpec,
    AssetSpec,
    EconPolicy,
    TechScalers,
    OperationCosts,
    UnitCapacities,
    TransportCapacities,
    RegionWeatherCell,
    ExogenousSpec,
    ScenarioConfig,
    estimate_assets_count,
    estimate_milp_size,
    estimate_solve_time_hours,
    passes_budget_guard,
    compute_flexibility_metrics,
    compute_difficulty_indicators,
    build_meta,
)


# ============================================================
# P8: Latin Hypercube Sampling for continuous parameters
# ============================================================

def latin_hypercube_sample(n_samples: int, n_dims: int, seed: int = None) -> np.ndarray:
    """
    Generate Latin Hypercube Samples in [0, 1]^n_dims.
    
    Args:
        n_samples: Number of samples to generate
        n_dims: Number of dimensions
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (n_samples, n_dims) with values in [0, 1]
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    samples = np.zeros((n_samples, n_dims))
    for dim in range(n_dims):
        # Create n_samples intervals of equal probability
        intervals = np.arange(n_samples) / n_samples
        # Sample uniformly within each interval
        points = intervals + rng.uniform(0, 1.0 / n_samples, n_samples)
        # Shuffle to break correlation between dimensions
        rng.shuffle(points)
        samples[:, dim] = points
    
    return samples


def map_lhs_to_ranges(lhs_samples: np.ndarray, ranges: List[Tuple[float, float]]) -> np.ndarray:
    """Map LHS samples from [0,1] to specified ranges."""
    mapped = np.zeros_like(lhs_samples)
    for dim, (lo, hi) in enumerate(ranges):
        mapped[:, dim] = lo + lhs_samples[:, dim] * (hi - lo)
    return mapped


# ============================================================
# Extended dataclass to hold DR parameters for diversity
# ============================================================

@dataclass
class DRScalers:
    """Extended DR parameters for diversity tracking."""
    dr_max_shed_share: float
    dr_duration_hours: float
    dr_num_blocks: int
    dr_rebound_decay: float
    dr_rebound_tolerance: float
    dr_max_events: int
    dr_min_duration: int
    dr_ramp_limit_factor: float


# ============================================================
# P3 + P4: Enriched diversity vector
# ============================================================

def compute_regional_weather_stats(region_weather: Dict[str, RegionWeatherCell], 
                                    weather_choices: List[str]) -> Dict[str, Any]:
    """
    Compute statistics about regional weather heterogeneity.
    
    Returns:
        - profile_histogram: normalized counts of each weather profile
        - entropy: Shannon entropy of profile distribution
        - spread_stats: min/mean/max of regional spread intensity
    """
    n_regions = len(region_weather)
    if n_regions == 0:
        return {
            "profile_histogram": [0.0] * len(weather_choices),
            "entropy": 0.0,
            "spread_min": 0.0,
            "spread_mean": 0.0,
            "spread_max": 0.0,
        }
    
    # Count profiles
    profile_counts = Counter(cell.weather_profile for cell in region_weather.values())
    histogram = [profile_counts.get(p, 0) / n_regions for p in weather_choices]
    
    # Shannon entropy
    probs = np.array([c for c in histogram if c > 0])
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # Spread intensity stats
    spreads = [cell.weather_spread_intensity for cell in region_weather.values()]
    
    return {
        "profile_histogram": histogram,
        "entropy": entropy,
        "spread_min": min(spreads),
        "spread_mean": np.mean(spreads),
        "spread_max": max(spreads),
    }


def enriched_diversity_vector(cfg: ScenarioConfig, space: Dict[str, Any], 
                               dr_scalers: Optional[DRScalers] = None) -> np.ndarray:
    """
    Build enriched diversity vector covering all axes from P3+P4:
    
    A. Topology/size (hardness proxies)
    B. Asset mix (degrees-of-freedom)
    C. Tech scalers that change MILP difficulty
    D. Exogenous & policy drivers
    E. Regional weather heterogeneity (P4)
    """
    vector: List[float] = []
    weather_choices = space["exogenous"]["weather_profiles"]
    demand_choices = space["exogenous"]["demand_profiles"]
    cb_choices = space["economics_policy"]["cross_border_policy"]
    
    # ---- A. Topology / size (hardness proxies) ----
    vector.append(float(cfg.graph.regions))  # |R|
    vector.append(float(sum(cfg.graph.zones_per_region)))  # |Z|
    vector.append(float(np.mean(cfg.graph.zones_per_region)))  # mean zones per region
    vector.append(float(np.std(cfg.graph.zones_per_region)) if len(cfg.graph.zones_per_region) > 1 else 0.0)  # std
    vector.append(float(cfg.graph.intertie_density))
    vector.append(float(cfg.graph.neighbor_nations))
    
    # ---- B. Asset mix (degrees-of-freedom) ----
    counts = estimate_assets_count(cfg.assets)
    zones = sum(cfg.graph.zones_per_region)
    
    # Totals
    vector.append(float(counts["thermal"]))
    vector.append(float(counts["nuclear"]))
    vector.append(float(counts["solar"]))
    vector.append(float(counts["wind"]))
    vector.append(float(counts["battery"]))
    vector.append(float(counts["hydro_pumped"]))
    vector.append(float(counts["hydro_reservoir"]))
    vector.append(float(counts["dr"]))
    
    # Densities (per zone)
    vector.append(float(counts["thermal"]) / max(zones, 1))
    vector.append(float(counts["dr"]) / max(zones, 1))
    vector.append(float(counts["battery"] + counts["hydro_pumped"]) / max(zones, 1))  # storage per zone
    
    # Ratios
    vre_total = counts["solar"] + counts["wind"]
    dispatchable = counts["thermal"] + counts["nuclear"]
    vector.append(float(vre_total) / max(dispatchable, 1))  # VRE/dispatchable ratio
    
    # Storage power / peak demand proxy
    avg_battery_mw = 50.0
    avg_pumped_mw = 200.0
    storage_power = counts["battery"] * avg_battery_mw + counts["hydro_pumped"] * avg_pumped_mw
    peak_demand_proxy = zones * 500.0 * cfg.exogenous.demand_scale_factor
    vector.append(storage_power / max(peak_demand_proxy, 1))
    
    # DR headroom proxy
    dr_headroom = zones * 500.0 * cfg.tech.dr_max_shed_share * cfg.exogenous.demand_scale_factor
    vector.append(dr_headroom / max(peak_demand_proxy, 1))
    
    # ---- C. Tech scalers that change MILP difficulty ----
    vector.append(float(cfg.tech.thermal_ramp_pct))
    vector.append(float(cfg.tech.battery_e_to_p_hours))
    vector.append(float(cfg.tech.battery_roundtrip_eff))
    vector.append(float(cfg.tech.battery_final_soc_tolerance))
    vector.append(float(cfg.tech.pumped_final_level_tolerance))
    
    # DR structure (if extended scalers provided)
    if dr_scalers is not None:
        vector.append(float(dr_scalers.dr_max_shed_share))
        vector.append(float(dr_scalers.dr_duration_hours))
        vector.append(float(dr_scalers.dr_max_events))
        vector.append(float(dr_scalers.dr_min_duration))
        vector.append(float(dr_scalers.dr_ramp_limit_factor))
    else:
        vector.append(float(cfg.tech.dr_max_shed_share))
        vector.append(float(cfg.tech.dr_duration_hours))
        vector.append(5.0)  # default dr_max_events
        vector.append(2.0)  # default dr_min_duration
        vector.append(0.5)  # default dr_ramp_limit_factor
    
    # ---- D. Exogenous & policy drivers ----
    # Demand profile (one-hot)
    vector.extend(one_hot(cfg.exogenous.demand_profile, demand_choices))
    
    # Weather profile (one-hot) - dominant profile
    vector.extend(one_hot(cfg.exogenous.weather_profile, weather_choices))
    
    # Continuous exogenous
    vector.append(float(cfg.exogenous.demand_scale_factor))
    vector.append(float(cfg.exogenous.inflow_factor))
    vector.append(float(cfg.exogenous.weather_spread_intensity))
    
    # Cross-border policy (one-hot)
    vector.extend(one_hot(cfg.econ_policy.cross_border_policy, cb_choices))
    
    # Economic scalers
    vector.append(float(cfg.econ_policy.import_export_caps_factor))
    vector.append(float(cfg.econ_policy.co2_price))
    vector.append(float(cfg.econ_policy.price_cap))
    
    # ---- E. Regional weather heterogeneity (P4) ----
    weather_stats = compute_regional_weather_stats(cfg.exogenous.region_weather, weather_choices)
    vector.extend(weather_stats["profile_histogram"])  # normalized histogram
    vector.append(weather_stats["entropy"])  # diversity measure
    vector.append(weather_stats["spread_min"])
    vector.append(weather_stats["spread_mean"])
    vector.append(weather_stats["spread_max"])
    
    return np.array(vector, dtype=float)


def compute_diversity_bounds(space: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute min/max bounds for the enriched diversity vector.
    Must match the structure of enriched_diversity_vector().
    """
    weather_choices = space["exogenous"]["weather_profiles"]
    demand_choices = space["exogenous"]["demand_profiles"]
    cb_choices = space["economics_policy"]["cross_border_policy"]
    
    mins_list: List[float] = []
    maxs_list: List[float] = []
    
    struct = space["structure"]
    assets = space["assets"]
    tech = space["techno_params_scalers"]
    exo = space["exogenous"]
    econ = space["economics_policy"]
    
    # Max possible zones
    max_regions = struct["regions"][1]
    max_zones_per_region = struct["zones_per_region"][1]
    max_zones = max_regions * max_zones_per_region
    
    # ---- A. Topology ----
    mins_list.extend([struct["regions"][0], struct["zones_per_region"][0], struct["zones_per_region"][0], 0.0,
                      struct["intertie_density"][0], struct["neighbor_nations"][0]])
    maxs_list.extend([struct["regions"][1], max_zones, struct["zones_per_region"][1], 10.0,
                      struct["intertie_density"][1], struct["neighbor_nations"][1]])
    
    # ---- B. Asset mix totals (8) ----
    for key in ["thermal_per_zone", "nuclear_per_region", "solar_per_zone", "wind_per_zone", 
                "battery_per_zone", "hydro_pumped_per_region", "hydro_reservoir_per_region", "dr_per_zone"]:
        if "per_zone" in key:
            mins_list.append(0.0)
            maxs_list.append(float(assets[key][1] * max_zones))
        else:  # per_region
            mins_list.append(0.0)
            maxs_list.append(float(assets[key][1] * max_regions))
    
    # Densities (3): thermal/zone, dr/zone, storage/zone
    mins_list.extend([0.0, 0.0, 0.0])
    maxs_list.extend([float(assets["thermal_per_zone"][1]), 
                      float(assets["dr_per_zone"][1]),
                      float(assets["battery_per_zone"][1] + assets["hydro_pumped_per_region"][1])])
    
    # Ratios (3): VRE/disp, storage/demand, DR/demand
    mins_list.extend([0.0, 0.0, 0.0])
    maxs_list.extend([20.0, 1.0, 0.5])  # reasonable upper bounds
    
    # ---- C. Tech scalers (10) ----
    mins_list.extend([tech["thermal_ramp_pct"][0], tech["battery_e_to_p_hours"][0],
                      tech["battery_roundtrip_eff"][0], tech["battery_final_soc_tolerance"][0],
                      tech["pumped_final_level_tolerance"][0]])
    maxs_list.extend([tech["thermal_ramp_pct"][1], tech["battery_e_to_p_hours"][1],
                      tech["battery_roundtrip_eff"][1], tech["battery_final_soc_tolerance"][1],
                      tech["pumped_final_level_tolerance"][1]])
    
    # DR structure (5)
    mins_list.extend([tech["dr_max_shed_share"][0], tech["dr_duration_hours"][0],
                      tech.get("dr_max_events", [2, 10])[0], tech.get("dr_min_duration", [1, 3])[0],
                      tech.get("dr_ramp_limit_factor", [0.25, 1])[0]])
    maxs_list.extend([tech["dr_max_shed_share"][1], tech["dr_duration_hours"][1],
                      tech.get("dr_max_events", [2, 10])[1], tech.get("dr_min_duration", [1, 3])[1],
                      tech.get("dr_ramp_limit_factor", [0.25, 1])[1]])
    
    # ---- D. Exogenous & policy ----
    # Demand profile one-hot
    mins_list.extend([0.0] * len(demand_choices))
    maxs_list.extend([1.0] * len(demand_choices))
    
    # Weather profile one-hot
    mins_list.extend([0.0] * len(weather_choices))
    maxs_list.extend([1.0] * len(weather_choices))
    
    # Continuous exogenous (3)
    mins_list.extend([exo["demand_scale_factor"][0], exo["inflow_factor"][0], exo["weather_spread_intensity"][0]])
    maxs_list.extend([exo["demand_scale_factor"][1], exo["inflow_factor"][1], exo["weather_spread_intensity"][1]])
    
    # Cross-border policy one-hot
    mins_list.extend([0.0] * len(cb_choices))
    maxs_list.extend([1.0] * len(cb_choices))
    
    # Economic (3)
    mins_list.extend([econ["import_export_caps_factor"][0], econ["co2_price_eur_per_t"][0], 
                      econ["price_cap_eur_per_mwh"][0]])
    maxs_list.extend([econ["import_export_caps_factor"][1], econ["co2_price_eur_per_t"][1],
                      econ["price_cap_eur_per_mwh"][1]])
    
    # ---- E. Regional weather heterogeneity ----
    # Histogram (len weather_choices)
    mins_list.extend([0.0] * len(weather_choices))
    maxs_list.extend([1.0] * len(weather_choices))
    
    # Entropy, spread min/mean/max (4)
    mins_list.extend([0.0, exo["weather_spread_intensity"][0], exo["weather_spread_intensity"][0], 
                      exo["weather_spread_intensity"][0]])
    maxs_list.extend([np.log(len(weather_choices)), exo["weather_spread_intensity"][1],
                      exo["weather_spread_intensity"][1], exo["weather_spread_intensity"][1]])
    
    return np.array(mins_list, dtype=float), np.array(maxs_list, dtype=float)


def normalized_distance(a: np.ndarray, b: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> float:
    """Compute normalized Euclidean distance."""
    denom = np.maximum(maxs - mins, 1e-9)
    return float(np.linalg.norm((a - b) / denom))


# ============================================================
# P5: Pool + Greedy K-Center Selection
# ============================================================

def greedy_k_center_selection(vectors: np.ndarray, mins: np.ndarray, maxs: np.ndarray, 
                               k: int, seed: int = None) -> List[int]:
    """
    Select k diverse scenarios using greedy farthest-point (k-center) algorithm.
    
    x_{t+1} = argmax_{x in Pool} min_{x' in Selected} d(x, x')
    
    Args:
        vectors: Array of shape (n_candidates, n_dims) with diversity vectors
        mins, maxs: Normalization bounds
        k: Number of scenarios to select
        seed: Random seed for initial point selection
        
    Returns:
        List of indices of selected scenarios
    """
    n_candidates = len(vectors)
    if k >= n_candidates:
        return list(range(n_candidates))
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Normalize all vectors
    denom = np.maximum(maxs - mins, 1e-9)
    normalized = (vectors - mins) / denom
    
    # Start with random point
    selected_indices: List[int] = [rng.randint(n_candidates)]
    
    # Track minimum distance to selected set for each candidate
    min_dists_to_selected = np.full(n_candidates, np.inf)
    
    while len(selected_indices) < k:
        last_selected = selected_indices[-1]
        last_vec = normalized[last_selected]
        
        # Update min distances with distance to newly selected point
        for i in range(n_candidates):
            if i not in selected_indices:
                dist = np.linalg.norm(normalized[i] - last_vec)
                min_dists_to_selected[i] = min(min_dists_to_selected[i], dist)
        
        # Select candidate with maximum min-distance to selected set
        min_dists_to_selected[selected_indices] = -1  # Exclude already selected
        next_idx = int(np.argmax(min_dists_to_selected))
        selected_indices.append(next_idx)
    
    return selected_indices


# ============================================================
# P6: Stratification / Quotas
# ============================================================

@dataclass
class StratificationBins:
    """Define bins for stratified sampling."""
    topology_bins: List[str] = field(default_factory=lambda: ["small", "medium", "large", "very_large"])
    stress_bins: List[str] = field(default_factory=lambda: ["low", "medium_low", "medium", "medium_high", "high"])
    flexibility_bins: List[str] = field(default_factory=lambda: ["low", "medium", "high", "very_high"])
    hardness_bins: List[str] = field(default_factory=lambda: ["easy", "medium", "hard", "very_hard"])


def compute_bin_assignment(cfg: ScenarioConfig, n_binary: int, space: Dict[str, Any]) -> Dict[str, str]:
    """
    Assign a scenario to stratification bins.
    
    Returns dict with keys: topology, stress, flexibility, hardness, weather, demand, cross_border
    """
    zones = sum(cfg.graph.zones_per_region)
    counts = estimate_assets_count(cfg.assets)
    
    # Topology bin based on zone count quartiles
    if zones <= 50:
        topology = "small"
    elif zones <= 120:
        topology = "medium"
    elif zones <= 250:
        topology = "large"
    else:
        topology = "very_large"
    
    # Stress bin based on demand scale factor quintiles
    dsf = cfg.exogenous.demand_scale_factor
    dsf_range = space["exogenous"]["demand_scale_factor"]
    dsf_normalized = (dsf - dsf_range[0]) / (dsf_range[1] - dsf_range[0])
    if dsf_normalized < 0.2:
        stress = "low"
    elif dsf_normalized < 0.4:
        stress = "medium_low"
    elif dsf_normalized < 0.6:
        stress = "medium"
    elif dsf_normalized < 0.8:
        stress = "medium_high"
    else:
        stress = "high"
    
    # Flexibility bin based on storage power
    storage_count = counts["battery"] + counts["hydro_pumped"]
    if storage_count <= 5:
        flexibility = "low"
    elif storage_count <= 15:
        flexibility = "medium"
    elif storage_count <= 30:
        flexibility = "high"
    else:
        flexibility = "very_high"
    
    # Hardness bin based on binary variable count
    if n_binary < 5000:
        hardness = "easy"
    elif n_binary < 12000:
        hardness = "medium"
    elif n_binary < 18000:
        hardness = "hard"
    else:
        hardness = "very_hard"
    
    return {
        "topology": topology,
        "stress": stress,
        "flexibility": flexibility,
        "hardness": hardness,
        "weather": cfg.exogenous.weather_profile,
        "demand": cfg.exogenous.demand_profile,
        "cross_border": cfg.econ_policy.cross_border_policy,
    }


def compute_bin_quotas(target: int, space: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """
    Compute target quotas for each bin to ensure balanced sampling.
    
    Uses uniform distribution across bins by default.
    """
    bins = StratificationBins()
    
    quotas = {
        "topology": {b: target // len(bins.topology_bins) for b in bins.topology_bins},
        "stress": {b: target // len(bins.stress_bins) for b in bins.stress_bins},
        "flexibility": {b: target // len(bins.flexibility_bins) for b in bins.flexibility_bins},
        "hardness": {b: target // len(bins.hardness_bins) for b in bins.hardness_bins},
    }
    
    return quotas


def select_with_stratification(candidates: List[Tuple[int, Dict[str, str]]], 
                                vectors: np.ndarray,
                                mins: np.ndarray, maxs: np.ndarray,
                                target: int, space: Dict[str, Any]) -> List[int]:
    """
    Select scenarios with stratification constraints using modified k-center.
    
    First fills quotas for each bin, then uses k-center for remaining slots.
    """
    quotas = compute_bin_quotas(target, space)
    bin_counts: Dict[str, Dict[str, int]] = {
        key: {b: 0 for b in bins} for key, bins in quotas.items()
    }
    
    selected_indices: List[int] = []
    remaining_candidates = list(range(len(candidates)))
    
    # Phase 1: Fill quotas for underrepresented bins
    for idx, (orig_idx, bins) in enumerate(candidates):
        if len(selected_indices) >= target:
            break
            
        # Check if this candidate helps fill any quota
        helps_quota = False
        for bin_type, bin_val in bins.items():
            if bin_type in quotas:
                if bin_counts[bin_type].get(bin_val, 0) < quotas[bin_type].get(bin_val, 0):
                    helps_quota = True
                    break
        
        if helps_quota:
            selected_indices.append(idx)
            remaining_candidates.remove(idx)
            for bin_type, bin_val in bins.items():
                if bin_type in bin_counts and bin_val in bin_counts[bin_type]:
                    bin_counts[bin_type][bin_val] += 1
    
    # Phase 2: Fill remaining slots with k-center on remaining candidates
    if len(selected_indices) < target and remaining_candidates:
        remaining_vectors = vectors[remaining_candidates]
        additional_needed = target - len(selected_indices)
        
        if len(remaining_candidates) <= additional_needed:
            selected_indices.extend(remaining_candidates)
        else:
            additional_indices = greedy_k_center_selection(
                remaining_vectors, mins, maxs, additional_needed
            )
            selected_indices.extend([remaining_candidates[i] for i in additional_indices])
    
    return selected_indices


# ============================================================
# P8: LHS-based continuous parameter sampling
# ============================================================

# Continuous parameters for LHS (from P8)
LHS_PARAMS = [
    ("demand_scale_factor", "exogenous"),
    ("inflow_factor", "exogenous"),
    ("co2_price_eur_per_t", "economics_policy"),
    ("intertie_density", "structure"),
    ("thermal_ramp_pct", "techno_params_scalers"),
    ("battery_e_to_p_hours", "techno_params_scalers"),
    ("import_export_caps_factor", "economics_policy"),
    ("weather_spread_intensity", "exogenous"),
]


def generate_lhs_param_grid(space: Dict[str, Any], n_samples: int, seed: int) -> np.ndarray:
    """
    Generate LHS samples for continuous parameters.
    
    Returns array of shape (n_samples, len(LHS_PARAMS))
    """
    ranges = []
    for param_name, section in LHS_PARAMS:
        bounds = space[section][param_name]
        ranges.append((bounds[0], bounds[1]))
    
    lhs_unit = latin_hypercube_sample(n_samples, len(LHS_PARAMS), seed)
    return map_lhs_to_ranges(lhs_unit, ranges)


# ============================================================
# Extended Samplers with LHS support
# ============================================================

def sample_graph_lhs(space: Dict[str, Any], lhs_row: np.ndarray) -> GraphSpec:
    """Sample graph with LHS for intertie_density."""
    structure = space["structure"]
    regions = rand_int(*structure["regions"])
    zones_per_region = [rand_int(*structure["zones_per_region"]) for _ in range(regions)]
    
    # Get intertie_density from LHS
    intertie_idx = [p[0] for p in LHS_PARAMS].index("intertie_density")
    intertie_density = float(lhs_row[intertie_idx])
    
    return GraphSpec(
        regions=regions,
        zones_per_region=zones_per_region,
        intertie_density=intertie_density,
        neighbor_nations=rand_int(*structure["neighbor_nations"]),
    )


def sample_econ_lhs(space: Dict[str, Any], lhs_row: np.ndarray) -> EconPolicy:
    """Sample economics with LHS for co2_price and import_export_caps_factor."""
    econ_cfg = space["economics_policy"]
    
    co2_idx = [p[0] for p in LHS_PARAMS].index("co2_price_eur_per_t")
    caps_idx = [p[0] for p in LHS_PARAMS].index("import_export_caps_factor")
    
    return EconPolicy(
        co2_price=float(lhs_row[co2_idx]),
        price_cap=rand_float(*econ_cfg["price_cap_eur_per_mwh"]),
        cross_border_policy=rand_choice(econ_cfg["cross_border_policy"]),
        import_export_caps_factor=float(lhs_row[caps_idx]),
    )


def sample_tech_lhs(space: Dict[str, Any], lhs_row: np.ndarray) -> TechScalers:
    """Sample tech scalers with LHS for thermal_ramp_pct and battery_e_to_p_hours."""
    tech_cfg = space["techno_params_scalers"]
    
    ramp_idx = [p[0] for p in LHS_PARAMS].index("thermal_ramp_pct")
    e2p_idx = [p[0] for p in LHS_PARAMS].index("battery_e_to_p_hours")
    
    return TechScalers(
        thermal_marg_cost=rand_float(*tech_cfg["thermal_marg_cost"]),
        thermal_ramp_pct=float(lhs_row[ramp_idx]),
        battery_roundtrip_eff=rand_float(*tech_cfg["battery_roundtrip_eff"]),
        battery_e_to_p_hours=float(lhs_row[e2p_idx]),
        dr_max_shed_share=rand_float(*tech_cfg["dr_max_shed_share"]),
        dr_duration_hours=rand_float(*tech_cfg["dr_duration_hours"]),
        hydro_reservoir_head_eff=rand_float(*tech_cfg["hydro_reservoir_head_eff"]),
        battery_initial_soc_fraction=rand_float(*tech_cfg["battery_initial_soc_fraction"]),
        battery_final_soc_tolerance=rand_float(*tech_cfg["battery_final_soc_tolerance"]),
        battery_self_discharge_per_hour=rand_float(*tech_cfg["battery_self_discharge_per_hour"]),
        pumped_initial_level_fraction=rand_float(*tech_cfg["pumped_initial_level_fraction"]),
        pumped_final_level_tolerance=rand_float(*tech_cfg["pumped_final_level_tolerance"]),
        pumped_self_discharge_per_hour=rand_float(*tech_cfg["pumped_self_discharge_per_hour"]),
    )


def sample_exogenous_lhs(space: Dict[str, Any], graph: GraphSpec, lhs_row: np.ndarray) -> ExogenousSpec:
    """Sample exogenous with LHS for demand_scale_factor, inflow_factor, weather_spread_intensity."""
    exo_cfg = space["exogenous"]
    variants = normalize_weights(exo_cfg.get("zone_profile_variants"))
    mix_range = resolve_range(exo_cfg.get("zone_profile_mix_weight"), (0.4, 0.8), lo=0.0, hi=1.0, min_span=0.05)
    phase_range = resolve_range(exo_cfg.get("zone_profile_phase_shift_hours"), (-2.5, 2.5))
    noise_range = resolve_range(exo_cfg.get("zone_profile_noise_std"), (0.02, 0.06), lo=0.0, min_span=0.003)
    curvature_range = resolve_range(exo_cfg.get("zone_profile_curvature_exp"), (0.9, 1.18), lo=0.1, min_span=0.01)

    weather_choices = exo_cfg["weather_profiles"]
    
    # LHS indices
    dsf_idx = [p[0] for p in LHS_PARAMS].index("demand_scale_factor")
    inflow_idx = [p[0] for p in LHS_PARAMS].index("inflow_factor")
    spread_idx = [p[0] for p in LHS_PARAMS].index("weather_spread_intensity")
    
    base_spread = float(lhs_row[spread_idx])
    
    # Sample regional weather with heterogeneity preserved
    region_weather: Dict[str, RegionWeatherCell] = {}
    region_profiles: List[str] = []
    for ridx in range(graph.regions):
        region_name = f"R{ridx + 1}"
        profile = rand_choice(weather_choices)
        # Add variation around LHS spread value
        spread = base_spread * rand_float(0.8, 1.2)
        spread = max(exo_cfg["weather_spread_intensity"][0], 
                     min(exo_cfg["weather_spread_intensity"][1], spread))
        region_weather[region_name] = RegionWeatherCell(
            weather_profile=profile,
            weather_spread_intensity=spread,
        )
        region_profiles.append(profile)

    # Determine dominant profile but preserve heterogeneity in region_weather
    if region_profiles:
        profile_counts: Dict[str, int] = {}
        for profile in region_profiles:
            profile_counts[profile] = profile_counts.get(profile, 0) + 1
        dominant_profile = max(profile_counts.items(), key=lambda item: (item[1], item[0]))[0]
    else:
        dominant_profile = rand_choice(weather_choices)

    return ExogenousSpec(
        weather_profile=dominant_profile,
        weather_spread_intensity=base_spread,
        demand_profile=rand_choice(exo_cfg["demand_profiles"]),
        demand_scale_factor=float(lhs_row[dsf_idx]),
        inflow_factor=float(lhs_row[inflow_idx]),
        zone_profile_variants=variants,
        zone_profile_mix_weight=mix_range,
        zone_profile_phase_shift_hours=phase_range,
        zone_profile_noise_std=noise_range,
        zone_profile_curvature_exp=curvature_range,
        region_weather=region_weather,
    )


def sample_dr_scalers(space: Dict[str, Any]) -> DRScalers:
    """Sample extended DR scalers."""
    tech_cfg = space["techno_params_scalers"]
    return DRScalers(
        dr_max_shed_share=rand_float(*tech_cfg["dr_max_shed_share"]),
        dr_duration_hours=rand_float(*tech_cfg["dr_duration_hours"]),
        dr_num_blocks=rand_int(*tech_cfg.get("dr_num_blocks", [2, 4])),
        dr_rebound_decay=rand_float(*tech_cfg.get("dr_rebound_decay", [0.01, 0.15])),
        dr_rebound_tolerance=rand_float(*tech_cfg.get("dr_rebound_tolerance", [0.1, 5])),
        dr_max_events=rand_int(*tech_cfg.get("dr_max_events", [2, 10])),
        dr_min_duration=rand_int(*tech_cfg.get("dr_min_duration", [1, 3])),
        dr_ramp_limit_factor=rand_float(*tech_cfg.get("dr_ramp_limit_factor", [0.25, 1])),
    )


# Import remaining samplers from v1
from .generator_v1 import (
    sample_assets,
    sample_operation_costs,
    sample_unit_capacities,
    sample_transport_capacities,
    sample_mip_gap,
)


# ============================================================
# Main Generator V2: Pool + K-Center with Stratification
# ============================================================

def generate_scenarios_v2(space_path: str, out_dir: str, 
                           pool_multiplier: int = 20,
                           use_lhs: bool = True,
                           use_stratification: bool = True) -> Dict[str, Any]:
    """
    Generate diverse MILP scenarios using optimized diversity methods.
    
    Args:
        space_path: Path to scenario_space.yaml
        out_dir: Output directory for scenarios
        pool_multiplier: Generate pool_multiplier * target candidates before selection
        use_lhs: Use Latin Hypercube Sampling for continuous params
        use_stratification: Use stratified selection with quotas
        
    Returns:
        Manifest dict with generation statistics
    """
    space = load_space(space_path)
    seed = space["global"]["seed"]
    set_seed(seed)
    
    output_dir = pathlib.Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target = space["global"]["target_scenarios"]
    pool_size = target * pool_multiplier
    
    print(f"[V2] Generating candidate pool of {pool_size} scenarios...")
    
    # Pre-generate LHS samples if enabled
    if use_lhs:
        lhs_grid = generate_lhs_param_grid(space, pool_size, seed)
    else:
        lhs_grid = None
    
    # Compute diversity bounds
    mins, maxs = compute_diversity_bounds(space)
    
    # Phase 1: Generate candidate pool passing budget guard
    candidates: List[Tuple[ScenarioConfig, DRScalers, np.ndarray, int, int, int, float, Dict[str, str]]] = []
    attempts = 0
    max_attempts = pool_size * 5
    
    while len(candidates) < pool_size and attempts < max_attempts:
        attempts += 1
        
        # Sample scenario
        if use_lhs and lhs_grid is not None:
            lhs_idx = len(candidates) % len(lhs_grid)
            lhs_row = lhs_grid[lhs_idx]
            graph = sample_graph_lhs(space, lhs_row)
            econ = sample_econ_lhs(space, lhs_row)
            tech = sample_tech_lhs(space, lhs_row)
            exo = sample_exogenous_lhs(space, graph, lhs_row)
        else:
            from .generator_v1 import sample_graph, sample_econ, sample_tech, sample_exogenous
            graph = sample_graph(space)
            econ = sample_econ(space)
            tech = sample_tech(space)
            exo = sample_exogenous(space, graph)
        
        assets = sample_assets(space, graph)
        costs = sample_operation_costs(space)
        unit_caps = sample_unit_capacities(space)
        transport_caps = sample_transport_capacities(space)
        mip_gap = sample_mip_gap(space)
        dr_scalers = sample_dr_scalers(space)

        cfg = ScenarioConfig(
            id=str(uuid.uuid4()),
            horizon_hours=space["global"]["horizon_hours"],
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

        # Check budget guard
        vars_total, cons_total, n_binary = estimate_milp_size(cfg)
        est_hours = estimate_solve_time_hours(vars_total, cons_total, n_binary, space)
        
        if not passes_budget_guard(vars_total, cons_total, n_binary, est_hours, space):
            continue
        
        # Compute diversity vector
        div_vec = enriched_diversity_vector(cfg, space, dr_scalers)
        
        # Compute bin assignment for stratification
        bins = compute_bin_assignment(cfg, n_binary, space)
        
        candidates.append((cfg, dr_scalers, div_vec, vars_total, cons_total, n_binary, est_hours, bins))
    
    print(f"[V2] Generated {len(candidates)} valid candidates from {attempts} attempts")
    
    if len(candidates) == 0:
        print("[V2] ERROR: No valid candidates generated!")
        return {"count": 0, "target": target, "error": "No valid candidates"}
    
    # Phase 2: Select diverse subset using k-center with optional stratification
    print(f"[V2] Selecting {target} diverse scenarios...")
    
    vectors = np.array([c[2] for c in candidates])
    bin_assignments = [(i, c[7]) for i, c in enumerate(candidates)]
    
    if use_stratification:
        selected_indices = select_with_stratification(
            bin_assignments, vectors, mins, maxs, target, space
        )
    else:
        selected_indices = greedy_k_center_selection(vectors, mins, maxs, target, seed)
    
    print(f"[V2] Selected {len(selected_indices)} scenarios")
    
    # Phase 3: Write selected scenarios to disk
    for rank, idx in enumerate(selected_indices):
        cfg, dr_scalers, div_vec, vars_total, cons_total, n_binary, est_hours, bins = candidates[idx]
        
        meta = build_meta(cfg, vars_total, cons_total, est_hours)
        flexibility_metrics = compute_flexibility_metrics(cfg)
        difficulty_indicators = compute_difficulty_indicators(cfg, vars_total, cons_total, est_hours)
        
        # Add stratification bins to meta
        meta["stratification_bins"] = bins
        
        # Add DR scalers to meta
        meta["dr_scalers"] = asdict(dr_scalers)
        
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

        cfg_path = output_dir / f"scenario_{rank + 1:05d}.json"
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # Build manifest with statistics
    manifest: Dict[str, Any] = {
        "count": len(selected_indices),
        "target": target,
        "pool_size": len(candidates),
        "attempts": attempts,
        "method": "pool_k_center_v2",
        "use_lhs": use_lhs,
        "use_stratification": use_stratification,
        "T": int(space["global"]["horizon_hours"] * 60 / space["global"]["dt_minutes"]),
    }

    if selected_indices:
        selected_cfgs = [candidates[i][0] for i in selected_indices]
        total_regions = [cfg.graph.regions for cfg in selected_cfgs]
        total_zones = [sum(cfg.graph.zones_per_region) for cfg in selected_cfgs]
        total_vars = [candidates[i][3] for i in selected_indices]
        total_cons = [candidates[i][4] for i in selected_indices]
        total_hours = [candidates[i][6] for i in selected_indices]
        
        manifest["stats"] = {
            "avg_regions": round(float(np.mean(total_regions)), 2),
            "avg_zones": round(float(np.mean(total_zones)), 2),
            "avg_vars_est": round(float(np.mean(total_vars)), 0),
            "avg_cons_est": round(float(np.mean(total_cons)), 0),
            "avg_est_cpu_hours": round(float(np.mean(total_hours)), 2),
            "min_zones": int(np.min(total_zones)),
            "max_zones": int(np.max(total_zones)),
        }
        
        # Bin distribution stats
        bin_dist: Dict[str, Dict[str, int]] = {}
        for idx in selected_indices:
            bins = candidates[idx][7]
            for key, val in bins.items():
                if key not in bin_dist:
                    bin_dist[key] = {}
                bin_dist[key][val] = bin_dist[key].get(val, 0) + 1
        manifest["bin_distribution"] = bin_dist

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
