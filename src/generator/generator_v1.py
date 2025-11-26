from __future__ import annotations

import json
import random
import uuid
import pathlib
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Tuple

import numpy as np
import yaml

DEFAULT_ZONE_PROFILE_VARIANTS = {
    "evening_peak": 0.28,
    "double_peak": 0.16,
    "office_peak": 0.20,
    "industrial": 0.14,
    "night_shift": 0.10,
    "residential_morning": 0.12,
}

# ---------- Utilities

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def rand_int(lo: int, hi: int) -> int:
    return int(random.randint(lo, hi))


def rand_float(lo: float, hi: float) -> float:
    return float(random.uniform(lo, hi))


def rand_choice(seq: List[Any]) -> Any:
    return random.choice(seq)


def one_hot(value: str, choices: List[str]) -> np.ndarray:
    vec = np.zeros(len(choices), dtype=float)
    vec[choices.index(value)] = 1.0
    return vec


def resolve_range(
    raw: Any,
    default: Tuple[float, float],
    *,
    lo: float | None = None,
    hi: float | None = None,
    min_span: float | None = None,
) -> Tuple[float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        lower, upper = float(raw[0]), float(raw[1])
    else:
        lower, upper = default
    if lower > upper:
        lower, upper = upper, lower
    if lo is not None:
        lower, upper = max(lo, lower), max(lo, upper)
    if hi is not None:
        lower, upper = min(hi, lower), min(hi, upper)
    if min_span is not None and abs(upper - lower) < min_span:
        centre = 0.5 * (lower + upper)
        lower, upper = centre - 0.5 * min_span, centre + 0.5 * min_span
    if lower > upper:
        lower, upper = upper, lower
    return lower, upper


def normalize_weights(raw: Dict[str, Any] | None) -> Dict[str, float]:
    if not raw:
        weights = dict(DEFAULT_ZONE_PROFILE_VARIANTS)
    else:
        weights = {
            str(k): float(v)
            for k, v in raw.items()
            if isinstance(v, (int, float)) and float(v) > 0.0
        }
        if not weights:
            weights = dict(DEFAULT_ZONE_PROFILE_VARIANTS)
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def load_space(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---------- Scenario spec dataclasses

@dataclass
class GraphSpec:
    regions: int
    zones_per_region: List[int]
    sites_per_zone: List[int]
    intertie_density: float
    neighbor_nations: int


@dataclass
class AssetSpec:
    thermal_per_site: List[int]
    solar_per_site: List[int]
    wind_per_site: List[int]
    battery_per_site: List[int]
    dr_per_site: List[int]
    nuclear_per_region: List[int]
    hydro_reservoir_per_region: List[int]
    hydro_ror_per_zone: List[int]
    hydro_pumped_per_region: List[int]


@dataclass
class EconPolicy:
    co2_price: float
    price_cap: float
    cross_border_policy: str
    import_export_caps_factor: float


@dataclass
class TechScalers:
    thermal_marg_cost: float
    thermal_ramp_pct: float
    battery_roundtrip_eff: float
    battery_e_to_p_hours: float
    dr_max_shed_share: float
    dr_duration_hours: float
    hydro_reservoir_head_eff: float
    battery_initial_soc_fraction: float
    battery_final_soc_tolerance: float
    battery_self_discharge_per_hour: float
    pumped_initial_level_fraction: float
    pumped_final_level_tolerance: float
    pumped_self_discharge_per_hour: float


@dataclass
class OperationCosts:
    thermal_fuel_eur_per_mwh: float
    nuclear_fuel_eur_per_mwh: float
    demand_response_cost_eur_per_mwh: float
    value_of_lost_load_eur_per_mwh: float
    renewable_spill_cost_eur_per_mwh: float
    hydro_spill_cost_eur_per_mwh: float
    overgen_spill_cost_eur_per_mwh: float
    battery_cycle_cost_eur_per_mwh: float
    pumped_cycle_cost_eur_per_mwh: float


@dataclass
class RegionWeatherCell:
    weather_profile: str
    weather_spread_intensity: float


@dataclass
class ExogenousSpec:
    weather_profile: str
    weather_spread_intensity: float
    demand_profile: str
    demand_scale_factor: float
    inflow_factor: float
    zone_profile_variants: Dict[str, float]
    zone_profile_mix_weight: Tuple[float, float]
    zone_profile_phase_shift_hours: Tuple[float, float]
    zone_profile_noise_std: Tuple[float, float]
    zone_profile_curvature_exp: Tuple[float, float]
    region_weather: Dict[str, RegionWeatherCell] = field(default_factory=dict)


@dataclass
class ScenarioConfig:
    id: str
    horizon_hours: int
    dt_minutes: int
    graph: GraphSpec
    assets: AssetSpec
    econ_policy: EconPolicy
    tech: TechScalers
    costs: OperationCosts
    exogenous: ExogenousSpec
    mip_gap_target_pct: float

# ---------- Samplers

def sample_graph(space: Dict[str, Any]) -> GraphSpec:
    structure = space["structure"]
    regions = rand_int(*structure["regions"])
    zones_per_region = [rand_int(*structure["zones_per_region"]) for _ in range(regions)]
    sites_per_zone: List[int] = []
    for count in zones_per_region:
        sites_per_zone.extend(rand_int(*structure["sites_per_zone"]) for _ in range(count))
    return GraphSpec(
        regions=regions,
        zones_per_region=zones_per_region,
        sites_per_zone=sites_per_zone,
        intertie_density=rand_float(*structure["intertie_density"]),
        neighbor_nations=rand_int(*structure["neighbor_nations"]),
    )


def sample_assets(space: Dict[str, Any], graph: GraphSpec) -> AssetSpec:
    assets_cfg = space["assets"]
    ns = len(graph.sites_per_zone)
    nz = sum(graph.zones_per_region)

    per_site = lambda bounds: [rand_int(*bounds) for _ in range(ns)]
    per_zone = lambda bounds: [rand_int(*bounds) for _ in range(nz)]
    per_region = lambda bounds: [rand_int(*bounds) for _ in range(len(graph.zones_per_region))]

    return AssetSpec(
        thermal_per_site=per_site(assets_cfg["thermal_per_site"]),
        solar_per_site=per_site(assets_cfg["solar_per_site"]),
        wind_per_site=per_site(assets_cfg["wind_per_site"]),
        battery_per_site=per_site(assets_cfg["battery_per_site"]),
        dr_per_site=per_site(assets_cfg["dr_per_site"]),
        nuclear_per_region=per_region(assets_cfg["nuclear_per_region"]),
        hydro_reservoir_per_region=per_region(assets_cfg["hydro_reservoir_per_region"]),
        hydro_ror_per_zone=per_zone(assets_cfg["hydro_ror_per_zone"]),
        hydro_pumped_per_region=per_region(assets_cfg["hydro_pumped_per_region"]),
    )


def sample_econ(space: Dict[str, Any]) -> EconPolicy:
    econ_cfg = space["economics_policy"]
    return EconPolicy(
        co2_price=rand_float(*econ_cfg["co2_price_eur_per_t"]),
        price_cap=rand_float(*econ_cfg["price_cap_eur_per_mwh"]),
        cross_border_policy=rand_choice(econ_cfg["cross_border_policy"]),
        import_export_caps_factor=rand_float(*econ_cfg["import_export_caps_factor"]),
    )


def sample_tech(space: Dict[str, Any]) -> TechScalers:
    tech_cfg = space["techno_params_scalers"]
    return TechScalers(
        thermal_marg_cost=rand_float(*tech_cfg["thermal_marg_cost"]),
        thermal_ramp_pct=rand_float(*tech_cfg["thermal_ramp_pct"]),
        battery_roundtrip_eff=rand_float(*tech_cfg["battery_roundtrip_eff"]),
        battery_e_to_p_hours=rand_float(*tech_cfg["battery_e_to_p_hours"]),
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


def sample_operation_costs(space: Dict[str, Any]) -> OperationCosts:
    cost_cfg = space["operation_costs"]
    return OperationCosts(
        thermal_fuel_eur_per_mwh=rand_float(*cost_cfg["thermal_fuel_eur_per_mwh"]),
        nuclear_fuel_eur_per_mwh=rand_float(*cost_cfg["nuclear_fuel_eur_per_mwh"]),
        demand_response_cost_eur_per_mwh=rand_float(*cost_cfg["demand_response_cost_eur_per_mwh"]),
        value_of_lost_load_eur_per_mwh=rand_float(*cost_cfg["value_of_lost_load_eur_per_mwh"]),
        renewable_spill_cost_eur_per_mwh=rand_float(*cost_cfg["renewable_spill_cost_eur_per_mwh"]),
        hydro_spill_cost_eur_per_mwh=rand_float(*cost_cfg["hydro_spill_cost_eur_per_mwh"]),
        overgen_spill_cost_eur_per_mwh=rand_float(*cost_cfg["overgen_spill_cost_eur_per_mwh"]),
        battery_cycle_cost_eur_per_mwh=rand_float(*cost_cfg["battery_cycle_cost_eur_per_mwh"]),
        pumped_cycle_cost_eur_per_mwh=rand_float(*cost_cfg["pumped_cycle_cost_eur_per_mwh"]),
    )


def sample_exogenous(space: Dict[str, Any], graph: GraphSpec) -> ExogenousSpec:
    exo_cfg = space["exogenous"]
    variants = normalize_weights(exo_cfg.get("zone_profile_variants"))
    mix_range = resolve_range(exo_cfg.get("zone_profile_mix_weight"), (0.4, 0.8), lo=0.0, hi=1.0, min_span=0.05)
    phase_range = resolve_range(exo_cfg.get("zone_profile_phase_shift_hours"), (-2.5, 2.5))
    noise_range = resolve_range(exo_cfg.get("zone_profile_noise_std"), (0.02, 0.06), lo=0.0, min_span=0.003)
    curvature_range = resolve_range(exo_cfg.get("zone_profile_curvature_exp"), (0.9, 1.18), lo=0.1, min_span=0.01)

    weather_choices = exo_cfg["weather_profiles"]
    spread_bounds = exo_cfg["weather_spread_intensity"]
    region_weather: Dict[str, RegionWeatherCell] = {}
    region_profiles: List[str] = []
    region_spreads: List[float] = []
    for ridx in range(graph.regions):
        region_name = f"R{ridx + 1}"
        profile = rand_choice(weather_choices)
        spread = rand_float(*spread_bounds)
        region_weather[region_name] = RegionWeatherCell(
            weather_profile=profile,
            weather_spread_intensity=spread,
        )
        region_profiles.append(profile)
        region_spreads.append(spread)

    if region_profiles:
        profile_counts: Dict[str, int] = {}
        for profile in region_profiles:
            profile_counts[profile] = profile_counts.get(profile, 0) + 1
        dominant_profile = max(profile_counts.items(), key=lambda item: (item[1], item[0]))[0]
        avg_spread = float(sum(region_spreads) / len(region_spreads))
        avg_spread = max(spread_bounds[0], min(spread_bounds[1], avg_spread))
    else:
        dominant_profile = rand_choice(weather_choices)
        avg_spread = rand_float(*spread_bounds)

    return ExogenousSpec(
        weather_profile=dominant_profile,
        weather_spread_intensity=avg_spread,
        demand_profile=rand_choice(exo_cfg["demand_profiles"]),
        demand_scale_factor=rand_float(*exo_cfg["demand_scale_factor"]),
        inflow_factor=rand_float(*exo_cfg["inflow_factor"]),
        zone_profile_variants=variants,
        zone_profile_mix_weight=mix_range,
        zone_profile_phase_shift_hours=phase_range,
        zone_profile_noise_std=noise_range,
        zone_profile_curvature_exp=curvature_range,
        region_weather=region_weather,
    )


def sample_mip_gap(space: Dict[str, Any]) -> float:
    lo, hi = space["budget_guard"]["mip_gap_target"]
    return rand_float(lo, hi)

# ---------- Complexity estimators

def estimate_assets_count(assets: AssetSpec) -> Dict[str, int]:
    return {
        "thermal": sum(assets.thermal_per_site),
        "solar": sum(assets.solar_per_site),
        "wind": sum(assets.wind_per_site),
        "battery": sum(assets.battery_per_site),
        "dr": sum(assets.dr_per_site),
        "nuclear": sum(assets.nuclear_per_region),
        "hydro_reservoir": sum(assets.hydro_reservoir_per_region),
        "hydro_ror": sum(assets.hydro_ror_per_zone),
        "hydro_pumped": sum(assets.hydro_pumped_per_region),
    }


def estimate_milp_size(cfg: ScenarioConfig) -> Tuple[int, int]:
    T = int(cfg.horizon_hours * 60 / cfg.dt_minutes)
    counts = estimate_assets_count(cfg.assets)

    vars_per_step = (
        4 * counts["thermal"]
        + 2 * counts["solar"]
        + 2 * counts["wind"]
        + 3 * counts["battery"]
        + 3 * counts["dr"]
        + 4 * counts["nuclear"]
        + 3 * counts["hydro_reservoir"]
        + 2 * counts["hydro_ror"]
        + 3 * counts["hydro_pumped"]
    )
    zones = sum(cfg.graph.zones_per_region)
    sites = len(cfg.graph.sites_per_zone)
    overhead = 20 * zones + 8 * sites

    vars_total = T * (vars_per_step + overhead)
    cons_total = int(vars_total * 1.3)
    return vars_total, cons_total


def estimate_solve_time_hours(vars_total: int, cons_total: int, space: Dict[str, Any]) -> float:
    te = space["budget_guard"]["time_estimator"]
    sec = (
        te["base_intercept_sec"]
        + te["per_1k_vars_sec"] * (vars_total / 1000.0)
        + te["per_1k_cons_sec"] * (cons_total / 1000.0)
    )
    sec *= te.get("branching_penalty", 1.0)
    return sec / 3600.0


def passes_budget_guard(vars_total: int, cons_total: int, est_hours: float, space: Dict[str, Any]) -> bool:
    guard = space["budget_guard"]
    if vars_total > guard["max_vars_per_scenario"]:
        return False
    if cons_total > guard["max_cons_per_scenario"]:
        return False
    if est_hours > guard["reject_if_est_cpu_hours_gt"]:
        return False
    return True

# ---------- Diversity metrics

def scenario_meta_vector(cfg: ScenarioConfig, space: Dict[str, Any]) -> np.ndarray:
    vector: List[float] = []
    vector.append(float(cfg.graph.regions))
    vector.append(float(np.mean(cfg.graph.zones_per_region)))
    vector.append(float(cfg.graph.intertie_density))
    vector.append(float(cfg.econ_policy.co2_price))
    vector.append(float(cfg.econ_policy.price_cap))
    vector.append(float(cfg.exogenous.demand_scale_factor))
    vector.append(float(np.mean(cfg.assets.solar_per_site)))
    vector.append(float(np.mean(cfg.assets.wind_per_site)))
    vector.extend(cfg.exogenous.zone_profile_mix_weight)
    vector.append(float(cfg.costs.thermal_fuel_eur_per_mwh))
    vector.append(float(cfg.costs.nuclear_fuel_eur_per_mwh))
    vector.append(float(cfg.costs.demand_response_cost_eur_per_mwh))
    vector.append(float(cfg.costs.value_of_lost_load_eur_per_mwh))
    vector.append(float(cfg.costs.renewable_spill_cost_eur_per_mwh))
    vector.append(float(cfg.costs.hydro_spill_cost_eur_per_mwh))
    vector.append(float(cfg.costs.overgen_spill_cost_eur_per_mwh))
    vector.extend(one_hot(cfg.exogenous.weather_profile, space["exogenous"]["weather_profiles"]))
    vector.extend(one_hot(cfg.econ_policy.cross_border_policy, space["economics_policy"]["cross_border_policy"]))
    return np.array(vector, dtype=float)


def normalized_distance(a: np.ndarray, b: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> float:
    denom = np.maximum(maxs - mins, 1e-9)
    return np.linalg.norm((a - b) / denom)


def compute_flexibility_metrics(cfg: ScenarioConfig) -> Dict[str, Any]:
    """
    Compute flexibility metrics for a scenario.
    
    Estimates total storage capacity, DR capacity, ramp capability, etc.
    These help categorize scenarios by their operational flexibility.
    """
    counts = estimate_assets_count(cfg.assets)
    zones = sum(cfg.graph.zones_per_region)
    
    # Storage estimates (batteries + pumped hydro)
    # Assume typical capacities based on realistic power system data
    avg_battery_power_mw = 50.0  # MW per battery unit
    avg_battery_energy_mwh = avg_battery_power_mw * cfg.tech.battery_e_to_p_hours
    avg_pumped_power_mw = 200.0  # MW per pumped hydro unit
    avg_pumped_energy_mwh = avg_pumped_power_mw * 8.0  # ~8 hours typical
    
    total_battery_power = counts["battery"] * avg_battery_power_mw
    total_battery_energy = counts["battery"] * avg_battery_energy_mwh
    total_pumped_power = counts["hydro_pumped"] * avg_pumped_power_mw
    total_pumped_energy = counts["hydro_pumped"] * avg_pumped_energy_mwh
    
    total_storage_power_mw = total_battery_power + total_pumped_power
    total_storage_capacity_mwh = total_battery_energy + total_pumped_energy
    
    # Demand response estimates
    # DR is per zone, estimate based on typical load
    avg_zone_demand_mw = 500.0 * cfg.exogenous.demand_scale_factor
    total_dr_capacity_mw = zones * avg_zone_demand_mw * cfg.tech.dr_max_shed_share
    
    # Ramping capability (thermal + nuclear)
    avg_thermal_capacity_mw = 300.0  # MW per thermal unit
    avg_nuclear_capacity_mw = 1000.0  # MW per nuclear unit
    thermal_total_capacity = counts["thermal"] * avg_thermal_capacity_mw
    nuclear_total_capacity = counts["nuclear"] * avg_nuclear_capacity_mw
    
    # Ramp rates per timestep (dt_minutes)
    dt_hours = cfg.dt_minutes / 60.0
    thermal_ramp_per_step = thermal_total_capacity * cfg.tech.thermal_ramp_pct * dt_hours
    nuclear_ramp_per_step = nuclear_total_capacity * 0.05 * dt_hours  # Nuclear ramps slowly
    
    total_ramp_up_mw_per_step = thermal_ramp_per_step + nuclear_ramp_per_step
    total_ramp_down_mw_per_step = total_ramp_up_mw_per_step  # Symmetric
    
    # Flexibility ratio: storage + DR relative to total dispatchable capacity
    dispatchable_capacity = thermal_total_capacity + nuclear_total_capacity
    flex_capacity = total_storage_power_mw + total_dr_capacity_mw
    thermal_flex_ratio = flex_capacity / dispatchable_capacity if dispatchable_capacity > 0 else 0.0
    
    return {
        "total_storage_power_mw": round(total_storage_power_mw, 1),
        "total_storage_capacity_mwh": round(total_storage_capacity_mwh, 1),
        "battery_power_mw": round(total_battery_power, 1),
        "battery_capacity_mwh": round(total_battery_energy, 1),
        "pumped_power_mw": round(total_pumped_power, 1),
        "pumped_capacity_mwh": round(total_pumped_energy, 1),
        "total_dr_capacity_mw": round(total_dr_capacity_mw, 1),
        "total_ramp_up_mw_per_step": round(total_ramp_up_mw_per_step, 1),
        "total_ramp_down_mw_per_step": round(total_ramp_down_mw_per_step, 1),
        "thermal_flex_ratio": round(thermal_flex_ratio, 3),
        "dr_duration_hours": round(cfg.tech.dr_duration_hours, 2),
        "battery_roundtrip_efficiency": round(cfg.tech.battery_roundtrip_eff, 3),
    }


def compute_difficulty_indicators(cfg: ScenarioConfig, vars_total: int, cons_total: int, est_hours: float) -> Dict[str, Any]:
    """
    Compute difficulty indicators for a scenario.
    
    Estimates VRE penetration, demand volatility, problem complexity, etc.
    These help categorize scenarios by expected solver difficulty.
    """
    counts = estimate_assets_count(cfg.assets)
    zones = sum(cfg.graph.zones_per_region)
    
    # VRE penetration estimate
    avg_solar_capacity_mw = 100.0  # MW per solar site
    avg_wind_capacity_mw = 150.0  # MW per wind site
    avg_thermal_capacity_mw = 300.0
    avg_nuclear_capacity_mw = 1000.0
    
    vre_capacity = counts["solar"] * avg_solar_capacity_mw + counts["wind"] * avg_wind_capacity_mw
    thermal_capacity = counts["thermal"] * avg_thermal_capacity_mw + counts["nuclear"] * avg_nuclear_capacity_mw
    hydro_capacity = counts["hydro_reservoir"] * 500.0 + counts["hydro_ror"] * 200.0
    
    total_capacity = vre_capacity + thermal_capacity + hydro_capacity
    vre_penetration_pct = (vre_capacity / total_capacity * 100.0) if total_capacity > 0 else 0.0
    
    # Net demand volatility estimate (based on weather/demand profile types)
    # Stormy/sunny have high variability, calm/overcast are moderate
    weather_volatility_map = {
        "calm_winter": 0.12,
        "stormy_winter": 0.25,
        "sunny_summer": 0.22,
        "overcast_summer": 0.10,
        "mixed": 0.18,
    }
    base_volatility = weather_volatility_map.get(cfg.exogenous.weather_profile, 0.15)
    
    # Demand profile volatility
    demand_volatility_map = {
        "wkday_peak": 0.15,
        "wkend_flat": 0.08,
        "cold_snap": 0.20,
        "heatwave": 0.22,
        "shoulder": 0.10,
    }
    demand_volatility = demand_volatility_map.get(cfg.exogenous.demand_profile, 0.15)
    
    # Combined volatility
    net_demand_volatility = round(np.sqrt(base_volatility**2 + demand_volatility**2), 3)
    
    # Peak to valley ratio estimate (typical power system)
    # Higher demand scale and volatile weather → higher ratio
    base_peak_valley = 1.8
    peak_to_valley_ratio = round(base_peak_valley * cfg.exogenous.demand_scale_factor, 2)
    
    # Complexity score based on problem size and structure
    # Categories: trivial, easy, medium, hard, very_hard
    if vars_total < 10000:
        complexity_score = "trivial"
    elif vars_total < 30000:
        complexity_score = "easy"
    elif vars_total < 70000:
        complexity_score = "medium"
    elif vars_total < 120000:
        complexity_score = "hard"
    else:
        complexity_score = "very_hard"
    
    # Additional indicators
    n_binary_vars = counts["thermal"] * int(cfg.horizon_hours * 60 / cfg.dt_minutes)
    n_timesteps = int(cfg.horizon_hours * 60 / cfg.dt_minutes)
    
    return {
        "vre_penetration_pct": round(vre_penetration_pct, 2),
        "net_demand_volatility": net_demand_volatility,
        "peak_to_valley_ratio": peak_to_valley_ratio,
        "estimated_milp_solve_time_seconds": round(est_hours * 3600, 1),
        "complexity_score": complexity_score,
        "n_binary_variables": n_binary_vars,
        "n_timesteps": n_timesteps,
        "n_zones": zones,
        "weather_profile": cfg.exogenous.weather_profile,
        "demand_profile": cfg.exogenous.demand_profile,
    }

# ---------- Metadata helpers

def build_meta(cfg: ScenarioConfig, vars_total: int, cons_total: int, est_hours: float) -> Dict[str, Any]:
    asset_counts = estimate_assets_count(cfg.assets)
    zones = sum(cfg.graph.zones_per_region)
    sites = len(cfg.graph.sites_per_zone)
    return {
        "regions": cfg.graph.regions,
        "zones": zones,
        "sites": sites,
        "intertie_density": cfg.graph.intertie_density,
        "neighbor_nations": cfg.graph.neighbor_nations,
        "assets": asset_counts,
        "demand_scale_factor": cfg.exogenous.demand_scale_factor,
        "zone_profile_variants": cfg.exogenous.zone_profile_variants,
        "zone_profile_mix_weight": list(cfg.exogenous.zone_profile_mix_weight),
        "weather_profile": cfg.exogenous.weather_profile,
        "cross_border_policy": cfg.econ_policy.cross_border_policy,
        "price_cap": cfg.econ_policy.price_cap,
        "co2_price": cfg.econ_policy.co2_price,
        "import_export_caps_factor": cfg.econ_policy.import_export_caps_factor,
        "operation_costs": asdict(cfg.costs),
        "tech_scalers": asdict(cfg.tech),
        "estimates": {
            "vars_total": vars_total,
            "cons_total": cons_total,
            "est_cpu_hours": est_hours,
        },
    }

# ---------- Main generator

def generate_scenarios(space_path: str, out_dir: str):
    space = load_space(space_path)
    set_seed(space["global"]["seed"])
    output_dir = pathlib.Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target = space["global"]["target_scenarios"]

    accepted: List[ScenarioConfig] = []
    meta_vectors: List[np.ndarray] = []

    exo_cfg = space["exogenous"]
    mix_bounds = resolve_range(exo_cfg.get("zone_profile_mix_weight"), (0.4, 0.8), lo=0.0, hi=1.0, min_span=0.05)
    cost_cfg = space["operation_costs"]

    weather_choices = exo_cfg["weather_profiles"]
    cb_choices = space["economics_policy"]["cross_border_policy"]

    mins = np.array([
        space["structure"]["regions"][0],
        space["structure"]["zones_per_region"][0],
        space["structure"]["intertie_density"][0],
        space["economics_policy"]["co2_price_eur_per_t"][0],
        space["economics_policy"]["price_cap_eur_per_mwh"][0],
        exo_cfg["demand_scale_factor"][0],
        space["assets"]["solar_per_site"][0],
        space["assets"]["wind_per_site"][0],
        mix_bounds[0],
        mix_bounds[0],
        cost_cfg["thermal_fuel_eur_per_mwh"][0],
        cost_cfg["nuclear_fuel_eur_per_mwh"][0],
        cost_cfg["demand_response_cost_eur_per_mwh"][0],
        cost_cfg["value_of_lost_load_eur_per_mwh"][0],
        cost_cfg["renewable_spill_cost_eur_per_mwh"][0],
        cost_cfg["hydro_spill_cost_eur_per_mwh"][0],
        cost_cfg["overgen_spill_cost_eur_per_mwh"][0],
        *[0.0] * len(weather_choices),
        *[0.0] * len(cb_choices),
    ])

    maxs = np.array([
        space["structure"]["regions"][1],
        space["structure"]["zones_per_region"][1],
        space["structure"]["intertie_density"][1],
        space["economics_policy"]["co2_price_eur_per_t"][1],
        space["economics_policy"]["price_cap_eur_per_mwh"][1],
        exo_cfg["demand_scale_factor"][1],
        space["assets"]["solar_per_site"][1],
        space["assets"]["wind_per_site"][1],
        mix_bounds[1],
        mix_bounds[1],
        cost_cfg["thermal_fuel_eur_per_mwh"][1],
        cost_cfg["nuclear_fuel_eur_per_mwh"][1],
        cost_cfg["demand_response_cost_eur_per_mwh"][1],
        cost_cfg["value_of_lost_load_eur_per_mwh"][1],
        cost_cfg["renewable_spill_cost_eur_per_mwh"][1],
        cost_cfg["hydro_spill_cost_eur_per_mwh"][1],
        cost_cfg["overgen_spill_cost_eur_per_mwh"][1],
        *[1.0] * len(weather_choices),
        *[1.0] * len(cb_choices),
    ])

    attempts = 0
    max_attempts = target * 25

    while len(accepted) < target and attempts < max_attempts:
        attempts += 1

        graph = sample_graph(space)
        assets = sample_assets(space, graph)
        econ = sample_econ(space)
        tech = sample_tech(space)
        costs = sample_operation_costs(space)
        exo = sample_exogenous(space, graph)
        mip_gap = sample_mip_gap(space)

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
            mip_gap_target_pct=mip_gap,
        )

        vars_total, cons_total = estimate_milp_size(cfg)
        est_hours = estimate_solve_time_hours(vars_total, cons_total, space)
        if not passes_budget_guard(vars_total, cons_total, est_hours, space):
            continue

        if space["diversity"]["selection_method"] == "greedy_cover":
            vec = scenario_meta_vector(cfg, space)
            if meta_vectors:
                dists = [normalized_distance(vec, existing, mins, maxs) for existing in meta_vectors]
                if max(dists) < space["diversity"]["min_pairwise_distance"]:
                    continue
            meta_vectors.append(vec)

        accepted.append(cfg)

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

        cfg_path = output_dir / f"scenario_{len(accepted):05d}.json"
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    manifest: Dict[str, Any] = {
        "count": len(accepted),
        "target": target,
        "attempts": attempts,
        "T": int(space["global"]["horizon_hours"] * 60 / space["global"]["dt_minutes"]),
        "note": "Generated with greedy-cover & budget guard.",
    }

    if accepted:
        total_regions = [cfg.graph.regions for cfg in accepted]
        total_zones = [sum(cfg.graph.zones_per_region) for cfg in accepted]
        total_sites = [len(cfg.graph.sites_per_zone) for cfg in accepted]
        total_vars = []
        total_cons = []
        total_hours = []
        for cfg in accepted:
            v, c = estimate_milp_size(cfg)
            total_vars.append(v)
            total_cons.append(c)
            total_hours.append(estimate_solve_time_hours(v, c, space))
        manifest["stats"] = {
            "avg_regions": round(float(np.mean(total_regions)), 2),
            "avg_zones": round(float(np.mean(total_zones)), 2),
            "avg_sites": round(float(np.mean(total_sites)), 2),
            "avg_vars_est": round(float(np.mean(total_vars)), 0),
            "avg_cons_est": round(float(np.mean(total_cons)), 0),
            "avg_est_cpu_hours": round(float(np.mean(total_hours)), 2),
        }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
