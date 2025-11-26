from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from uuid import UUID

import numpy as np

from .defaults import (
    BATTERY,
    DEMAND_RESPONSE_COST,
    HYDRO_RES,
    HYDRO_ROR,
    IMPORT_BASE_CAPACITY,
    NUCLEAR,
    PUMPED,
    SOLAR,
    WIND,
    THERMAL,
    TRANS,
    VALUE_OF_LOST_LOAD,
)
from .profiles import (
    build_demand_profile,
    build_inflow_profile,
    build_runofriver_profile,
    build_solar_profile,
    build_wind_profile,
)


@dataclass(frozen=True)
class NetworkLine:
    name: str
    from_zone: str
    to_zone: str
    capacity_mw: float
    distance_km: float


@dataclass
class ScenarioData:
    scenario_id: str
    periods: List[int]
    dt_hours: float
    zones: List[str]
    region_of_zone: Dict[str, str]
    region_weather_profile: Dict[str, str]
    region_weather_spread: Dict[str, float]

    demand: Dict[Tuple[str, int], float]
    peak_demand: Dict[str, float]

    thermal_capacity: Dict[str, float]
    thermal_min_power: Dict[str, float]
    thermal_cost: Dict[str, float]
    thermal_ramp: Dict[str, float]
    thermal_initial_output: Dict[str, float]

    nuclear_capacity: Dict[str, float]
    nuclear_min_power: Dict[str, float]
    nuclear_cost: Dict[str, float]

    solar_capacity: Dict[str, float]
    solar_available: Dict[Tuple[str, int], float]
    wind_capacity: Dict[str, float]
    wind_available: Dict[Tuple[str, int], float]
    res_capacity: Dict[str, float]
    res_available: Dict[Tuple[str, int], float]

    dr_limit: Dict[Tuple[str, int], float]

    battery_power: Dict[str, float]
    battery_energy: Dict[str, float]
    battery_initial: Dict[str, float]
    battery_eta_charge: float
    battery_eta_discharge: float
    battery_cycle_cost: Dict[str, float]
    battery_retention: Dict[str, float]
    battery_final_min: Dict[str, float]
    battery_final_max: Dict[str, float]

    hydro_res_capacity: Dict[str, float]
    hydro_res_energy: Dict[str, float]
    hydro_initial: Dict[str, float]
    hydro_inflow_power: Dict[Tuple[str, int], float]
    hydro_release_efficiency: float
    hydro_spill_cost: float

    hydro_ror_generation: Dict[Tuple[str, int], float]

    pumped_power: Dict[str, float]
    pumped_energy: Dict[str, float]
    pumped_initial: Dict[str, float]
    pumped_eta_charge: float
    pumped_eta_discharge: float
    pumped_cycle_cost: Dict[str, float]
    pumped_retention: Dict[str, float]
    pumped_final_min: Dict[str, float]
    pumped_final_max: Dict[str, float]

    dr_cost_per_mwh: float
    voll: float
    variable_spill_cost: float
    import_capacity: float
    import_cost: float
    export_cost: float
    overgen_spill_penalty: float
    import_anchor_zone: str

    lines: Dict[str, NetworkLine]
    lines_from_index: Dict[str, Tuple[str, ...]]
    lines_to_index: Dict[str, Tuple[str, ...]]


_DEFAULT_ZONE_VARIANTS = {
    "evening_peak": 0.25,
    "double_peak": 0.15,
    "office_peak": 0.2,
    "industrial": 0.1,
    "night_shift": 0.1,
    "residential_morning": 0.2,
}


def _base_solar_share(weather_profile: str | None) -> float:
    profile = (weather_profile or "").lower()
    if "sunny" in profile or "summer" in profile:
        return 0.68
    if "storm" in profile or "wind" in profile:
        return 0.32
    if "calm" in profile:
        return 0.56
    if "overcast" in profile:
        return 0.45
    if "mixed" in profile:
        return 0.5
    return 0.5


def _split_legacy_res_counts(
    res_counts: list[int],
    weather_profile: str | None,
    *,
    zone_weather_profiles: list[str] | None = None,
) -> tuple[list[int], list[int]]:
    if not res_counts:
        return ([], [])
    solar: list[int] = []
    wind: list[int] = []
    for idx, units in enumerate(res_counts):
        units_int = int(max(0, round(units)))
        if units_int == 0:
            solar.append(0)
            wind.append(0)
            continue
        zone_profile = None
        if zone_weather_profiles is not None and idx < len(zone_weather_profiles):
            zone_profile = zone_weather_profiles[idx]
        base_share = _base_solar_share(zone_profile or weather_profile)
        offset = ((idx * 37) % 11 - 5) / 55.0
        share = float(np.clip(base_share + offset, 0.1, 0.9))
        solar_units = int(round(units_int * share))
        solar_units = max(0, min(solar_units, units_int))
        wind_units = units_int - solar_units
        solar.append(solar_units)
        wind.append(wind_units)
    return solar, wind


def _uuid_to_seed(uid: str) -> int:
    try:
        return int(UUID(uid).int & 0xFFFFFFFF)
    except ValueError:
        return abs(hash(uid)) & 0xFFFFFFFF


def _iter_zone_names(zones_per_region: Iterable[int]) -> List[str]:
    zone_names: List[str] = []
    for ridx, count in enumerate(zones_per_region, start=1):
        for zidx in range(1, count + 1):
            zone_names.append(f"R{ridx}Z{zidx}")
    return zone_names


def _distribute_per_region(values: Iterable[int], zones_per_region: Iterable[int]) -> List[int]:
    per_zone: List[int] = []
    for val, count in zip(values, zones_per_region):
        count = int(count)
        if count <= 0:
            per_zone.extend([0] * count)
            continue
        base, remainder = divmod(int(val), count)
        for idx in range(count):
            per_zone.append(base + (1 if idx < remainder else 0))
    return per_zone


def _build_lines(zones: List[str], zones_per_region: List[int], intertie_density: float) -> Dict[str, NetworkLine]:
    lines: Dict[str, NetworkLine] = {}
    base_capacity = TRANS.base_capacity_mw * (0.6 + 0.8 * intertie_density)
    
    # Distance parameters (typical power system)
    # Lower intertie_density → more sparse → longer distances
    intra_regional_base_km = 80.0 + 40.0 * (1.0 - intertie_density)  # 80-120 km
    inter_regional_base_km = 200.0 + 100.0 * (1.0 - intertie_density)  # 200-300 km
    
    zone_cursor = 0
    line_id = 0

    # Intra-regional lines (connect zones within same region)
    for count in zones_per_region:
        region_zones = zones[zone_cursor: zone_cursor + count]
        for left, right in zip(region_zones[:-1], region_zones[1:]):
            line_id += 1
            # Add variance: ±20% around base distance
            distance_km = intra_regional_base_km * (0.8 + 0.4 * np.random.random())
            lines[f"L{line_id:03d}"] = NetworkLine(
                f"L{line_id:03d}", left, right, base_capacity, distance_km
            )
        zone_cursor += count

    # Inter-regional lines (connect first zones of adjacent regions)
    region_starts: List[str] = []
    cursor = 0
    for count in zones_per_region:
        region_starts.append(zones[cursor])
        cursor += count
    for left, right in zip(region_starts[:-1], region_starts[1:]):
        line_id += 1
        # Inter-regional lines are longer with less variance
        distance_km = inter_regional_base_km * (0.9 + 0.2 * np.random.random())
        lines[f"L{line_id:03d}"] = NetworkLine(
            f"L{line_id:03d}", left, right, base_capacity * 0.8, distance_km
        )

    return lines


def _normalize_variant_weights(raw_variants: Dict[str, float] | None) -> Dict[str, float]:
    if not raw_variants:
        return dict(_DEFAULT_ZONE_VARIANTS)
    cleaned = {str(k): float(v) for k, v in raw_variants.items() if isinstance(v, (int, float)) and float(v) > 0.0}
    if not cleaned:
        return dict(_DEFAULT_ZONE_VARIANTS)
    total = sum(cleaned.values())
    return {k: v / total for k, v in cleaned.items()}


def _extract_range(value, default: Tuple[float, float]) -> Tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi
    return default


def _bounded_range(lo: float, hi: float, *, lower=None, upper=None, min_span: float | None = None) -> Tuple[float, float]:
    if lower is not None:
        lo = max(lower, lo)
        hi = max(lower, hi)
    if upper is not None:
        lo = min(upper, lo)
        hi = min(upper, hi)
    if min_span is not None and abs(hi - lo) < min_span:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5 * min_span, mid + 0.5 * min_span
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _make_zone_demand_profile(
    base_profile_name: str,
    base_profile: np.ndarray,
    periods: int,
    dt_hours: float,
    zone_seed: int,
    variant_weights: Dict[str, float],
    mix_range: Tuple[float, float],
    phase_range: Tuple[float, float],
    noise_range: Tuple[float, float],
    curvature_range: Tuple[float, float],
) -> np.ndarray:
    zone_rng = np.random.default_rng(zone_seed)

    variant_names = list(variant_weights.keys())
    variant_probs = np.array([variant_weights[name] for name in variant_names], dtype=float)
    if base_profile_name not in variant_weights:
        variant_names.append(base_profile_name)
        variant_probs = np.append(variant_probs, 0.05)
    variant_probs = variant_probs / variant_probs.sum()
    variant_choice = zone_rng.choice(variant_names, p=variant_probs)

    variant_profile = build_demand_profile(variant_choice, periods, dt_hours, zone_rng)

    mix = zone_rng.uniform(mix_range[0], mix_range[1])
    profile = mix * variant_profile + (1.0 - mix) * base_profile

    if phase_range[0] != 0.0 or phase_range[1] != 0.0:
        shift_hours = zone_rng.uniform(phase_range[0], phase_range[1])
        if dt_hours > 0:
            shift_steps = int(round(shift_hours / dt_hours))
            if shift_steps:
                profile = np.roll(profile, shift_steps)

    curvature = zone_rng.uniform(curvature_range[0], curvature_range[1])
    profile = np.power(np.clip(profile, 1e-3, None), curvature)

    noise_scale = zone_rng.uniform(noise_range[0], noise_range[1])
    if noise_scale > 0:
        profile += zone_rng.normal(scale=noise_scale, size=periods)

    profile = np.clip(profile, 0.05, None)
    profile /= max(profile.max(), 1e-6)
    return profile


def load_scenario_data(path: Path) -> ScenarioData:
    scenario = json.loads(Path(path).read_text(encoding="utf-8"))

    dt_hours = scenario["dt_minutes"] / 60.0
    periods_count = int(scenario["horizon_hours"] * 60 / scenario["dt_minutes"])
    periods = list(range(periods_count))

    graph = scenario["graph"]
    assets = scenario["assets"]
    econ = scenario["econ_policy"]
    tech = scenario["tech"]
    costs = scenario["operation_costs"]
    exo = scenario["exogenous"]

    zones_per_region: List[int] = graph["zones_per_region"]
    zones = _iter_zone_names(zones_per_region)
    region_of_zone: Dict[str, str] = {}
    zone_index = 0
    for ridx, count in enumerate(zones_per_region, start=1):
        region_name = f"R{ridx}"
        for _ in range(count):
            region_of_zone[zones[zone_index]] = region_name
            zone_index += 1

    scenario_seed = _uuid_to_seed(scenario["id"])
    rng = np.random.default_rng(scenario_seed)
    region_names = sorted(set(region_of_zone.values()))
    raw_region_weather = exo.get("region_weather") or {}
    region_weather_profile: Dict[str, str] = {}
    region_weather_spread: Dict[str, float] = {}
    region_solar_base: Dict[str, np.ndarray] = {}
    region_wind_base: Dict[str, np.ndarray] = {}
    for ridx, region_name in enumerate(region_names):
        cell = raw_region_weather.get(region_name) or {}
        profile_name = cell.get("weather_profile") or exo.get("weather_profile") or "default"
        spread_value = cell.get("weather_spread_intensity", exo.get("weather_spread_intensity"))
        try:
            spread_float = float(spread_value)
        except (TypeError, ValueError):
            spread_float = float(exo.get("weather_spread_intensity", 1.0) or 1.0)
        spread_float = max(0.0, spread_float)
        region_weather_profile[region_name] = profile_name
        region_weather_spread[region_name] = spread_float
        base_seed = scenario_seed + 104729 * (ridx + 1)
        solar_rng = np.random.default_rng(base_seed)
        wind_rng = np.random.default_rng(base_seed + 1)
        region_solar_base[region_name] = build_solar_profile(profile_name, periods_count, dt_hours, solar_rng, spread_float)
        region_wind_base[region_name] = build_wind_profile(profile_name, periods_count, dt_hours, wind_rng, spread_float)


    base_demand_profile = build_demand_profile(exo["demand_profile"], periods_count, dt_hours, rng)

    variant_weights = _normalize_variant_weights(exo.get("zone_profile_variants"))
    mix_range = _extract_range(exo.get("zone_profile_mix_weight"), (0.35, 0.75))
    mix_range = _bounded_range(mix_range[0], mix_range[1], lower=0.0, upper=1.0, min_span=0.05)
    phase_range = _extract_range(exo.get("zone_profile_phase_shift_hours"), (-3.0, 3.0))
    noise_range = _extract_range(exo.get("zone_profile_noise_std"), (0.03, 0.08))
    noise_range = _bounded_range(max(0.0, noise_range[0]), max(0.0, noise_range[1]), lower=0.0, min_span=0.005)
    curvature_range = _extract_range(exo.get("zone_profile_curvature_exp"), (0.9, 1.15))
    curvature_range = _bounded_range(max(0.1, curvature_range[0]), max(0.11, curvature_range[1]), lower=0.1, min_span=0.01)

    zone_profiles: Dict[str, np.ndarray] = {}

    solar_profiles: Dict[str, np.ndarray] = {}
    wind_profiles: Dict[str, np.ndarray] = {}
    for idx, zone in enumerate(zones):
        zone_seed = scenario_seed + 7919 * (idx + 1)
        zone_profiles[zone] = _make_zone_demand_profile(
            exo["demand_profile"],
            base_demand_profile.copy(),
            periods_count,
            dt_hours,
            zone_seed,
            variant_weights,
            mix_range,
            phase_range,
            noise_range,
            curvature_range,
        )
        region_name = region_of_zone.get(zone)
        if region_name is None and region_names:
            region_name = region_names[0]
        spread_val = region_weather_spread.get(
            region_name,
            float(exo.get("weather_spread_intensity", 1.0) or 1.0),
        )
        spread_val = max(0.0, float(spread_val))
        solar_scale = rng.uniform(0.85, 1.15)
        solar_noise = rng.normal(scale=0.05 * spread_val, size=periods_count)
        base_solar = region_solar_base.get(region_name)
        if base_solar is None:
            profile_name = region_weather_profile.get(region_name, exo.get("weather_profile") or "default")
            fallback_rng = np.random.default_rng(scenario_seed + 65537 * (idx + 1))
            base_solar = build_solar_profile(profile_name, periods_count, dt_hours, fallback_rng, spread_val)
            region_solar_base[region_name] = base_solar
        solar_profiles[zone] = np.clip(base_solar * solar_scale + solar_noise, 0.0, 1.0)

        wind_noise = rng.normal(scale=0.1 * spread_val, size=periods_count)
        wind_shift = rng.integers(-3, 4) if periods_count > 0 else 0
        base_wind = region_wind_base.get(region_name)
        if base_wind is None:
            profile_name = region_weather_profile.get(region_name, exo.get("weather_profile") or "default")
            fallback_rng = np.random.default_rng(scenario_seed + 73421 * (idx + 1))
            base_wind = build_wind_profile(profile_name, periods_count, dt_hours, fallback_rng, spread_val)
            region_wind_base[region_name] = base_wind
        wind_profiles[zone] = np.clip(np.roll(base_wind, wind_shift) + wind_noise, 0.0, 1.0)

    total_zones = len(zones)
    zone_weather_profiles = [region_weather_profile.get(region_of_zone.get(zone), exo.get("weather_profile")) for zone in zones]
    if region_weather_spread:
        avg_region_spread = float(np.mean(list(region_weather_spread.values())))
    else:
        avg_region_spread = float(exo.get("weather_spread_intensity", 1.0) or 1.0)
    raw_solar_per_site = assets.get("solar_per_site")
    raw_wind_per_site = assets.get("wind_per_site")

    def _norm_units(values):
        if values is None:
            return None
        seq = list(values)
        if len(seq) < total_zones:
            seq.extend([0] * (total_zones - len(seq)))
        elif len(seq) > total_zones:
            seq = seq[:total_zones]
        return [int(max(0, round(seq[idx]))) for idx in range(total_zones)]

    solar_units_per_zone = _norm_units(raw_solar_per_site)
    wind_units_per_zone = _norm_units(raw_wind_per_site)

    res_counts = list(assets.get("res_per_site", []))
    if len(res_counts) < total_zones:
        res_counts.extend([0] * (total_zones - len(res_counts)))
    elif len(res_counts) > total_zones:
        res_counts = res_counts[:total_zones]

    if solar_units_per_zone is None and wind_units_per_zone is None:
        solar_units_per_zone, wind_units_per_zone = _split_legacy_res_counts(
            res_counts,
            exo.get("weather_profile"),
            zone_weather_profiles=zone_weather_profiles,
        )
    elif solar_units_per_zone is None:
        solar_units_per_zone = []
        for idx in range(total_zones):
            total_units = int(max(0, round(res_counts[idx]))) if res_counts else 0
            solar_units_per_zone.append(max(0, total_units - wind_units_per_zone[idx]))
    elif wind_units_per_zone is None:
        wind_units_per_zone = []
        for idx in range(total_zones):
            total_units = int(max(0, round(res_counts[idx]))) if res_counts else 0
            wind_units_per_zone.append(max(0, total_units - solar_units_per_zone[idx]))

    if not solar_units_per_zone:
        solar_units_per_zone = [0] * total_zones
    if not wind_units_per_zone:
        wind_units_per_zone = [0] * total_zones

    nuclear_per_zone = _distribute_per_region(assets["nuclear_per_region"], zones_per_region) if assets.get("nuclear_per_region") else [0] * sum(zones_per_region)
    hydro_reservoir_per_zone = _distribute_per_region(assets["hydro_reservoir_per_region"], zones_per_region) if assets.get("hydro_reservoir_per_region") else [0] * sum(zones_per_region)
    hydro_pumped_per_zone = _distribute_per_region(assets["hydro_pumped_per_region"], zones_per_region) if assets.get("hydro_pumped_per_region") else [0] * sum(zones_per_region)

    runofriver_profile = build_runofriver_profile(periods_count, dt_hours, avg_region_spread, rng)
    inflow_profile = build_inflow_profile(periods_count, dt_hours, exo["inflow_factor"], rng)

    demand_scale = scenario["exogenous"]["demand_scale_factor"]
    co2_price = econ["co2_price"]
    import_cap = max(0.0, econ["import_export_caps_factor"]) * IMPORT_BASE_CAPACITY * (
        len(zones_per_region) + max(1, graph["neighbor_nations"])
    )
    import_cost = econ["price_cap"] * 0.9
    export_cost = import_cost * 0.2
    variable_spill_cost = costs["renewable_spill_cost_eur_per_mwh"]
    hydro_spill_cost_value = costs["hydro_spill_cost_eur_per_mwh"]
    overgen_spill_penalty = costs["overgen_spill_cost_eur_per_mwh"]

    battery_eta = math.sqrt(max(0.05, tech["battery_roundtrip_eff"]))
    pumped_eta = math.sqrt(max(0.05, PUMPED.efficiency))

    battery_cycle_cost_value = float(costs.get("battery_cycle_cost_eur_per_mwh", 0.0) or 0.0)
    pumped_cycle_cost_value = float(costs.get("pumped_cycle_cost_eur_per_mwh", 0.0) or 0.0)

    battery_initial_base = float(tech.get("battery_initial_soc_fraction", 0.5) or 0.5)
    battery_final_tol = float(tech.get("battery_final_soc_tolerance", 0.1) or 0.1)
    battery_self_discharge_rate = max(0.0, float(tech.get("battery_self_discharge_per_hour", 0.0) or 0.0))

    pumped_initial_base = float(tech.get("pumped_initial_level_fraction", 0.5) or 0.5)
    pumped_final_tol = float(tech.get("pumped_final_level_tolerance", 0.1) or 0.1)
    pumped_self_discharge_rate = max(0.0, float(tech.get("pumped_self_discharge_per_hour", 0.0) or 0.0))

    def _retention_factor(rate_per_hour: float) -> float:
        rate = max(0.0, min(0.999, rate_per_hour))
        if rate <= 0.0:
            return 1.0
        return float(math.exp(math.log(1.0 - rate) * dt_hours))

    demand: Dict[Tuple[str, int], float] = {}
    peak_demand: Dict[str, float] = {}
    thermal_capacity: Dict[str, float] = {}
    thermal_min: Dict[str, float] = {}
    thermal_cost: Dict[str, float] = {}
    thermal_ramp: Dict[str, float] = {}
    thermal_initial: Dict[str, float] = {}

    nuclear_capacity: Dict[str, float] = {}
    nuclear_min: Dict[str, float] = {}
    nuclear_cost: Dict[str, float] = {}

    solar_capacity: Dict[str, float] = {}
    solar_available: Dict[Tuple[str, int], float] = {}
    wind_capacity: Dict[str, float] = {}
    wind_available: Dict[Tuple[str, int], float] = {}
    res_capacity: Dict[str, float] = {}
    res_available: Dict[Tuple[str, int], float] = {}

    dr_limit: Dict[Tuple[str, int], float] = {}

    battery_power: Dict[str, float] = {}
    battery_energy: Dict[str, float] = {}
    battery_initial: Dict[str, float] = {}
    battery_cycle_cost_map: Dict[str, float] = {}
    battery_retention: Dict[str, float] = {}
    battery_final_min: Dict[str, float] = {}
    battery_final_max: Dict[str, float] = {}

    hydro_res_capacity: Dict[str, float] = {}
    hydro_res_energy: Dict[str, float] = {}
    hydro_initial: Dict[str, float] = {}
    hydro_inflow: Dict[Tuple[str, int], float] = {}

    hydro_ror_generation: Dict[Tuple[str, int], float] = {}

    pumped_power: Dict[str, float] = {}
    pumped_energy: Dict[str, float] = {}
    pumped_initial: Dict[str, float] = {}
    pumped_cycle_cost_map: Dict[str, float] = {}
    pumped_retention: Dict[str, float] = {}
    pumped_final_min: Dict[str, float] = {}
    pumped_final_max: Dict[str, float] = {}

    dr_cost_value = costs["demand_response_cost_eur_per_mwh"]
    voll_value = costs["value_of_lost_load_eur_per_mwh"]
    thermal_fuel_cost = costs["thermal_fuel_eur_per_mwh"]
    nuclear_fuel_cost = costs["nuclear_fuel_eur_per_mwh"]

    for idx, zone in enumerate(zones):
        zone_profile = zone_profiles[zone]

        base_peak = 180.0 * demand_scale * (1.0 + 0.1 * (idx % 3))
        base_peak *= 1.0 + 0.1 * rng.standard_normal()
        peak_demand[zone] = max(80.0, base_peak)

        for t in periods:
            demand[(zone, t)] = peak_demand[zone] * zone_profile[t]

        thermal_units = max(0, assets["thermal_per_site"][idx])
        thermal_capacity[zone] = thermal_units * THERMAL.unit_capacity_mw
        thermal_min[zone] = thermal_capacity[zone] * (THERMAL.min_power_fraction if thermal_capacity[zone] > 0 else 0.0)
        thermal_cost[zone] = thermal_fuel_cost + tech["thermal_marg_cost"] + co2_price * THERMAL.emission_rate_t_per_mwh
        thermal_ramp[zone] = thermal_capacity[zone] * tech["thermal_ramp_pct"]
        thermal_initial[zone] = 0.0

        nuclear_units = max(0, nuclear_per_zone[idx])
        nuclear_capacity[zone] = nuclear_units * NUCLEAR.unit_capacity_mw
        nuclear_min[zone] = nuclear_capacity[zone] * (NUCLEAR.min_power_fraction if nuclear_capacity[zone] > 0 else 0.0)
        nuclear_cost[zone] = nuclear_fuel_cost

        solar_units = max(0, solar_units_per_zone[idx])
        solar_capacity[zone] = solar_units * SOLAR.unit_capacity_mw
        solar_profile = solar_profiles[zone]
        for t in periods:
            solar_available[(zone, t)] = solar_capacity[zone] * solar_profile[t]

        wind_units = max(0, wind_units_per_zone[idx])
        wind_capacity[zone] = wind_units * WIND.unit_capacity_mw
        wind_profile = wind_profiles[zone]
        for t in periods:
            wind_available[(zone, t)] = wind_capacity[zone] * wind_profile[t]

        res_capacity[zone] = solar_capacity[zone] + wind_capacity[zone]
        for t in periods:
            res_available[(zone, t)] = solar_available.get((zone, t), 0.0) + wind_available[(zone, t)]

        dr_units = max(0, assets["dr_per_site"][idx])
        share = tech["dr_max_shed_share"] * (0.5 + 0.5 * min(dr_units, 4))
        for t in periods:
            dr_limit[(zone, t)] = demand[(zone, t)] * min(share, 0.8)

        bat_units = max(0, assets["battery_per_site"][idx])
        battery_power[zone] = bat_units * BATTERY.power_per_unit_mw
        energy_cap = battery_power[zone] * max(1.0, tech["battery_e_to_p_hours"])
        battery_energy[zone] = energy_cap
        if energy_cap > 1e-6:
            init_low = max(0.05, battery_initial_base - 0.1)
            init_high = min(0.95, battery_initial_base + 0.1)
            if init_low > init_high:
                init_low, init_high = init_high, init_low
            init_frac = float(rng.uniform(init_low, init_high))
            init_frac = float(np.clip(init_frac, 0.05, 0.95))
            battery_initial[zone] = energy_cap * init_frac
            final_min_frac = max(0.0, min(init_frac - battery_final_tol, 1.0))
            final_max_frac = min(1.0, init_frac + battery_final_tol)
            if final_max_frac < final_min_frac + 1e-4:
                final_max_frac = min(1.0, final_min_frac + 1e-4)
            battery_final_min[zone] = energy_cap * final_min_frac
            battery_final_max[zone] = energy_cap * final_max_frac
            battery_cycle_cost_map[zone] = battery_cycle_cost_value
            battery_retention[zone] = _retention_factor(battery_self_discharge_rate)
        else:
            battery_initial[zone] = 0.0
            battery_final_min[zone] = 0.0
            battery_final_max[zone] = 0.0
            battery_cycle_cost_map[zone] = 0.0
            battery_retention[zone] = 1.0

        hydro_res_units = max(0, hydro_reservoir_per_zone[idx])
        hydro_res_capacity[zone] = hydro_res_units * HYDRO_RES.power_per_unit_mw
        hydro_res_energy[zone] = hydro_res_capacity[zone] * HYDRO_RES.energy_hours
        hydro_initial[zone] = 0.5 * hydro_res_energy[zone]
        for t in periods:
            inflow_power = hydro_res_capacity[zone] * 0.25 * inflow_profile[t]
            hydro_inflow[(zone, t)] = inflow_power

        hydro_ror_units = max(0, assets["hydro_ror_per_zone"][idx])
        for t in periods:
            hydro_ror_generation[(zone, t)] = hydro_ror_units * HYDRO_ROR.output_per_unit_mw * runofriver_profile[t]

        pumped_units = max(0, hydro_pumped_per_zone[idx])
        pumped_power[zone] = pumped_units * PUMPED.power_per_unit_mw
        pumped_cap = pumped_power[zone] * PUMPED.energy_hours
        pumped_energy[zone] = pumped_cap
        if pumped_cap > 1e-6:
            init_low = max(0.05, pumped_initial_base - 0.12)
            init_high = min(0.95, pumped_initial_base + 0.12)
            if init_low > init_high:
                init_low, init_high = init_high, init_low
            init_frac = float(rng.uniform(init_low, init_high))
            init_frac = float(np.clip(init_frac, 0.05, 0.95))
            pumped_initial[zone] = pumped_cap * init_frac
            final_min_frac = max(0.0, min(init_frac - pumped_final_tol, 1.0))
            final_max_frac = min(1.0, init_frac + pumped_final_tol)
            if final_max_frac < final_min_frac + 1e-4:
                final_max_frac = min(1.0, final_min_frac + 1e-4)
            pumped_final_min[zone] = pumped_cap * final_min_frac
            pumped_final_max[zone] = pumped_cap * final_max_frac
            pumped_cycle_cost_map[zone] = pumped_cycle_cost_value
            pumped_retention[zone] = _retention_factor(pumped_self_discharge_rate)
        else:
            pumped_initial[zone] = 0.0
            pumped_final_min[zone] = 0.0
            pumped_final_max[zone] = 0.0
            pumped_cycle_cost_map[zone] = 0.0
            pumped_retention[zone] = 1.0

    lines = _build_lines(zones, zones_per_region, graph["intertie_density"])
    from_idx: Dict[str, List[str]] = defaultdict(list)
    to_idx: Dict[str, List[str]] = defaultdict(list)
    for lid, line in lines.items():
        from_idx[line.from_zone].append(lid)
        to_idx[line.to_zone].append(lid)

    return ScenarioData(
        scenario_id=scenario["id"],
        periods=periods,
        dt_hours=dt_hours,
        zones=zones,
        region_of_zone=region_of_zone,
        region_weather_profile=region_weather_profile,
        region_weather_spread=region_weather_spread,
        demand=demand,
        peak_demand=peak_demand,
        thermal_capacity=thermal_capacity,
        thermal_min_power=thermal_min,
        thermal_cost=thermal_cost,
        thermal_ramp=thermal_ramp,
        thermal_initial_output=thermal_initial,
        nuclear_capacity=nuclear_capacity,
        nuclear_min_power=nuclear_min,
        nuclear_cost=nuclear_cost,
        solar_capacity=solar_capacity,
        solar_available=solar_available,
        wind_capacity=wind_capacity,
        wind_available=wind_available,
        res_capacity=res_capacity,
        res_available=res_available,
        dr_limit=dr_limit,
        battery_power=battery_power,
        battery_energy=battery_energy,
        battery_initial=battery_initial,
        battery_eta_charge=battery_eta,
        battery_eta_discharge=battery_eta,
        battery_cycle_cost=battery_cycle_cost_map,
        battery_retention=battery_retention,
        battery_final_min=battery_final_min,
        battery_final_max=battery_final_max,
        hydro_res_capacity=hydro_res_capacity,
        hydro_res_energy=hydro_res_energy,
        hydro_initial=hydro_initial,
        hydro_inflow_power=hydro_inflow,
        hydro_release_efficiency=1.0,
        hydro_spill_cost=hydro_spill_cost_value,
        hydro_ror_generation=hydro_ror_generation,
        pumped_power=pumped_power,
        pumped_energy=pumped_energy,
        pumped_initial=pumped_initial,
        pumped_eta_charge=pumped_eta,
        pumped_eta_discharge=pumped_eta,
        pumped_cycle_cost=pumped_cycle_cost_map,
        pumped_retention=pumped_retention,
        pumped_final_min=pumped_final_min,
        pumped_final_max=pumped_final_max,
        dr_cost_per_mwh=dr_cost_value,
        voll=voll_value,
        variable_spill_cost=variable_spill_cost,
        import_capacity=import_cap,
        import_cost=import_cost,
        export_cost=export_cost,
        overgen_spill_penalty=overgen_spill_penalty,
        import_anchor_zone=zones[0],
        lines=lines,
        lines_from_index={k: tuple(v) for k, v in from_idx.items()},
        lines_to_index={k: tuple(v) for k, v in to_idx.items()},
    )

