"""Synthetic operating days on the same 24-hour grid, without solving a MILP.

The preselected anchor is the first numbered medium-family input. Its installed
equipment, network, initial states, hydro, prices and demand-response rules are
retained. Only demand, solar/wind availability and their derived limits vary.
The sampled regimes and amplitudes are an experimental design, not observed
calendar dates or estimated probabilities of real weather.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock
from uuid import UUID

import numpy as np

from research.paper1_revision.campaign import load_physics
from research.paper1_revision.realized.features import from_instance
from src.milp.model import build_uc_model
from src.milp.profiles import build_demand_profile, build_solar_profile, build_wind_profile
from src.milp.scenario_loader import ScenarioData
from src.operational_exit.physics import apply_cost_units


DEFAULT_ANCHOR = (Path(__file__).resolve().parents[2] / 'outputs'
                  / 'medium_criticality_scenarios' / 'scenario_00001.json')
RULE_DIRECTIONS = {
    'residual_peak_mw': 1,
    'upward_ramp_mw_per_hour': 1,
    'capacity_margin_mw': -1,
    'initial_storage_mwh': -1,
}

# Demand response is a fixed *share* of the varying load. Its hourly MW limits
# and the ramp defined as a fixed share of peak MW therefore vary by design.
_DAILY_FIELDS = frozenset({
    'region_weather_profile', 'demand', 'peak_demand', 'solar_available',
    'wind_available', 'res_available', 'dr_limit', 'dr_block_limit', 'dr_ramp_limit',
})
_ANCHOR_LOAD_LOCK = Lock()


def _json_tree(value):
    """Convert tuple-indexed input tables to stable JSON, without rounding MW."""
    if isinstance(value, dict):
        return [(key, _json_tree(item)) for key, item in sorted(value.items())]
    if is_dataclass(value):
        return {f.name: _json_tree(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _field_hash(data: ScenarioData, names) -> str:
    payload = {name: _json_tree(getattr(data, name)) for name in sorted(names)}
    encoded = json.dumps(payload, ensure_ascii=True, separators=(',', ':'), allow_nan=False)
    return sha256(encoded.encode('utf-8')).hexdigest()


def static_sha256(data: ScenarioData) -> str:
    """Hash all realized fixed fields, including hydro series, network and initial storage."""
    return _field_hash(data, [f.name for f in fields(data) if f.name not in _DAILY_FIELDS])


def profiles_sha256(data: ScenarioData) -> str:
    """Identify the three input MW chronologies by value, independently of day label."""
    return _field_hash(data, ('demand', 'solar_available', 'wind_available'))


def _draw_regime(rng):
    return dict(
        demand_profile=str(rng.choice(['flat', 'evening_peak', 'double_peak', 'office_peak'])),
        weather_profile=str(rng.choice(['continental', 'coastal', 'storm', 'winter'])),
        peak_load_multiplier=float(rng.uniform(0.75, 1.25)),
        solar_availability_multiplier=float(rng.uniform(0.40, 1.00)),
        wind_availability_multiplier=float(rng.uniform(0.35, 1.00)),
    )


def day_parameters(seed: int) -> dict:
    """Document ex-ante choices; every draw uses only the supplied day seed."""
    return _draw_regime(np.random.default_rng(seed))


def load_anchor(anchor_path: Path) -> ScenarioData:
    """Load the same physical network, including every realized line distance.

    The historical loader uses its own UUID-derived generator for profiles, but
    draws line distances from numpy's legacy global stream. Isolate that draw
    here, leaving the archived loader and the caller's random state unchanged.
    """
    path = Path(anchor_path)
    identifier = json.loads(path.read_text(encoding='utf-8-sig'))['id']
    try:
        distance_seed = UUID(identifier).int & 0xffffffff
    except (ValueError, TypeError) as exc:
        raise ValueError("A fixed-day anchor needs a UUID id for reproducible physics.") from exc
    with _ANCHOR_LOAD_LOCK:
        state = np.random.get_state()
        try:
            np.random.seed(distance_seed)
            data = load_physics(path)
        finally:
            np.random.set_state(state)
    if len(data.periods) != 24 or data.dt_hours != 1.0:
        raise ValueError('The fixed-day pilot requires 24 one-hour periods.')
    return data


def load_day(anchor_path: Path, seed: int) -> ScenarioData:
    """Reload the exact anchor and replace only its hourly operating conditions.

    A daily load shape is mixed equally with each zone's original normalized
    shape. Regional weather profiles receive weight 0.75 and the zone's anchor
    availability profile 0.25, before multiplication by installed MW and the
    daily solar/wind amplitude. The physical loader's unit profiles are reused.
    """
    anchor = load_anchor(anchor_path)

    rng = np.random.default_rng(seed)
    regime = _draw_regime(rng)
    demand_shape = build_demand_profile(regime['demand_profile'], 24, 1., rng)
    weather = regime['weather_profile']
    regions = sorted(set(anchor.region_of_zone.values()))
    solar_by_region, wind_by_region = {}, {}
    for r in regions:
        spread = anchor.region_weather_spread[r]
        solar_by_region[r] = build_solar_profile(weather, 24, 1., rng, spread)
        wind_by_region[r] = build_wind_profile(weather, 24, 1., rng, spread)

    day = replace(anchor, demand={}, peak_demand={}, solar_available={}, wind_available={},
                  res_available={}, dr_limit={}, dr_block_limit={}, dr_ramp_limit={},
                  region_weather_profile=dict.fromkeys(regions, weather))
    for z in anchor.zones:
        original_peak = float(anchor.peak_demand[z])
        original = np.array([anchor.demand[z, t] for t in anchor.periods], dtype=float)
        if original_peak <= 0 or np.any(original <= 0):
            raise ValueError(f'Expected strictly positive benchmark demand in zone {z}.')
        shape = 0.5 * (original / original_peak + demand_shape)
        day.peak_demand[z] = original_peak * regime['peak_load_multiplier']
        demand = day.peak_demand[z] * shape / shape.max()
        day.dr_ramp_limit[z] = anchor.dr_ramp_limit[z] * regime['peak_load_multiplier']
        region = anchor.region_of_zone[z]
        for pos, t in enumerate(anchor.periods):
            k = (z, t)
            day.demand[k] = float(demand[pos])
            scale = day.demand[k] / anchor.demand[k]
            day.dr_limit[k] = anchor.dr_limit[k] * scale
            for block in range(anchor.dr_num_blocks):
                key = (z, t, block)
                day.dr_block_limit[key] = anchor.dr_block_limit[key] * scale
            day.solar_available[k] = float(regime['solar_availability_multiplier'] * (
                0.75 * anchor.solar_capacity[z] * solar_by_region[region][pos]
                + 0.25 * anchor.solar_available[k]))
            day.wind_available[k] = float(regime['wind_availability_multiplier'] * (
                0.75 * anchor.wind_capacity[z] * wind_by_region[region][pos]
                + 0.25 * anchor.wind_available[k]))
            day.res_available[k] = day.solar_available[k] + day.wind_available[k]
    return day


def inspect_day(data: ScenarioData) -> dict:
    """Compute frozen predictor inputs and physical inspection rules before solving.

    Residual demand subtracts available solar and wind only. The margin is
    thermal + nuclear capacity + run-of-river output at this residual peak
    minus the residual peak. It is a nameplate screening proxy in MW.
    Initial stock sums battery and pumped storage MWh (fixed by design).
    """
    model = build_uc_model(data)
    apply_cost_units(model)
    realized, diagnostics = from_instance(data, model)
    trio = [diagnostics['zones'], diagnostics['free_binary_variables'], realized['B4']]
    dimensions = [diagnostics[k] for k in ('zones', 'steps', 'lines', 'variables',
                                           'free_binary_variables', 'active_constraints')]
    residual = [sum(data.demand[z, t] - data.res_available[z, t] for z in data.zones)
                for t in data.periods]
    peak_index = int(np.argmax(residual))
    peak = float(residual[peak_index])
    peak_period = data.periods[peak_index]
    firm = sum(data.thermal_capacity.values()) + sum(data.nuclear_capacity.values())
    run_of_river = sum(data.hydro_ror_generation[z, peak_period] for z in data.zones)
    return dict(
        static_sha256=static_sha256(data), profiles_sha256=profiles_sha256(data),
        compact_trio=trio, model_dimensions=dimensions,
        realized_features={'real_' + key: val for key, val in realized.items()},
        demand_mwh=float(sum(data.demand.values()) * data.dt_hours), dt_hours=float(data.dt_hours),
        residual_peak_mw=peak,
        upward_ramp_mw_per_hour=float(max(0., np.max(np.diff(residual))) / data.dt_hours),
        capacity_margin_mw=float(firm + run_of_river - peak),
        initial_storage_mwh=float(sum(data.battery_initial.values()) + sum(data.pumped_initial.values())),
    )
