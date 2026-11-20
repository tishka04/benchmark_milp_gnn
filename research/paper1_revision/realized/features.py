"""Read model inputs and count declared variables without creating any solver."""
from __future__ import annotations
import time
import numpy as np
from pyomo.environ import Var, Constraint, value
from ..campaign import load_physics

IDS = tuple([f'A{i}' for i in range(1, 13)] + [f'B{i}' for i in range(1, 11)])
DEFINITIONS = dict(zip(IDS, [
    'Available solar+wind energy / demand energy (not dispatched energy)',
    'Population std of aggregate residual demand / max(mean demand, 1 MW)',
    'Actual aggregate demand peak / max(actual valley, 1 MW)',
    'Max absolute step change in residual demand / max(mean demand, 1 MW) / dt_hours; 0 for one step',
    'Mean demand / zones / 500 MW nominal comparison scale',
    'Actual demand peak / max(thermal+nuclear capacity, 1 MW)',
    '1 - installed battery+pumped power / max(positive residual peak, 1 MW)',
    '1 - installed battery+pumped energy / max(positive residual energy, 1 MWh)',
    '1 - sum thermal ramp MW per transition / max(absolute residual step change, 1 MW)',
    '1 - total DR deferral-cap energy / max(total demand energy, 1 MWh); ignores intertemporal attainability',
    'Import/export capacity / max(demand peak, 1 MW); capacity, not realized trade reliance',
    'Demand peak / max(sum transport-line capacities, 1 MW); aggregate proxy, not a congestion certificate',
    'Actual number of zones',
    'Actual number of time steps',
    'Declared binary variables neither explicitly fixed nor fixed by equal bounds; before solver presolve',
    'Zones with nonzero thermal capacity / zones; aggregated commitment density',
    'Sum thermal+nuclear minimum power / max(sum thermal+nuclear capacity, 1 MW); thermal minima conditional on commitment',
    'Thermal start cost once per active zone / max(full-capacity thermal absolute variable cost over horizon, 1 EUR)',
    'Total thermal capacity / max(total per-transition thermal ramp, 1 MW)',
    'Energy-weighted installed storage duration / relative admissible terminal window; window fraction floor 0.01; absent storage gives 0',
    'Actual number of transport lines / zones',
    'Population coefficient of variation of incident transport capacity across zones; zero if no capacity',
]))


def free_variable(v):
    return not v.fixed and not (v.lb is not None and v.ub is not None and value(v.lb) == value(v.ub))


def from_instance(data, model):
    """All inputs precede optimization; no dispatch/solution values are consulted."""
    zones, periods = data.zones, data.periods
    if not zones or not periods or data.dt_hours <= 0:
        raise ValueError('Empty spatial/temporal support')
    total = lambda name: sum(float(v) for v in getattr(data, name).values())
    aggregate = lambda name: np.array([sum(getattr(data, name)[z, t] for z in zones) for t in periods], float)
    demand = aggregate('demand')
    renewable = aggregate('solar_available') + aggregate('wind_available')
    residual = demand - renewable - aggregate('hydro_ror_generation')
    if not np.isfinite(demand).all() or np.any(demand < 0):
        raise ValueError('Invalid demand')
    dt = float(data.dt_hours)
    horizon = dt * len(periods)
    mean, peak, valley = float(demand.mean()), float(demand.max()), float(demand.min())
    demand_energy = float(demand.sum() * dt)
    positive = np.maximum(residual, 0)
    change = float(np.max(np.abs(np.diff(residual)))) if len(periods) > 1 else 0.
    firm = total('thermal_capacity') + total('nuclear_capacity')
    power = total('battery_power') + total('pumped_power')
    energy = total('battery_energy') + total('pumped_energy')
    ramp = total('thermal_ramp')
    variables = list(model.component_data_objects(Var, active=True))
    free = [v for v in variables if free_variable(v)]
    binaries = [v for v in free if v.is_binary()]
    active_thermal = [z for z in zones if data.thermal_capacity[z] > 1e-6]
    startup = sum(data.thermal_startup_cost[z] for z in active_thermal)
    thermal_cost = sum(abs(data.thermal_cost[z]) * data.thermal_capacity[z] * horizon for z in zones)
    storage_terms, storage_weights = [], []
    for prefix in ['battery', 'pumped']:
        for z in zones:
            cap = getattr(data, prefix + '_energy')[z]
            p = getattr(data, prefix + '_power')[z]
            if cap <= 1e-6 or p <= 1e-6:
                continue
            lower_con = getattr(model, prefix + '_final_min_con')
            upper_con = getattr(model, prefix + '_final_max_con')
            lower = getattr(data, prefix + '_final_min')[z] if z in lower_con else 0.
            upper = min(cap, getattr(data, prefix + '_final_max')[z]) if z in upper_con else cap
            if upper < lower - 1e-6:
                raise ValueError('Inconsistent terminal storage window')
            width = max(upper - lower, 0.) / cap
            storage_terms.append((cap / p) / max(width, .01))
            storage_weights.append(cap)
    incident = {z: 0. for z in zones}
    for line in data.lines.values():
        incident[line.from_zone] += line.capacity_mw
        incident[line.to_zone] += line.capacity_mw
    strengths = np.array(list(incident.values()))
    feature_values = [
        float(renewable.sum() * dt) / max(demand_energy, 1.),
        float(residual.std()) / max(mean, 1.), peak / max(valley, 1.),
        change / max(mean, 1.) / dt, mean / len(zones) / 500., peak / max(firm, 1.),
        1 - power / max(float(positive.max()), 1.),
        1 - energy / max(float(positive.sum() * dt), 1.),
        1 - ramp / max(change, 1.),
        1 - total('dr_limit') * dt / max(demand_energy, 1.),
        data.import_capacity / max(peak, 1.),
        peak / max(sum(l.capacity_mw for l in data.lines.values()), 1.),
        float(len(zones)), float(len(periods)), float(len(binaries)),
        len(active_thermal) / len(zones),
        (total('thermal_min_power') + total('nuclear_min_power')) / max(firm, 1.),
        startup / max(thermal_cost, 1.), total('thermal_capacity') / max(ramp, 1.),
        float(np.average(storage_terms, weights=storage_weights)) if storage_terms else 0.,
        len(data.lines) / len(zones),
        float(strengths.std() / strengths.mean()) if strengths.mean() > 0 else 0.,
    ]
    if not np.isfinite(feature_values).all():
        raise ValueError('Nonfinite realized descriptor')
    diagnostics = dict(zones=len(zones), steps=len(periods), dt_hours=dt, horizon_hours=horizon,
        lines=len(data.lines), variables=len(variables), free_variables=len(free),
        binary_variables=sum(v.is_binary() for v in variables), free_binary_variables=len(binaries),
        active_constraints=sum(1 for _ in model.component_data_objects(Constraint, active=True)),
        storage_power_mw=power, storage_energy_mwh=energy,
        installed_storage_duration_hours=energy / power if power > 0 else 0.,
        actual_demand_peak_mw=peak, actual_demand_valley_mw=valley,
        definition='Declared model after structural fixes, before solver presolve; no solve.')
    return dict(zip(IDS, feature_values)), diagnostics


def extract(path):
    from src.milp.model import build_uc_model
    from src.operational_exit.physics import apply_cost_units
    begin = time.perf_counter()
    data = load_physics(path)
    loaded = time.perf_counter()
    model = build_uc_model(data)
    apply_cost_units(model)
    built = time.perf_counter()
    features, diagnostics = from_instance(data, model)
    end = time.perf_counter()
    diagnostics.update(load_seconds=loaded - begin, build_seconds=built - loaded,
                       descriptor_seconds=end - built, total_seconds=end - begin)
    return features, diagnostics
