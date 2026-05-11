from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from pyomo.environ import SolverFactory, TerminationCondition, value
from pyomo.core import TransformationFactory
from pyomo.opt import SolverStatus

from .model import build_uc_model
from .scenario_loader import ScenarioData, load_scenario_data

# Default time limit: 2 hours in seconds
DEFAULT_TIME_LIMIT_SECONDS = 60 * 10 # 10 minutes


def _configure_solver_time_limit(solver, solver_name: str, time_limit_seconds: float) -> None:
    """Configure time limit for various solvers."""
    solver_name_lower = solver_name.lower()
    
    if "highs" in solver_name_lower:
        solver.options["time_limit"] = time_limit_seconds
    elif "gurobi" in solver_name_lower:
        solver.options["TimeLimit"] = time_limit_seconds
    elif "cplex" in solver_name_lower:
        solver.options["timelimit"] = time_limit_seconds
    elif "cbc" in solver_name_lower:
        solver.options["seconds"] = time_limit_seconds
    elif "glpk" in solver_name_lower:
        solver.options["tmlim"] = time_limit_seconds
    else:
        # Generic attempt - many solvers accept these
        solver.options["time_limit"] = time_limit_seconds
        solver.options["TimeLimit"] = time_limit_seconds


def _is_timeout_termination(termination_condition: TerminationCondition) -> bool:
    """Check if termination was due to time limit."""
    timeout_conditions = {
        TerminationCondition.maxTimeLimit,
        TerminationCondition.maxIterations,
    }
    return termination_condition in timeout_conditions


@dataclass
class SolveSummary:
    objective: float
    termination_condition: TerminationCondition
    solver_status: SolverStatus
    solve_seconds: float | None = None

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "objective": float(self.objective),
            "termination": self.termination_condition.name,
            "status": self.solver_status.name,
        }
        if self.solve_seconds is not None:
            payload["solve_seconds"] = float(self.solve_seconds)
        return payload

def _relax_integrality(model) -> None:
    TransformationFactory("core.relax_integer_vars").apply_to(model)


def _sum_param_times_var(model, param, var) -> float:
    total = 0.0
    for z in model.Z:
        for t in model.T:
            total += float(value(param[z] * var[z, t]))
    return total


def _sum_var(model, var) -> float:
    total = 0.0
    for z in model.Z:
        for t in model.T:
            total += float(value(var[z, t]))
    return total


def _sum_time_var(model, var) -> float:
    total = 0.0
    for t in model.T:
        total += float(value(var[t]))
    return total


def _aggregate_zone_series(detail: Dict[str, Any], key: str) -> list[float]:
    zones = detail["zones"]
    horizon = len(detail["time_steps"])
    return [
        sum(detail[key][zone][idx] for zone in zones)
        for idx in range(horizon)
    ]


def _compute_cost_components(model) -> Dict[str, float]:
    components: Dict[str, float] = {}
    components["thermal_fuel"] = _sum_param_times_var(model, model.thermal_cost, model.p_thermal)
    components["nuclear_fuel"] = _sum_param_times_var(model, model.nuclear_cost, model.p_nuclear)

    scalar_voll = float(value(model.voll))
    scalar_res_spill = float(value(model.res_spill_cost))
    scalar_hydro_spill = float(value(model.hydro_spill_cost))
    scalar_overgen = float(value(model.overgen_spill_cost))
    scalar_import = float(value(model.import_cost))
    scalar_export = float(value(model.export_cost))

    # Calculate DR cost using tiered blocks
    dr_total_cost = 0.0
    for z in model.Z:
        for t in model.T:
            for k in model.DR_BLOCKS:
                block_cost = float(value(model.dr_block_cost[k]))
                block_shed = float(value(model.dr_shed_block[z, t, k]))
                dr_total_cost += block_cost * block_shed
    components["demand_response"] = dr_total_cost
    components["unserved_energy"] = scalar_voll * _sum_var(model, model.unserved)

    solar_spill = scalar_res_spill * _sum_var(model, model.spill_solar)
    wind_spill = scalar_res_spill * _sum_var(model, model.spill_wind)
    components["solar_spill"] = solar_spill
    components["wind_spill"] = wind_spill

    components["hydro_spill"] = scalar_hydro_spill * _sum_var(model, model.h_spill)
    components["overgen_spill"] = scalar_overgen * _sum_var(model, model.overgen_spill)
    components["imports"] = scalar_import * _sum_time_var(model, model.net_import)
    components["exports"] = scalar_export * _sum_time_var(model, model.net_export)

    battery_cycle_total = 0.0
    for z in model.Z:
        cycle_cost = float(value(model.battery_cycle_cost[z]))
        if cycle_cost <= 0.0:
            continue
        total_throughput = 0.0
        for t in model.T:
            total_throughput += float(value(model.b_charge[z, t] + model.b_discharge[z, t]))
        battery_cycle_total += cycle_cost * total_throughput
    if battery_cycle_total:
        components["battery_cycle"] = battery_cycle_total

    pumped_cycle_total = 0.0
    for z in model.Z:
        cycle_cost = float(value(model.pumped_cycle_cost[z]))
        if cycle_cost <= 0.0:
            continue
        total_throughput = 0.0
        for t in model.T:
            total_throughput += float(value(model.pumped_charge[z, t] + model.pumped_discharge[z, t]))
        pumped_cycle_total += cycle_cost * total_throughput
    if pumped_cycle_total:
        components["pumped_cycle"] = pumped_cycle_total
    return components


def _coerce_array(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    try:
        return np.asarray(values, dtype=float)
    except Exception:
        return None


def _set_warm_value(var_obj, key, raw_value: Any, *, binary: bool = False) -> int:
    try:
        var = var_obj[key]
    except Exception:
        return 0
    if getattr(var, "is_fixed", lambda: False)():
        return 0
    try:
        value_f = float(raw_value)
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(value_f):
        return 0
    if binary:
        value_f = 1.0 if value_f >= 0.5 else 0.0
    lb = getattr(var, "lb", None)
    ub = getattr(var, "ub", None)
    if lb is not None:
        value_f = max(float(lb), value_f)
    if ub is not None:
        value_f = min(float(ub), value_f)
    var.value = value_f
    return 1


def _apply_mip_warm_start(model, warm_start_data: Dict[str, Any] | None) -> int:
    """
    Seed the exact MILP with values from a previous projected candidate.

    Supports:
    - `continuous_vars`: output of LPWorkerTwoStage._extract_solution()
    - `binary_sample`: raw sampled EBM tensor [Z, T, 7] as a fallback source
    """
    if not warm_start_data:
        return 0

    continuous_vars = warm_start_data.get("continuous_vars", warm_start_data)
    if not isinstance(continuous_vars, dict):
        continuous_vars = {}
    binary_sample = _coerce_array(warm_start_data.get("binary_sample"))

    zones = list(model.Z)
    periods = list(model.T)
    n_zones = len(zones)
    n_timesteps = len(periods)
    n_set = 0

    def get_2d(name: str) -> np.ndarray | None:
        arr = _coerce_array(continuous_vars.get(name))
        if arr is None or arr.ndim != 2:
            return None
        if arr.shape[0] < n_zones or arr.shape[1] < n_timesteps:
            return None
        return arr

    def get_1d(name: str) -> np.ndarray | None:
        arr = _coerce_array(continuous_vars.get(name))
        if arr is None:
            return None
        arr = arr.reshape(-1)
        if arr.shape[0] < n_timesteps:
            return None
        return arr

    two_d_continuous = [
        "p_thermal", "p_nuclear", "p_solar", "p_wind",
        "b_charge", "b_discharge", "b_soc",
        "pumped_charge", "pumped_discharge", "pumped_level",
        "h_release", "h_level", "h_spill",
        "dr_shed", "dr_rebound", "unserved",
        "spill_solar", "spill_wind", "overgen_spill",
    ]
    two_d_binary = [
        "u_thermal", "v_thermal_startup",
        "b_charge_mode", "pumped_charge_mode", "dr_active",
    ]
    one_d_continuous = ["net_import", "net_export"]
    one_d_binary = ["import_mode"]

    for name in two_d_continuous:
        arr = get_2d(name)
        if arr is None or not hasattr(model, name):
            continue
        var_obj = getattr(model, name)
        for z_idx, zone in enumerate(zones):
            for t in periods:
                n_set += _set_warm_value(var_obj, (zone, t), arr[z_idx, t], binary=False)

    for name in two_d_binary:
        arr = get_2d(name)
        if arr is None or not hasattr(model, name):
            continue
        var_obj = getattr(model, name)
        for z_idx, zone in enumerate(zones):
            for t in periods:
                n_set += _set_warm_value(var_obj, (zone, t), arr[z_idx, t], binary=True)

    for name in one_d_continuous:
        arr = get_1d(name)
        if arr is None or not hasattr(model, name):
            continue
        var_obj = getattr(model, name)
        for t in periods:
            n_set += _set_warm_value(var_obj, t, arr[t], binary=False)

    for name in one_d_binary:
        arr = get_1d(name)
        if arr is None or not hasattr(model, name):
            continue
        var_obj = getattr(model, name)
        for t in periods:
            n_set += _set_warm_value(var_obj, t, arr[t], binary=True)

    if binary_sample is not None and binary_sample.ndim == 3 and binary_sample.shape[0] >= n_zones and binary_sample.shape[1] >= n_timesteps:
        if hasattr(model, "u_thermal") and "u_thermal" not in continuous_vars:
            for z_idx, zone in enumerate(zones):
                for t in periods:
                    n_set += _set_warm_value(model.u_thermal, (zone, t), binary_sample[z_idx, t, 6], binary=True)
        if hasattr(model, "dr_active") and "dr_active" not in continuous_vars:
            for z_idx, zone in enumerate(zones):
                for t in periods:
                    n_set += _set_warm_value(model.dr_active, (zone, t), binary_sample[z_idx, t, 4], binary=True)
        if hasattr(model, "b_charge_mode") and "b_charge_mode" not in continuous_vars:
            for z_idx, zone in enumerate(zones):
                for t in periods:
                    charge_mode = 1.0 if binary_sample[z_idx, t, 0] >= binary_sample[z_idx, t, 1] else 0.0
                    n_set += _set_warm_value(model.b_charge_mode, (zone, t), charge_mode, binary=True)
        if hasattr(model, "pumped_charge_mode") and "pumped_charge_mode" not in continuous_vars:
            for z_idx, zone in enumerate(zones):
                for t in periods:
                    charge_mode = 1.0 if binary_sample[z_idx, t, 2] >= binary_sample[z_idx, t, 3] else 0.0
                    n_set += _set_warm_value(model.pumped_charge_mode, (zone, t), charge_mode, binary=True)
        if hasattr(model, "v_thermal_startup") and "v_thermal_startup" not in continuous_vars:
            for z_idx, zone in enumerate(zones):
                prev_u = 0.0
                for t in periods:
                    current_u = 1.0 if binary_sample[z_idx, t, 6] >= 0.5 else 0.0
                    startup = 1.0 if current_u > prev_u else 0.0
                    n_set += _set_warm_value(model.v_thermal_startup, (zone, t), startup, binary=True)
                    prev_u = current_u

    # Derive binary helpers from continuous warm-start if they were absent.
    u_thermal = get_2d("u_thermal")
    if u_thermal is not None and hasattr(model, "v_thermal_startup") and "v_thermal_startup" not in continuous_vars:
        for z_idx, zone in enumerate(zones):
            prev_u = 0.0
            for t in periods:
                current_u = 1.0 if u_thermal[z_idx, t] >= 0.5 else 0.0
                startup = 1.0 if current_u > prev_u else 0.0
                n_set += _set_warm_value(model.v_thermal_startup, (zone, t), startup, binary=True)
                prev_u = current_u

    if hasattr(model, "b_charge_mode") and "b_charge_mode" not in continuous_vars:
        b_charge = get_2d("b_charge")
        b_discharge = get_2d("b_discharge")
        if b_charge is not None and b_discharge is not None:
            for z_idx, zone in enumerate(zones):
                for t in periods:
                    charge_mode = 1.0 if b_charge[z_idx, t] >= b_discharge[z_idx, t] else 0.0
                    n_set += _set_warm_value(model.b_charge_mode, (zone, t), charge_mode, binary=True)

    if hasattr(model, "pumped_charge_mode") and "pumped_charge_mode" not in continuous_vars:
        p_charge = get_2d("pumped_charge")
        p_discharge = get_2d("pumped_discharge")
        if p_charge is not None and p_discharge is not None:
            for z_idx, zone in enumerate(zones):
                for t in periods:
                    charge_mode = 1.0 if p_charge[z_idx, t] >= p_discharge[z_idx, t] else 0.0
                    n_set += _set_warm_value(model.pumped_charge_mode, (zone, t), charge_mode, binary=True)

    if hasattr(model, "dr_active") and "dr_active" not in continuous_vars:
        dr_shed = get_2d("dr_shed")
        if dr_shed is not None:
            for z_idx, zone in enumerate(zones):
                for t in periods:
                    active = 1.0 if dr_shed[z_idx, t] >= 0.05 else 0.0
                    n_set += _set_warm_value(model.dr_active, (zone, t), active, binary=True)

    if hasattr(model, "import_mode") and "import_mode" not in continuous_vars:
        net_import = get_1d("net_import")
        net_export = get_1d("net_export")
        if net_import is not None or net_export is not None:
            for t in periods:
                imp = float(net_import[t]) if net_import is not None else 0.0
                exp = float(net_export[t]) if net_export is not None else 0.0
                import_mode = 1.0 if imp >= exp else 0.0
                n_set += _set_warm_value(model.import_mode, t, import_mode, binary=True)

    return n_set


def _collect_duals_zone_time(model, constraint) -> Dict[Tuple[str, int], float]:
    duals: Dict[Tuple[str, int], float] = {}
    for z in model.Z:
        for t in model.T:
            entry = constraint[z, t]
            duals[(str(z), int(t))] = float(model.dual.get(entry, 0.0))
    return duals


def _collect_duals_line_time(model, constraint) -> Dict[Tuple[str, int], float]:
    duals: Dict[Tuple[str, int], float] = {}
    for l in model.L:
        for t in model.T:
            entry = constraint[l, t]
            duals[(str(l), int(t))] = float(model.dual.get(entry, 0.0))
    return duals


def _build_zone_dispatch_df(detail: Dict[str, Any]):
    import pandas as pd

    zones = detail["zones"]
    time_steps = detail["time_steps"]
    time_hours = detail["time_hours"]
    records = []
    zone_series_keys = [
        "demand",
        "thermal",
        "nuclear",
        "solar",
        "wind",
        "solar_spill",
        "wind_spill",
        "hydro_release",
        "hydro_ror",
        "hydro_spill",
        "battery_charge",
        "battery_discharge",
        "battery_soc",
        "pumped_charge",
        "pumped_discharge",
        "pumped_level",
        "demand_response",
        "unserved",
        "overgen_spill",
    ]

    for idx, step in enumerate(time_steps):
        hour = float(time_hours[idx])
        for zone in zones:
            row = {
                "time_step": int(step),
                "time_hours": hour,
                "zone": zone,
            }
            for key in zone_series_keys:
                row[key] = float(detail[key][zone][idx])
            records.append(row)

    return pd.DataFrame.from_records(records)


def _build_system_dispatch_df(detail: Dict[str, Any]):
    import pandas as pd

    time_steps = detail["time_steps"]
    time_hours = detail["time_hours"]

    aggregated = {
        "thermal_total": _aggregate_zone_series(detail, "thermal"),
        "nuclear_total": _aggregate_zone_series(detail, "nuclear"),
        "solar_total": _aggregate_zone_series(detail, "solar"),
        "wind_total": _aggregate_zone_series(detail, "wind"),
        "hydro_release_total": _aggregate_zone_series(detail, "hydro_release"),
        "hydro_ror_total": _aggregate_zone_series(detail, "hydro_ror"),
        "battery_discharge_total": _aggregate_zone_series(detail, "battery_discharge"),
        "pumped_discharge_total": _aggregate_zone_series(detail, "pumped_discharge"),
        "demand_total": _aggregate_zone_series(detail, "demand"),
        "demand_response_total": _aggregate_zone_series(detail, "demand_response"),
        "unserved_total": _aggregate_zone_series(detail, "unserved"),
        "solar_spill_total": _aggregate_zone_series(detail, "solar_spill"),
        "wind_spill_total": _aggregate_zone_series(detail, "wind_spill"),
        "overgen_spill_total": _aggregate_zone_series(detail, "overgen_spill"),
        "battery_soc_total": _aggregate_zone_series(detail, "battery_soc"),
        "pumped_level_total": _aggregate_zone_series(detail, "pumped_level"),
        "hydro_spill_total": _aggregate_zone_series(detail, "hydro_spill"),
    }

    net_import = detail["net_import"]["values"]
    net_export = detail["net_export"]["values"]

    records = []
    for idx, step in enumerate(time_steps):
        row = {
            "time_step": int(step),
            "time_hours": float(time_hours[idx]),
            "net_import": float(net_import[idx]),
            "net_export": float(net_export[idx]),
        }
        for key, series in aggregated.items():
            row[key] = float(series[idx])
        records.append(row)

    return pd.DataFrame.from_records(records)


def _export_dispatch(detail: Dict[str, Any], csv_prefix: Path | None, hdf_path: Path | None) -> None:
    if not (csv_prefix or hdf_path):
        return
    if detail is None:
        raise RuntimeError("Dispatch detail not captured; rerun with capture_detail=True")

    zone_df = _build_zone_dispatch_df(detail)
    system_df = _build_system_dispatch_df(detail)

    if csv_prefix:
        csv_prefix.parent.mkdir(parents=True, exist_ok=True)
        zone_df.to_csv(csv_prefix.with_name(csv_prefix.name + "_zone.csv"), index=False)
        system_df.to_csv(csv_prefix.with_name(csv_prefix.name + "_system.csv"), index=False)

    if hdf_path:
        hdf_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import pandas as pd  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pandas is required for HDF export") from exc
        zone_df.to_hdf(hdf_path, key="zone", mode="w")
        system_df.to_hdf(hdf_path, key="system", mode="a")


def solve_scenario(
    scenario_path: Path,
    solver_name: str = "highs",
    tee: bool = False,
    capture_detail: bool = False,
    export_csv_prefix: Path | None = None,
    export_hdf: Path | None = None,
    warm_start_data: Dict[str, Any] | None = None,
    time_limit_seconds: float | None = DEFAULT_TIME_LIMIT_SECONDS,
) -> Dict[str, Any]:
    data: ScenarioData = load_scenario_data(Path(scenario_path))
    solver = SolverFactory(solver_name)
    
    # Configure time limit if specified
    if time_limit_seconds is not None and time_limit_seconds > 0:
        _configure_solver_time_limit(solver, solver_name, time_limit_seconds)

    mip_model = build_uc_model(data, enable_duals=False)
    warm_start_count = 0
    if warm_start_data:
        if hasattr(solver, "config") and hasattr(solver.config, "warmstart"):
            solver.config.warmstart = True
        warm_start_count = _apply_mip_warm_start(mip_model, warm_start_data)
    mip_start = time.perf_counter()
    mip_results = solver.solve(mip_model, tee=tee)
    mip_elapsed = time.perf_counter() - mip_start
    mip_reported = getattr(mip_results.solver, 'time', None)
    try:
        mip_reported_seconds = float(mip_reported) if mip_reported is not None else None
    except (TypeError, ValueError):
        mip_reported_seconds = None
    mip_time_seconds = mip_reported_seconds if mip_reported_seconds and mip_reported_seconds > 0 else mip_elapsed
    mip_termination = mip_results.solver.termination_condition
    mip_timed_out = _is_timeout_termination(mip_termination)
    
    # Handle timeout case - use infinity for objective if no feasible solution
    try:
        mip_objective = value(mip_model.obj)
    except Exception:
        mip_objective = float('inf') if mip_timed_out else float('nan')
    
    mip_summary = SolveSummary(
        objective=mip_objective,
        termination_condition=mip_termination,
        solver_status=mip_results.solver.status,
        solve_seconds=mip_time_seconds,
    )
    
    # Compute cost components only if we have a valid solution
    if mip_timed_out and mip_objective == float('inf'):
        cost_components = {}
    else:
        cost_components = _compute_cost_components(mip_model)

    detail_payload = None
    # Skip detail capture if solver timed out without a feasible solution
    if capture_detail and not (mip_timed_out and mip_objective == float('inf')):
        periods = list(mip_model.T)
        zones = [str(z) for z in mip_model.Z]
        horizon = len(periods)
        solar_dispatch = {zone: [float(value(mip_model.p_solar[zone, t])) for t in periods] for zone in zones}
        wind_dispatch = {zone: [float(value(mip_model.p_wind[zone, t])) for t in periods] for zone in zones}
        solar_spill = {zone: [float(value(mip_model.spill_solar[zone, t])) for t in periods] for zone in zones}
        wind_spill = {zone: [float(value(mip_model.spill_wind[zone, t])) for t in periods] for zone in zones}
        detail_payload = {
            "time_steps": [int(t) for t in periods],
            "time_hours": [float(t * data.dt_hours) for t in periods],
            "dt_hours": float(data.dt_hours),
            "zones": zones,
            "lines": [str(lid) for lid in mip_model.L],
            "demand": {
                zone: [float(data.demand[(zone, t)]) for t in periods]
                for zone in zones
            },
            "thermal": {zone: [float(value(mip_model.p_thermal[zone, t])) for t in periods] for zone in zones},
            "nuclear": {zone: [float(value(mip_model.p_nuclear[zone, t])) for t in periods] for zone in zones},
            "solar": solar_dispatch,
            "wind": wind_dispatch,
            "solar_spill": solar_spill,
            "wind_spill": wind_spill,
            "hydro_release": {zone: [float(value(mip_model.h_release[zone, t])) for t in periods] for zone in zones},
            "hydro_ror": {zone: [float(value(mip_model.hydro_ror[zone, t])) for t in periods] for zone in zones},
            "hydro_spill": {zone: [float(value(mip_model.h_spill[zone, t])) for t in periods] for zone in zones},
            "hydro_level": {zone: [float(value(mip_model.h_level[zone, t])) for t in periods] for zone in zones},
            "battery_charge": {zone: [float(value(mip_model.b_charge[zone, t])) for t in periods] for zone in zones},
            "battery_discharge": {zone: [float(value(mip_model.b_discharge[zone, t])) for t in periods] for zone in zones},
            "battery_soc": {zone: [float(value(mip_model.b_soc[zone, t])) for t in periods] for zone in zones},
            "pumped_charge": {zone: [float(value(mip_model.pumped_charge[zone, t])) for t in periods] for zone in zones},
            "pumped_discharge": {zone: [float(value(mip_model.pumped_discharge[zone, t])) for t in periods] for zone in zones},
            "pumped_level": {zone: [float(value(mip_model.pumped_level[zone, t])) for t in periods] for zone in zones},
            "demand_response": {zone: [float(value(mip_model.dr_shed[zone, t])) for t in periods] for zone in zones},
            "dr_rebound": {zone: [float(value(mip_model.dr_rebound[zone, t])) for t in periods] for zone in zones},
            "dr_active": {zone: [float(value(mip_model.dr_active[zone, t])) for t in periods] for zone in zones},
            "unserved": {zone: [float(value(mip_model.unserved[zone, t])) for t in periods] for zone in zones},
            "overgen_spill": {zone: [float(value(mip_model.overgen_spill[zone, t])) for t in periods] for zone in zones},
            "battery_charge_mode": {zone: [float(value(mip_model.b_charge_mode[zone, t])) for t in periods] for zone in zones},
            "pumped_charge_mode": {zone: [float(value(mip_model.pumped_charge_mode[zone, t])) for t in periods] for zone in zones},
            "thermal_commitment": {zone: [float(value(mip_model.u_thermal[zone, t])) for t in periods] for zone in zones},
            "thermal_startup": {zone: [float(value(mip_model.v_thermal_startup[zone, t])) for t in periods] for zone in zones},
            # Nuclear is always ON (must-run baseload) - no commitment binaries
            "net_import": {"values": [float(value(mip_model.net_import[t])) for t in periods]},
            "net_export": {"values": [float(value(mip_model.net_export[t])) for t in periods]},
            "import_mode": {"values": [float(value(mip_model.import_mode[t])) for t in periods]},
            "flows": {
                str(line): [float(value(mip_model.flow[line, t])) for t in periods]
                for line in mip_model.L
            },
            "dr_shed_blocks": {
                f"{zone}_block_{k}": [float(value(mip_model.dr_shed_block[zone, t, k])) for t in periods]
                for zone in zones for k in mip_model.DR_BLOCKS
            },
        }

        anchor_zone = data.import_anchor_zone
        net_import_series = detail_payload.get("net_import", {}).get("values")
        if net_import_series is not None:
            net_import_list = [float(val) for val in net_import_series]
        else:
            net_import_list = [0.0 for _ in periods]

        zero_template = [0.0 for _ in periods]

        pre_dispatch = {
            "thermal": {zone: list(values) for zone, values in detail_payload["thermal"].items()},
            "nuclear": {zone: list(values) for zone, values in detail_payload["nuclear"].items()},
            "solar": {zone: list(values) for zone, values in detail_payload["solar"].items()},
            "wind": {zone: list(values) for zone, values in detail_payload["wind"].items()},
            "hydro_release": {zone: list(values) for zone, values in detail_payload["hydro_release"].items()},
            "hydro_ror": {zone: [float(value(mip_model.hydro_ror[zone, t])) for t in periods] for zone in zones},
            "demand_response": {zone: list(values) for zone, values in detail_payload["demand_response"].items()},
            "battery_charge": {zone: list(values) for zone, values in detail_payload["battery_charge"].items()},
            "battery_discharge": {zone: list(values) for zone, values in detail_payload["battery_discharge"].items()},
            "pumped_charge": {zone: list(values) for zone, values in detail_payload["pumped_charge"].items()},
            "pumped_discharge": {zone: list(values) for zone, values in detail_payload["pumped_discharge"].items()},
            "net_import": {
                zone: (list(net_import_list) if zone == anchor_zone else list(zero_template))
                for zone in zones
            },
            "unserved": {zone: list(zero_template) for zone in zones},
        }

        detail_payload["pre_dispatch"] = pre_dispatch

    lp_model = build_uc_model(data, enable_duals=True)
    _relax_integrality(lp_model)
    lp_start = time.perf_counter()
    lp_results = solver.solve(lp_model, tee=tee)
    lp_elapsed = time.perf_counter() - lp_start
    lp_reported = getattr(lp_results.solver, 'time', None)
    try:
        lp_reported_seconds = float(lp_reported) if lp_reported is not None else None
    except (TypeError, ValueError):
        lp_reported_seconds = None
    lp_time_seconds = lp_reported_seconds if lp_reported_seconds and lp_reported_seconds > 0 else lp_elapsed
    lp_summary = SolveSummary(
        objective=value(lp_model.obj),
        termination_condition=lp_results.solver.termination_condition,
        solver_status=lp_results.solver.status,
        solve_seconds=lp_time_seconds,
    )

    duals: Dict[str, Dict[Tuple[str, int], float]] = {}
    duals: Dict[str, Dict[Tuple[str, int], float]] = {}
    duals["power_balance"] = _collect_duals_zone_time(lp_model, lp_model.power_balance)
    duals["flow_upper"] = _collect_duals_line_time(lp_model, lp_model.flow_upper)
    duals["flow_lower"] = _collect_duals_line_time(lp_model, lp_model.flow_lower)
    duals["battery_soc_limit"] = _collect_duals_zone_time(lp_model, lp_model.battery_soc_limit)
    duals["pumped_level_cap"] = _collect_duals_zone_time(lp_model, lp_model.pumped_level_cap)
    duals["hydro_level_cap"] = _collect_duals_zone_time(lp_model, lp_model.hydro_level_cap)

    if detail_payload is not None:
        _export_dispatch(detail_payload, export_csv_prefix, export_hdf)

    # Build result with timeout classification
    result = {
        "scenario_id": data.scenario_id,
        "mip": mip_summary,
        "lp": lp_summary,
        "cost_components": cost_components,
        "lp_duals": duals,
        "periods": data.periods,
        "zones": data.zones,
        "detail": detail_payload,
    }
    if warm_start_data:
        result["warm_start"] = {
            "requested": True,
            "n_initialized_vars": int(warm_start_count),
        }
    
    # Add timeout classification if applicable
    if mip_timed_out:
        result["feasibility_status"] = "not feasible in operational time"
        result["scenario_classification"] = "extreme (for benchmark only)"
        result["time_limit_seconds"] = time_limit_seconds
    
    return result
