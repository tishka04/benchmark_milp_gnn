from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Dict, Tuple

from pyomo.environ import SolverFactory, TerminationCondition, value
from pyomo.core import TransformationFactory
from pyomo.opt import SolverStatus

from .model import build_uc_model
from .scenario_loader import ScenarioData, load_scenario_data

# Default time limit: 2 hours in seconds
DEFAULT_TIME_LIMIT_SECONDS = 2 * 60 * 60  # 14400 seconds


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
    time_limit_seconds: float | None = DEFAULT_TIME_LIMIT_SECONDS,
) -> Dict[str, Any]:
    data: ScenarioData = load_scenario_data(Path(scenario_path))
    solver = SolverFactory(solver_name)
    
    # Configure time limit if specified
    if time_limit_seconds is not None and time_limit_seconds > 0:
        _configure_solver_time_limit(solver, solver_name, time_limit_seconds)

    mip_model = build_uc_model(data, enable_duals=False)
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
            "nuclear_commitment": {zone: [float(value(mip_model.u_nuclear[zone, t])) for t in periods] for zone in zones},
            "nuclear_startup": {zone: [float(value(mip_model.v_nuclear_startup[zone, t])) for t in periods] for zone in zones},
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
    
    # Add timeout classification if applicable
    if mip_timed_out:
        result["feasibility_status"] = "not feasible in operational time"
        result["scenario_classification"] = "extreme (for benchmark only)"
        result["time_limit_seconds"] = time_limit_seconds
    
    return result
