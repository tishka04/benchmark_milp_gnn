from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from pyomo.environ import SolverFactory, TerminationCondition, value
from pyomo.core import TransformationFactory
from pyomo.opt import SolverStatus

from .model import build_uc_model
from .scenario_loader import ScenarioData, load_scenario_data


@dataclass
class SolveSummary:
    objective: float
    termination_condition: TerminationCondition
    solver_status: SolverStatus

    def as_dict(self) -> Dict[str, Any]:
        return {
            "objective": float(self.objective),
            "termination": self.termination_condition.name,
            "status": self.solver_status.name,
        }


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

    scalar_dr_cost = float(value(model.dr_cost))
    scalar_voll = float(value(model.voll))
    scalar_res_spill = float(value(model.res_spill_cost))
    scalar_hydro_spill = float(value(model.hydro_spill_cost))
    scalar_overgen = float(value(model.overgen_spill_cost))
    scalar_import = float(value(model.import_cost))
    scalar_export = float(value(model.export_cost))

    components["demand_response"] = scalar_dr_cost * _sum_var(model, model.dr_shed)
    components["unserved_energy"] = scalar_voll * _sum_var(model, model.unserved)

    solar_spill = scalar_res_spill * _sum_var(model, model.spill_solar)
    wind_spill = scalar_res_spill * _sum_var(model, model.spill_wind)
    components["solar_spill"] = solar_spill
    components["wind_spill"] = wind_spill
    components["renewable_spill"] = solar_spill + wind_spill

    components["hydro_spill"] = scalar_hydro_spill * _sum_var(model, model.h_spill)
    components["overgen_spill"] = scalar_overgen * _sum_var(model, model.overgen_spill)
    components["imports"] = scalar_import * _sum_time_var(model, model.net_import)
    components["exports"] = scalar_export * _sum_time_var(model, model.net_export)
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
        "renewable",
        "solar_spill",
        "wind_spill",
        "renewable_spill",
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
        "renewable_total": _aggregate_zone_series(detail, "renewable"),
        "hydro_release_total": _aggregate_zone_series(detail, "hydro_release"),
        "hydro_ror_total": _aggregate_zone_series(detail, "hydro_ror"),
        "battery_discharge_total": _aggregate_zone_series(detail, "battery_discharge"),
        "pumped_discharge_total": _aggregate_zone_series(detail, "pumped_discharge"),
        "demand_total": _aggregate_zone_series(detail, "demand"),
        "demand_response_total": _aggregate_zone_series(detail, "demand_response"),
        "unserved_total": _aggregate_zone_series(detail, "unserved"),
        "solar_spill_total": _aggregate_zone_series(detail, "solar_spill"),
        "wind_spill_total": _aggregate_zone_series(detail, "wind_spill"),
        "renewable_spill_total": _aggregate_zone_series(detail, "renewable_spill"),
        "overgen_spill_total": _aggregate_zone_series(detail, "overgen_spill"),
        "battery_soc_total": _aggregate_zone_series(detail, "battery_soc"),
        "pumped_level_total": _aggregate_zone_series(detail, "pumped_level"),
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
) -> Dict[str, Any]:
    data: ScenarioData = load_scenario_data(Path(scenario_path))
    solver = SolverFactory(solver_name)

    mip_model = build_uc_model(data, enable_duals=False)
    mip_results = solver.solve(mip_model, tee=tee)
    mip_summary = SolveSummary(
        objective=value(mip_model.obj),
        termination_condition=mip_results.solver.termination_condition,
        solver_status=mip_results.solver.status,
    )

    cost_components = _compute_cost_components(mip_model)

    detail_payload = None
    if capture_detail:
        periods = list(mip_model.T)
        zones = [str(z) for z in mip_model.Z]
        horizon = len(periods)
        solar_dispatch = {zone: [float(value(mip_model.p_solar[zone, t])) for t in periods] for zone in zones}
        wind_dispatch = {zone: [float(value(mip_model.p_wind[zone, t])) for t in periods] for zone in zones}
        solar_spill = {zone: [float(value(mip_model.spill_solar[zone, t])) for t in periods] for zone in zones}
        wind_spill = {zone: [float(value(mip_model.spill_wind[zone, t])) for t in periods] for zone in zones}
        renewable_dispatch = {
            zone: [solar_dispatch[zone][idx] + wind_dispatch[zone][idx] for idx in range(horizon)]
            for zone in zones
        }
        renewable_spill = {
            zone: [solar_spill[zone][idx] + wind_spill[zone][idx] for idx in range(horizon)]
            for zone in zones
        }
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
            "renewable": renewable_dispatch,
            "solar_spill": solar_spill,
            "wind_spill": wind_spill,
            "renewable_spill": renewable_spill,
            "hydro_release": {zone: [float(value(mip_model.h_release[zone, t])) for t in periods] for zone in zones},
            "hydro_ror": {zone: [float(value(mip_model.hydro_ror[zone, t])) for t in periods] for zone in zones},
            "hydro_spill": {zone: [float(value(mip_model.h_spill[zone, t])) for t in periods] for zone in zones},
            "battery_charge": {zone: [float(value(mip_model.b_charge[zone, t])) for t in periods] for zone in zones},
            "battery_discharge": {zone: [float(value(mip_model.b_discharge[zone, t])) for t in periods] for zone in zones},
            "battery_soc": {zone: [float(value(mip_model.b_soc[zone, t])) for t in periods] for zone in zones},
            "pumped_charge": {zone: [float(value(mip_model.pumped_charge[zone, t])) for t in periods] for zone in zones},
            "pumped_discharge": {zone: [float(value(mip_model.pumped_discharge[zone, t])) for t in periods] for zone in zones},
            "pumped_level": {zone: [float(value(mip_model.pumped_level[zone, t])) for t in periods] for zone in zones},
            "demand_response": {zone: [float(value(mip_model.dr_shed[zone, t])) for t in periods] for zone in zones},
            "unserved": {zone: [float(value(mip_model.unserved[zone, t])) for t in periods] for zone in zones},
            "overgen_spill": {zone: [float(value(mip_model.overgen_spill[zone, t])) for t in periods] for zone in zones},
            "net_import": {"values": [float(value(mip_model.net_import[t])) for t in periods]},
            "net_export": {"values": [float(value(mip_model.net_export[t])) for t in periods]},
            "flows": {
                str(line): [float(value(mip_model.flow[line, t])) for t in periods]
                for line in mip_model.L
            },
        }

    lp_model = build_uc_model(data, enable_duals=True)
    _relax_integrality(lp_model)
    lp_results = solver.solve(lp_model, tee=tee)
    lp_summary = SolveSummary(
        objective=value(lp_model.obj),
        termination_condition=lp_results.solver.termination_condition,
        solver_status=lp_results.solver.status,
    )

    duals: Dict[str, Dict[Tuple[str, int], float]] = {}
    duals["power_balance"] = _collect_duals_zone_time(lp_model, lp_model.power_balance)
    duals["flow_upper"] = _collect_duals_line_time(lp_model, lp_model.flow_upper)
    duals["flow_lower"] = _collect_duals_line_time(lp_model, lp_model.flow_lower)
    duals["battery_soc_limit"] = _collect_duals_zone_time(lp_model, lp_model.battery_soc_limit)
    duals["pumped_level_cap"] = _collect_duals_zone_time(lp_model, lp_model.pumped_level_cap)
    duals["hydro_level_cap"] = _collect_duals_zone_time(lp_model, lp_model.hydro_level_cap)

    if detail_payload is not None:
        _export_dispatch(detail_payload, export_csv_prefix, export_hdf)

    return {
        "scenario_id": data.scenario_id,
        "mip": mip_summary,
        "lp": lp_summary,
        "cost_components": cost_components,
        "lp_duals": duals,
        "periods": data.periods,
        "zones": data.zones,
        "detail": detail_payload,
    }
