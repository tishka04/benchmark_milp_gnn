"""Independent full-dispatch equations and physical costs (MW, MWh, currency).

No solver/model expressions are used by validate_dispatch. Extraction is strict:
missing variables, NaNs, and fractional decisions are never repaired by validation.
"""
from __future__ import annotations

import numpy as np
from .common import PHYSICS

ZONE_VARS = ("p_thermal", "u_thermal", "v_thermal_startup", "p_nuclear", "p_solar",
             "spill_solar", "p_wind", "spill_wind", "dr_shed", "dr_recover", "dr_rebound",
             "dr_active", "unserved", "b_charge", "b_discharge", "b_soc", "b_charge_mode",
             "h_release", "h_spill", "h_level", "pumped_charge", "pumped_discharge",
             "pumped_level", "pumped_charge_mode", "overgen_spill")
SYSTEM_VARS = ("net_import", "net_export", "import_mode")
BINARY_VARS = ("u_thermal", "v_thermal_startup", "dr_active", "b_charge_mode",
               "pumped_charge_mode", "import_mode")
REPORT_NAMES = dict(zip(ZONE_VARS, (
    "thermal", "thermal_commitment", "thermal_startup", "nuclear", "solar", "solar_spill",
    "wind", "wind_spill", "demand_response", "dr_recover", "dr_rebound", "dr_active",
    "unserved", "battery_charge", "battery_discharge", "battery_soc", "battery_charge_mode",
    "hydro_release", "hydro_spill", "hydro_level", "pumped_charge", "pumped_discharge",
    "pumped_level", "pumped_charge_mode", "overgen_spill")))


def extract_dispatch(model):
    from pyomo.environ import value
    zones, times, lines = list(model.Z), list(model.T), list(model.L)
    v = {name: [[float(value(getattr(model, name)[z, t])) for t in times] for z in zones]
         for name in ZONE_VARS}
    v.update({name: [float(value(getattr(model, name)[t])) for t in times] for name in SYSTEM_VARS})
    v["flow"] = [[float(value(model.flow[l, t])) for t in times] for l in lines]
    v["dr_shed_block"] = [[[float(value(model.dr_shed_block[z, t, k])) for k in model.DR_BLOCKS]
                            for t in times] for z in zones]
    return dict(physics=PHYSICS, zones=zones, periods=times, lines=lines, values=v)


def from_report(detail, data):
    """Exact conversion only; absent flows/blocks/binaries invalidate old reports."""
    z, ts, ls = data.zones, data.periods, list(data.lines)
    if detail["zones"] != z or detail["time_steps"] != ts or detail["dt_hours"] != data.dt_hours:
        raise ValueError("Report axes/time step mismatch")
    vals = {k: [detail[REPORT_NAMES[k]][zone] for zone in z] for k in ZONE_VARS}
    vals.update({k: detail[k]["values"] for k in SYSTEM_VARS})
    vals["flow"] = [detail["flows"][line] for line in ls]
    vals["dr_shed_block"] = [[[detail["dr_shed_blocks"][f"{zone}_block_{k}"][t]
                              for k in range(data.dr_num_blocks)] for t in ts] for zone in z]
    return dict(physics=PHYSICS, zones=z, periods=ts, lines=ls, values=vals)


def to_report(dispatch, data):
    d = dict(zones=data.zones, time_steps=data.periods, dt_hours=data.dt_hours,
             time_hours=[t * data.dt_hours for t in data.periods], lines=list(data.lines),
             dr_formulation="load_shifting_v1", thermal_formulation="thermal_transitions_v1")
    v = dispatch["values"]
    d.update({REPORT_NAMES[k]: dict(zip(data.zones, v[k])) for k in ZONE_VARS})
    d.update({k: {"values": v[k]} for k in SYSTEM_VARS})
    d["flows"] = dict(zip(data.lines, v["flow"]))
    d["dr_shed_blocks"] = {f"{z}_block_{k}": [row[k] for row in v["dr_shed_block"][i]]
                            for i, z in enumerate(data.zones) for k in range(data.dr_num_blocks)}
    return d


def apply_cost_units(model):
    """Correct the old power-sum objective locally; preserve repair penalties.

    Identical at dt=1. Startup is a per-event cost and is never time-scaled.
    This wrapper leaves the historical builder and its callers unchanged.
    """
    from pyomo.environ import value, Objective
    if getattr(model, "_operational_cost_units", False):
        return
    dt = float(value(model.dt_hours))
    base = model.base_obj_expr if hasattr(model, "base_obj_expr") else model.obj.expr
    starts = sum(model.thermal_startup_cost[z] * model.v_thermal_startup[z, t]
                 for z in model.Z for t in model.T)
    corrected = dt * (base - starts) + starts
    for objective in model.component_data_objects(Objective, active=True):
        objective.set_value(objective.expr + corrected - base)
    if not model.obj.active:
        model.obj.set_value(corrected)
    if hasattr(model, "base_obj_expr"):
        model.base_obj_expr = corrected
    model._operational_cost_units = True


def physical_cost(data, v):
    dt = data.dt_hours
    col = lambda name: np.array([getattr(data, name)[z] for z in data.zones])[:, None]
    out = {
        "thermal_fuel": np.sum(col("thermal_cost") * v["p_thermal"]) * dt,
        "thermal_startup": np.sum(col("thermal_startup_cost") * v["v_thermal_startup"]),
        "nuclear_fuel": np.sum(col("nuclear_cost") * v["p_nuclear"]) * dt,
        "demand_response": np.sum(v["dr_shed_block"] * np.array(
            [data.dr_block_cost[k] for k in range(data.dr_num_blocks)])) * dt,
        "unserved_energy": np.sum(v["unserved"]) * data.voll * dt,
        "renewable_spill": np.sum(v["spill_solar"] + v["spill_wind"]) * data.variable_spill_cost * dt,
        "hydro_spill": np.sum(v["h_spill"]) * data.hydro_spill_cost * dt,
        "overgen_spill": np.sum(v["overgen_spill"]) * data.overgen_spill_penalty * dt,
        "battery_cycle": np.sum(col("battery_cycle_cost") * (v["b_charge"] + v["b_discharge"])) * dt,
        "pumped_cycle": np.sum(col("pumped_cycle_cost") * (v["pumped_charge"] + v["pumped_discharge"])) * dt,
        "imports": np.sum(v["net_import"]) * data.import_cost * dt,
        "exports": np.sum(v["net_export"]) * data.export_cost * dt,
    }
    return {k: float(x) for k, x in out.items()}


def validate_dispatch(data, dispatch, physical_tol=1e-3, integer_tol=1e-5):
    from src.thermal import scenario_thermal_limits
    try:
        if (dispatch["physics"] != PHYSICS or dispatch["zones"] != data.zones
                or dispatch["periods"] != data.periods or dispatch["lines"] != list(data.lines)):
            raise ValueError("dispatch axes or physics mismatch")
        Z, T, L = len(data.zones), len(data.periods), len(data.lines)
        shapes = {**{k: (Z, T) for k in ZONE_VARS}, **{k: (T,) for k in SYSTEM_VARS},
                  "flow": (L, T), "dr_shed_block": (Z, T, data.dr_num_blocks)}
        v = {k: np.asarray(dispatch["values"][k], dtype=float) for k in shapes}
        if L == 0 and v["flow"].size == 0:
            v["flow"] = v["flow"].reshape(0, T)
        for k, shape in shapes.items():
            if v[k].shape != shape or not np.isfinite(v[k]).all():
                raise ValueError(f"missing/nonfinite/malformed dispatch variable: {k}")
        dt = data.dt_hours
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("invalid dt_hours")
        res = {}
        def le(name, lhs, rhs=0):
            res[name] = max(res.get(name, 0), float(np.max(np.asarray(lhs) - rhs, initial=0)))
        def eq(name, lhs, rhs=0):
            le(name, np.abs(np.asarray(lhs) - rhs))
        def col(name):
            return np.array([getattr(data, name)[z] for z in data.zones])[:, None]
        def series(name):
            return np.array([[getattr(data, name)[z, t] for t in data.periods] for z in data.zones])
        def prev(x, initial):
            return np.concatenate((np.broadcast_to(initial, (Z, 1)), x[:, :-1]), axis=1)
        for k in shapes:
            if k != "flow":
                le("nonnegative:" + k, -v[k])
        integ = max(float(np.max(np.abs(v[k] - np.rint(v[k])), initial=0)) for k in BINARY_VARS)
        for k in BINARY_VARS:
            le("binary_bounds:" + k, v[k], 1)
        limits = scenario_thermal_limits(data)
        tc = lambda name: np.array([getattr(limits[z], name) for z in data.zones])[:, None]
        p, u, start = v["p_thermal"], v["u_thermal"], v["v_thermal_startup"]
        pu, pp = prev(u, tc("initial_on")), prev(p, col("thermal_initial_output"))
        stop = pu - u + start
        le("thermal_capacity", p, col("thermal_capacity") * u)
        le("thermal_min", col("thermal_min_power") * u, p)
        le("startup_lower", u - pu, start)
        le("startup_on", start, u)
        le("startup_previous_off", start, 1 - pu)
        le("thermal_ramp_up", p - pp, col("thermal_ramp") * pu + tc("startup") * start)
        le("thermal_ramp_down", pp - p, col("thermal_ramp") * u + tc("shutdown") * stop)
        le("nuclear_cap", v["p_nuclear"], col("nuclear_capacity"))
        le("nuclear_min", col("nuclear_min_power"), v["p_nuclear"])
        le("nuclear_ramp", np.abs(np.diff(v["p_nuclear"], axis=1)), .5 * col("nuclear_capacity"))
        for kind in ("solar", "wind"):
            eq(kind + "_balance", v["p_" + kind] + v["spill_" + kind], series(kind + "_available"))
        blocks = np.array([[[data.dr_block_limit[z, t, k] for k in range(data.dr_num_blocks)]
                            for t in data.periods] for z in data.zones])
        le("dr_blocks_cap", v["dr_shed_block"], blocks)
        eq("dr_blocks_sum", v["dr_shed_block"].sum(axis=2), v["dr_shed"])
        lim, active = series("dr_limit"), v["dr_active"]
        le("dr_power", v["dr_shed"], lim)
        le("dr_active_upper", v["dr_shed"], lim * active)
        le("dr_active_lower", .1 * active, v["dr_shed"])
        le("dr_recovery_cap", v["dr_recover"], lim * data.dr_recovery_limit_factor * (1 - active))
        debt = prev(v["dr_rebound"], 0)
        eq("dr_energy", v["dr_rebound"], debt + dt * (v["dr_shed"] - v["dr_recover"]))
        le("dr_causality", dt * v["dr_recover"], debt)
        eq("dr_terminal", v["dr_rebound"][:, -1])
        le("dr_active_periods", active.sum(axis=1, keepdims=True), col("dr_max_events"))
        le("dr_ramp", np.abs(np.diff(v["dr_shed"], axis=1)), col("dr_ramp_limit"))
        for prefix, c, d, e, mode in (("battery", "b_charge", "b_discharge", "b_soc", "b_charge_mode"),
                                      ("pumped", "pumped_charge", "pumped_discharge", "pumped_level", "pumped_charge_mode")):
            cap, energy = col(prefix + "_power"), col(prefix + "_energy")
            le(prefix + "_charge", v[c], cap * v[mode])
            le(prefix + "_discharge", v[d], cap * (1 - v[mode]))
            expected = col(prefix + "_retention") * prev(v[e], col(prefix + "_initial")) + dt * (
                getattr(data, prefix + "_eta_charge") * v[c] - v[d] / max(getattr(data, prefix + "_eta_discharge"), 1e-4))
            if prefix == "pumped":
                expected = np.where(energy <= 1e-6, 0, expected)
            eq(prefix + "_energy", v[e], expected)
            le(prefix + "_capacity", v[e], energy)
            for sign, bound in ((1, "max"), (-1, "min")):
                bnd = col(prefix + "_final_" + bound)
                keep = (energy[:, 0] > 1e-6) & (bnd[:, 0] > 1e-6)
                le(prefix + "_final_" + bound, sign * v[e][keep, -1], sign * bnd[keep, 0])
        le("hydro_release_cap", v["h_release"], col("hydro_res_capacity"))
        expected = prev(v["h_level"], col("hydro_initial")) + dt * (series("hydro_inflow_power") - v["h_release"] - v["h_spill"])
        eq("hydro_energy", v["h_level"], np.where(col("hydro_res_energy") <= 1e-6, 0, expected))
        le("hydro_capacity", v["h_level"], col("hydro_res_energy"))
        le("import_exclusivity", v["net_import"], data.import_capacity * v["import_mode"])
        le("export_exclusivity", v["net_export"], data.import_capacity * (1 - v["import_mode"]))
        net = np.zeros((Z, T))
        for i, line in enumerate(data.lines.values()):
            le("line_capacity:" + line.name, np.abs(v["flow"][i]), line.capacity_mw)
            net[data.zones.index(line.from_zone)] -= v["flow"][i]
            net[data.zones.index(line.to_zone)] += v["flow"][i]
        net[data.zones.index(data.import_anchor_zone)] += v["net_import"] - v["net_export"]
        generation = sum(v[k] for k in ("p_thermal", "p_nuclear", "p_solar", "p_wind", "h_release", "b_discharge", "pumped_discharge"))
        eq("nodal_balance", generation + series("hydro_ror_generation") + net + v["dr_shed"] + v["unserved"],
           series("demand") + v["dr_recover"] + v["b_charge"] + v["pumped_charge"] + v["overgen_spill"])
        for name, cap in (("u_thermal", "thermal_capacity"), ("v_thermal_startup", "thermal_capacity"),
                          ("b_charge_mode", "battery_power"), ("pumped_charge_mode", "pumped_power")):
            eq("zero_capacity:" + name, v[name][col(cap)[:, 0] <= 1e-6])
        eq("zero_dr", v["dr_active"][lim <= 1e-6])
        if data.import_capacity <= 1e-6:
            eq("zero_import", v["import_mode"])
        costs = physical_cost(data, v)
        if not all(np.isfinite(x) for x in costs.values()):
            raise ValueError("nonfinite physical cost")
        violations = {k: x for k, x in res.items() if x > physical_tol}
        return dict(valid=not violations and integ <= integer_tol, physical_tolerance=physical_tol,
                    integer_tolerance=integer_tol, max_physical_residual=max(res.values(), default=0),
                    max_integrality_residual=integ, violations=violations, cost=sum(costs.values()),
                    cost_components=costs, ens_mwh=float(v["unserved"].sum() * dt),
                    overgen_mwh=float(v["overgen_spill"].sum() * dt),
                    renewable_spill_mwh=float((v["spill_solar"] + v["spill_wind"]).sum() * dt),
                    hydro_spill_mwh=float(v["h_spill"].sum() * dt))
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return dict(valid=False, error=f"{type(exc).__name__}: {exc}")
