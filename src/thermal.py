"""Shared one-period thermal startup/shutdown convention (MW per transition).

``ramp`` retains the repository's MW-per-period convention while online.
Startup and shutdown limits default to min(Pmax, max(Pmin, ramp)). This is
an explicit single-period UC approximation, not a multi-period startup model.
"""
from dataclasses import dataclass
import math

THERMAL_FORMULATION = "thermal_transitions_v1"


@dataclass(frozen=True)
class ThermalLimits:
    capacity: float
    minimum: float
    ramp: float
    startup: float
    shutdown: float
    initial_power: float
    initial_on: int


def thermal_limits(capacity, minimum, ramp, initial_power=0.0, *,
                   startup=None, shutdown=None, initial_on=None):
    """Validate parameters, including a consistent state before period zero."""
    capacity, minimum, ramp, initial_power = map(float, (capacity, minimum, ramp, initial_power))
    default = min(capacity, max(minimum, ramp))
    startup = default if startup is None else float(startup)
    shutdown = default if shutdown is None else float(shutdown)
    initial_on = int(initial_power > 0.0) if initial_on is None else float(initial_on)
    if not all(math.isfinite(x) for x in
               (capacity, minimum, ramp, initial_power, startup, shutdown, initial_on)):
        raise ValueError("Thermal parameters must be finite")
    if not (0 <= minimum <= capacity and ramp >= 0):
        raise ValueError("Thermal parameters require 0 <= Pmin <= Pmax and ramp >= 0")
    if not (minimum <= startup <= capacity and minimum <= shutdown <= capacity):
        raise ValueError("One-period thermal startup/shutdown limits must lie in [Pmin, Pmax]")
    if initial_on not in (0, 1) or (capacity == 0 and initial_on != 0):
        raise ValueError("Thermal initial commitment must be binary and off when capacity is zero")
    if not minimum * initial_on <= initial_power <= capacity * initial_on:
        raise ValueError("Thermal initial output is inconsistent with initial commitment/Pmin/Pmax")
    return ThermalLimits(capacity, minimum, ramp, startup, shutdown, initial_power, int(initial_on))


def scenario_thermal_limits(data):
    """Resolve optional per-zone overrides; legacy scenarios keep their inputs."""
    return {
        z: thermal_limits(
            data.thermal_capacity[z], data.thermal_min_power[z], data.thermal_ramp[z],
            data.thermal_initial_output[z],
            startup=data.thermal_startup_limit.get(z),
            shutdown=data.thermal_shutdown_limit.get(z),
            initial_on=data.thermal_initial_on.get(z),
        ) for z in data.zones
    }


def dispatch_bounds(unit, previous_power, previous_on):
    """Return online interval and whether switching off is allowed this period."""
    if previous_on:
        return (max(unit.minimum, previous_power - unit.ramp),
                min(unit.capacity, previous_power + unit.ramp),
                previous_power <= unit.shutdown + 1e-8)
    return unit.minimum, unit.startup, True


def project_dispatch(unit, previous_power, previous_on, requested, want_on):
    """Thermally consistent heuristic proposal; not a whole-system certificate."""
    lower, upper, can_stop = dispatch_bounds(unit, previous_power, previous_on)
    if unit.capacity == 0 or (not want_on and can_stop):
        return 0.0, 0.0
    return min(upper, max(lower, float(requested))), 1.0


def thermal_metrics(power, commitment, startup, *, capacity, minimum, ramp,
                    startup_limit, shutdown_limit, initial_power, initial_on, tolerance=1e-3):
    """Independent array check of the same thermal equations; no solver call."""
    import numpy as np
    p, u, v = (np.asarray(x, dtype=float) for x in (power, commitment, startup))
    if p.ndim != 2 or not p.size or u.shape != p.shape or v.shape != p.shape:
        raise ValueError("Thermal schedules must share a nonempty [zone, time] shape")
    params = [np.asarray(x, dtype=float).reshape(-1, 1) for x in
              (capacity, minimum, ramp, startup_limit, shutdown_limit, initial_power, initial_on)]
    if any(x.shape != (p.shape[0], 1) for x in params):
        raise ValueError("Thermal parameters must have one value per zone")
    if not all(np.isfinite(x).all() for x in (p, u, v, *params)):
        return {"thermal_feasible": False, "thermal_nonfinite": True}
    cap, low, rate, su, sd, p0, u0 = params
    pp, pu = np.concatenate([p0, p[:, :-1]], axis=1), np.concatenate([u0, u[:, :-1]], axis=1)
    w = pu - u + v
    pos = lambda x: float(np.maximum(x, 0).max(initial=0))
    errors = {
        "thermal_capacity_violation_mw": pos(p - cap * u),
        "thermal_minimum_violation_mw": pos(low * u - p),
        "thermal_absent_unit_commitment": pos(u * (cap == 0)),
        "thermal_binary_violation": max(float(np.abs(u - np.rint(u)).max()),
                                         float(np.abs(v - np.rint(v)).max()),
                                         pos(-u), pos(u - 1), pos(-v), pos(v - 1)),
        "thermal_startup_violation": max(pos(u - pu - v), pos(v - u), pos(v + pu - 1)),
        "thermal_ramp_up_violation_mw": pos(p - pp - rate * pu - su * v),
        "thermal_ramp_down_violation_mw": pos(pp - p - rate * u - sd * w),
    }
    return {**errors, "thermal_feasible": all(x <= tolerance for x in errors.values())}
