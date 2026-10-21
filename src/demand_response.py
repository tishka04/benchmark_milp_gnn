"""Shared load-shifting contract, diagnostics and solver-free scheduling.

These utilities never solve a model or launch training. Binary DR channels
denote deferral only; recovery is a continuous dispatch decision.
"""
from __future__ import annotations

import numpy as np

DR_FORMULATION = "load_shifting_v1"


class FormulationMismatchError(ValueError):
    """An artifact belongs to a different (or unspecified) physical model."""


def require_current_formulation(payload, context="artifact"):
    """Reject legacy artifacts, including unversioned checkpoints and caches."""
    if not hasattr(payload, "get"):
        raise FormulationMismatchError(f"{context}: unversioned artifact; regenerate for load shifting")
    version = payload.get("dr_formulation")
    if version is None:
        for key in ("detail", "metadata", "meta"):
            nested = payload.get(key)
            if isinstance(nested, np.ndarray) and nested.ndim == 0:
                nested = nested.item()
            if isinstance(nested, dict) and "dr_formulation" in nested:
                version = nested["dr_formulation"]
                break
    if isinstance(version, np.ndarray) and version.size == 1:
        version = version.item()
    if version != DR_FORMULATION:
        raise FormulationMismatchError(
            f"{context}: expected dr_formulation={DR_FORMULATION!r}, got {version!r}. "
            "Regenerate this artifact from load-shifting data; do not relabel "
            "historical shedding artifacts."
        )
    return DR_FORMULATION


def _array(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def shifting_metrics(deferred, recovered, dt_hours, *, backlog=None, active=None,
                     down_limit=None, up_limit=None, max_periods=None, ramp=None,
                     tolerance=1e-5):
    """Diagnostics for complete [zone, time] trajectories, not time slices.

    ``dr_feasible`` covers DR constraints only, not the rest of the UC model.
    Missing recovery must be rejected by callers, never filled with zero.
    """
    down, up = _array(deferred), _array(recovered)
    dt = float(dt_hours)
    if down.ndim != 2 or down.shape != up.shape or not down.size:
        raise ValueError("DR arrays must have the same non-empty [zone, time] shape")
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(down).all() or not np.isfinite(up).all():
        raise ValueError("DR trajectories and positive dt_hours must be finite")
    energy = np.cumsum((down - up) * dt, axis=1)
    previous = np.concatenate([np.zeros((down.shape[0], 1)), energy[:, :-1]], axis=1)
    positive_max = lambda x: float(np.maximum(x, 0).max(initial=0))
    errors = {
        "dr_negative_power_mw": positive_max(-np.minimum(down, up)),
        "dr_causality_violation_mwh": positive_max(up * dt - previous),
        "dr_terminal_backlog_mwh": float(np.abs(energy[:, -1]).max()),
        "dr_simultaneous_mw": positive_max(np.minimum(down, up)),
    }
    if backlog is not None:
        stored = _array(backlog)
        if stored.shape != down.shape or not np.isfinite(stored).all():
            raise ValueError("DR backlog must be finite with shape [zone, time]")
        errors["dr_backlog_residual_mwh"] = float(np.abs(stored - energy).max())
    if active is not None:
        a = _array(active)
        if a.shape != down.shape or not np.isfinite(a).all():
            raise ValueError("DR activation must be finite with shape [zone, time]")
        errors["dr_activation_violation_mw"] = max(
            positive_max(down * (a < 0.5)),
            positive_max(up * (a >= 0.5)),
            positive_max(0.1 * (a >= 0.5) - down),
        )
        errors["dr_binary_violation"] = float(np.minimum(np.abs(a), np.abs(a - 1)).max())
    if down_limit is not None:
        errors["dr_deferral_capacity_violation_mw"] = positive_max(down - _array(down_limit))
    if up_limit is not None:
        errors["dr_recovery_capacity_violation_mw"] = positive_max(up - _array(up_limit))
    if max_periods is not None:
        used = (_array(active) >= 0.5).sum(axis=1) if active is not None else (down > tolerance).sum(axis=1)
        errors["dr_active_period_violation"] = positive_max(used - _array(max_periods))
    if ramp is not None and down.shape[1] > 1:
        errors["dr_ramp_violation_mw"] = positive_max(np.abs(np.diff(down, axis=1)) - _array(ramp).reshape(-1, 1))
    return {
        **errors,
        "dr_deferred_mwh": float(down.sum() * dt),
        "dr_recovered_mwh": float(up.sum() * dt),
        "dr_energy_error_mwh": float(np.abs((down - up).sum(axis=1) * dt).max()),
        "dr_feasible": all(error <= tolerance for error in errors.values()),
    }


def validate_dr_report(report, n_timesteps=None):
    """Check provenance, full horizon and the DR physical identities of labels."""
    require_current_formulation(report, "MILP report")
    detail = report.get("detail", report)
    zones = report.get("zones") or detail.get("zones") or list(detail.get("demand_response", {}))
    if not zones:
        raise ValueError("Load-shifting report has no zones")
    for key in ("demand_response", "dr_recover", "dr_rebound"):
        if key not in detail or any(z not in detail[key] for z in zones):
            raise ValueError(f"Load-shifting report missing {key} for one or more zones")
    down = np.asarray([detail["demand_response"][z] for z in zones], dtype=float)
    up = np.asarray([detail["dr_recover"][z] for z in zones], dtype=float)
    if n_timesteps is not None and down.shape[1] != n_timesteps:
        raise ValueError("Load-shifting labels must retain the complete horizon; no truncation/padding")
    active = detail.get("dr_active")
    metrics = shifting_metrics(
        down, up, detail["dt_hours"],
        backlog=[detail["dr_rebound"][z] for z in zones],
        active=[active[z] for z in zones] if active is not None else None,
        tolerance=1e-3,
    )
    if not metrics["dr_feasible"]:
        raise ValueError(f"Invalid load-shifting labels: {metrics}")
    return zones


def schedule_shifting(requested, recovery_headroom, down_limit, up_limit, dt_hours,
                      max_periods, ramp):
    """Conservative greedy postponement with reserved future recovery.

    Reserve recovery only from forecast headroom. Each deferral is bounded by
    the ramp limit, a conservative subset of the MILP ramp constraints. This
    is a heuristic, not an optimisation or feasibility certificate for UC.
    """
    req = np.maximum(_array(requested), 0)
    headroom = np.maximum(_array(recovery_headroom), 0)
    down_cap, up_cap = np.maximum(_array(down_limit), 0), np.maximum(_array(up_limit), 0)
    if req.ndim != 2 or any(a.shape != req.shape for a in (headroom, down_cap, up_cap)):
        raise ValueError("Scheduling inputs must share [zone, time] shape")
    if not all(np.isfinite(a).all() for a in (req, headroom, down_cap, up_cap)):
        raise ValueError("Scheduling inputs must be finite")
    if not np.isfinite(dt_hours) or dt_hours <= 0:
        raise ValueError("dt_hours must be positive")
    Z, T = req.shape
    budgets = np.broadcast_to(_array(max_periods), (Z,))
    ramps = np.broadcast_to(_array(ramp), (Z,))
    down, up = np.zeros_like(req), np.zeros_like(req)
    for z in range(Z):
        used = 0
        for t in range(T - 1):
            if up[z, t] > 0 or used >= budgets[z]:
                continue
            spare = np.maximum(np.minimum(headroom[z], up_cap[z]) - up[z], 0)
            spare[:t + 1] = 0
            spare[down[z] > 0] = 0
            amount = min(req[z, t], down_cap[z, t], ramps[z], spare.sum())
            if amount < 0.1:
                continue
            down[z, t] = amount
            used += 1
            remaining = amount
            for future in range(t + 1, T):
                recovered = min(remaining, spare[future])
                up[z, future] += recovered
                remaining -= recovered
                if remaining <= 1e-10:
                    break
    backlog = np.cumsum((down - up) * dt_hours, axis=1)
    return down, up, backlog


def physics_dr_schedule(physics, requested=None):
    """Build a forecast-based schedule using exact MILP DR caps when available."""
    demand = _array(physics.demand)
    zero = np.zeros_like(demand)
    free = sum(_array(getattr(physics, key)) if getattr(physics, key) is not None else zero
               for key in ("solar_available", "wind_available", "hydro_ror"))
    down_cap = getattr(physics, "dr_limit_mw", None)
    if down_cap is None:
        cap = getattr(physics, "dr_capacity_mw", None)
        down_cap = zero if cap is None else np.minimum(_array(cap)[:, None], demand)
    up_cap = getattr(physics, "dr_recovery_limit_mw", None)
    if up_cap is None:
        up_cap = down_cap
    budgets = getattr(physics, "dr_max_active_periods", None)
    if budgets is None:
        budgets = int(physics.dr_max_duration_hours / physics.dt_hours)
    ramp = getattr(physics, "dr_ramp_mw", None)
    if ramp is None:
        ramp = _array(down_cap).max(axis=1)
    request = np.maximum(demand - free, 0) if requested is None else requested
    return schedule_shifting(request, np.maximum(free - demand, 0), down_cap, up_cap,
                             physics.dt_hours, budgets, ramp)
