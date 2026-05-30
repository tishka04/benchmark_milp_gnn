"""Critical-zone selection, heuristic fill, and per-arm commitment producers.

Operationalises Section 2--3 of ``docs/partial_grid_recovery_protocol.md``.

The four protocol arms differ only in the commitment vector ``u`` they produce;
the post-fault redispatch (``contingency_redispatch.py``) is shared:

* Arm A -- Full MILP            : :func:`commitment_full_milp`
* Arm B -- Partial MILP + fill  : :func:`commitment_partial_milp`
* Arm C -- Pipeline (full)      : :func:`commitment_from_binary_tensor`
* Arm D -- Pipeline-on-S* + fill: :func:`commitment_from_binary_tensor` restricted to S*,
  merged with :func:`heuristic_fill_commitment`.

Critical-zone subsets ``S*`` are precomputed *before* solving (no peeking at the
post-contingency outcome), via three selection rules: ``top_k`` criticality,
``graph_centrality``, and ``random``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from pyomo.environ import value

from ..milp.model import build_uc_model
from ..milp.scenario_loader import ScenarioData
from .contingency_redispatch import Commitment, solve_and_load


# --------------------------------------------------------------------------- #
# Per-zone criticality proxies and S* selection
# --------------------------------------------------------------------------- #
def zone_peak_to_firm(data: ScenarioData) -> Dict[str, float]:
    """Per-zone peak-load / firm-capacity ratio (higher = more critical)."""
    scores: Dict[str, float] = {}
    for z in data.zones:
        peak = float(data.peak_demand.get(z, 0.0))
        firm = float(data.thermal_capacity.get(z, 0.0)) + float(data.nuclear_capacity.get(z, 0.0))
        scores[z] = peak / max(firm, 1.0)
    return scores


def zone_line_degree(data: ScenarioData) -> Dict[str, float]:
    """Per-zone line-incidence degree weighted by capacity (centrality proxy)."""
    deg: Dict[str, float] = {z: 0.0 for z in data.zones}
    for ln in data.lines.values():
        if ln.from_zone in deg:
            deg[ln.from_zone] += ln.capacity_mw
        if ln.to_zone in deg:
            deg[ln.to_zone] += ln.capacity_mw
    return deg


def select_critical_zones(
    data: ScenarioData,
    rule: str = "top_k",
    keep_fraction: float = 0.4,
    seed: int = 0,
) -> List[str]:
    """Return the critical zone subset ``S*`` under the given selection rule."""
    zones = list(data.zones)
    k = max(1, math.ceil(keep_fraction * len(zones)))
    if rule == "top_k":
        scores = zone_peak_to_firm(data)
        ranked = sorted(zones, key=lambda z: scores[z], reverse=True)
        return ranked[:k]
    if rule == "graph_centrality":
        scores = zone_line_degree(data)
        ranked = sorted(zones, key=lambda z: scores[z], reverse=True)
        return ranked[:k]
    if rule == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(zones), size=k, replace=False)
        return [zones[i] for i in sorted(idx)]
    raise ValueError(f"unknown selection rule: {rule!r}")


# --------------------------------------------------------------------------- #
# Heuristic fill for Z \ S*  (protocol 3, option ii / "all-on for must-runs")
# --------------------------------------------------------------------------- #
def heuristic_fill_commitment(data: ScenarioData, zones_fill: Sequence[str]) -> Commitment:
    """Cheap myopic fill: thermal all-on where capacity exists, storage idle/discharge,
    no DR activation. This is the most-feasible reference fill of protocol 3."""
    c = Commitment()
    periods = list(range(len(data.periods)))
    for z in zones_fill:
        has_thermal = float(data.thermal_capacity.get(z, 0.0)) > 1e-6
        for t in periods:
            c.u_thermal[(z, t)] = 1.0 if has_thermal else 0.0
            c.b_charge_mode[(z, t)] = 0.0       # discharge-enabled
            c.pumped_charge_mode[(z, t)] = 0.0
            c.dr_active[(z, t)] = 0.0
    return c


# --------------------------------------------------------------------------- #
# Commitment extraction from a solved Pyomo model
# --------------------------------------------------------------------------- #
def _commitment_from_model(model) -> Commitment:
    c = Commitment()
    for z in model.Z:
        for t in model.T:
            c.u_thermal[(z, t)] = round(float(value(model.u_thermal[z, t])))
            if (z, t) in model.b_charge_mode:
                c.b_charge_mode[(z, t)] = round(float(value(model.b_charge_mode[z, t])))
            if (z, t) in model.pumped_charge_mode:
                c.pumped_charge_mode[(z, t)] = round(float(value(model.pumped_charge_mode[z, t])))
            if (z, t) in model.dr_active:
                c.dr_active[(z, t)] = round(float(value(model.dr_active[z, t])))
    for t in model.T:
        if t in model.import_mode:
            c.import_mode[t] = round(float(value(model.import_mode[t])))
    return c


def _solve_uc(model, solver_name: str, time_limit: float):
    return solve_and_load(solver_name, model, time_limit)


# --------------------------------------------------------------------------- #
# Arm A: Full MILP
# --------------------------------------------------------------------------- #
def commitment_full_milp(
    data: ScenarioData, *, solver_name: str = "appsi_highs", time_limit: float = 120.0
) -> Tuple[Commitment, bool, str]:
    model = build_uc_model(data)
    res, ok = _solve_uc(model, solver_name, time_limit)
    term = str(res.solver.termination_condition)
    if not ok:
        return Commitment(), False, term
    return _commitment_from_model(model), True, term


# --------------------------------------------------------------------------- #
# Arm B: Partial MILP on S* + heuristic fill elsewhere
# --------------------------------------------------------------------------- #
def commitment_partial_milp(
    data: ScenarioData,
    critical_zones: Sequence[str],
    *,
    solver_name: str = "appsi_highs",
    time_limit: float = 120.0,
) -> Tuple[Commitment, bool, str]:
    """Solve UC with binaries on ``Z \\ S*`` fixed to the heuristic fill, optimising
    only the binaries on ``S*``. The continuous dispatch stays coupled network-wide,
    so the sub-region solve still respects system-wide balance."""
    keep = set(critical_zones)
    fill_zones = [z for z in data.zones if z not in keep]
    fill = heuristic_fill_commitment(data, fill_zones)

    model = build_uc_model(data)
    for (z, t), val in fill.u_thermal.items():
        if (z, t) in model.u_thermal and not model.u_thermal[z, t].is_fixed():
            if float(data.thermal_capacity.get(z, 0.0)) > 1e-6 or val == 0.0:
                model.u_thermal[z, t].fix(float(val))
    for (z, t), val in fill.b_charge_mode.items():
        if (z, t) in model.b_charge_mode and not model.b_charge_mode[z, t].is_fixed():
            if float(data.battery_power.get(z, 0.0)) > 1e-6 or val == 0.0:
                model.b_charge_mode[z, t].fix(float(val))
    for (z, t), val in fill.pumped_charge_mode.items():
        if (z, t) in model.pumped_charge_mode and not model.pumped_charge_mode[z, t].is_fixed():
            if float(data.pumped_power.get(z, 0.0)) > 1e-6 or val == 0.0:
                model.pumped_charge_mode[z, t].fix(float(val))
    for (z, t), val in fill.dr_active.items():
        if (z, t) in model.dr_active and not model.dr_active[z, t].is_fixed():
            model.dr_active[z, t].fix(float(val))

    res, ok = _solve_uc(model, solver_name, time_limit)
    term = str(res.solver.termination_condition)
    if not ok:
        return Commitment(), False, term
    return _commitment_from_model(model), True, term


# --------------------------------------------------------------------------- #
# Arms C/D: from a pipeline binary tensor [Z, T, F]
# --------------------------------------------------------------------------- #
# Feature order matches src/eval/pipeline_runner._BINARY_FEATURE_NAMES:
#   0 battery_charge, 1 battery_discharge, 2 pumped_charge, 3 pumped_discharge,
#   4 dr_active, 5 thermal_startup, 6 thermal_on
def commitment_from_binary_tensor(
    u_bin: np.ndarray,
    zone_names: Sequence[str],
    n_periods: int,
) -> Commitment:
    """Convert a pipeline binary candidate ``[Z, T, F]`` to a :class:`Commitment`."""
    arr = np.asarray(u_bin)
    if arr.ndim != 3:
        raise ValueError(f"expected [Z,T,F] tensor, got shape {arr.shape}")
    Z = min(len(zone_names), arr.shape[0])
    T = min(n_periods, arr.shape[1])
    c = Commitment()
    for zi in range(Z):
        z = zone_names[zi]
        for t in range(T):
            c.u_thermal[(z, t)] = 1.0 if arr[zi, t, 6] > 0.5 else 0.0
            c.b_charge_mode[(z, t)] = 1.0 if arr[zi, t, 0] > arr[zi, t, 1] else 0.0
            c.pumped_charge_mode[(z, t)] = 1.0 if arr[zi, t, 2] > arr[zi, t, 3] else 0.0
            c.dr_active[(z, t)] = 1.0 if arr[zi, t, 4] > 0.5 else 0.0
    return c


def commitment_pipeline_on_critical(
    data: ScenarioData,
    pipeline_commitment: Commitment,
    critical_zones: Sequence[str],
) -> Commitment:
    """Arm D: pipeline commitment on ``S*`` overlaid on heuristic fill elsewhere."""
    keep = set(critical_zones)
    fill_zones = [z for z in data.zones if z not in keep]
    fill = heuristic_fill_commitment(data, fill_zones)
    return pipeline_commitment.restricted_to(keep).merged_with(fill)
