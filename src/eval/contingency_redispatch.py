"""Post-fault DC power-flow redispatch for the partial-grid recovery protocol.

This module operationalises Section 4--5 of
``docs/partial_grid_recovery_protocol.md``. Given a *fixed* unit-commitment
vector ``u`` (from any of the four protocol arms) it:

1. Builds an LP redispatch model from the same :class:`ScenarioData` physics as
   the MILP oracle (:func:`src.milp.model.build_uc_model`), with all commitment
   binaries fixed and only the continuous dispatch variables free.
2. Replaces the transport-style line bounds with an *approximate DC power flow*:
   each line ``l`` is given a susceptance ``B_l = 1 / x_l`` with reactance
   ``x_l`` proportional to its ``distance_km`` (standard when only line lengths
   are known), and ``flow[l,t] = B_l * (theta[from] - theta[to])``. Because DC
   flows depend only on *relative* susceptances, the absolute scaling cancels.
3. Removes the outaged line(s) and softens the remaining thermal limits with an
   overload penalty ``c_ovl`` so that post-fault overloads and cascades become
   observable (a pure transport model cannot express them).

It exposes a single-line N-1 sweep (:func:`run_n1_sweep`) and a sequential
cascade simulator (:func:`run_cascade`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pyomo.core import TransformationFactory
from pyomo.environ import (
    Constraint,
    NonNegativeReals,
    Objective,
    Reals,
    SolverFactory,
    TerminationCondition,
    Var,
    minimize,
    value,
)

from ..milp.model import build_uc_model
from ..milp.scenario_loader import ScenarioData


# --------------------------------------------------------------------------- #
# Commitment container
# --------------------------------------------------------------------------- #
@dataclass
class Commitment:
    """A frozen unit-commitment vector for a scenario.

    Keys are ``(zone, t)`` with ``t`` the 0-indexed period. Values are 0/1.
    ``import_mode`` is intentionally optional: like the cascading LP worker, the
    redispatch leaves the import/export decision free unless a value is given.
    """

    u_thermal: Dict[Tuple[str, int], float] = field(default_factory=dict)
    b_charge_mode: Dict[Tuple[str, int], float] = field(default_factory=dict)
    pumped_charge_mode: Dict[Tuple[str, int], float] = field(default_factory=dict)
    dr_active: Dict[Tuple[str, int], float] = field(default_factory=dict)
    import_mode: Dict[int, float] = field(default_factory=dict)

    def restricted_to(self, zones_keep: Iterable[str]) -> "Commitment":
        keep = set(zones_keep)
        def _f(d):
            return {(z, t): v for (z, t), v in d.items() if z in keep}
        return Commitment(
            u_thermal=_f(self.u_thermal),
            b_charge_mode=_f(self.b_charge_mode),
            pumped_charge_mode=_f(self.pumped_charge_mode),
            dr_active=_f(self.dr_active),
            import_mode=dict(self.import_mode),
        )

    def merged_with(self, other: "Commitment") -> "Commitment":
        """Return ``self`` overlaid on ``other`` (``self`` wins on conflicts)."""
        def _m(a, b):
            out = dict(b)
            out.update(a)
            return out
        return Commitment(
            u_thermal=_m(self.u_thermal, other.u_thermal),
            b_charge_mode=_m(self.b_charge_mode, other.b_charge_mode),
            pumped_charge_mode=_m(self.pumped_charge_mode, other.pumped_charge_mode),
            dr_active=_m(self.dr_active, other.dr_active),
            import_mode=_m(self.import_mode, other.import_mode),
        )


# --------------------------------------------------------------------------- #
# Network susceptances (approximate DC power flow)
# --------------------------------------------------------------------------- #
def line_susceptances(data: ScenarioData) -> Dict[str, float]:
    """Per-line susceptance ``B_l = 1 / x_l`` with ``x_l`` proportional to length.

    The reactances are normalised by the mean line length so that the resulting
    susceptances are O(1); only relative values matter for DC flow distribution.
    """
    lengths = {lid: max(float(ln.distance_km), 1e-3) for lid, ln in data.lines.items()}
    if not lengths:
        return {}
    mean_len = sum(lengths.values()) / len(lengths)
    return {lid: mean_len / x for lid, x in lengths.items()}


# --------------------------------------------------------------------------- #
# Redispatch model
# --------------------------------------------------------------------------- #
@dataclass
class RedispatchResult:
    feasible: bool
    termination: str
    objective: float
    lol_mwh: float                # involuntary load shed (unserved) in MWh
    dr_mwh: float                 # voluntary demand response in MWh
    overload_mwh: float           # total over-capacity flow integrated over time
    n_overloaded_lines: int       # distinct lines with |flow| > limit at any t
    cascade_risk_score: float     # sum_(l,t) max(|flow|/limit - 1, 0)
    redispatch_cost: float        # objective minus shed/overload penalties
    solve_seconds: float
    tripped_lines: Tuple[str, ...] = ()
    overloaded_lines: Tuple[str, ...] = ()


def _fix(var, key, val):
    if key in var:
        v = var[key]
        if not v.is_fixed():
            v.fix(float(val))


def build_redispatch_model(
    data: ScenarioData,
    commitment: Commitment,
    outaged_lines: Sequence[str] = (),
    *,
    overload_penalty: float = 5_000.0,
    fix_modes: bool = False,
    fix_import_mode: bool = False,
):
    """Build the fixed-commitment DC-PF redispatch LP for a given outage set.

    By default only the *thermal commitment* ``u_thermal`` is frozen -- this is the
    classic security-constrained economic dispatch (SCED) contingency model where
    the unit-commitment (which thermal plants are online) is fixed and all fast,
    continuous flexibility (storage, DR, exchanges) is re-dispatchable post-fault.
    Set ``fix_modes=True`` to additionally freeze the storage/DR mode binaries
    (closer to a fully frozen plan, but can be infeasible for imperfect plans).
    """
    m = build_uc_model(data, enable_duals=False)
    outaged = set(outaged_lines)

    # 1) Freeze the thermal commitment (capacity-aware, mirroring the LP worker).
    for (z, t), val in commitment.u_thermal.items():
        has_cap = float(data.thermal_capacity.get(z, 0.0)) > 1e-6
        if has_cap or float(val) == 0.0:
            _fix(m.u_thermal, (z, t), round(float(val)))
    if fix_modes:
        for (z, t), val in commitment.b_charge_mode.items():
            if float(data.battery_power.get(z, 0.0)) > 1e-6 or float(val) == 0.0:
                _fix(m.b_charge_mode, (z, t), round(float(val)))
        for (z, t), val in commitment.pumped_charge_mode.items():
            if float(data.pumped_power.get(z, 0.0)) > 1e-6 or float(val) == 0.0:
                _fix(m.pumped_charge_mode, (z, t), round(float(val)))
        for (z, t), val in commitment.dr_active.items():
            if float(data.dr_limit.get((z, t), 0.0)) > 1e-6 or float(val) == 0.0:
                _fix(m.dr_active, (z, t), round(float(val)))
    if fix_import_mode:
        for t, val in commitment.import_mode.items():
            _fix(m.import_mode, t, round(float(val)))

    # 2) Approximate DC power flow: theta[z,t], slack bus = first zone.
    susc = line_susceptances(data)
    slack_zone = data.zones[0]
    m.theta = Var(m.Z, m.T, within=Reals, initialize=0.0)
    for t in m.T:
        m.theta[slack_zone, t].fix(0.0)

    def _dcpf_rule(model, l, t):
        if l in outaged:
            return model.flow[l, t] == 0.0
        ln = data.lines[l]
        return model.flow[l, t] == susc[l] * (model.theta[ln.from_zone, t] - model.theta[ln.to_zone, t])

    m.dcpf = Constraint(m.L, m.T, rule=_dcpf_rule)

    # 3) Soften the thermal line limits with an overload penalty.
    m.flow_upper.deactivate()
    m.flow_lower.deactivate()
    m.ovl_pos = Var(m.L, m.T, within=NonNegativeReals)
    m.ovl_neg = Var(m.L, m.T, within=NonNegativeReals)

    def _soft_upper(model, l, t):
        if l in outaged:
            return Constraint.Skip
        return model.flow[l, t] <= data.lines[l].capacity_mw + model.ovl_pos[l, t]

    def _soft_lower(model, l, t):
        if l in outaged:
            return Constraint.Skip
        return -data.lines[l].capacity_mw - model.ovl_neg[l, t] <= model.flow[l, t]

    m.soft_flow_upper = Constraint(m.L, m.T, rule=_soft_upper)
    m.soft_flow_lower = Constraint(m.L, m.T, rule=_soft_lower)

    # 4) Augment the objective with the overload penalty.
    base_obj_expr = m.obj.expr
    m.obj.deactivate()
    m.obj_contingency = Objective(
        expr=base_obj_expr
        + overload_penalty * sum(m.ovl_pos[l, t] + m.ovl_neg[l, t] for l in m.L for t in m.T),
        sense=minimize,
    )

    # 5) Relax any remaining integrality so this is a pure LP.
    TransformationFactory("core.relax_integer_vars").apply_to(m)

    # Record line capacities for post-solve overload accounting.
    m._line_caps = {lid: float(ln.capacity_mw) for lid, ln in data.lines.items()}
    return m


def solve_and_load(solver_name: str, model, time_limit: float):
    """Solve ``model`` and load primals, handling the APPSI HiGHS quirks.

    Returns ``(results, ok)`` where ``ok`` is True iff a usable solution loaded.
    Mirrors the pattern used by the cascading LP worker.
    """
    solver = SolverFactory(solver_name)
    sl = solver_name.lower()
    if time_limit and time_limit > 0:
        try:
            if solver_name == "appsi_highs":
                solver.config.time_limit = float(time_limit)
            elif "highs" in sl:
                solver.options["time_limit"] = float(time_limit)
            elif "gurobi" in sl:
                solver.options["TimeLimit"] = float(time_limit)
            elif "cbc" in sl:
                solver.options["seconds"] = float(time_limit)
        except Exception:
            pass

    results = solver.solve(model, tee=False, load_solutions=False)
    tc = results.solver.termination_condition
    ok = tc in (
        TerminationCondition.optimal,
        TerminationCondition.feasible,
        TerminationCondition.locallyOptimal,
        TerminationCondition.globallyOptimal,
        TerminationCondition.maxTimeLimit,
    )
    if ok:
        loaded = False
        if hasattr(solver, "load_vars"):
            try:
                solver.load_vars()
                loaded = True
            except Exception:
                loaded = False
        if not loaded:
            try:
                if getattr(results, "solution_loader", None) is not None:
                    results.solution_loader.load_vars()
                    loaded = True
            except Exception:
                loaded = False
        if not loaded:
            try:
                model.solutions.load_from(results)
                loaded = True
            except Exception:
                loaded = False
        ok = loaded
    return results, ok


def solve_redispatch(
    model,
    *,
    solver_name: str = "appsi_highs",
    time_limit: float = 120.0,
    overload_tol_mw: float = 1.0,
) -> RedispatchResult:
    data_dt = float(value(model.dt_hours))
    t0 = time.perf_counter()
    try:
        res, feasible = solve_and_load(solver_name, model, time_limit)
        term = str(res.solver.termination_condition)
    except Exception as exc:  # pragma: no cover - solver edge cases
        return RedispatchResult(
            feasible=False, termination=f"error:{type(exc).__name__}", objective=float("nan"),
            lol_mwh=float("nan"), dr_mwh=float("nan"), overload_mwh=float("nan"),
            n_overloaded_lines=-1, cascade_risk_score=float("nan"),
            redispatch_cost=float("nan"), solve_seconds=time.perf_counter() - t0,
        )
    solve_seconds = time.perf_counter() - t0

    if not feasible:
        return RedispatchResult(
            feasible=False, termination=term, objective=float("nan"),
            lol_mwh=float("nan"), dr_mwh=float("nan"), overload_mwh=float("nan"),
            n_overloaded_lines=-1, cascade_risk_score=float("nan"),
            redispatch_cost=float("nan"), solve_seconds=solve_seconds,
        )

    lol = sum(float(value(model.unserved[z, t])) for z in model.Z for t in model.T) * data_dt
    dr = sum(float(value(model.dr_shed[z, t])) for z in model.Z for t in model.T) * data_dt
    overload_mwh = 0.0
    cascade_score = 0.0
    overloaded: set = set()
    caps = getattr(model, "_line_caps", {})
    for l in model.L:
        cap = caps.get(l, None)
        if cap is None or cap <= 1e-9:
            continue
        for t in model.T:
            ov = float(value(model.ovl_pos[l, t])) + float(value(model.ovl_neg[l, t]))
            if ov > overload_tol_mw:
                overload_mwh += ov * data_dt
                cascade_score += ov / cap
                overloaded.add(l)
    obj = float(value(model.obj_contingency))
    return RedispatchResult(
        feasible=True,
        termination=term,
        objective=obj,
        lol_mwh=lol,
        dr_mwh=dr,
        overload_mwh=overload_mwh,
        n_overloaded_lines=len(overloaded),
        cascade_risk_score=cascade_score,
        redispatch_cost=obj,
        solve_seconds=solve_seconds,
        overloaded_lines=tuple(sorted(overloaded)),
    )


# --------------------------------------------------------------------------- #
# N-1 sweep and cascade
# --------------------------------------------------------------------------- #
def contingency_line_set(data: ScenarioData, mode: str = "critical", top_k: int = 12) -> List[str]:
    """Return the set of lines to test as single outages.

    ``"n1"`` is the full single-line N-1 set; ``"critical"`` keeps the ``top_k``
    lines by base-case capacity (a proxy for loading) to bound runtime.
    """
    lids = list(data.lines.keys())
    if mode == "n1" or len(lids) <= top_k:
        return lids
    ranked = sorted(lids, key=lambda l: data.lines[l].capacity_mw, reverse=True)
    return ranked[:top_k]


def run_n1_sweep(
    data: ScenarioData,
    commitment: Commitment,
    *,
    contingency_lines: Optional[Sequence[str]] = None,
    overload_penalty: float = 5_000.0,
    fix_modes: bool = False,
    solver_name: str = "appsi_highs",
    time_limit: float = 120.0,
    deadline: Optional[float] = None,
) -> Dict[str, RedispatchResult]:
    """Solve one redispatch per single-line outage. Returns ``{line: result}``.

    The base case (no outage) is included under the key ``"__base__"``.

    ``deadline`` is an absolute ``time.perf_counter()`` instant: outages still
    pending when it passes are skipped (the sweep returns what it has). Each
    solve's own time limit is also capped to the remaining budget so a single
    solve cannot overrun the deadline by more than its presolve overhead.
    """
    if contingency_lines is None:
        contingency_lines = contingency_line_set(data, mode="critical")
    out: Dict[str, RedispatchResult] = {}
    for outage in ["__base__", *contingency_lines]:
        tl = time_limit
        if deadline is not None:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            tl = min(time_limit, remaining)
        outaged = () if outage == "__base__" else (outage,)
        model = build_redispatch_model(
            data, commitment, outaged, overload_penalty=overload_penalty, fix_modes=fix_modes
        )
        res = solve_redispatch(
            model, solver_name=solver_name, time_limit=tl
        )
        res.tripped_lines = tuple(outaged)
        out[outage] = res
    return out


def run_cascade(
    data: ScenarioData,
    commitment: Commitment,
    seed_line: str,
    *,
    overload_threshold: float = 0.0,
    max_iterations: int = 5,
    overload_penalty: float = 5_000.0,
    fix_modes: bool = False,
    solver_name: str = "appsi_highs",
    time_limit: float = 120.0,
    deadline: Optional[float] = None,
) -> List[RedispatchResult]:
    """Sequential cascade: trip overloaded lines and redispatch until stable.

    A line is tripped in the next round if its post-redispatch overload exceeds
    ``overload_threshold`` (fraction of capacity). Stops when no new line trips,
    after ``max_iterations`` rounds, or when ``deadline`` (an absolute
    ``time.perf_counter()`` instant) passes. Returns the per-round results.
    """
    tripped: List[str] = [seed_line]
    history: List[RedispatchResult] = []
    for _ in range(max_iterations):
        tl = time_limit
        if deadline is not None:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            tl = min(time_limit, remaining)
        model = build_redispatch_model(
            data, commitment, tuple(tripped), overload_penalty=overload_penalty, fix_modes=fix_modes
        )
        res = solve_redispatch(model, solver_name=solver_name, time_limit=tl)
        res.tripped_lines = tuple(tripped)
        history.append(res)
        if not res.feasible:
            break
        # Identify newly overloaded lines (beyond threshold) to trip next round.
        new_trips: List[str] = []
        caps = model._line_caps
        for l in model.L:
            if l in tripped:
                continue
            cap = caps.get(l, 0.0)
            if cap <= 1e-9:
                continue
            max_ov = max(
                float(value(model.ovl_pos[l, t])) + float(value(model.ovl_neg[l, t]))
                for t in model.T
            )
            if max_ov / cap > overload_threshold:
                new_trips.append(l)
        if not new_trips:
            break
        tripped = tripped + new_trips
    return history
