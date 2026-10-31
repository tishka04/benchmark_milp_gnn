"""Streaming HiGHS incumbents for matched budgets. Imports never start a solve."""
import math
import time

FEASIBILITY_TOLERANCE = 1e-3


def linear_validator(lp):
    """Compile the original solver matrix once, before presolve, inside the budget."""
    import numpy as np
    from scipy.sparse import csc_matrix, csr_matrix

    a = lp.a_matrix_
    fmt = str(a.format_).split(".")[-1]
    if fmt not in {"kColwise", "kRowwise"}:
        raise ValueError(f"Unsupported HiGHS matrix format: {fmt}")
    constructor = csc_matrix if fmt == "kColwise" else csr_matrix
    matrix = constructor((a.value_, a.index_, a.start_), shape=(lp.num_row_, lp.num_col_))
    row_lo, row_hi = np.asarray(lp.row_lower_), np.asarray(lp.row_upper_)
    col_lo, col_hi = np.asarray(lp.col_lower_), np.asarray(lp.col_upper_)
    types = list(lp.integrality_) or [0] * lp.num_col_
    if any(int(t) not in (0, 1) for t in types):
        raise ValueError("Only continuous and integer columns are supported")
    integers = np.array([int(t) == 1 for t in types])
    costs = np.asarray(lp.col_cost_)

    def check(values):
        x = np.asarray(values, dtype=float)
        if x.shape != (lp.num_col_,) or not np.isfinite(x).all():
            return math.inf, math.inf
        rows = matrix @ x
        if not np.isfinite(rows).all():
            return math.inf, math.inf
        violation = max(0.0, float(np.max(row_lo - rows, initial=0)),
                        float(np.max(rows - row_hi, initial=0)),
                        float(np.max(col_lo - x, initial=0)),
                        float(np.max(x - col_hi, initial=0)),
                        float(np.max(np.abs(x[integers] - np.rint(x[integers])), initial=0)))
        return float(costs @ x + lp.offset_), violation
    return check


def solve_streaming(job, publish, *, clock=time.monotonic):
    """Fresh MILP, no warm start; publish validated incumbents before the deadline."""
    from pathlib import Path
    import highspy
    from pyomo.environ import value
    from pyomo.contrib.appsi.solvers.highs import Highs
    from src.milp.model import build_uc_model
    from src.milp.scenario_loader import load_scenario_data

    model = build_uc_model(load_scenario_data(Path(job["scenario_file"])), enable_duals=False)
    if clock() >= job["deadline"]:
        return
    adapter = Highs()
    adapter.set_instance(model)
    native = adapter._solver_model
    lp = native.getLp()
    if lp.sense_ != highspy.ObjSense.kMinimize:
        raise ValueError("Expected a minimization MILP")
    validate = linear_validator(lp)
    columns = adapter._pyomo_var_to_solver_var_map
    unserved = [columns[id(v)] for v in model.unserved.values() if id(v) in columns]
    overgen = [columns[id(v)] for v in model.overgen_spill.values() if id(v) in columns]
    dt = float(value(model.dt_hours))
    best = math.inf
    errors = []

    def retain(values, status, dual_bound=None, gap=None):
        nonlocal best
        if clock() >= job["deadline"]:
            return
        objective, violation = validate(values)
        if not math.isfinite(objective) or violation > FEASIBILITY_TOLERANCE or objective >= best:
            return
        row = dict(scenario_id=job["scenario_id"], family=job["family_alias"],
                   dr_formulation="load_shifting_v1", success=True, lp_objective=objective,
                   status=status, max_constraint_violation=violation,
                   feasibility_tolerance=FEASIBILITY_TOLERANCE,
                   unserved_mwh=sum(float(values[i]) for i in unserved) * dt,
                   overgen_spill_mwh=sum(float(values[i]) for i in overgen) * dt,
                   dual_bound=dual_bound, mip_gap=gap, configured_highs_threads=job["threads"])
        row["slack_mwh"] = row["unserved_mwh"] + row["overgen_spill_mwh"]
        if publish("milp", row):
            best = objective

    def incumbent(event):
        try:
            retain(event.data_out.mip_solution, "incumbent",
                   event.data_out.mip_dual_bound, event.data_out.mip_gap)
        except Exception as exc:
            errors.append(exc)
            event.interrupt()

    def interrupt(event):
        if errors or clock() >= job["deadline"]:
            event.interrupt()

    native.cbMipImprovingSolution.subscribe(incumbent)
    native.cbMipInterrupt.subscribe(interrupt)
    native.cbSimplexInterrupt.subscribe(interrupt)
    native.cbIpmInterrupt.subscribe(interrupt)
    remaining = job["deadline"] - clock()
    if remaining <= 0:
        return
    for name, setting in {"threads": job["threads"], "parallel": "on" if job["threads"] > 1 else "off",
                          "time_limit": remaining, "random_seed": job["seed"],
                          "output_flag": False}.items():
        if native.setOptionValue(name, setting) == highspy.HighsStatus.kError:
            raise ValueError(f"HiGHS rejected option {name}={setting}")
    status = native.run()
    if errors:
        raise errors[0]
    if status == highspy.HighsStatus.kError:
        raise RuntimeError(f"HiGHS error: {native.getModelStatus()}")
    solution = native.getSolution()
    if solution.value_valid:
        info = native.getInfo()
        retain(solution.col_value, str(native.getModelStatus()), info.mip_dual_bound, info.mip_gap)
