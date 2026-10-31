"""Native HiGHS reference/label solver, and a strict adapter to the common cascade."""
from __future__ import annotations

import math
import time
from .physics import apply_cost_units, extract_dispatch, validate_dispatch, to_report
from .common import PHYSICS


def solve_data(data, *, threads=16, seconds=600, seed=42, deadline=None, relaxation=False):
    """seconds is solver-only for references; deadline additionally caps short runs.

    No warm-start values or snapshots are imported. A native value_valid flag
    alone is insufficient: the independent full validator must also pass.
    """
    import highspy
    from pyomo.contrib.appsi.solvers.highs import Highs
    from src.milp.model import build_uc_model
    model = build_uc_model(data)
    apply_cost_units(model)
    adapter = Highs()
    adapter.set_instance(model)
    native = adapter._solver_model
    remaining = seconds if deadline is None else min(seconds, deadline - time.perf_counter())
    if remaining <= 0:
        return dict(status="no_incumbent", complete=True, valid=False)
    for k, v in dict(threads=threads, parallel="on" if threads > 1 else "off", time_limit=remaining,
                     random_seed=seed, output_flag=False).items():
        if native.setOptionValue(k, v) == highspy.HighsStatus.kError:
            raise RuntimeError(f"HiGHS rejected {k}")
    if deadline is not None:
        def stop(event):
            if time.perf_counter() >= deadline:
                event.interrupt()
        for cb in (native.cbMipInterrupt, native.cbSimplexInterrupt, native.cbIpmInterrupt):
            cb.subscribe(stop)
    start = time.perf_counter()
    status = native.run()
    elapsed = time.perf_counter() - start
    if status == highspy.HighsStatus.kError:
        raise RuntimeError(f"HiGHS infrastructure error: {native.getModelStatus()}")
    info, solution = native.getInfo(), native.getSolution()
    clean = lambda x: float(x) if math.isfinite(float(x)) else None
    result = dict(complete=True, status=str(native.getModelStatus()), valid=False,
                  solver_seconds=elapsed, solver_limit=seconds, threads=threads,
                  bound=clean(info.mip_dual_bound), gap=clean(info.mip_gap),
                  proven_optimal=native.getModelStatus() == highspy.HighsModelStatus.kOptimal,
                  physics=PHYSICS, lp=dict(objective=None, status="skipped", termination="not_requested"), lp_duals={})
    if not solution.value_valid:
        result.update(status="no_incumbent", solver_status=str(native.getModelStatus()))
        return result
    columns = adapter._pyomo_var_to_solver_var_map
    from pyomo.environ import Var
    for var in model.component_data_objects(Var):
        if id(var) in columns:
            var.set_value(solution.col_value[columns[id(var)]], skip_validation=True)
    dispatch = extract_dispatch(model)
    check = validate_dispatch(data, dispatch)
    objective = clean(info.objective_function_value)
    if check.get("valid") and not math.isclose(check["cost"], objective, rel_tol=1e-9, abs_tol=1e-3):
        raise ValueError("Native objective differs from physical cost")
    result.update(valid=check["valid"], dispatch=dispatch, validation=check, objective=objective)
    result["mip"] = dict(objective=objective, termination="optimal" if result["proven_optimal"] else "maxTimeLimit" if "Time" in result["status"] else result["status"],
                         status="ok", solve_seconds=elapsed, bound=result["bound"], gap=result["gap"])
    result.update(detail=to_report(dispatch, data), cost_components=check.get("cost_components", {}),
                  periods=data.periods, zones=data.zones, scenario_id=data.scenario_id,
                  dr_formulation="load_shifting_v1", thermal_formulation="thermal_transitions_v1")
    if relaxation:
        from pyomo.core import TransformationFactory
        from pyomo.environ import SolverFactory, value
        relaxed = build_uc_model(data)
        apply_cost_units(relaxed)
        TransformationFactory("core.relax_integer_vars").apply_to(relaxed)
        solver = SolverFactory("highs")
        solver.options.update({"threads": threads, "time_limit": seconds})
        r = solver.solve(relaxed, load_solutions=False)
        if len(r.solution):
            relaxed.solutions.load_from(r)
            result["lp"] = dict(objective=float(value(relaxed.obj)), status=str(r.solver.status), termination=str(r.solver.termination_condition))
    return result


def make_repair_worker(settings):
    from src.milp.lp_worker_two_stage import LPWorkerTwoStage
    class OperationalRepair(LPWorkerTwoStage):
        deadline = math.inf

        def _solve_with_timeout(self, model, time_limit):
            remaining = min(time_limit, self.deadline - time.perf_counter())
            if remaining <= 0:
                raise TimeoutError("cooperative repair deadline")
            apply_cost_units(model)
            return super()._solve_with_timeout(model, remaining)

        def _extract_solution(self, model):
            solution = super()._extract_solution(model)
            solution["operational_dispatch"] = extract_dispatch(model)
            return solution
    worker = OperationalRepair("", solver_name="appsi_highs", verbose=False, **{
        k: v for k, v in settings.items() if k != "max_stages"})
    if "highs" not in worker.solver_name:
        raise RuntimeError("HiGHS required; solver fallback forbidden")
    return worker
