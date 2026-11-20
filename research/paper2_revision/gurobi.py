"""Independent direct Gurobi comparison; the repair backend stays HiGHS."""
from __future__ import annotations
import math
import time

def solve(data, *, seconds, deadline, seed, env):
    import gurobipy as gp
    from pyomo.environ import Var, Constraint, value
    from pyomo.repn import generate_standard_repn
    from src.milp.model import build_uc_model
    from src.operational_exit.physics import apply_cost_units, extract_dispatch, validate_dispatch
    from src.operational_exit.common import PHYSICS
    model = build_uc_model(data); apply_cost_units(model)
    with gp.Model(env=env) as native:
        native.Params.OutputFlag=0; native.Params.Threads=16; native.Params.Seed=seed
        variables={}
        for var in model.component_data_objects(Var):
            lo=value(var.lb) if var.lb is not None else -gp.GRB.INFINITY
            hi=value(var.ub) if var.ub is not None else gp.GRB.INFINITY
            if var.fixed:
                lo=hi=float(value(var))
            variables[id(var)]=native.addVar(lb=lo,ub=hi,
                vtype=gp.GRB.BINARY if var.is_binary() else gp.GRB.INTEGER if var.is_integer() else gp.GRB.CONTINUOUS)
        def linear(expr):
            rep=generate_standard_repn(expr)
            if not rep.is_linear():
                raise ValueError("Gurobi adapter accepts only linear expressions")
            constant=float(value(rep.constant)) if rep.constant is not None else 0.
            return gp.LinExpr([float(value(c)) for c in rep.linear_coefs],
                              [variables[id(v)] for v in rep.linear_vars])+constant
        native.setObjective(linear(model.obj.expr),gp.GRB.MINIMIZE)
        for con in model.component_data_objects(Constraint,active=True):
            expr=linear(con.body)
            if con.equality:
                native.addConstr(expr==float(value(con.lower)))
            else:
                if con.has_lb(): native.addConstr(expr>=float(value(con.lower)))
                if con.has_ub(): native.addConstr(expr<=float(value(con.upper)))
        remaining=min(seconds,deadline-time.perf_counter())
        if remaining<=0:
            return dict(status="no_incumbent",valid=False,physics=PHYSICS)
        native.Params.TimeLimit=remaining
        def stop(m,where):
            if time.perf_counter()>=deadline: m.terminate()
        native.optimize(stop)
        result=dict(status="no_incumbent",valid=False,physics=PHYSICS,
                    solver_status=int(native.Status),solver_seconds=float(native.Runtime),
                    solver_limit=seconds,threads=16,proven_optimal=native.Status==gp.GRB.OPTIMAL,
                    bound=float(native.ObjBound) if math.isfinite(native.ObjBound) else None)
        if native.SolCount:
            for var in model.component_data_objects(Var):
                var.set_value(variables[id(var)].X,skip_validation=True)
            dispatch=extract_dispatch(model); check=validate_dispatch(data,dispatch)
            if check["valid"] and not math.isclose(check["cost"],native.ObjVal,rel_tol=1e-9,abs_tol=1e-3):
                raise ValueError("Gurobi objective differs from physical cost")
            result.update(status="solution" if check["valid"] else "invalid_dispatch",
                          valid=check["valid"],dispatch=dispatch,validation=check,objective=float(native.ObjVal))
        return result
