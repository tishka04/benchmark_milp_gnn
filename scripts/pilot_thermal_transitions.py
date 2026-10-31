"""Bounded before/after pilot: exactly two saved test scenarios per family.

No training, neural inference, old-result writes or complete campaign launches.
Prepare fixes the cohort first; run uses one isolated 16-thread MILP at a time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/load_shifting_v1"
DEFAULT_OUT = ROOT / "outputs/thermal_transition_pilot_20261027"
FAMILIES = ["low_criticality_scenarios", "medium_criticality_scenarios", "high_criticality_scenarios",
            "persona_vre_battery_scenarios", "persona_network_operator_scenarios", "persona_mathematician_scenarios"]
SOURCE_NAMES = ["src/thermal.py", "src/milp/model.py", "src/milp/scenario_loader.py", "src/milp/defaults.py",
                "src/milp/profiles.py", "src/milp/solver_resources.py", "src/milp/solve.py",
                "src/demand_response.py", "scripts/pilot_thermal_transitions.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def finite(x):
        if isinstance(x, dict):
            return {k: finite(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [finite(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(finite(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def prepare(out):
    if (out / "manifest.json").exists():
        raise ValueError("Pilot already prepared; use --run to resume its fixed cohort")
    index = BASE / "evaluation/budgets_v1/criticality_audit_20261027/scenario_data.csv"
    with index.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    selected = []
    for family in FAMILIES:
        for blocked in (True, False):
            pool = [r for r in rows if r["family"] == family and (r["thermal_startup_blocked"] == "True") == blocked]
            optimal = [r for r in pool if r["ref_optimal"] == "True"]
            pool = optimal or pool
            ens = [r for r in pool if float(r["unserved_mwh"]) > .001]
            if blocked and ens:
                pool = ens
            pool.sort(key=lambda r: (int(r["n_zones"]), r["scenario_id"]))
            r = pool[(len(pool)-1)//2]
            scenario = BASE / family / (r["scenario_id"] + ".json")
            reference = BASE / family / "reports" / scenario.name
            old = json.loads(reference.read_text(encoding="utf-8"))
            provenance = old["batch_checkpoint"]["provenance"]
            assert provenance["scenario_sha256"] == sha(scenario)
            assert provenance["solver"] == "highs" and provenance["highs_threads"] == 16
            assert provenance["time_limit_seconds"] == 600
            assert old.get("thermal_formulation") is None, "Expected the historical reference"
            detail = old["detail"]
            for column, key in (("unserved_mwh", "unserved"), ("thermal_mwh", "thermal"),
                                ("overgen_mwh", "overgen_spill"), ("demand_mwh", "demand")):
                actual = sum(sum(series) for series in detail[key].values()) * detail["dt_hours"]
                assert math.isclose(actual, float(r[column]), rel_tol=1e-8, abs_tol=1e-5), column
            selected.append(dict(
                family=family, scenario_id=r["scenario_id"], group="blocked" if blocked else "control",
                scenario_path=str(scenario.relative_to(ROOT)), reference_path=str(reference.relative_to(ROOT)),
                scenario_sha256=sha(scenario), reference_sha256=sha(reference),
                n_zones=int(r["n_zones"]), n_lines=int(r["n_lines"]), ramp_pct=float(r["thermal_ramp_pct"]),
                old_objective=float(old["mip"]["objective"]), old_seconds=float(old["mip"]["solve_seconds"]),
                old_status=old["mip"]["termination"], old_unserved_mwh=float(r["unserved_mwh"]),
                old_overgen_mwh=float(r["overgen_mwh"]), old_thermal_mwh=float(r["thermal_mwh"]),
                demand_mwh=float(r["demand_mwh"]), old_code_hash=provenance["solve_code_sha256"],
                selection_pool_size=len(pool),
            ))
    assert len(selected) == 12
    manifest = dict(protocol="thermal_real_scenarios_pilot_v1", threads=16, mip_limit_seconds=600,
                    per_process_watchdog_seconds=750, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    selection="Two per family: blocked and unblocked; prefer old optimal references, then ENS-positive for blocked when available; lower median by zone count, ties by scenario ID. Fixed before new solves; diagnostic, not representative.",
                    old_index_sha256=sha(index), sources={name: sha(ROOT/name) for name in SOURCE_NAMES}, cases=selected)
    write_json(out / "manifest.json", manifest)
    print(json.dumps(selected, indent=2, ensure_ascii=False))


def verify(manifest):
    assert len(manifest["cases"]) == 12 and manifest["threads"] == 16 and manifest["mip_limit_seconds"] == 600
    for name, expected in manifest["sources"].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f"Source changed after pilot preparation: {name}")
    for c in manifest["cases"]:
        assert sha(ROOT/c["scenario_path"]) == c["scenario_sha256"]
        assert sha(ROOT/c["reference_path"]) == c["reference_sha256"]


def worker(out, index):
    manifest = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    verify(manifest)
    c = manifest["cases"][index]
    target = out / "cases" / c["family"] / (c["scenario_id"] + ".json")
    os.environ["BENCHMARK_HIGHS_THREADS"] = str(manifest["threads"])
    import highspy
    import numpy as np
    from pyomo.environ import Constraint, SolverFactory, Var, value
    from src.milp.model import build_uc_model
    from src.milp.scenario_loader import load_scenario_data
    from src.milp.solve import _compute_cost_components
    from src.milp.solver_resources import configure_highs_threads
    from src.thermal import THERMAL_FORMULATION, thermal_metrics
    from src.demand_response import shifting_metrics

    data = load_scenario_data(ROOT/c["scenario_path"])
    start = time.perf_counter()
    model = build_uc_model(data)
    build_seconds = time.perf_counter()-start
    solver = SolverFactory("highs")
    resources = configure_highs_threads(solver, "highs")
    solver.options["time_limit"] = manifest["mip_limit_seconds"]
    start = time.perf_counter()
    solved = solver.solve(model, load_solutions=False, tee=True)
    api_seconds = time.perf_counter()-start
    reported = getattr(solved.solver, "time", None)
    seconds = float(reported) if reported is not None and float(reported) > 0 else api_seconds
    result = dict(family=c["family"], scenario_id=c["scenario_id"], thermal_formulation=THERMAL_FORMULATION,
                  scenario_sha256=c["scenario_sha256"], reference_sha256=c["reference_sha256"],
                  status=str(solved.solver.termination_condition), solver_status=str(solved.solver.status),
                  solver_resources=resources, highs_version=highspy.Highs().version(),
                  build_seconds=build_seconds, solve_seconds=seconds, solver_api_seconds=api_seconds,
                  feasible=False, objective=None, unserved_mwh=None, overgen_mwh=None, thermal_mwh=None)
    info = solver._solver_model.getInfo()
    result.update(dual_bound=float(info.mip_dual_bound), mip_gap=float(info.mip_gap), mip_nodes=int(info.mip_node_count))
    if len(solved.solution):
        model.solutions.load_from(solved)
        zones, periods = list(model.Z), list(model.T)
        def matrix(name):
            obj = getattr(model, name)
            return np.array([[value(obj[z,t]) for t in periods] for z in zones], dtype=float)
        def vector(name):
            obj = getattr(model, name)
            return np.array([value(obj[z]) for z in zones], dtype=float)
        arrays = {name: matrix(name) for name in ("p_thermal", "u_thermal", "v_thermal_startup", "w_thermal_shutdown",
                  "unserved", "overgen_spill", "demand", "dr_shed", "dr_recover", "dr_rebound", "dr_active")}
        physical = thermal_metrics(
            arrays["p_thermal"], arrays["u_thermal"], arrays["v_thermal_startup"],
            capacity=vector("thermal_capacity"), minimum=vector("thermal_min"), ramp=vector("thermal_ramp"),
            startup_limit=vector("thermal_startup_limit"), shutdown_limit=vector("thermal_shutdown_limit"),
            initial_power=vector("thermal_initial"), initial_on=vector("thermal_initial_on"))
        dr = shifting_metrics(arrays["dr_shed"], arrays["dr_recover"], data.dt_hours,
                              backlog=arrays["dr_rebound"], active=arrays["dr_active"],
                              down_limit=matrix("dr_limit"), up_limit=matrix("dr_recovery_limit"),
                              max_periods=vector("dr_max_events"), ramp=vector("dr_ramp_limit"), tolerance=1e-3)
        max_violation, worst, n_constraints = 0.0, None, 0
        for con in model.component_data_objects(Constraint, active=True):
            body = value(con.body)
            violation = max(0.0, value(con.lower)-body if con.has_lb() else 0.0,
                            body-value(con.upper) if con.has_ub() else 0.0)
            n_constraints += 1
            if not math.isfinite(body):
                violation = math.inf
            if violation > max_violation:
                max_violation, worst = violation, con.name
        max_variable_violation, unused = 0.0, 0
        for var in model.component_data_objects(Var, active=True):
            if var.value is None:
                unused += 1
                continue
            x = float(var.value)
            error = max(0.0, value(var.lb)-x if var.has_lb() else 0.0,
                        x-value(var.ub) if var.has_ub() else 0.0,
                        abs(x-round(x)) if var.is_integer() else 0.0)
            max_variable_violation = max(max_variable_violation, error if math.isfinite(x) else math.inf)
        result.update(objective=float(value(model.obj)), unserved_mwh=float(arrays["unserved"].sum()*data.dt_hours),
                      overgen_mwh=float(arrays["overgen_spill"].sum()*data.dt_hours),
                      thermal_mwh=float(arrays["p_thermal"].sum()*data.dt_hours),
                      thermal_startups=float(arrays["v_thermal_startup"].sum()),
                      thermal_shutdowns=float(arrays["w_thermal_shutdown"].sum()),
                      thermal_metrics=physical, dr_metrics=dr,
                      max_constraint_violation=max_violation, worst_constraint=worst,
                      max_variable_violation=max_variable_violation, n_constraints_checked=n_constraints,
                      unused_variables=unused, cost_components=_compute_cost_components(model),
                      detail=dict(zones=zones, dt_hours=data.dt_hours, **{k:v.tolist() for k,v in arrays.items()}))
        result["feasible"] = bool(physical["thermal_feasible"] and dr["dr_feasible"] and
                                  max_violation <= 1e-3 and max_variable_violation <= 1e-3)
    write_json(target, result)
    print("PILOT_RESULT " + json.dumps({k:v for k,v in result.items() if k not in ("detail", "cost_components")}), flush=True)


def run(out):
    manifest = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    verify(manifest)
    start = time.perf_counter()
    for i, c in enumerate(manifest["cases"]):
        target = out / "cases" / c["family"] / (c["scenario_id"]+".json")
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            assert existing["scenario_sha256"] == c["scenario_sha256"]
            print(f"[{i+1}/12] Already completed: {c['family']}/{c['scenario_id']}", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{i+1}/12] {c['family']}/{c['scenario_id']} ({c['group']}, {c['n_zones']} zones); old={c['old_seconds']:.1f}s", flush=True)
        before = time.perf_counter()
        with target.with_suffix(".log").open("w", encoding="utf-8") as log:
            env = dict(os.environ, BENCHMARK_HIGHS_THREADS="16", OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            child = subprocess.Popen([sys.executable, "-B", "-u", "-m", "scripts.pilot_thermal_transitions",
                                      "--output", str(out), "--worker", str(i)], cwd=ROOT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
            try:
                child.wait(timeout=manifest["per_process_watchdog_seconds"])
            except BaseException:
                child.kill()
                child.wait()
                raise
        if child.returncode or not target.exists():
            raise RuntimeError(f"Pilot worker failed; see {target.with_suffix('.log')}")
        result = json.loads(target.read_text(encoding="utf-8"))
        result["process_wall_seconds"] = time.perf_counter()-before
        write_json(target, result)
        print(f"  {result['status']}; valid={result['feasible']}; new={result['solve_seconds']:.1f}s; cost={result['objective']}; ENS={result['unserved_mwh']} MWh", flush=True)
    verify(manifest)
    write_json(out/"run_complete.json", dict(n=12, invocation_wall_seconds=time.perf_counter()-start,
                                             inputs_unchanged=True, training=False, neural_inference=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", type=int)
    args = parser.parse_args()
    out = args.output.resolve()
    if out == BASE or BASE in out.parents:
        raise ValueError("Pilot outputs must be separate from the original campaign")
    if args.prepare:
        prepare(out)
    elif args.run:
        run(out)
    else:
        worker(out, args.worker)


if __name__ == "__main__":
    main()
