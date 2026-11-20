"""Explicit observation grid, common criticality and independently validated costs."""
from __future__ import annotations
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import copy
import math
import time
from .core import (ROOT, campaign, config, read, record, sha, put, sealed, frozen,
                   scheduled, digest, economic, gap, FAMILIES)
from .cases import load_data

def feature_row(path, family, data=None):
    """Only pre-solve inputs. Runtime measured by the online router, not reused here."""
    from src.analysis.criticality_index import compute_criticality, NORM_BOUNDS
    raw = read(path); descriptor = copy.deepcopy(raw)
    data = load_data(path) if data is None else data
    condition = raw.get("_three_papers", {}).get("condition", "base")
    # Preserve the published generator-proxy definitions. Explicitly update size/horizon
    # descriptors; topology and physical capacity changes may be invisible to this index.
    descriptor.setdefault("meta", {})["zones"] = len(data.zones)
    descriptor["horizon_hours"] = len(data.periods)*data.dt_hours
    if condition == "size":
        descriptor["meta"]["assets"] = {k: 2*v for k,v in descriptor["meta"].get("assets", {}).items()}
        diff = descriptor.setdefault("difficulty_indicators", {})
        diff["n_binary_variables"] = 2*diff.get("n_binary_variables", 0)
    if condition.startswith("horizon"):
        diff = descriptor.setdefault("difficulty_indicators", {})
        diff["n_binary_variables"] = diff.get("n_binary_variables", 0)*len(data.periods)/24
    c = compute_criticality(descriptor, alpha=.5)  # same weights for every family
    components = {**c.stress_normalized, **c.hardness_normalized}
    raw_components = {**asdict(c.stress_metrics), **asdict(c.hardness_metrics)}
    row = dict(family=family, n_zones=len(data.zones), n_steps=len(data.periods),
               n_lines=len(data.lines), zone_steps=len(data.zones)*len(data.periods),
               stress=c.stress_index, hardness=c.hardness_index, criticality=c.criticality_index,
               thermal_mw=sum(data.thermal_capacity.values()),
               battery_mw=sum(data.battery_power.values()),
               demand_mwh=sum(data.demand.values())*data.dt_hours,
               active_thermal_zones=int(sum(x>0 for x in data.thermal_capacity.values())),
               index_scope="generator_proxy_common_v1", **components)
    for k, value in raw_components.items():
        row["raw_"+k] = value
        lo, hi = NORM_BOUNDS[k]
        row["sat_"+k] = int(value <= lo or value >= hi)
    return row

def accepted(r, budget, data):
    from src.operational_exit.physics import validate_dispatch
    if r is None or not r.get("complete"):
        return None, {}
    success = bool(r.get("success") and not r.get("forced") and r.get("all_processes_exited")
                   and r.get("returncode") == 0 and isinstance(r.get("exit_s"), (float,int))
                   and math.isfinite(r["exit_s"]) and r["exit_s"] <= budget)
    check = {}
    if success:
        result = r.get("result") or {}
        check = validate_dispatch(data, result.get("dispatch"))
        stored = (result.get("validation") or {}).get("cost")
        if (not check.get("valid") or stored is None or
            not math.isclose(check["cost"], stored, rel_tol=1e-9, abs_tol=1e-3)):
            raise ValueError("An accepted outcome fails physical/cost revalidation")
    if r.get("success") and not success:
        raise ValueError("An accepted outcome fails the process-exit contract")
    return success, check

def reference(result, data):
    from src.operational_exit.physics import validate_dispatch
    if not result or not result.get("valid"):
        return dict(reference_cost=None, reference_bound=None, reference_certified=False)
    if result.get("physics") != config()["physics"]:
        raise ValueError("Reference formulation mismatch")
    if result.get("solver_limit") != 600 or result.get("threads") != 16:
        raise ValueError("Reference resources differ from the common reference protocol")
    check = validate_dispatch(data, result.get("dispatch"))
    if not check["valid"]:
        raise ValueError("Reference failed physical validation")
    bound = result.get("bound")
    if bound is not None and not math.isfinite(bound):
        bound = None
    if bound is not None and bound > check["cost"] + 1e-3:
        raise ValueError("Reference lower bound exceeds feasible reference cost")
    return dict(reference_cost=check["cost"], reference_bound=bound,
                reference_certified=bool(result.get("proven_optimal")))

def observation(case, item, r, ref, features, data):
    success, check = accepted(r, item["budget"], data)
    cost = check.get("cost") if success else None
    rel, delta = gap(cost, ref["reference_cost"])
    row = dict(group=case.get("group", case["key"]),
               partition=case.get("partition", "historical_development"),
               condition=case.get("condition", "historical"), **features, **item,
               observed=bool(r and r.get("complete")), feasible=success,
               status=r["status"] if r else "not_run", cost=cost,
               ens_mwh=check.get("ens_mwh"), relative_gap=rel, absolute_gap_eur=delta, **ref)
    for threshold in config()["quality_thresholds"]:
        row[f"economic_{round(100*threshold)}"] = economic(success, cost, ref["reference_cost"], threshold)
    for name in ("exit_s", "preload_s", "compute_s", "finalization_s"):
        row[name] = r.get(name) if r else None
    result = (r or {}).get("result") or {}
    row["selected_method"] = result.get("selected_method",item["method"])
    row.update({"stage_"+k: v for k, v in result.get("stage_seconds", {}).items()})
    for k in ("repair_changed_fraction", "repair_stage", "generated_candidates", "unfinished_candidates"):
        row[k] = result.get(k)
    decision=result.get("routing_decision") or {}
    for name in ("risk_supported", "fallback", "predicted_success", "predicted_cost", "predicted_time",
                 "prediction_allowance_s", "method_allowance_s", "outside_budget_support",
                 "interpolated_budget", "selection_overrun", "calibration_status",
                 "outside_feature_support", "feature_support_warnings"):
        row["routing_"+name]=decision.get(name)
    return row

def extract_main(run):
    run = Path(run); snap = frozen(run)
    m = read(campaign()/"manifest.json")
    cases = {c["key"]: c for c in m["cohort"]}
    @lru_cache(maxsize=4)
    def case_inputs(key):
        c = cases[key]
        if sha(c["file"]) != c["scenario_sha256"]:
            raise ValueError("Scenario changed")
        data = load_data(c["file"])
        p = campaign()/"references"/c["family"]/(c["scenario_id"]+".json")
        r = record(p)
        if r["sha256"] != snap["references"][str(p)] or r["identity"]["manifest_hash"] != digest(m):
            raise ValueError("Reference provenance mismatch")
        return data, feature_row(c["file"], c["family"]), reference(r["result"], data)
    rows = []
    # Sorting improves reuse; it never selects observations by outcome.
    for _, item, identity, p in sorted(scheduled(), key=lambda x: x[1]["key"]):
        r = record(p, identity)
        if r["sha256"] != snap["records"][str(p)]:
            raise ValueError("Source record changed since freeze")
        data, features, ref = case_inputs(item["key"])
        rows.append(observation(cases[item["key"]], item, r, ref, features, data))
    put(run/"main_rows.json", sealed(dict(freeze=snap["sha256"], rows=rows)))

def audit_dimensions(run):
    """Actual training shapes and actual free model decisions, no solver invocation."""
    import numpy as np
    from pyomo.environ import Var, Constraint
    from src.milp.model import build_uc_model
    from src.analysis.criticality_index import NORM_BOUNDS
    run = Path(run); snap = frozen(run); m = read(campaign()/"manifest.json")
    dataset = record(campaign()/"dataset.json")
    shapes = []
    for entry in dataset["entries"]:
        if sha(entry["file"]) != entry["sha256"]:
            raise ValueError("Training dataset changed")
        with np.load(entry["file"], allow_pickle=False) as arr:
            z,t,f = arr["u_star"].shape
        shapes.append(dict(key=entry["key"], partition=entry["partition"], zones=z, steps=t, features=f))
    dimensions = []
    for c in m["cohort"]:
        model = build_uc_model(load_data(c["file"]))
        variables = list(model.component_data_objects(Var))
        dimensions.append(dict(key=c["key"], variables=len(variables),
            integer_total=sum(v.is_integer() for v in variables),
            integer_free=sum(v.is_integer() and not v.fixed for v in variables),
            constraints=sum(1 for _ in model.component_data_objects(Constraint, active=True))))
    put(run/"dimension_audit.json", sealed(dict(freeze=snap["sha256"], training_shapes=shapes,
        evaluation_dimensions=dimensions, normalization_bounds=NORM_BOUNDS,
        warning="Equipment counts and graph nodes are not individual commitment decisions; compare free integers.")))

def extract_extension(run, partition, suite):
    from .experiments import identities, output_path
    from .cases import select
    run=Path(run); snap=frozen(run)
    cases={c["key"]:c for c in select(run,partition)}
    rows=[]
    @lru_cache(maxsize=2)
    def inputs(key):
        c=cases[key]
        if sha(c["file"]) != c["scenario_sha256"]: raise ValueError("Extension case changed")
        data=load_data(c["file"])
        if partition=="gurobi120":
            p=campaign()/"references"/c["family"]/(c["scenario_id"]+".json")
            rr=record(p)
            if rr["sha256"] != snap["references"][str(p)]: raise ValueError("Historical reference changed")
            ref=reference(rr["result"],data)
        else:
            p=run/"references"/(digest(c)+".json")
            rr=record(p,optional=True)
            if rr and (rr["identity"]["case"]!=c or rr["identity"]["freeze"]!=snap["sha256"]):
                raise ValueError("Extension reference identity mismatch")
            ref=reference(rr.get("result") if rr else None,data)
        return data,feature_row(c["file"],c["family"]),ref
    for c,item,identity in identities(run,partition,suite):
        data,features,ref=inputs(c["key"])
        r=record(output_path(run,identity),identity,optional=True)
        if r is None and (run/"attempts"/digest(identity)[:16]).exists():
            r=dict(complete=False,status="interrupted_or_inflight_no_observed_terminal_outcome")
        rows.append(observation(c,item,r,ref,features,data))
    put(run/f"rows_{partition}_{suite}.json",sealed(dict(freeze=snap["sha256"],rows=rows)))
