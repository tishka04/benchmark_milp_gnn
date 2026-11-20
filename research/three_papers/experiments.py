"""User-launched serial runs. All writes remain in the new research directory."""
from __future__ import annotations
from pathlib import Path
import uuid
from .core import (config, frozen, campaign, read, put, record, sealed, digest, sha,
                   METHODS, seed, writable, verify)
from .cases import select

SUITES={"four":METHODS,"ablation":("pipeline","perturb16"),"gurobi":("gurobi",),
        "multi":("mh_jepa","mh_pipeline"),"router":("router",),
        "routing_baselines":tuple("router_"+k for k in ("highs","highs_rh","fixed","family","index","logistic","tree"))}

def guard_test(run,partition,method=None):
    if partition in ("final","controlled_test"):
        decision=verify(read(Path(run)/"design_lock.json"))
        if decision["cases"]!=sha(Path(run)/"cases.json"):
            raise ValueError("Test design changed")
        for p,h in decision["artifacts"].items():
            if sha(p)!=h: raise ValueError("Final decisions/model changed")
        needed=[]
        if method=="router" or (method and method.startswith("router_")):
            needed=[Path(run)/"routing"/(("selector" if method=="router" else method.removeprefix("router_"))+".joblib")]
        if method and method.startswith("mh_"):
            needed=[Path(run)/"multi_horizon"/(name+"_best.pt") for name in ("ae","jepa","ebm")]
        if any(str(p.resolve()) not in decision["artifacts"] for p in needed):
            raise ValueError("Requested model was not part of the final design lock")

def bind_models(run,method,snap):
    paths=dict(snap["model_paths"]); hashes=dict(snap["models"])
    if method.startswith("mh_"):
        for name in ("ae","jepa","ebm"):
            p=Path(run)/"multi_horizon"/(name+"_best.pt")
            receipt=verify(read(Path(run)/"multi_horizon"/(name+"_receipt.json")))
            if receipt["freeze"]!=snap["sha256"] or receipt["model_sha256"]!=sha(p):
                raise ValueError("Multi-horizon model provenance mismatch")
            paths["mh_"+name]=str(p.resolve()); hashes["mh_"+name]=sha(p)
    if method=="router" or method.startswith("router_"):
        kind="selector" if method=="router" else method.removeprefix("router_")
        p=Path(run)/"routing"/(kind+".joblib")
        receipt=verify(read(Path(run)/"routing"/"model_receipt.json"))
        expected=receipt["model_sha256"] if kind=="selector" else receipt["baseline_models"][kind]
        if receipt["freeze"]!=snap["sha256"] or expected!=sha(p):
            raise ValueError("Router provenance mismatch")
        paths["router"]=str(p.resolve()); hashes["router"]=sha(p)
    return paths,hashes

def identities(run,partition,suite):
    from .requirements import request_for, profile_signature, SCHEMA, deadlines
    snap=frozen(run)
    if suite not in SUITES: raise ValueError(suite)
    routing = suite in ("router", "routing_baselines")
    enabled = [None]
    if routing:
        receipt=verify(read(Path(run)/"routing"/"model_receipt.json"))
        if receipt.get("schema") != SCHEMA or receipt["profile_signature"] != profile_signature():
            raise ValueError("Router/profile schema mismatch")
        enabled=receipt["enabled_requests"]
    bindings={m:bind_models(run,m,snap)[1] for m in SUITES[suite]}
    for i,c in enumerate(select(run,partition)):
        for budget in (deadlines() if routing else config()["extension_budgets"]):
            methods=SUITES[suite]
            methods=methods[i%len(methods):]+methods[:i%len(methods)]
            for method in methods:
                if c["expected_horizon"]>24 and (method in ("jepa","pipeline","perturb16","router") or method.startswith("router_")):
                    continue
                for requested in enabled:
                    if routing and requested["deadline_s"] != budget:
                        continue
                    pid = requested["profile_id"] if routing else None
                    item=dict(key=c["key"],method=method,budget=budget,repetition=0,seed=seed(c["group"],0))
                    if pid is not None:
                        item.update(profile_id=pid,request=request_for(pid,float(budget)))
                    identity=dict(**item,protocol=config()["protocol"],freeze=snap["sha256"],
                                  case_sha256=c["scenario_sha256"],models=bindings[method],
                                  timing_policy="remaining_allowance_service_v2" if routing else "unchanged_budget_scaled_reserve",
                                  suite=suite)
                    yield c,item,identity

def output_path(run,identity):
    return Path(run)/"experiments"/(digest(identity)+".json")

def settings(run,method,snap,budget,reference=False):
    from src.operational_exit.budget_reserve import effective_reserve
    m=read(campaign()/"manifest.json")
    paths,hashes=bind_models(run,method,snap)
    worker_method="router" if method.startswith("router_") else method
    return dict(method=worker_method,manifest_hash=snap["manifest_hash"],repair=m["repair"],
                reserve_s=30. if reference else effective_reserve(budget,snap["calibration"]["reserve_s"]),
                extension_sources=snap["sources"],model_hashes=hashes,base_reserve_s=snap["calibration"]["reserve_s"],
                **paths,**({"solver_seconds":config()["reference_seconds"]} if reference else {}))

def execute(run,partition,suite):
    from src.operational_exit.supervisor import run_experiment
    run=Path(run); snap=frozen(run); guard_test(run,partition)
    for c,item,identity in identities(run,partition,suite):
        guard_test(run,partition,item["method"])
        p=output_path(run,identity)
        old=record(p,identity,optional=True)
        if old is not None:
            if not old.get("complete"): raise ValueError("Unexpected incomplete terminal record")
            continue  # failures are terminal too
        if sha(c["file"])!=c["scenario_sha256"]: raise ValueError("Case changed")
        attempt=run/"attempts"/digest(identity)[:16]/uuid.uuid4().hex[:12]
        request=dict(item,identity=identity,file=c["file"],scenario_sha256=c["scenario_sha256"],
                     family=c["family"])
        result=run_experiment(settings(run,item["method"],snap,item["budget"]),request,attempt,
                              preload_timeout=600,worker_module="research.three_papers.worker")
        result.update(attempt=str(attempt),case=c,resources="16 CPU solver threads total; RTX 4050 for learned methods")
        put(p,sealed(result))
        print(c["key"],item["method"],item["budget"],item.get("profile_id"),result["status"],flush=True)

def references(run,partition):
    from src.operational_exit.supervisor import run_experiment
    run=Path(run); snap=frozen(run); guard_test(run,partition)
    if partition=="gurobi120": raise ValueError("Use the unchanged existing common references")
    for c in select(run,partition):
        p=run/"references"/(digest(c)+".json")
        identity=dict(case=c,freeze=snap["sha256"],solver="highs",seconds=600,threads=16,
                      wall_cap=config()["reference_wall_cap_seconds"])
        if record(p,identity,optional=True) is not None: continue
        if sha(c["file"])!=c["scenario_sha256"]: raise ValueError("Case changed")
        budget=config()["reference_wall_cap_seconds"]
        req=dict(key=c["key"],method="highs",budget=budget,repetition=0,seed=seed(c["group"],"reference"),
                 identity=identity,file=c["file"],scenario_sha256=c["scenario_sha256"],family=c["family"])
        attempt=run/"ref_attempts"/digest(identity)[:16]/uuid.uuid4().hex[:12]
        outcome=run_experiment(settings(run,"highs",snap,budget,reference=True),req,attempt,
                              worker_module="research.three_papers.worker")
        result=outcome.get("result") or {}
        if not outcome.get("success"):
            result=dict(valid=False,status=outcome["status"],physics=config()["physics"])
        put(p,sealed(dict(identity=identity,complete=True,result=result,supervision=outcome)))
        print("reference",c["key"],result.get("status"),flush=True)

def lock_design(run):
    run=Path(run); snap=frozen(run)
    from .requirements import profiles, profile_signature
    required=[run/"main_rows.json",run/"routing"/"diagnostic.json"]
    gate=verify(read(required[-1]))
    if gate.get("profile_signature") != profile_signature(): raise ValueError("Profile diagnostic changed")
    if gate["proceed"] and not (run/"routing"/"model_receipt.json").exists():
        raise ValueError("Complete gated router development before the final lock")
    if (run/"routing"/"model_receipt.json").exists():
        receipt=verify(read(run/"routing"/"model_receipt.json"))
        for p,h in receipt["artifacts"].items():
            if sha(p)!=h: raise ValueError("Router artifact changed")
        required.append(run/"routing"/"model_receipt.json")
    optional=list((run/"routing").glob("*.joblib"))
    optional += [run/"multi_horizon"/(s+"_best.pt") for s in ("ae","jepa","ebm")]
    artifacts={str(p.resolve()):sha(p) for p in required+[p for p in optional if p.exists()]}
    put(run/"design_lock.json",sealed(dict(freeze=snap["sha256"],cases=sha(run/"cases.json"),artifacts=artifacts,
        profiles=profiles(),profile_signature=profile_signature(),
        routing_settings=config()["service_routing"],final_partitions=["final","controlled_test"],
        disclosure="The 600 historical scenarios are development data. Final outcomes were not used in this lock.")))
