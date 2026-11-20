"""Explicit user-launched runs. The final lock has no routing dependency."""
from pathlib import Path
import uuid
from .core import (config, frozen, campaign, read, put, record, sealed, digest, sha,
                   seed, verify, environment)
from .cases import select
from .design import schedule

def model_directory(run, training_seed=None):
    return Path(run)/"multi_horizon"/str(training_seed or config()["training_seeds"][0])

def guard_test(run, partition, method=None, training_seed=None):
    if partition not in ("final", "controlled_test"):
        return
    run=Path(run); snap=frozen(run)
    decision=verify(read(run/"design_lock.json"))
    if decision["freeze"] != snap["sha256"] or decision["cases"] != sha(run/"cases.json"):
        raise ValueError("Final design changed")
    for path, expected in decision["artifacts"].items():
        if sha(path) != expected:
            raise ValueError("Locked development/model artifact changed")
    if method and method.startswith("mh_"):
        for stage in ("ae", "jepa", "ebm"):
            p=model_directory(run,training_seed)/(stage+"_best.pt")
            if str(p.resolve()) not in decision["artifacts"]:
                raise ValueError("Model was not included in the final lock")

def bind_models(run, method, snap, training_seed=None):
    paths=dict(snap["model_paths"]); hashes=dict(snap["models"])
    if method.startswith("mh_"):
        for stage in ("ae", "jepa", "ebm"):
            folder=model_directory(run,training_seed); p=folder/(stage+"_best.pt")
            receipt=verify(read(folder/(stage+"_receipt.json")))
            if receipt["freeze"]!=snap["sha256"] or receipt["model_sha256"]!=sha(p):
                raise ValueError("Multi-horizon model provenance mismatch")
            paths["mh_"+stage]=str(p.resolve()); hashes["mh_"+stage]=sha(p)
    return paths,hashes

def identities(run, partition, suite, **options):
    snap=frozen(run)
    cache={}
    for case,item in schedule(select(run,partition),suite,**options):
        key=(item["method"],item.get("training_seed"))
        if key not in cache:
            cache[key]=bind_models(run,key[0],snap,key[1])[1]
        identity=dict(**item,protocol=config()["protocol"],freeze=snap["sha256"],
                      case_sha256=case["scenario_sha256"],models=cache[key],
                      timing_policy="unchanged_budget_scaled_reserve")
        if suite=="gurobi":
            identity["gurobi_environment"]=verify(read(Path(run)/"gurobi_environment.json"))["sha256"]
        yield case,item,identity

def output_path(run, identity):
    return Path(run)/"experiments"/(digest(identity)+".json")

def settings(run, method, snap, budget, reference=False, training_seed=None):
    from src.operational_exit.budget_reserve import effective_reserve
    m=read(campaign()/"manifest.json")
    paths,hashes=bind_models(run,method,snap,training_seed)
    return dict(method=method,manifest_hash=snap["manifest_hash"],repair=m["repair"],
                reserve_s=30. if reference else effective_reserve(budget,snap["calibration"]["reserve_s"]),
                extension_sources=snap["sources"],model_hashes=hashes,
                **paths,**({"solver_seconds":config()["reference_seconds"]} if reference else {}))

def execute(run, partition, suite, **options):
    from src.operational_exit.supervisor import run_experiment
    run=Path(run); snap=frozen(run); guard_test(run,partition)
    jobs=list(identities(run,partition,suite,**options))
    remaining=sum(not output_path(run,identity).exists() for _,_,identity in jobs)
    allocated=sum(item["budget"] for _,item,_ in jobs)/3600
    left=sum(item["budget"] for _,item,identity in jobs if not output_path(run,identity).exists())/3600
    print(f"{partition}/{suite}: {len(jobs)} trials, {remaining} remaining; "
          f"deadline allocations {allocated:.2f} h total / {left:.2f} h remaining, excluding preload (not elapsed-time forecasts)",flush=True)
    for case,item,identity in jobs:
        guard_test(run,partition,item["method"],item.get("training_seed"))
        p=output_path(run,identity); old=record(p,identity,optional=True)
        if old is not None:
            if not old.get("complete"): raise ValueError("Incomplete terminal record")
            continue  # observed failures are terminal too
        if sha(case["file"])!=case["scenario_sha256"]: raise ValueError("Case changed")
        attempt=run/"attempts"/digest(identity)[:16]/uuid.uuid4().hex[:12]
        request=dict(item,identity=identity,file=case["file"],scenario_sha256=case["scenario_sha256"],family=case["family"])
        result=run_experiment(settings(run,item["method"],snap,item["budget"],training_seed=item.get("training_seed")),
                              request,attempt,preload_timeout=600,worker_module="research.paper2_revision.worker")
        if (result.get("result") or {}).get("status")=="infrastructure_error":
            raise RuntimeError(f"Gurobi infrastructure/license error; no scientific failure recorded. Inspect {attempt}")
        result.update(attempt=str(attempt),case=case,resources="16 CPU solver threads total; RTX 4050 for learned methods")
        put(p,sealed(result))
        print(case["key"],item["method"],item["budget"],result["status"],flush=True)
    name=f"{partition}_{suite}_{digest(options)[:10]}"
    put(run/"completion"/(name+".json"),sealed(dict(partition=partition,suite=suite,options=options,
        freeze=snap["sha256"],records={str(output_path(run,i)):sha(output_path(run,i)) for _,_,i in jobs})))

def references(run, partition):
    from src.operational_exit.supervisor import run_experiment
    run=Path(run); snap=frozen(run); guard_test(run,partition)
    if partition=="gurobi120": raise ValueError("Historical references are reused")
    cases=select(run,partition)
    remaining=sum(not (run/"references"/(digest(case)+".json")).exists() for case in cases)
    cap=config()['reference_wall_cap_seconds']
    print(f"{partition}: {len(cases)} references, {remaining} remaining; wall cap {cap} s each; "
          f"{len(cases)*cap/3600:.2f} h total / {remaining*cap/3600:.2f} h remaining allocated, excluding preload",flush=True)
    for case in cases:
        p=run/"references"/(digest(case)+".json")
        identity=dict(case=case,freeze=snap["sha256"],solver="highs",seconds=600,threads=16,
                      wall_cap=config()["reference_wall_cap_seconds"])
        old=record(p,identity,optional=True)
        if old is not None:
            if not old.get("complete"): raise ValueError("Incomplete reference record")
            continue
        if sha(case["file"])!=case["scenario_sha256"]: raise ValueError("Case changed")
        budget=config()["reference_wall_cap_seconds"]
        req=dict(key=case["key"],method="highs",budget=budget,repetition=0,seed=seed(case["group"],"reference"),
                 identity=identity,file=case["file"],scenario_sha256=case["scenario_sha256"],family=case["family"])
        attempt=run/"ref_attempts"/digest(identity)[:16]/uuid.uuid4().hex[:12]
        outcome=run_experiment(settings(run,"highs",snap,budget,reference=True),req,attempt,
                              worker_module="research.paper2_revision.worker")
        result=outcome.get("result") or {}
        if not outcome.get("success"):
            result=dict(valid=False,status=outcome["status"],physics=config()["physics"])
        put(p,sealed(dict(identity=identity,complete=True,result=result,supervision=outcome)))
        print("reference",case["key"],result.get("status"),flush=True)

def gurobi_preflight(run):
    """Optional installation/license check; no model solve."""
    import importlib.metadata
    import gurobipy as gp
    with gp.Env(params={"OutputFlag":0}):
        version=gp.gurobi.version()
    obj=sealed(dict(package=importlib.metadata.version("gurobipy"),engine=list(version),
                    adapter_sha256=sha(Path(__file__).with_name("gurobi.py")),threads=16,
                    disclosure="License initialization passed; model-size entitlement is checked by the synthetic test and actual runs."))
    put(Path(run)/"gurobi_environment.json",obj)
    return obj

def lock_design(run):
    run=Path(run); snap=frozen(run)
    required=[run/"cases.json",run/"input_audit.json",run/"multi_horizon"/"dataset.json"]
    for partition,suite in (("development","four"),("development","ablation"),
                            ("controlled_development","four"),("controlled_development","multi")):
        completions=list((run/"completion").glob(f"{partition}_{suite}_*.json"))
        if not completions: raise ValueError(f"Complete {partition}/{suite} before locking")
        for p in completions:
            receipt=verify(read(p))
            if receipt["freeze"]!=snap["sha256"]: raise ValueError("Development freeze mismatch")
            for path,h in receipt["records"].items():
                if sha(path)!=h: raise ValueError("Development record changed")
        required.extend(completions)
    for training_seed in config()["training_seeds"]:
        for stage in ("ae","jepa","ebm"):
            folder=model_directory(run,training_seed)
            required.extend([folder/(stage+"_best.pt"),folder/(stage+"_receipt.json")])
    artifacts={str(p.resolve()):sha(p) for p in required}
    put(run/"design_lock.json",sealed(dict(freeze=snap["sha256"],cases=sha(run/"cases.json"),
        artifacts=artifacts,primary_inference=config()["primary_inference"],
        disclosure="No router dependency. Primary model seed fixed before final evaluation; no test-driven changes.")))
