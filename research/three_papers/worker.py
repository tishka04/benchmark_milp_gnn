"""New worker, reusing the unchanged Windows process-tree supervisor.

Adapted from src/operational_exit/worker.py: same repair cascade, selection and
cooperative joins. Adds isolated data adapter, timing, EBM control and routing.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time
import traceback

def uc_from_dispatch(dispatch):
    import numpy as np
    v=dispatch["values"]
    b=np.asarray(v["b_charge"])>1e-3; bd=np.asarray(v["b_discharge"])>1e-3
    p=np.asarray(v["pumped_charge"])>1e-3; pd=np.asarray(v["pumped_discharge"])>1e-3
    return np.stack((b,bd,p,pd,np.asarray(v["dr_active"])>=.5,np.asarray(v["v_thermal_startup"])>=.5,
                     np.asarray(v["u_thermal"])>=.5,np.broadcast_to(v["import_mode"],b.shape)>=.5),axis=-1).astype(np.float32)

def repair_process(conn,settings):
    import pandas
    import torch
    from .cases import install_loader, load_data
    install_loader()
    from src.operational_exit.solving import make_repair_worker
    from src.operational_exit.physics import validate_dispatch
    torch.set_num_threads(1)
    os.environ["BENCHMARK_HIGHS_THREADS"]=str(settings["threads"])
    try:
        worker=make_repair_worker(settings["repair"])
        conn.send({"ready":True}); job=conn.recv()
        if job is None: return
        worker.scenarios_dir=str(Path(job["file"]).parent)
        worker.deadline=job["compute_deadline"]
        options=getattr(worker.solver,"highs_options",None)
        if options is None: options=worker.solver.options
        options["random_seed"]=job["seed"]
        data=load_data(job["file"]); worker.dt_hours=data.dt_hours
        start=time.perf_counter()
        r=worker.solve(Path(job["file"]).stem,torch.from_numpy(job["candidate"][:,:,:7]),
                       max_stages=3,candidate_time_budget_s=max(.001,worker.deadline-time.perf_counter()))
        dispatch=(r.continuous_vars or {}).get("operational_dispatch")
        check=validate_dispatch(data,dispatch) if dispatch else {"valid":False}
        status="solution" if check["valid"] else "invalid_dispatch" if dispatch else "no_incumbent"
        if not dispatch and r.status=="error":
            status="cooperative_deadline" if "deadline" in r.message else "repair_error"
        changed=None
        if dispatch:
            # All seven supplied targets; structural no-asset targets are included and reported.
            changed=float((uc_from_dispatch(dispatch)[:,:,:7]!=job["candidate"][:,:,:7]).mean())
        conn.send(dict(status=status,dispatch=dispatch,validation=check,repair_status=r.status,
                       repair_stage=str(r.stage_used),repair_message=r.message,candidate=job["index"],
                       repair_changed_fraction=changed,repair_seconds=time.perf_counter()-start))
    except TimeoutError:
        conn.send(dict(status="cooperative_deadline",validation={"valid":False}))
    except Exception as exc:
        conn.send(dict(status="repair_error",error=repr(exc),validation={"valid":False}))
    finally:
        conn.close()

def main():
    control=sys.stdout
    def event(name,**kw):
        control.write(json.dumps(dict(event=name,**kw))+"\n"); control.flush()
    event("BOOT",pid=os.getpid())
    from src.operational_exit.supervisor import enroll_current_process
    enroll_current_process(json.loads(sys.stdin.readline())["job_name"])
    event("ENROLLED",pid=os.getpid())
    settings=json.loads(sys.stdin.readline())
    from .core import sha, source_hashes
    if source_hashes()!=settings["extension_sources"]:
        raise ValueError("Extension source changed")
    if any(sha(settings[k])!=v for k,v in settings["model_hashes"].items()):
        raise ValueError("Model changed")
    import pandas
    import numpy as np
    import torch
    import highspy
    from .cases import install_loader, load_data
    install_loader()
    from src.operational_exit.common import atomic_json, sealed, seed_for
    from src.operational_exit.physics import validate_dispatch
    from src.operational_exit.solving import solve_data
    from src.operational_exit.worker import select_candidates
    from src.heuristics.rolling_horizon import rolling_horizon_heuristic, HeuristicConfig
    from src.ebm.feasibility import load_physics_from_scenario
    from .fixed_generator import Generator
    torch.set_num_threads(1)
    method=settings["method"]; generators={}; pools={}; env=None; router=None
    try:
        if method=="gurobi":
            import gurobipy as gp
            env=gp.Env(params={"OutputFlag":0})
        if method=="router":
            import joblib
            router=joblib.load(settings["router"])
            for name in ("jepa","pipeline"):
                generators[name]=Generator(dict(settings,method=name))
        elif method in ("jepa","pipeline","perturb16"):
            generators[method]=Generator(settings)
        elif method in ("mh_jepa","mh_pipeline"):
            from .multihorizon import MultiGenerator
            generators[method]=MultiGenerator(settings)
        if method not in ("highs","gurobi"):
            import multiprocessing as mp
            ctx=mp.get_context("spawn")
            counts=(1,16) if method=="router" else (16,) if method in ("pipeline","perturb16","mh_pipeline") else (1,)
            for count in counts:
                pools[count]=[]
                for _ in range(count):
                    parent,child=ctx.Pipe()
                    proc=ctx.Process(target=repair_process,args=(child,dict(settings,threads=1 if count==16 else 16)))
                    proc.start(); child.close(); pools[count].append((parent,proc))
                for conn,proc in pools[count]:
                    if not conn.poll(240) or not conn.recv().get("ready"):
                        raise RuntimeError("Repair preload failed")
        if generators: torch.cuda.reset_peak_memory_stats()
        event("READY",metadata=dict(method=method,scenario_preloaded=False,
              repair_processes=sum(len(p) for p in pools.values()),static_portfolio_preloaded=bool(router)))
        req=json.loads(sys.stdin.readline()); deadline=req["deadline"]-settings["reserve_s"]
        stage={}; sent=set(); result=dict(status="no_incumbent")
        routing_decision=None; method_req=req
        try:
            start=time.perf_counter()
            if sha(req["file"])!=req["scenario_sha256"]: raise ValueError("Scenario changed")
            data=load_data(req["file"]); stage["scenario_read"]=time.perf_counter()-start
            if router is not None:
                from .extract import feature_row
                from .routing import choose
                from .requirements import check_request
                from src.operational_exit.budget_reserve import effective_reserve
                request=check_request(req["request"])
                if request["deadline_s"] != float(req["budget"]):
                    raise ValueError("Request deadline differs from execution identity")
                start=time.perf_counter()
                features=feature_row(req["file"],req["family"],data=data)
                stage["features"]=time.perf_counter()-start
                from .core import config
                allowance=max(0.,req["deadline"]-time.perf_counter()-
                              config()["service_routing"]["selection_reserve_s"])
                start=time.perf_counter()
                routing_decision=choose(router,features,allowance,request)
                stage["routing"]=time.perf_counter()-start
                actual_allowance=max(0.,req["deadline"]-time.perf_counter())
                routing_decision["method_allowance_s"]=actual_allowance
                routing_decision["selection_overrun"]=actual_allowance<allowance
                if routing_decision["selection_overrun"]:
                    routing_decision["risk_supported"]=False
                method=routing_decision["method"]
                # Same reserve rule, applied to the remaining method allowance.
                if actual_allowance<=0: raise TimeoutError("Routing exhausted allowance")
                deadline=req["deadline"]-effective_reserve(actual_allowance,settings["base_reserve_s"])
                method_req=dict(req,budget=actual_allowance)
                stage["features_and_routing"]=stage["features"]+stage["routing"]
            if time.perf_counter()>=deadline: raise TimeoutError("No compute allowance")
            if method in ("highs","gurobi"):
                start=time.perf_counter()
                seconds=settings.get("solver_seconds",method_req["budget"])
                if method=="highs":
                    result=solve_data(data,threads=16,seconds=seconds,seed=req["seed"],deadline=deadline)
                else:
                    from .gurobi import solve
                    result=solve(data,seconds=seconds,seed=req["seed"],deadline=deadline,env=env)
                result["status"]="solution" if result.get("valid") else result["status"]
                for k in ("detail","lp","lp_duals"): result.pop(k,None)
                stage["direct_solve"]=time.perf_counter()-start
            else:
                generator=generators.get(method)
                start=time.perf_counter()
                if generator:
                    candidates,seeds=generator.candidates(data,method_req,deadline)
                    stage.update(generator.stage_seconds)
                else:
                    physics=load_physics_from_scenario(Path(req["file"]).stem,str(Path(req["file"]).parent))
                    candidates=rolling_horizon_heuristic(physics,HeuristicConfig()).numpy()[None]
                    seeds=[seed_for(req["key"],req["repetition"],0)]
                    stage["generation"]=time.perf_counter()-start
                if time.perf_counter()>=deadline: raise TimeoutError("Generation exhausted allowance")
                pool=pools[len(candidates)]; start=time.perf_counter()
                for i,(conn,proc) in enumerate(pool):
                    conn.send(dict(file=req["file"],compute_deadline=deadline,candidate=candidates[i],index=i,seed=seeds[i]))
                    sent.add(id(conn))
                pending=set(range(len(pool))); completed=[]
                while pending and time.perf_counter()<deadline:
                    for i in list(pending):
                        try:
                            if pool[i][0].poll():
                                completed.append(pool[i][0].recv()); pending.remove(i)
                        except (EOFError,BrokenPipeError):
                            completed.append(dict(status="repair_crash",validation={"valid":False})); pending.remove(i)
                    if pending: time.sleep(.001)
                result=select_candidates(completed)
                result.update(candidate_diagnostics=[{k:v for k,v in r.items() if k!="dispatch"} for r in completed],
                              generated_candidates=len(candidates),unfinished_candidates=len(pending),candidate_seeds=seeds)
                stage["repair_wait"]=time.perf_counter()-start
        except TimeoutError:
            result=dict(status="cooperative_deadline")
        except Exception as exc:
            traceback.print_exc(); result=dict(status="worker_error",error=repr(exc))
        compute_end=time.perf_counter()
        pool=[x for p in pools.values() for x in p]
        for conn,proc in pool:
            if id(conn) not in sent: conn.send(None)
        # No forced stop can become a success. Drain only to unblock writers; never select late candidates.
        while any(proc.is_alive() for conn,proc in pool):
            for conn,proc in pool:
                try:
                    if conn.poll(): conn.recv()
                except (EOFError,BrokenPipeError): pass
            time.sleep(.001)
        for conn,proc in pool: proc.join(); conn.close()
        start=time.perf_counter()
        if result.get("dispatch"):
            result["validation"]=validate_dispatch(data,result["dispatch"])
            if not result["validation"]["valid"]: result["status"]="invalid_dispatch"
        stage["validation"]=time.perf_counter()-start
        if env is not None: env.dispose(); env=None
        if generators:
            torch.cuda.synchronize(); result["gpu_peak_bytes"]=torch.cuda.max_memory_allocated()
        result.update(identity=req["identity"],compute_end=compute_end,stage_seconds=stage,selected_method=method,
                      routing_decision=routing_decision,request=req.get("request"))
        atomic_json(req["result_file"],sealed(result),deadline=req["deadline"])
    except Exception as exc:
        event("ERROR",error=repr(exc)); raise
    finally:
        if env is not None: env.dispose()

if __name__=="__main__":
    main()
