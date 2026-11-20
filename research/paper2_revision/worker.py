"""Isolated timed worker using the historical process-tree supervisor."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

def uc_from_dispatch(dispatch):
    import numpy as np
    v=dispatch["values"]; b=np.asarray(v["b_charge"])>1e-3
    return np.stack((b,np.asarray(v["b_discharge"])>1e-3,
        np.asarray(v["pumped_charge"])>1e-3,np.asarray(v["pumped_discharge"])>1e-3,
        np.asarray(v["dr_active"])>=.5,np.asarray(v["v_thermal_startup"])>=.5,
        np.asarray(v["u_thermal"])>=.5,np.broadcast_to(v["import_mode"],b.shape)>=.5),axis=-1).astype(np.float32)

def repair_change(data, candidate, dispatch):
    """Only applicable decisions enter the primary repair-change denominator."""
    import numpy as np
    z,t=len(data.zones),len(data.periods)
    mask=np.zeros((z,t,7),dtype=bool)
    for i,zone in enumerate(data.zones):
        mask[i,:,0:2]=data.battery_power[zone]>0 and data.battery_energy[zone]>0
        mask[i,:,2:4]=data.pumped_power[zone]>0 and data.pumped_energy[zone]>0
        mask[i,:,4]=[data.dr_limit[zone,period]>0 for period in data.periods]
        mask[i,:,5:7]=data.thermal_capacity[zone]>0
    changed=uc_from_dispatch(dispatch)[:,:,:7]!=candidate[:,:,:7]
    return dict(repair_changed_fraction=float(changed[mask].mean()) if mask.any() else None,
                repair_applicable_decisions=int(mask.sum()),repair_changed_count=int(changed[mask].sum()),
                repair_changed_fraction_all=float(changed.mean()))

def repair_process(conn, settings):
    import pandas  # Load Arrow DLLs before Torch on Windows.
    import torch
    from .cases import install_loader, load_data
    install_loader()
    from src.operational_exit.solving import make_repair_worker
    from src.operational_exit.physics import validate_dispatch
    torch.set_num_threads(1); os.environ["BENCHMARK_HIGHS_THREADS"]=str(settings["threads"])
    try:
        worker=make_repair_worker(settings["repair"])
        conn.send({"ready":True}); job=conn.recv()
        if job is None: return
        worker.scenarios_dir=str(Path(job["file"]).parent); worker.deadline=job["compute_deadline"]
        options=getattr(worker.solver,"highs_options",None)
        if options is None: options=worker.solver.options
        options["random_seed"]=job["seed"]
        data=load_data(job["file"]); worker.dt_hours=data.dt_hours; start=time.perf_counter()
        r=worker.solve(Path(job["file"]).stem,torch.from_numpy(job["candidate"][:,:,:7]),
                       max_stages=3,candidate_time_budget_s=max(.001,worker.deadline-time.perf_counter()))
        repair_end=time.perf_counter(); dispatch=(r.continuous_vars or {}).get("operational_dispatch")
        check=validate_dispatch(data,dispatch) if dispatch else {"valid":False}
        status="solution" if check["valid"] else "invalid_dispatch" if dispatch else "no_incumbent"
        if not dispatch and r.status=="error":
            status="cooperative_deadline" if "deadline" in r.message else "repair_error"
        changes=repair_change(data,job["candidate"],dispatch) if dispatch else {}
        conn.send(dict(status=status,dispatch=dispatch,validation=check,repair_status=r.status,
                       repair_stage=str(r.stage_used),repair_message=r.message,candidate=job["index"],
                       repair_seconds=repair_end-start,candidate_validation_seconds=time.perf_counter()-repair_end,
                       **changes))
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
    enroll_current_process(json.loads(sys.stdin.readline())["job_name"]); event("ENROLLED",pid=os.getpid())
    settings=json.loads(sys.stdin.readline())
    from .core import sha, source_hashes
    if source_hashes()!=settings["extension_sources"]: raise ValueError("Extension source changed")
    if any(sha(settings[k])!=v for k,v in settings["model_hashes"].items()): raise ValueError("Model changed")
    import pandas
    import torch
    from .cases import install_loader, load_data
    install_loader()
    from .physics_adapter import physics_from_data
    from src.operational_exit.common import atomic_json, sealed, seed_for
    from src.operational_exit.physics import validate_dispatch
    from src.operational_exit.solving import solve_data
    from src.operational_exit.worker import select_candidates
    from src.heuristics.rolling_horizon import rolling_horizon_heuristic, HeuristicConfig
    from .fixed_generator import Generator
    torch.set_num_threads(1)
    method=settings["method"]; generator=None; pool=[]; env=None
    try:
        if method=="gurobi":
            import gurobipy as gp
            env=gp.Env(params={"OutputFlag":0})
        elif method in ("jepa","pipeline","perturb16"):
            generator=Generator(settings)
        elif method.startswith("mh_"):
            from .multihorizon import MultiGenerator
            generator=MultiGenerator(settings)
        if method not in ("highs","gurobi"):
            import multiprocessing as mp
            ctx=mp.get_context("spawn")
            count=16 if method in ("pipeline","perturb16","mh_pipeline","mh_perturb16") else 1
            for _ in range(count):
                parent,child=ctx.Pipe(); proc=ctx.Process(target=repair_process,args=(child,dict(settings,threads=1 if count==16 else 16)))
                proc.start(); child.close(); pool.append((parent,proc))
            for conn,proc in pool:
                if not conn.poll(240) or not conn.recv().get("ready"): raise RuntimeError("Repair preload failed")
        if generator: torch.cuda.reset_peak_memory_stats()
        event("READY",metadata=dict(method=method,scenario_preloaded=False,repair_processes=len(pool)))
        req=json.loads(sys.stdin.readline()); deadline=req["deadline"]-settings["reserve_s"]
        stage={}; sent=False; result=dict(status="no_incumbent")
        try:
            start=time.perf_counter()
            if sha(req["file"])!=req["scenario_sha256"]: raise ValueError("Scenario changed")
            data=load_data(req["file"]); stage["scenario_read"]=time.perf_counter()-start
            if time.perf_counter()>=deadline: raise TimeoutError("No compute allowance")
            if method in ("highs","gurobi"):
                start=time.perf_counter(); seconds=settings.get("solver_seconds",req["budget"])
                if method=="highs": result=solve_data(data,threads=16,seconds=seconds,seed=req["seed"],deadline=deadline)
                else:
                    from .gurobi import solve
                    result=solve(data,seconds=seconds,seed=req["seed"],deadline=deadline,env=env)
                result["status"]="solution" if result.get("valid") else result["status"]
                for k in ("detail","lp","lp_duals"): result.pop(k,None)
                stage["direct_solve"]=time.perf_counter()-start
            else:
                start=time.perf_counter()
                if generator:
                    candidates,seeds=generator.candidates(data,req,deadline); stage.update(generator.stage_seconds)
                else:
                    # Direct conversion of the SAME realized data: no second loader/fallback.
                    candidates=rolling_horizon_heuristic(physics_from_data(data),HeuristicConfig()).numpy()[None]
                    seeds=[seed_for(req["key"],req["repetition"],0)]
                    stage["generation"]=time.perf_counter()-start
                if time.perf_counter()>=deadline: raise TimeoutError("Generation exhausted allowance")
                if len(candidates)!=len(pool): raise ValueError("Candidate/resource mismatch")
                start=time.perf_counter()
                for i,(conn,proc) in enumerate(pool):
                    conn.send(dict(file=req["file"],compute_deadline=deadline,candidate=candidates[i],index=i,seed=seeds[i]))
                sent=True; pending=set(range(len(pool))); completed=[]
                while pending and time.perf_counter()<deadline:
                    for i in list(pending):
                        try:
                            if pool[i][0].poll():
                                candidate=pool[i][0].recv()
                                if time.perf_counter()<deadline: completed.append(candidate)
                                pending.remove(i)
                        except (EOFError,BrokenPipeError):
                            completed.append(dict(status="repair_crash",validation={"valid":False})); pending.remove(i)
                    if pending: time.sleep(.001)
                result=select_candidates(completed)
                result.update(candidate_diagnostics=[{k:v for k,v in c.items() if k!="dispatch"} for c in completed],
                              generated_candidates=len(candidates),unfinished_candidates=len(pending),candidate_seeds=seeds)
                stage["repair_wait"]=time.perf_counter()-start
        except TimeoutError:
            result=dict(status="cooperative_deadline")
        except Exception as exc:
            traceback.print_exc()
            infrastructure=method=="gurobi" and isinstance(exc,gp.GurobiError)
            result=dict(status="infrastructure_error" if infrastructure else "worker_error",error=repr(exc))
        compute_end=time.perf_counter(); start=time.perf_counter()
        if not sent:
            for conn,proc in pool: conn.send(None)
        while any(proc.is_alive() for conn,proc in pool):
            for conn,proc in pool:
                try:
                    if conn.poll(): conn.recv()
                except (EOFError,BrokenPipeError): pass
            time.sleep(.001)
        for conn,proc in pool: proc.join(); conn.close()
        stage["child_closure"]=time.perf_counter()-start
        start=time.perf_counter()
        if result.get("dispatch"):
            result["validation"]=validate_dispatch(data,result["dispatch"])
            if not result["validation"]["valid"]: result["status"]="invalid_dispatch"
        stage["validation"]=time.perf_counter()-start
        if env is not None: env.dispose(); env=None
        if generator:
            torch.cuda.synchronize(); result["gpu_peak_bytes"]=torch.cuda.max_memory_allocated()
        result.update(identity=req["identity"],compute_end=compute_end,stage_seconds=stage,selected_method=method)
        atomic_json(req["result_file"],sealed(result),deadline=req["deadline"])
    except Exception as exc:
        event("ERROR",error=repr(exc)); raise
    finally:
        if env is not None: env.dispose()

if __name__=="__main__": main()
