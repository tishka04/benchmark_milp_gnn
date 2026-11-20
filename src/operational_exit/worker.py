"""One experiment per process tree. No scenario is available before READY."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import traceback


def select_candidates(completed):
    valid = [r for r in completed if r.get("validation", {}).get("valid")]
    if valid:
        return min(valid, key=lambda r: (r["validation"]["cost"], r.get("candidate", 0)))
    for reason in ("repair_error", "repair_crash", "invalid_dispatch", "cooperative_deadline"):
        if any(r.get("status") == reason for r in completed):
            return dict(status=reason)
    return dict(status="no_incumbent")


def repair_process(conn, settings):
    # Import order avoids Windows Arrow/Torch DLL conflicts.
    import pandas  # noqa: F401
    import torch
    from .solving import make_repair_worker
    from .physics import validate_dispatch
    from src.milp.scenario_loader import load_scenario_data
    torch.set_num_threads(1)
    os.environ["BENCHMARK_HIGHS_THREADS"] = str(settings["threads"])
    try:
        worker = make_repair_worker(settings["repair"])
        conn.send({"ready": True})
        job = conn.recv()
        if job is None:
            return
        worker.scenarios_dir = str(Path(job["file"]).parent)
        worker.deadline = job["compute_deadline"]
        options = getattr(worker.solver, "highs_options", worker.solver.options)
        options["random_seed"] = job["seed"]
        data = load_scenario_data(Path(job["file"]))
        worker.dt_hours = data.dt_hours
        result = worker.solve(Path(job["file"]).stem, torch.from_numpy(job["candidate"][:, :, :7]),
                              max_stages=3, candidate_time_budget_s=max(.001, worker.deadline - time.perf_counter()))
        dispatch = (result.continuous_vars or {}).get("operational_dispatch")
        check = validate_dispatch(data, dispatch) if dispatch else {"valid": False}
        status = "solution" if check["valid"] else "invalid_dispatch" if dispatch else "no_incumbent"
        if not dispatch and result.status == "error":
            status = "cooperative_deadline" if "deadline" in result.message else "repair_error"
        conn.send(dict(status=status,
                       dispatch=dispatch, validation=check, repair_status=result.status,
                       repair_stage=str(result.stage_used), repair_message=result.message, candidate=job["index"]))
    except TimeoutError:
        conn.send(dict(status="cooperative_deadline", validation={"valid": False}))
    except Exception as exc:
        conn.send(dict(status="repair_error", error=repr(exc), validation={"valid": False}))
    finally:
        conn.close()


def main():
    control = sys.stdout
    def event(name, **kw):
        control.write(json.dumps(dict(event=name, **kw)) + "\n")
        control.flush()
    event("BOOT", pid=os.getpid())
    from .supervisor import enroll_current_process
    enroll_current_process(json.loads(sys.stdin.readline())["job_name"])
    event("ENROLLED", pid=os.getpid())
    settings = json.loads(sys.stdin.readline())
    if "source_hashes" in settings:
        from .protocol import source_signature
        from .common import sha
        if source_signature() != settings["source_hashes"]:
            raise ValueError("Code/configuration changed during campaign")
        if any(sha(settings[name]) != expected for name, expected in settings["model_hashes"].items()):
            raise ValueError("Model changed after protocol freeze")
    import pandas  # noqa: F401
    import numpy as np
    import torch
    import highspy  # noqa: F401 — native library preloaded without a scenario.
    from pyomo.contrib.appsi.solvers.highs import Highs  # noqa: F401
    from src.milp.model import build_uc_model  # noqa: F401
    from .common import atomic_json, sealed, sha, seed_for
    from .physics import validate_dispatch
    from .solving import solve_data
    from src.milp.scenario_loader import load_scenario_data
    from src.heuristics.rolling_horizon import rolling_horizon_heuristic, HeuristicConfig
    from src.ebm.feasibility import load_physics_from_scenario
    torch.set_num_threads(1)
    method = settings["method"]
    pool = []
    generator = None
    try:
        if method in ("pipeline", "jepa"):
            from .generation import Generator
            generator = Generator(settings)
            torch.cuda.reset_peak_memory_stats()
        if method != "highs":
            import multiprocessing as mp
            context = mp.get_context("spawn")
            count = 16 if method == "pipeline" else 1
            for _ in range(count):
                parent, child = context.Pipe()
                process = context.Process(target=repair_process, args=(child, dict(settings, threads=1 if count == 16 else 16)))
                process.start()
                child.close()
                pool.append((parent, process))
            for conn, process in pool:
                if not conn.poll(240) or not conn.recv().get("ready"):
                    raise RuntimeError("Repair preload failed")
        event("READY", metadata=dict(method=method, repair_processes=len(pool), scenario_preloaded=False,
                                     gpu=torch.cuda.get_device_name(0) if generator else None))
        request = json.loads(sys.stdin.readline())
        compute_deadline = request["deadline"] - settings["reserve_s"]
        result = dict(status="no_incumbent")
        sent = False
        try:
            if sha(request["file"]) != request["scenario_sha256"]:
                raise ValueError("Scenario hash mismatch")
            data = load_scenario_data(Path(request["file"]))
            if time.perf_counter() >= compute_deadline:
                raise TimeoutError("no compute allowance after finalization reserve")
            if method == "highs":
                result = solve_data(data, threads=16, seconds=request["budget"], seed=request["seed"], deadline=compute_deadline)
                result["status"] = "solution" if result.get("valid") else result["status"]
                for k in ("detail", "lp", "lp_duals"):
                    result.pop(k, None)
            else:
                if generator:
                    candidates, seeds = generator.candidates(data, request, compute_deadline)
                else:
                    physics = load_physics_from_scenario(Path(request["file"]).stem, str(Path(request["file"]).parent))
                    candidates = rolling_horizon_heuristic(physics, HeuristicConfig()).numpy()[None]
                    seeds = [seed_for(request["key"], request["repetition"], 0)]
                if time.perf_counter() >= compute_deadline:
                    raise TimeoutError("generation/preparation exhausted compute time")
                for i, (conn, process) in enumerate(pool):
                    conn.send(dict(file=request["file"], compute_deadline=compute_deadline, candidate=candidates[i], index=i, seed=seeds[i]))
                sent = True
                pending = set(range(len(pool)))
                completed = []
                while pending and time.perf_counter() < compute_deadline:
                    for i in list(pending):
                        conn, process = pool[i]
                        try:
                            if conn.poll():
                                completed.append(conn.recv())
                                pending.remove(i)
                        except (EOFError, BrokenPipeError):
                            completed.append(dict(status="repair_crash", validation={"valid": False}))
                            pending.remove(i)
                    if pending:
                        time.sleep(.001)
                result = select_candidates(completed)
                result["candidate_statuses"] = [r["status"] for r in completed]
                result["candidate_diagnostics"] = [{k: v for k, v in r.items() if k != "dispatch"}
                                                    for r in completed]
                result["unfinished_candidates"] = len(pending)
                result["candidate_seeds"] = seeds
                result["generated_candidates"] = len(candidates)
        except TimeoutError:
            result = dict(status="cooperative_deadline")
        except Exception as exc:
            traceback.print_exc()
            result = dict(status="worker_error", error=repr(exc))
        compute_end = time.perf_counter()
        if not sent:
            for conn, process in pool:
                conn.send(None)
        # Cooperative solver limits expire before the global deadline. All
        # children are joined naturally. A blocked child makes the run fail.
        while any(process.is_alive() for conn, process in pool):
            # Drain unfinished candidate messages solely to release blocked
            # pipe writers. They can never replace the selected candidate.
            for conn, process in pool:
                try:
                    if conn.poll():
                        conn.recv()
                except (EOFError, BrokenPipeError):
                    pass
            time.sleep(.001)
        for conn, process in pool:
            process.join()
            conn.close()
        if result.get("dispatch"):
            result["validation"] = validate_dispatch(data, result["dispatch"])
            if not result["validation"]["valid"]:
                result["status"] = "invalid_dispatch"
        if generator:
            torch.cuda.synchronize()
            result["gpu_peak_bytes"] = torch.cuda.max_memory_allocated()
        result.update(identity=request["identity"], compute_end=compute_end)
        atomic_json(request["result_file"], sealed(result), deadline=request["deadline"])
    except Exception as exc:
        traceback.print_exc()
        event("ERROR", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
