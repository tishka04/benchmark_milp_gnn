"""GPU-batched candidates followed by concurrent CPU repairs, under one deadline.

This module does nothing on import. The PowerShell runner invokes it only in
steps 12/13. Each K uses one CUDA batch and K one-thread repair processes,
compared to a separate HiGHS solve allowed P threads. Cold process startup,
model loading, transfers and dispatch construction are timed; offline HTE
embedding generation and training are not. No time is divided by K, and the
deadline is never reset between generation and repair.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, fields
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

FORMULATION = "load_shifting_v1"
SCHEMA_VERSION = 2
PROTOCOL = "gpu_batch_cpu_repairs_v1"
DEFAULT_K_VALUES = (1, 2, 5, 8, 12, 16)
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_k_values(text, threads):
    values = sorted(set(int(x.strip()) for x in text.split(",") if x.strip()))
    if not values:
        values = [k for k in DEFAULT_K_VALUES if k <= threads]
    if threads < 1 or any(k < 1 or k > threads for k in values):
        raise ValueError("K must be between 1 and the explicitly configured HiGHS thread count")
    return values


def sample_decode_batch(sampler, ebm, ae, predictor, h_scenario, zone_mask,
                        k, thresholds, prior_mean=None, prior_std=None):
    """One JEPA prediction, K Langevin chains in one batch, one batched decode."""
    import torch
    from src.uc_jepa.latent_utils import apply_thresholds

    with torch.no_grad():
        anchor = predictor(h_scenario, zone_mask=zone_mask)["z_pred"]
    # Langevin needs gradients with respect to the latent, even in evaluation.
    latents, energies, _ = sampler.sample(
        ebm=ebm, z_anchor=anchor, h_scenario=h_scenario, zone_mask=zone_mask,
        n_samples=k, prior_mean=prior_mean, prior_std=prior_std,
    )
    with torch.no_grad():
        logits = ae.decode(
            latents[0], n_zones=h_scenario.shape[1], n_timesteps=h_scenario.shape[2],
            h_scenario=h_scenario.expand(k, -1, -1, -1),
        )
        if not torch.isfinite(logits).all() or not torch.isfinite(energies).all():
            raise ValueError("Non-finite GPU candidate logits or energies")
        candidates = apply_thresholds(torch.sigmoid(logits), thresholds)
    torch.cuda.synchronize()
    return candidates.detach().cpu().numpy(), energies[0].detach().cpu().numpy()


def generate_candidates_gpu(job):
    """Load one model stack on CUDA and export CPU arrays for the repair pool."""
    import numpy as np
    import torch
    from src.uc_jepa.data.uc_label_dataset import UCLabelDataset
    from src.uc_jepa.latent_utils import (
        load_frozen_ae, load_frozen_jepa, load_latent_ebm, seed_all, threshold_vector,
    )
    from src.uc_jepa.sampling.latent_langevin import LatentLangevinConfig, LatentLangevinSampler

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU-batch comparison; no CPU fallback")
    device = torch.device("cuda:0")
    seed_all(job["seed"])
    root = Path(job["root"]) / "uc_jepa"
    ebm, checkpoint = load_latent_ebm(str(root / "latent_ebm/latent_ebm_best.pt"), device)
    ae = load_frozen_ae(str(root / "autoencoder/uc_ae_best.pt"), device)
    predictor, jepa_cfg = load_frozen_jepa(str(root / "jepa/jepa_best.pt"), device)
    for parameter in ebm.parameters():
        parameter.requires_grad_(False)
    cfg = checkpoint.get("config", {}) or {}
    sampler_values = cfg.get("sampler", {}) or {}
    sampler_cfg = LatentLangevinConfig(**{
        f.name: sampler_values[f.name] for f in fields(LatentLangevinConfig)
        if f.name in sampler_values
    })
    sampler = LatentLangevinSampler(sampler_cfg)
    jepa_eval = jepa_cfg.get("eval", {}) or {}
    merged_thresholds = dict(jepa_eval.get("feature_thresholds", {}) or {}) | dict(
        checkpoint.get("thresholds", {}) or {})
    default_threshold = float((cfg.get("eval", {}) or {}).get(
        "threshold", jepa_eval.get("threshold", 0.5)))
    thresholds = threshold_vector(default_threshold, merged_thresholds, ae.n_features).to(device)
    prior_mean, prior_std = checkpoint.get("latent_prior_mean"), checkpoint.get("latent_prior_std")
    prior_mean = prior_mean.to(device) if isinstance(prior_mean, torch.Tensor) else None
    prior_std = prior_std.to(device) if isinstance(prior_std, torch.Tensor) else None
    sample = UCLabelDataset(roots=[job["label_file"]])[0]
    if sample.scenario_id != Path(job["scenario_file"]).stem:
        raise ValueError("Candidate label does not match the scenario")
    h = sample.h_scenario.unsqueeze(0).to(device)
    mask = torch.ones(1, sample.n_zones, device=device)
    torch.cuda.synchronize()
    if time.monotonic() >= job["deadline"]:
        raise TimeoutError("Model loading exhausted the common deadline")
    torch.cuda.reset_peak_memory_stats(device)
    batch_started = time.monotonic()
    candidates, energies = sample_decode_batch(
        sampler, ebm, ae, predictor, h, mask, job["k"], thresholds, prior_mean, prior_std,
    )
    batch_wall_s = time.monotonic() - batch_started
    if candidates.shape != (job["k"], sample.n_zones, sample.n_timesteps, ae.n_features):
        raise ValueError(f"Unexpected candidate batch shape: {candidates.shape}")
    if time.monotonic() >= job["deadline"]:
        raise TimeoutError("GPU generation exhausted the common deadline")
    folder = Path(job["artifacts"])
    folder.mkdir(parents=True)
    path = folder / "candidates.npz"
    temp = folder / "candidates.tmp"
    with temp.open("wb") as handle:
        np.savez(handle, candidates=candidates, energies=energies,
                 dr_formulation=FORMULATION, scenario_id=sample.scenario_id)
    temp.replace(path)
    return {
        "ready": True, "feasible": False, "objective": float("inf"), "status": "candidates_ready",
        "candidate_file": str(path), "n_candidates": job["k"],
        "gpu_batch_wall_s": batch_wall_s,  # synchronized compute + CPU transfer
        "gpu_device": torch.cuda.get_device_name(device), "cuda_version": torch.version.cuda,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "sampler": asdict(sampler_cfg), "thresholds": thresholds.detach().cpu().tolist(),
    }


def repair_candidate_cpu(job):
    import numpy as np
    import torch
    from src.demand_response import require_current_formulation
    from src.milp.lp_worker_two_stage import LPWorkerTwoStage
    from src.uc_jepa.workers.milp_warm_worker import _serialize_lp_result

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with np.load(job["candidate_file"], allow_pickle=False) as data:
        require_current_formulation(data, job["candidate_file"])
        if str(data["scenario_id"].item()) != Path(job["scenario_file"]).stem:
            raise ValueError("Candidate batch does not match the scenario")
        decoder = torch.from_numpy(data["candidates"][job["candidate_idx"], :, :, :7].copy())
        energy = float(data["energies"][job["candidate_idx"]])
    lp_worker = LPWorkerTwoStage(job["scenario_root"], solver_name="appsi_highs", verbose=False)
    options = getattr(lp_worker.solver, "highs_options", None)
    if options is None:
        options = lp_worker.solver.options
    options["random_seed"] = job["seed"]
    remaining = job["deadline"] - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Repair startup exhausted the common deadline")
    lp_result = lp_worker.solve(
        scenario_id=Path(job["scenario_file"]).stem, decoder_output=decoder,
        max_stages=3, candidate_time_budget_s=remaining,
    )
    if lp_result.status == "error":
        raise RuntimeError(lp_result.message)
    payload = _serialize_lp_result(lp_result)
    objective = float(payload["lp_objective"])
    return {
        "feasible": math.isfinite(objective), "objective": objective, "status": payload["status"],
        "candidate_idx": job["candidate_idx"], "energy": energy,
        "pipeline_result": payload, "solver_resources": lp_worker.solver_resources,
    }


def worker(job_file):
    """A fresh, isolated process. Never spawns another solver process."""
    job = json.loads(Path(job_file).read_text(encoding="utf-8"))
    cpu_start = time.process_time()
    try:
        # Imports, model loading and scenario construction are inside the budget.
        import pandas  # noqa: F401 -- avoid the local pyarrow/torch DLL conflict
        if job["kind"] == "gpu_generate":
            result = generate_candidates_gpu(job)
        elif job["kind"] == "repair":
            result = repair_candidate_cpu(job)
        elif job["kind"] == "milp":
            from pyomo.environ import Constraint, SolverFactory, Var, value
            from src.milp.model import build_uc_model
            from src.milp.scenario_loader import load_scenario_data
            from src.milp.solver_resources import configure_highs_threads

            model = build_uc_model(load_scenario_data(Path(job["scenario_file"])), enable_duals=False)
            solver = SolverFactory("appsi_highs")
            resources = configure_highs_threads(solver, "appsi_highs")
            remaining = job["deadline"] - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Model construction exhausted the common deadline")
            solver.highs_options["time_limit"] = remaining
            solver.highs_options["random_seed"] = job["seed"]
            solved = solver.solve(model, tee=False)
            objective = float(value(model.obj))
            # Validate the loaded incumbent, including integer and DR constraints.
            violation = 0.0
            for constraint in model.component_data_objects(Constraint, active=True):
                body = float(value(constraint.body))
                if not math.isfinite(body):
                    raise ValueError("Non-finite constraint value in the loaded incumbent")
                if constraint.has_lb():
                    violation = max(violation, float(value(constraint.lower)) - body)
                if constraint.has_ub():
                    violation = max(violation, body - float(value(constraint.upper)))
            for variable in model.component_data_objects(Var, active=True):
                if variable.value is None:
                    continue  # unused variables need not appear in the loaded solution
                v = float(variable.value)
                if not math.isfinite(v):
                    raise ValueError("Non-finite variable in the loaded incumbent")
                if variable.lb is not None:
                    violation = max(violation, float(value(variable.lb)) - v)
                if variable.ub is not None:
                    violation = max(violation, v - float(value(variable.ub)))
                if variable.is_integer():
                    violation = max(violation, abs(v - round(v)))
            result = {
                "feasible": math.isfinite(objective) and violation <= 1e-5,
                "objective": objective, "status": str(solved.solver.termination_condition),
                "max_constraint_violation": violation, "solver_resources": resources,
            }
        else:
            raise ValueError(f"Unknown worker kind: {job['kind']}")
    except TimeoutError as exc:
        result = {"feasible": False, "objective": float("inf"), "status": "deadline", "message": str(exc)}
    except Exception as exc:
        message = str(exc)
        # An unavailable incumbent is a benchmark outcome; programming and I/O
        # errors must stop the run instead of masquerading as infeasibility.
        no_incumbent = any(x in message.lower() for x in (
            "feasible solution was not found", "no solution can be loaded"))
        result = {
            "feasible": False, "objective": float("inf"),
            "status": "no_incumbent" if no_incumbent else "worker_error",
            "message": f"{type(exc).__name__}: {exc}",
        }
    result.update(
        dr_formulation=FORMULATION, cpu_seconds=time.process_time() - cpu_start,
        seed=job["seed"], configured_highs_threads=int(os.environ["BENCHMARK_HIGHS_THREADS"]),
    )
    atomic_json(job["result_file"], result)


def run_group(jobs, directory, budget, *, started=None, popen=None, clock=time.monotonic):
    """Launch all members before waiting; collect only results before deadline.

    The injectable process/clock interfaces allow solver-free regression tests.
    Partial groups are never reused as completed groups after a crash.
    """
    popen = subprocess.Popen if popen is None else popen
    phase_started = clock()
    started = phase_started if started is None else started
    directory = Path(directory) / f"attempt-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    deadline = started + budget
    active, records, logs = {}, {}, []
    try:
        for index, job in enumerate(jobs):
            if clock() >= deadline:
                records[index] = {"feasible": False, "objective": float("inf"), "status": "deadline_before_launch"}
                continue
            folder = directory / str(index)
            folder.mkdir()
            job = dict(job, deadline=deadline, result_file=str(folder / "result.json"),
                       artifacts=str(folder / "artifacts"))
            atomic_json(folder / "job.json", job)
            environment = os.environ.copy()
            threads = job["threads"]
            environment.update({key: "1" for key in THREAD_ENV})
            environment["BENCHMARK_HIGHS_THREADS"] = str(threads)
            environment["PYTHONHASHSEED"] = str(job["seed"])
            environment["PYTHONIOENCODING"] = "utf-8"
            environment["PYTHONUNBUFFERED"] = "1"
            if job.get("kind") != "gpu_generate":
                environment["CUDA_VISIBLE_DEVICES"] = "-1"
            log = (folder / "worker.log").open("w", encoding="utf-8")
            logs.append(log)
            process = popen(
                [sys.executable, "-u", "-B", "-m", "scripts.benchmark_parallel_best_of_k",
                 "--worker", str(folder / "job.json")],
                stdout=log, stderr=subprocess.STDOUT, env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            active[index] = (process, folder)
        while active and clock() < deadline:
            for index, (process, folder) in list(active.items()):
                if process.poll() is None:
                    continue
                observed = clock()
                if observed > deadline:
                    break
                result_path = folder / "result.json"
                if process.returncode != 0 or not result_path.exists():
                    raise RuntimeError(f"Worker failed; inspect {folder / 'worker.log'}")
                result = json.loads(result_path.read_text(encoding="utf-8"))
                observed = clock()
                if observed > deadline:
                    break
                if result["status"] == "worker_error":
                    raise RuntimeError(f"{result.get('message')}; inspect {folder / 'worker.log'}")
                result["available_after_s"] = observed - started
                records[index] = result
                del active[index]
            if active:
                time.sleep(min(0.02, max(0.0, deadline - clock())))
        decision_wall_s = min(clock() - started, budget)
        for index in active:
            records[index] = {"feasible": False, "objective": float("inf"), "status": "hard_timeout"}
    finally:
        # There are no worker subprocesses below these processes; HiGHS runs
        # in-process. Terminating the process also terminates its native threads.
        for process, _ in active.values():
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
        for log in logs:
            log.close()
    ordered = [records[i] for i in range(len(jobs))]
    valid = [r for r in ordered if r["feasible"] and math.isfinite(r["objective"])]
    best = min(valid, key=lambda r: r["objective"]) if valid else None
    return {
        "complete": True, "feasible": best is not None,
        "objective": best["objective"] if best else float("inf"),
        "best_candidate_idx": best.get("candidate_idx") if best else None,
        "decision_wall_s": decision_wall_s, "cleanup_wall_s": max(0.0, clock() - started - decision_wall_s),
        "phase_wall_s": max(0.0, started + decision_wall_s - phase_started),
        "first_feasible_s": min((r["available_after_s"] for r in valid), default=None),
        "cpu_seconds_completed_members": sum(r.get("cpu_seconds", 0.0) for r in ordered),
        "cpu_accounting_complete": all("cpu_seconds" in r for r in ordered),
        "members": ordered,
    }


def run_gpu_portfolio(common, k, seed, directory, budget, *, popen=None, clock=time.monotonic):
    """Two phases, one absolute deadline; K repairs each get all remaining time."""
    started = clock()
    directory = Path(directory)
    generation = run_group(
        [dict(common, kind="gpu_generate", k=k, threads=1, seed=seed)],
        directory / "generation", budget, started=started, popen=popen, clock=clock,
    )
    gpu = generation["members"][0]
    ready = bool(gpu.get("ready"))
    if ready:
        if gpu["n_candidates"] != k:
            raise ValueError("GPU worker returned an incomplete candidate batch")
        jobs = [dict(common, kind="repair", threads=1, seed=seed + i,
                     candidate_file=gpu["candidate_file"], candidate_idx=i)
                for i in range(k)]
        result = run_group(jobs, directory / "repairs", budget,
                           started=started, popen=popen, clock=clock)
        repair_wall_s = result["phase_wall_s"]
        result["cpu_seconds_completed_members"] += generation["cpu_seconds_completed_members"]
        result["cpu_accounting_complete"] &= generation["cpu_accounting_complete"]
        result["cleanup_wall_s"] += generation["cleanup_wall_s"]
    else:
        result = dict(generation)
        result["members"] = [
            {"feasible": False, "objective": float("inf"), "candidate_idx": i,
             "status": "gpu_generation_deadline"} for i in range(k)
        ]
        repair_wall_s = 0.0
    result.update(
        protocol=PROTOCOL, n_candidates_generated=k if ready else 0,
        n_repairs_completed=sum("cpu_seconds" in m for m in result["members"]),
        generation=gpu, generation_wall_s=generation["phase_wall_s"],
        repair_wall_s=repair_wall_s,
    )
    return result


def make_manifest(root, families, threads, budget, k_values):
    root = Path(root).resolve()
    cohort, inputs = [], {}
    for family in families:
        scenarios = sorted((root / family).glob("scenario_*.json"))
        if not scenarios:
            raise ValueError(f"No scenarios for {family}")
        for scenario in scenarios:
            label = root / "uc_jepa/dataset_eval" / family / f"{scenario.stem}_uc.npz"
            report = root / family / "reports" / scenario.name
            for path in (scenario, label, report):
                inputs[str(path.relative_to(root))] = sha256(path)
            cohort.append({"family": family, "scenario_id": scenario.stem})
    for name in ("autoencoder/uc_ae_best.pt", "jepa/jepa_best.pt", "latent_ebm/latent_ebm_best.pt"):
        inputs[f"uc_jepa/{name}"] = sha256(root / "uc_jepa" / name)
    sources = {str(p): sha256(p) for p in sorted(Path("src").rglob("*.py"))}
    sources["scripts/benchmark_parallel_best_of_k.py"] = sha256(__file__)
    return {
        "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL, "dr_formulation": FORMULATION,
        "root": str(root), "families": families, "cohort": cohort, "inputs_sha256": inputs,
        "source_sha256": sources, "highs_threads": threads, "k_values": k_values,
        "budget_seconds": budget, "repair_threads_per_candidate": 1,
        "pipeline_device": "cuda:0", "repair_device": "cpu",
        "selection": "lowest feasible dispatch objective",
        "timing": "one deadline for cold CUDA batch generation, transfers and CPU repairs; offline HTE embeddings excluded",
        "seeding": "scenario seed for each GPU batch; independent noise within batch; not nested prefixes across K",
        "cpu_time": "completed workers only; killed workers flagged as incomplete",
        "python": sys.version, "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "processor": platform.processor(), "machine": platform.machine(),
        "versions": {p: importlib.metadata.version(p) for p in ("highspy", "pyomo", "torch", "numpy")},
    }


def run_benchmark(args):
    root, out = Path(args.root).resolve(), Path(args.output_dir).resolve()
    k_values = parse_k_values(args.k_values, args.threads)
    manifest = make_manifest(root, args.families, args.threads, args.budget, k_values)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != manifest or not args.resume:
            raise ValueError("Existing benchmark differs or --resume is missing; use a fresh output directory")
    atomic_json(manifest_path, manifest)
    for item in manifest["cohort"]:
        family, sid = item["family"], item["scenario_id"]
        record_path = out / "cases" / family / f"{sid}.json"
        record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {
            **item, "dr_formulation": FORMULATION, "runs": {},
        }
        report = json.loads((root / family / "reports" / f"{sid}.json").read_text(encoding="utf-8"))
        if report.get("dr_formulation", (report.get("detail") or {}).get("dr_formulation")) != FORMULATION:
            raise ValueError(f"Legacy MILP reference: {family}/{sid}")
        record["reference_objective"] = float(report["mip"]["objective"])
        record["reference_status"] = report["mip"]["termination"]
        base_seed = int(hashlib.sha256(f"{family}/{sid}".encode()).hexdigest()[:8], 16) % 1_000_000_000
        common = {
            "root": str(root), "scenario_file": str(root / family / f"{sid}.json"),
            "scenario_root": str(root / family),
            "label_file": str(root / "uc_jepa/dataset_eval" / family / f"{sid}_uc.npz"),
        }
        if "milp" not in record:
            print(f"{family}/{sid}: HiGHS, {args.threads} threads, {args.budget}s", flush=True)
            record["milp"] = run_group(
                [dict(common, kind="milp", threads=args.threads, seed=base_seed)],
                out / "jobs" / family / sid / "milp", args.budget,
            )
            atomic_json(record_path, record)
        for k in k_values:
            if str(k) in record["runs"]:
                continue
            print(f"{family}/{sid}: CUDA batch K={k}, then {k} concurrent CPU repairs, {args.budget}s", flush=True)
            record["runs"][str(k)] = run_gpu_portfolio(
                common, k, base_seed, out / "jobs" / family / sid / f"k{k}", args.budget,
            )
            atomic_json(record_path, record)


def gap_percent(objective, reference):
    if not math.isfinite(objective) or not math.isfinite(reference) or abs(reference) < 1e-9:
        return None
    return 100 * (objective - reference) / abs(reference)


def comparison_rows(records, k_values):
    rows = []
    for record in records:
        baseline = record["milp"]
        common = baseline["feasible"] and all(record["runs"][str(k)]["feasible"] for k in k_values)
        for k in k_values:
            result = record["runs"][str(k)]
            paired = bool(result["feasible"] and baseline["feasible"])
            rows.append({
                "family": record["family"], "scenario_id": record["scenario_id"], "k": k,
                "feasible": result["feasible"], "milp_feasible": baseline["feasible"],
                "paired_feasible": paired, "common_all_k_feasible": common,
                "best_objective": result["objective"], "milp_budget_objective": baseline["objective"],
                "reference_objective": record["reference_objective"], "reference_status": record["reference_status"],
                "gap_vs_milp_budget_pct": gap_percent(result["objective"], baseline["objective"]) if paired else None,
                "cost_difference_vs_milp": result["objective"] - baseline["objective"] if paired else None,
                "gap_vs_reference_pct": gap_percent(result["objective"], record["reference_objective"]) if result["feasible"] else None,
                "wall_seconds": result["decision_wall_s"], "milp_wall_seconds": baseline["decision_wall_s"],
                "first_feasible_s": result["first_feasible_s"], "cleanup_wall_s": result["cleanup_wall_s"],
                "cpu_seconds_completed_members": result["cpu_seconds_completed_members"],
                "cpu_accounting_complete": result["cpu_accounting_complete"],
                "generation_wall_s": result.get("generation_wall_s"),
                "gpu_batch_wall_s": result.get("generation", {}).get("gpu_batch_wall_s"),
                "repair_wall_s": result.get("repair_wall_s"),
                "gpu_device": result.get("generation", {}).get("gpu_device"),
                "gpu_peak_allocated_bytes": result.get("generation", {}).get("gpu_peak_allocated_bytes"),
                "n_candidates_generated": result.get("n_candidates_generated"),
                "n_repairs_completed": result.get("n_repairs_completed"),
                "best_candidate_idx": result.get("best_candidate_idx"),
            })
    return rows


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(out):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("protocol") != PROTOCOL:
        raise ValueError("Incompatible comparison protocol; regenerate GPU-batch results at step 12")
    records = [json.loads((out / "cases" / x["family"] / f"{x['scenario_id']}.json").read_text(encoding="utf-8"))
               for x in manifest["cohort"]]
    rows = comparison_rows(records, manifest["k_values"])  # missing K is an error
    summaries = []
    for family in ["ALL"] + manifest["families"]:
        for k in manifest["k_values"]:
            group = [r for r in rows if r["k"] == k and (family == "ALL" or r["family"] == family)]
            if not group:
                continue
            paired = [r["gap_vs_milp_budget_pct"] for r in group if r["gap_vs_milp_budget_pct"] is not None]
            common = [r["gap_vs_milp_budget_pct"] for r in group
                      if r["common_all_k_feasible"] and r["gap_vs_milp_budget_pct"] is not None]
            summaries.append({
                "family": family, "k": k, "n_scenarios": len(group),
                "pipeline_feasibility_rate": statistics.mean(int(r["feasible"]) for r in group),
                "milp_feasibility_rate": statistics.mean(int(r["milp_feasible"]) for r in group),
                "n_paired_feasible": sum(r["paired_feasible"] for r in group),
                "n_paired_with_defined_percent_gap": len(paired),
                "median_gap_vs_milp_budget_pct_paired": statistics.median(paired) if paired else None,
                "n_common_all_k_feasible": sum(r["common_all_k_feasible"] for r in group),
                "n_common_with_defined_percent_gap": len(common),
                "median_gap_vs_milp_budget_pct_common_all_k": statistics.median(common) if common else None,
                "median_wall_seconds": statistics.median(r["wall_seconds"] for r in group),
                "median_milp_wall_seconds": statistics.median(r["milp_wall_seconds"] for r in group),
                "median_generation_wall_s": statistics.median(
                    r["generation_wall_s"] for r in group if r["generation_wall_s"] is not None
                ) if any(r["generation_wall_s"] is not None for r in group) else None,
                "median_repair_wall_s": statistics.median(
                    r["repair_wall_s"] for r in group if r["repair_wall_s"] is not None
                ) if any(r["repair_wall_s"] is not None for r in group) else None,
                "highs_threads": manifest["highs_threads"], "budget_seconds": manifest["budget_seconds"],
            })
    write_csv(out / "best_of_k_parallel_per_scenario.csv", rows)
    write_csv(out / "best_of_k_parallel_summary.csv", summaries)
    atomic_json(out / "best_of_k_parallel_summary.json", summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument("--output-dir")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--budget", type=float, default=120)
    parser.add_argument("--k-values", default="",
                        help="Comma-separated K values; default: 1,2,5,8,12,16 within --threads")
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.analyze_only:
        if not args.output_dir:
            parser.error("--output-dir is required")
        analyze(args.output_dir)
    else:
        if not args.root or not args.output_dir or not args.families or not args.threads:
            parser.error("--root, --output-dir, --families and positive --threads are required")
        if not math.isfinite(args.budget) or args.budget <= 0:
            parser.error("--budget must be positive and finite")
        run_benchmark(args)


if __name__ == "__main__":
    main()
