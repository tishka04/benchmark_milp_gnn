"""Isolated MILP / GPU-batch pipeline / JEPA comparison; no work on import.

The existing campaign never imports this module. Each budget is an independent
cold-process experiment. The parent alone starts/stops all workers and accepts
only outputs received and selected before a common absolute deadline.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from scripts.benchmark_parallel_best_of_k import (
    FORMULATION, THREAD_ENV, atomic_json, durable_checkpoint, make_manifest, sha256,
)
from scripts.feasible_candidate_selection import corrected_selection, finite, select_sample_index
from scripts.run_strict_standard_inference import fresh_attempt

PROTOCOL = "matched_budgets_gpu16_feasible_energy_v1"
DEFAULT_BUDGETS = (1, 2, 5, 10, 60, 120)
METHODS = ("milp", "pipeline", "jepa")
FAMILIES = (
    "low_criticality_scenarios", "medium_criticality_scenarios", "high_criticality_scenarios",
    "persona_vre_battery_scenarios", "persona_network_operator_scenarios", "persona_mathematician_scenarios",
)
NEW_SOURCES = (
    "scripts/benchmark_matched_budgets.py", "scripts/feasible_candidate_selection.py",
    "scripts/matched_budget_milp.py", "scripts/run_strict_standard_inference.py",
)
# Audited changes to parent checkpoint/resume I/O only. The worker, reducer,
# seeding, deadline and solver implementations are unchanged by these fixes.
RUNNER_SOURCE = "scripts/benchmark_matched_budgets.py"
COMPATIBLE_RUNNER_HASHES = {
    "b0ea206438abcd84a37ddc96fd9d3d5962bd793fb0226461d3a91019ec32dd26",
}
COMPATIBLE_SOURCE_TRANSITIONS = {
    "scripts/benchmark_parallel_best_of_k.py": {
        ("c87f022780249427c9a5ff8ab2517284c98f8f5927bcdf09f79e74e55a49ed16",
         "23f408b831ce9b22cf0004db7c481e988060f26242c78e7f60b1a33cb16bc5c1"),
    },
}


def parse_budgets(text):
    budgets = sorted(set(float(v.strip()) for v in text.split(",") if v.strip()))
    if not budgets or any(not math.isfinite(v) or v <= 0 for v in budgets):
        raise ValueError("Budgets must be positive finite seconds")
    return budgets


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=True).encode()).hexdigest()


def validate_resume_manifest(previous, current):
    """Allow audited orchestration fixes; retain all scientific provenance checks."""
    old, new = dict(previous), dict(current)
    old_sources = old.pop("source_sha256", {})
    new_sources = new.pop("source_sha256", {})
    incompatible = [name for name in sorted(old.keys() | new.keys()) if old.get(name) != new.get(name)]
    compatible_changes = []
    for source in sorted(old_sources.keys() | new_sources.keys()):
        before, after = old_sources.get(source), new_sources.get(source)
        if before == after:
            continue
        known_runner_fix = (source == RUNNER_SOURCE and before in COMPATIBLE_RUNNER_HASHES
                            and after == sha256(__file__))
        if (known_runner_fix or
                (before, after) in COMPATIBLE_SOURCE_TRANSITIONS.get(source, set())):
            compatible_changes.append(source)
        else:
            incompatible.append(f"source_sha256[{source}]")
    if incompatible:
        raise ValueError("Manifest mismatch: " + ", ".join(incompatible) +
                         ". Restore the original inputs/settings or use a separate output directory.")
    return compatible_changes


def save_parent_json(path, payload):
    """Parent checkpoints are outside the decision budget; fsync before publish."""
    durable_checkpoint(path, json.dumps(payload, indent=2, allow_nan=True).encode("utf-8"))


@contextmanager
def campaign_lock(root):
    """Share the legacy campaign lock; never run benchmarks concurrently with it."""
    path = Path(root) / "_pipeline/run.lock"
    if not path.parent.is_dir():
        raise ValueError("Missing prepared campaign inputs")
    handle = None
    locked = False
    try:
        try:
            handle = path.open("a+b")
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise RuntimeError("Another campaign holds run.lock. Wait for it to finish; no worker started.") from exc
        yield
    finally:
        if handle is not None:
            if locked and os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            handle.close()


def eligible(row):
    if not row or not row.get("success") or not finite(row.get("lp_objective")):
        return False
    if row.get("kind") == "milp":
        return finite(row.get("max_constraint_violation")) and row["max_constraint_violation"] <= 1e-3
    return bool((row.get("dr_metrics") or {}).get("dr_feasible"))


def select_repairs(rows):
    """Pure reducer, executed by the parent inside the common deadline."""
    if not rows:
        raise ValueError("At least one candidate slot is required")
    costs = [r["lp_objective"] if eligible(r) else math.inf for r in rows]
    energies = [r.get("energy", math.nan) if r else math.nan for r in rows]
    index = select_sample_index("energy", energies, costs)
    selected = dict(rows[index]) if rows and eligible(rows[index]) else {
        "success": False, "lp_objective": math.inf, "status": "no_feasible_candidate", "dr_metrics": {},
    }
    selected.update(best_sample_idx=index if finite(costs[index]) else None,
                    selection="energy_among_feasible", all_objectives=costs,
                    all_sample_ebm_energies=energies,
                    all_sample_statuses=[r.get("status") if r else "pending" for r in rows],
                    all_sample_slacks=[r.get("slack_mwh") if r else None for r in rows],
                    n_repairs_completed=sum(r is not None for r in rows))
    return selected


def run_phase(jobs, directory, started, deadline, *, reducer=None,
              popen=None, clock=None, sleep=None):
    """No nested children. Retry transient file locks without moving the deadline."""
    popen, clock, sleep = popen or subprocess.Popen, clock or time.monotonic, sleep or time.sleep
    directory = Path(directory)
    active, processes, logs = {}, [], []
    members = [{"row": None, "received_after_s": None, "completed": False} for _ in jobs]
    denied, retry_count = {}, 0
    aggregate, aggregate_at = None, None
    phase_started = clock()

    def read_json(path):
        nonlocal retry_count
        try:
            text = path.read_text(encoding="utf-8")
        except PermissionError as exc:
            now = clock()
            first = denied.setdefault(path, now)
            retry_count += 1
            if now - first >= 2:
                raise PermissionError(f"Worker output locked for 2s: {path}") from exc
            return None
        denied.pop(path, None)
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid worker JSON: {path}")
        return payload

    def snapshots(index, folder, job):
        nonlocal aggregate, aggregate_at
        paths = sorted(folder.glob("snapshot-*.json"), reverse=True)
        for rank, path in enumerate(paths):
            if clock() >= deadline:
                return False
            if path.stem <= members[index].get("snapshot", ""):
                return rank == 0
            payload = read_json(path)
            if payload is None:
                continue
            if clock() >= deadline:
                return False
            if (payload.get("method") != job["method"] or payload.get("kind") != job["kind"]
                    or path.stem != f"snapshot-{payload.get('sequence', 0):06d}"):
                raise ValueError("Snapshot method/kind/sequence mismatch")
            row = payload["row"]
            if (row.get("scenario_id") != job["scenario_id"] or row.get("family") != job["family_alias"]
                    or row.get("dr_formulation") != FORMULATION or row.get("kind") != job["kind"]):
                raise ValueError("Snapshot scenario/family/formulation mismatch")
            candidate = None
            if reducer:
                rows = [m["row"] for m in members]
                rows[index] = row
                candidate = reducer(rows)
            now = clock()  # selection, JSON reading and validation are all timed
            if now >= deadline:
                return False
            members[index].update(row=row, received_after_s=now - started, snapshot=path.stem)
            if reducer:
                aggregate, aggregate_at = candidate, now - started
            return rank == 0
        return not paths

    try:
        for index, source_job in enumerate(jobs):
            if clock() >= deadline:
                break
            folder = directory / f"{index:02d}"
            folder.mkdir(parents=True, exist_ok=False)
            job = dict(source_job, deadline=deadline, output_dir=str(folder / "a"),
                       snapshot_file=str(folder / "snapshot.json"), done_file=str(folder / "done.json"),
                       artifacts=str(folder / "a"))
            atomic_json(folder / "job.json", job)
            env = dict(os.environ, **{name: "1" for name in THREAD_ENV},
                       BENCHMARK_HIGHS_THREADS=str(job["threads"]), PYTHONUNBUFFERED="1",
                       PYTHONHASHSEED=str(job["seed"]), PYTHONIOENCODING="utf-8")
            if job["kind"] in ("milp", "repair"):
                env["CUDA_VISIBLE_DEVICES"] = "-1"
            if clock() >= deadline:
                break
            log = (folder / "worker.log").open("wb")
            logs.append(log)
            process = popen([sys.executable, "-u", "-B", "-m", "scripts.benchmark_matched_budgets",
                             "--worker", str(folder / "job.json")],
                            cwd=str(Path(__file__).resolve().parents[1]), env=env,
                            stdout=log, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            processes.append(process)
            active[index] = (process, folder, job)
        while active and clock() < deadline:
            for index, (process, folder, job) in list(active.items()):
                if clock() >= deadline:
                    break
                snapshots(index, folder, job)
                code = process.poll()
                if clock() >= deadline:
                    break
                if code is None:
                    continue
                if code != 0:
                    raise RuntimeError(f"Worker failed ({job['kind']}); see {folder / 'worker.log'}")
                done = read_json(folder / "done.json")
                if clock() >= deadline:
                    break
                if done is None:
                    continue
                if done.get("status") != "complete":
                    raise RuntimeError(f"Worker failed: {done.get('error', done)}")
                latest = snapshots(index, folder, job)
                if clock() >= deadline:
                    break
                if latest:
                    members[index]["completed"] = True
                    del active[index]
            if active:
                sleep(min(.02, max(0, deadline - clock())))
        observed = clock()
        decision = min(deadline - started, observed - started)
    finally:
        # Kill all remaining native solver threads before leaving this phase.
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.wait(timeout=10)
        for log in logs:
            log.close()
    return dict(members=members, aggregate=aggregate, aggregate_after_s=aggregate_at,
                decision_wall_s=decision, cleanup_wall_s=max(0, clock() - observed),
                hard_timeout=not all(m["completed"] for m in members),
                deadline_detection_lag_s=max(0, observed - deadline),
                phase_wall_s=max(0, observed - phase_started), file_access_retries=retry_count)


def run_experiment(common, method, k, budget, directory, *, phase=None, clock=None):
    phase, clock = phase or run_phase, clock or time.monotonic
    started = clock()
    deadline = started + budget
    directory = fresh_attempt(directory)
    job = dict(common, method=method)
    if method == "pipeline":
        generation = phase([dict(job, kind="generate", k=k, threads=1)], directory / "g",
                           started, deadline)
        gpu = generation["members"][0]["row"] or {}
        ready = gpu.get("ready", False)
        if ready and gpu.get("n_candidates") != k:
            raise ValueError("GPU returned an incomplete candidate batch")
        if ready and clock() < deadline:
            jobs = [dict(job, kind="repair", threads=1, seed=common["seed"] + i,
                         candidate_file=gpu["candidate_file"], candidate_idx=i) for i in range(k)]
            result = phase(jobs, directory / "r", started, deadline, reducer=select_repairs)
            row, received = result["aggregate"], result["aggregate_after_s"]
        else:
            result, row, received = dict(generation), None, None
            if clock() >= deadline:
                result.update(hard_timeout=True, decision_wall_s=budget)
        result.update(generation=gpu, generation_wall_s=generation["phase_wall_s"],
                      n_candidates_generated=k if ready else 0)
        if ready and row is not None:
            row.update(scenario_id=common["scenario_id"], family=common["family_alias"],
                       dr_formulation=FORMULATION, kind="pipeline")
    else:
        result = phase([dict(job, kind=method)], directory / "s", started, deadline)
        row = result["members"][0]["row"]
        received = result["members"][0]["received_after_s"]
    result.update(row=row, snapshot_available_after_s=received, budget_seconds=budget,
                  experiment_wall_s=max(0, clock() - started), protocol=PROTOCOL)
    result["cleanup_wall_s"] = max(0, result["experiment_wall_s"] - result["decision_wall_s"])
    return result


def worker(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    sequence = 0

    def publish(method, row):
        nonlocal sequence
        if method != job["method"] or time.monotonic() >= job["deadline"]:
            return False
        for key, expected in (("scenario_id", job["scenario_id"]), ("family", job["family_alias"]),
                              ("dr_formulation", FORMULATION)):
            if key in row and row[key] != expected:
                raise ValueError(f"Worker result identity mismatch: {key}")
        sequence += 1
        row = dict(row, scenario_id=job["scenario_id"], family=job["family_alias"],
                   dr_formulation=FORMULATION, kind=job["kind"])
        path = Path(job["snapshot_file"]).with_name(f"snapshot-{sequence:06d}.json")
        atomic_json(path, dict(method=method, kind=job["kind"], sequence=sequence, row=row))
        return time.monotonic() < job["deadline"]

    try:
        if job["kind"] == "milp":
            from scripts.matched_budget_milp import solve_streaming
            solve_streaming(job, publish)
        else:
            import pandas  # noqa: F401 -- local PyArrow/PyTorch DLL ordering
            if job["kind"] == "generate":
                from scripts.benchmark_parallel_best_of_k import generate_candidates_gpu
                publish("pipeline", generate_candidates_gpu(job))
            elif job["kind"] == "repair":
                from scripts.benchmark_parallel_best_of_k import repair_candidate_cpu
                result = repair_candidate_cpu(job)
                row = dict(result["pipeline_result"], success=result["feasible"],
                           energy=result["energy"], candidate_idx=result["candidate_idx"])
                publish("pipeline", row)
            elif job["kind"] == "jepa":
                import torch
                from src.uc_jepa.evaluation import evaluate_latent_pipeline as evaluation
                from src.uc_jepa.latent_utils import seed_all
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is required for this comparison")
                seed_all(job["seed"])
                root = Path(job["root"]) / "uc_jepa"
                with corrected_selection(evaluation):
                    evaluation.evaluate(
                        latent_ebm_path=str(root / "latent_ebm/latent_ebm_best.pt"),
                        ae_path=str(root / "autoencoder/uc_ae_best.pt"),
                        jepa_path=str(root / "jepa/jepa_best.pt"),
                        data_roots=[job["label_file"]], scenario_roots=[job["scenario_root"]],
                        output_dir=job["output_dir"], device="cuda:0", k=0, selection_policy="energy",
                        lp_solver="appsi_highs", lp_max_stages=3, analyze=False, direct_only=True,
                        deadline_monotonic=job["deadline"], result_callback=publish,
                        enforce_hard_lp_timeout=False,
                    )
            else:
                raise ValueError(f"Unknown worker kind: {job['kind']}")
        atomic_json(job["done_file"], {"status": "complete"})
    except TimeoutError:
        atomic_json(job["done_file"], {"status": "complete", "reason": "deadline"})
    except Exception as exc:
        traceback.print_exc()
        atomic_json(job["done_file"], {"status": "worker_error", "error": str(exc)})
        raise


def build_manifest(args):
    import importlib.metadata
    inventory = make_manifest(Path(args.root), args.families, args.threads, max(args.budgets), [args.k])
    base = {key: inventory[key] for key in (
        "root", "families", "cohort", "inputs_sha256", "source_sha256", "python", "platform",
        "logical_cpus", "processor", "machine", "versions", "dr_formulation",
    )}
    base["versions"]["scipy"] = importlib.metadata.version("scipy")
    if args.limit_per_family is not None:
        counts = {}
        cohort = []
        for item in base["cohort"]:
            family = item["family"]
            counts[family] = counts.get(family, 0) + 1
            if counts[family] <= args.limit_per_family:
                cohort.append(item)
        base["cohort"] = cohort
    for name in NEW_SOURCES:
        base["source_sha256"][name] = sha256(name)
    return dict(base, schema_version=1, protocol=PROTOCOL, budgets=args.budgets,
                methods=list(METHODS), k=args.k, milp_threads=args.threads,
                jepa_threads=args.threads, repair_workers=args.k, repair_threads=1,
                selection="energy_among_feasible",
                timing="independent cold processes; one deadline includes imports/models/GPU/CPU/selection; offline HTE excluded",
                method_order="rotating by scenario and budget index",
                checkpoint_policy="per scenario/budget/method; retain completed failures and timeouts",
                milp_incumbents="validated improving-solution callbacks; only parent reception before deadline",
                feasibility_tolerance=1e-3, zero_slack_tolerance_mwh=1e-3)


def identity(item, budget, method, manifest):
    return dict(item, budget_seconds=budget, method=method, k=manifest["k"], protocol=PROTOCOL)


def checkpoint_path(out, index, budget_index, method):
    return Path(out) / "cases" / f"{index:04d}" / f"b{budget_index:02d}" / f"{method}.json"


def load_checkpoint(path, expected, manifest_hash):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    claimed = payload.pop("sha256", None)
    if claimed != digest(payload) or payload.get("identity") != expected or payload.get("manifest_sha256") != manifest_hash:
        raise ValueError(f"Incompatible or corrupt completed experiment: {path}")
    outcome = payload["outcome"]
    budget = expected["budget_seconds"]
    if (outcome.get("protocol") != PROTOCOL or outcome.get("budget_seconds") != budget
            or not finite(outcome.get("decision_wall_s")) or not 0 <= outcome["decision_wall_s"] <= budget):
        raise ValueError(f"Invalid checkpoint timing: {path}")
    row = outcome.get("row")
    if row:
        received = outcome.get("snapshot_available_after_s")
        family = expected["family"].removesuffix("_criticality_scenarios")
        if (row.get("scenario_id") != expected["scenario_id"] or row.get("family") != family
                or row.get("kind") != expected["method"]
                or row.get("dr_formulation") != FORMULATION or not finite(received)
                or not 0 <= received < budget):
            raise ValueError(f"Invalid checkpoint result identity/reception: {path}")
    return outcome


def run_campaign(args):
    root, out = Path(args.root).resolve(), Path(args.output_dir).resolve()
    if out == root or root in out.parents and out.relative_to(root).parts[:1] != ("evaluation",):
        raise ValueError("Use a separate evaluation output directory")
    for name in ("latent_k", "parallel_best_of_k"):
        if any(part.startswith(name) for part in out.parts):
            raise ValueError("Refusing an existing campaign output directory")
    with campaign_lock(root):
        current_manifest = build_manifest(args)
        manifest = current_manifest
        path = out / "manifest.json"
        compatible_changes = []
        if path.exists():
            if not args.resume:
                raise ValueError("Existing comparison requires --resume; use a new output directory for a fresh run")
            manifest = json.loads(path.read_text(encoding="utf-8"))
            compatible_changes = validate_resume_manifest(manifest, current_manifest)
        elif out.exists() and any(out.iterdir()):
            raise ValueError("Nonempty output directory has no manifest")
        # Every checkpoint remains bound to the ORIGINAL manifest. Replacing its
        # hash with the current source inventory would invalidate completed work.
        manifest_hash = digest(manifest)
        execution_manifest_hash = digest(current_manifest)
        # Check every existing experiment before starting any new worker.
        cached = {}
        for i, item in enumerate(manifest["cohort"], 1):
            for b, budget in enumerate(args.budgets):
                for method in METHODS:
                    saved = checkpoint_path(out, i, b, method)
                    if saved.exists():
                        cached[i, b, method] = load_checkpoint(saved, identity(item, budget, method, manifest), manifest_hash)
        # Longest JEPA export path; GPU/repair files use short fixed names.
        longest = out / "w/9999/99/j/a001/s/00/a/family_results" / (
            "direct_baseline_family_persona_network_operator_scenarios_summary.json")
        if os.name == "nt" and len(str(longest)) >= 260:
            raise ValueError("Output path too long for Windows; choose a shorter --output-dir")
        if not path.exists():
            save_parent_json(path, manifest)
        else:
            save_parent_json(out / "resumes" / f"{time.time_ns()}.json", dict(
                original_manifest_sha256=manifest_hash,
                execution_manifest_sha256=execution_manifest_hash,
                compatible_source_changes=compatible_changes, manifest=current_manifest,
            ))
            if compatible_changes:
                print("Compatible resume fixes accepted: " + ", ".join(compatible_changes), flush=True)
        total = len(manifest["cohort"]) * len(args.budgets) * 3
        done = len(cached)
        print(f"Completed experiments reused: {done}/{total}", flush=True)
        for i, item in enumerate(manifest["cohort"], 1):
            family, sid = item["family"], item["scenario_id"]
            seed = int(hashlib.sha256(f"{family}/{sid}".encode()).hexdigest()[:8], 16) % 1_000_000_000
            common = dict(root=str(root), scenario_id=sid, family_alias=family.removesuffix("_criticality_scenarios"),
                          scenario_file=str(root / family / f"{sid}.json"), scenario_root=str(root / family),
                          label_file=str(root / "uc_jepa/dataset_eval" / family / f"{sid}_uc.npz"),
                          seed=seed, threads=args.threads)
            for b, budget in enumerate(args.budgets):
                offset = (i + b) % 3
                for method in METHODS[offset:] + METHODS[:offset]:
                    if (i, b, method) in cached:
                        continue
                    print(f"[{done + 1}/{total}] {family}/{sid} {method} {budget:g}s", flush=True)
                    outcome = run_experiment(common, method, args.k, budget,
                                             out / "w" / f"{i:04d}" / f"{b:02d}" / method[0])
                    payload = dict(identity=identity(item, budget, method, manifest),
                                   manifest_sha256=manifest_hash,
                                   execution_manifest_sha256=execution_manifest_hash, outcome=outcome)
                    payload["sha256"] = digest(payload)
                    save_parent_json(checkpoint_path(out, i, b, method), payload)
                    done += 1
        analyze(out)


def gap(cost, reference):
    return 100 * (cost - reference) / abs(reference) if finite(cost) and finite(reference) and abs(reference) > 1e-9 else None


def average(values):
    values = [v for v in values if finite(v)]
    return statistics.mean(values) if values else None


def median(values):
    values = [v for v in values if finite(v)]
    return statistics.median(values) if values else None


def summarize(rows, families, budgets):
    summaries, comparisons = [], []
    for family in list(families) + ["ALL"]:
        for budget in budgets:
            group = [r for r in rows if r["budget_seconds"] == budget and (family == "ALL" or r["family"] == family)]
            lookup = {(r["family"], r["scenario_id"], r["method"]): r for r in group}
            cases = {(r["family"], r["scenario_id"]) for r in group}
            common = [key for key in sorted(cases) if all(lookup[*key, m]["feasible"] for m in METHODS)]
            for method in METHODS:
                data = [r for r in group if r["method"] == method]
                complete = [r for r in data if r["completed"]]
                valid = [r for r in complete if r["feasible"]]
                summaries.append(dict(family=family, budget_seconds=budget, method=method,
                    n_planned=len(data), n_completed=len(complete), n_pending=len(data) - len(complete),
                    n_feasible=len(valid), feasibility_rate_completed=len(valid) / len(complete) if complete else None,
                    n_zero_slack=sum(r["zero_slack"] for r in valid),
                    n_timeouts=sum(r["hard_timeout"] for r in complete),
                    mean_wall_s=average([r["decision_wall_s"] for r in complete]),
                    mean_gap_reference_own_feasible=average([r["gap_reference_pct"] for r in valid]),
                    n_common_all_three=len(common),
                    mean_gap_reference_common=average([lookup[*key, method]["gap_reference_pct"] for key in common]),
                    median_gap_reference_common=median([lookup[*key, method]["gap_reference_pct"] for key in common])))
            for left, right in (("pipeline", "milp"), ("jepa", "milp"), ("pipeline", "jepa")):
                paired = [(lookup[*key, left], lookup[*key, right]) for key in sorted(cases)
                          if lookup[*key, left]["feasible"] and lookup[*key, right]["feasible"]]
                comparisons.append(dict(family=family, budget_seconds=budget, left=left, right=right,
                    n_paired=len(paired),
                    left_wins=sum(a["objective"] < b["objective"] - .001 for a, b in paired),
                    right_wins=sum(b["objective"] < a["objective"] - .001 for a, b in paired),
                    ties=sum(abs(a["objective"] - b["objective"]) <= .001 for a, b in paired),
                    mean_left_gap_reference=average([a["gap_reference_pct"] for a, _ in paired]),
                    mean_right_gap_reference=average([b["gap_reference_pct"] for _, b in paired]),
                    median_left_gap_reference=median([a["gap_reference_pct"] for a, _ in paired]),
                    median_right_gap_reference=median([b["gap_reference_pct"] for _, b in paired]),
                    mean_left_cost_difference=average([a["objective"] - b["objective"] for a, b in paired])))
    return summaries, comparisons


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(out):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Incompatible matched-budget protocol")
    manifest_hash = digest(manifest)
    rows = []
    for i, item in enumerate(manifest["cohort"], 1):
        ref_path = Path(manifest["root"]) / item["family"] / "reports" / f"{item['scenario_id']}.json"
        key = str(ref_path.relative_to(Path(manifest["root"])))
        if sha256(ref_path) != manifest["inputs_sha256"][key]:
            raise ValueError(f"Reference changed since benchmark: {ref_path}")
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        if ref.get("dr_formulation") != FORMULATION:
            raise ValueError("Legacy reference formulation")
        reference = ref["mip"]["objective"]
        for b, budget in enumerate(manifest["budgets"]):
            for method in METHODS:
                saved = checkpoint_path(out, i, b, method)
                outcome = load_checkpoint(saved, identity(item, budget, method, manifest), manifest_hash) if saved.exists() else None
                row = (outcome or {}).get("row") or {}
                ok = eligible(row)
                slack = row.get("slack_mwh")
                rows.append(dict(**item, budget_seconds=budget, method=method, completed=outcome is not None,
                    feasible=ok, zero_slack=bool(ok and finite(slack) and slack <= .001),
                    objective=row.get("lp_objective") if ok else None,
                    reference_objective=reference, reference_status=ref["mip"]["termination"],
                    gap_reference_pct=gap(row.get("lp_objective"), reference) if ok else None,
                    slack_mwh=slack if ok and finite(slack) else None,
                    status=row.get("status", "no_result" if outcome else "pending"),
                    hard_timeout=outcome["hard_timeout"] if outcome else False,
                    decision_wall_s=outcome["decision_wall_s"] if outcome else None,
                    cleanup_wall_s=outcome["cleanup_wall_s"] if outcome else None,
                    selected_candidate=row.get("best_sample_idx"),
                    n_repairs_completed=row.get("n_repairs_completed")))
    summaries, comparisons = summarize(rows, manifest["families"], manifest["budgets"])
    for name, data in (("per_scenario", rows), ("summary", summaries), ("paired_comparisons", comparisons)):
        write_csv(out / f"{name}.csv", data)
        atomic_json(out / f"{name}.json", data)
    print(f"Metrics written to {out} ({sum(r['completed'] for r in rows)}/{len(rows)} experiments complete)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--root", default="outputs/load_shifting_v1")
    parser.add_argument("--output-dir", default="outputs/load_shifting_v1/evaluation/budgets_v1")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--families", nargs="+", default=list(FAMILIES))
    parser.add_argument("--limit-per-family", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    args.budgets = parse_budgets(args.budgets)
    if args.threads < 1 or args.k < 1 or args.k > args.threads:
        parser.error("Require 1 <= K <= HiGHS threads")
    if args.limit_per_family is not None and args.limit_per_family < 1:
        parser.error("--limit-per-family must be positive")
    if len(set(args.families)) != len(args.families):
        parser.error("Duplicate families")
    if args.preview:
        counts = {f: min(len(list((Path(args.root) / f).glob('scenario_*.json'))), args.limit_per_family or sys.maxsize)
                  for f in args.families}
        print(json.dumps(dict(protocol=PROTOCOL, budgets=args.budgets, families=counts,
                              k=args.k, highs_threads=args.threads, jepa_threads=args.threads,
                              repair_threads=1, experiments=sum(counts.values()) * len(args.budgets) * 3,
                              output_dir=str(Path(args.output_dir).resolve())), indent=2))
    elif args.analyze_only:
        analyze(args.output_dir)
    else:
        run_campaign(args)


if __name__ == "__main__":
    main()
