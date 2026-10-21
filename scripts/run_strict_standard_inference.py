"""Isolate standard inference and JEPA in separately timed, killable processes.

Importing this module starts no experiments and imports no ML libraries.
The parent retains only snapshots received before each absolute deadline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from scripts.benchmark_parallel_best_of_k import (
    FORMULATION, THREAD_ENV, atomic_json, make_manifest, sha256,
)

PROTOCOL = "strict_standard_separate_v1"


def run_method(job, directory, budget, *, popen=None, clock=None, sleep=None):
    """Return the latest on-time snapshot, even if the next repair is killed."""
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("A positive finite per-scenario budget is required")
    popen, clock, sleep = popen or subprocess.Popen, clock or time.monotonic, sleep or time.sleep
    started = clock()
    deadline = started + budget
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    snapshot_path, done_path = directory / "snapshot.json", directory / "done.json"
    if list(directory.glob("snapshot-*.json")) or done_path.exists():
        raise ValueError(f"Refusing stale worker outputs: {directory}")
    job = dict(job, deadline=deadline, snapshot_file=str(snapshot_path),
               done_file=str(done_path), output_dir=str(directory / "artifacts"))
    job_path = directory / "job.json"
    atomic_json(job_path, job)
    env = dict(os.environ, **{name: "1" for name in THREAD_ENV},
               BENCHMARK_HIGHS_THREADS=str(job["threads"]), PYTHONUNBUFFERED="1")
    process, snapshot, received_at = None, None, None
    completed, timed_out = False, False
    log = (directory / "worker.log").open("wb")

    def read_snapshot():
        nonlocal snapshot, received_at
        if clock() >= deadline:
            return
        paths = sorted(directory.glob("snapshot-*.json"))
        if not paths:
            return
        # Immutable snapshots avoid replacing a file being read on Windows.
        latest = paths[-1]
        if snapshot is not None and latest.stem == f"snapshot-{snapshot['sequence']:06d}":
            return
        payload = json.loads(latest.read_text(encoding="utf-8"))
        now = clock()
        if now >= deadline:
            return
        if payload.get("method") != job["method"]:
            raise ValueError("Worker snapshot method mismatch")
        row = payload["row"]
        if row.get("dr_formulation") != FORMULATION or row.get("scenario_id") != job["scenario_id"]:
            raise ValueError("Worker snapshot scenario/formulation mismatch")
        # Only change reception time when a new snapshot is received.
        if snapshot is None or payload["sequence"] != snapshot["sequence"]:
            snapshot, received_at = payload, now - started

    try:
        if clock() < deadline:
            process = popen(
                [sys.executable, "-u", "-B", "-m", "scripts.run_strict_standard_inference",
                 "--worker", str(job_path)],
                cwd=str(Path(__file__).resolve().parents[1]), env=env,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        while process is not None and clock() < deadline:
            read_snapshot()
            code = process.poll()
            if clock() >= deadline:
                break
            if code is not None:
                if code != 0 or not done_path.exists():
                    raise RuntimeError(f"Worker failed ({job['method']}); see {directory / 'worker.log'}")
                done = json.loads(done_path.read_text(encoding="utf-8"))
                read_snapshot()  # final snapshot may have arrived just before exit
                if clock() >= deadline:
                    break
                if done.get("status") != "complete":
                    raise RuntimeError(f"Worker failed ({job['method']}): {done.get('error', done)}")
                completed = True
                break
            sleep(min(0.02, max(0.0, deadline - clock())))
        timed_out = not completed
        observed = clock()
        decision_time = min(budget, observed - started)
    finally:
        # Do not read outputs produced during termination or after the deadline.
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        log.close()
    finished = clock()
    return {
        "row": snapshot["row"] if snapshot else None,
        "hard_timeout": timed_out, "budget_seconds": budget,
        "decision_wall_s": decision_time,
        "cleanup_wall_s": max(0.0, finished - started - decision_time),
        "deadline_detection_lag_s": max(0.0, observed - deadline) if timed_out else 0.0,
        "snapshot_available_after_s": received_at,
    }


def run_pair(common, directory, pipeline_budget, jepa_budget, *, runner=None):
    """JEPA always gets a fresh deadline, including after a pipeline timeout."""
    runner = runner or run_method
    results = {}
    for method, budget in (("pipeline", pipeline_budget), ("jepa", jepa_budget)):
        print(f"  {method}: {budget:g}s", flush=True)
        results[method] = runner(dict(common, method=method), Path(directory) / method, budget)
    return results


def worker(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    try:
        # Import ordering avoids the local Windows PyArrow/PyTorch DLL conflict.
        import pandas  # noqa: F401
        import torch
        from src.uc_jepa.evaluation.evaluate_latent_pipeline import evaluate
        from src.uc_jepa.latent_utils import seed_all

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        seed_all(job["seed"])
        root = Path(job["root"]) / "uc_jepa"
        sequence = 0

        def publish(method, row):
            nonlocal sequence
            if method == job["method"] and time.monotonic() < job["deadline"]:
                sequence += 1
                path = Path(job["snapshot_file"]).with_name(f"snapshot-{sequence:06d}.json")
                atomic_json(path, {"method": method, "sequence": sequence, "row": row})

        evaluate(
            latent_ebm_path=str(root / "latent_ebm/latent_ebm_best.pt"),
            ae_path=str(root / "autoencoder/uc_ae_best.pt"),
            jepa_path=str(root / "jepa/jepa_best.pt"),
            data_roots=[job["label_file"]], scenario_roots=[job["scenario_root"]],
            output_dir=job["output_dir"], k=job["k"], selection_policy="energy",
            lp_solver="appsi_highs", lp_max_stages=3, analyze=False,
            skip_direct_baseline=job["method"] == "pipeline",
            direct_only=job["method"] == "jepa",
            deadline_monotonic=job["deadline"], result_callback=publish,
            # Whole-process termination is managed by the parent. No nested worker.
            enforce_hard_lp_timeout=False,
        )
        atomic_json(job["done_file"], {"status": "complete"})
    except Exception as exc:
        traceback.print_exc()
        atomic_json(job["done_file"], {"status": "worker_error", "error": str(exc)})
        raise


def result_row(outcome, fallback, *, method, k):
    """Preserve every scenario and use external wall time in standard metrics."""
    row = dict(outcome["row"] if outcome["row"] is not None else fallback)
    time_key = "time_total" if method == "pipeline" else "total_time_s"
    row["compute_time_before_snapshot_s"] = row.get(time_key, 0.0)
    row[time_key] = outcome["decision_wall_s"]
    row.update(
        timing_protocol=PROTOCOL, hard_timeout=outcome["hard_timeout"],
        budget_seconds=outcome["budget_seconds"], cleanup_wall_s=outcome["cleanup_wall_s"],
        deadline_detection_lag_s=outcome["deadline_detection_lag_s"],
        snapshot_available_after_s=outcome["snapshot_available_after_s"],
        requested_candidates=k if method == "pipeline" else 1,
    )
    if outcome["row"] is None:
        row.update(success=False, lp_objective=math.inf)
        row["lp_status" if method == "pipeline" else "status"] = (
            "hard_timeout" if outcome["hard_timeout"] else "no_result"
        )
    if outcome["hard_timeout"]:
        row["error_message"] = (
            "Deadline reached; only candidate results received before the deadline are retained."
        )
    return row


def export_results(root, out, manifest):
    """Build the legacy analysis inputs, including missing/timeout results."""
    import pandas as pd
    from dataclasses import asdict
    from src.eval.pipeline_runner import PipelineResult
    from src.uc_jepa.data.uc_label_dataset import UCLabelDataset
    from src.uc_jepa.evaluation.evaluate_latent_pipeline import (
        _family_alias, _augment_result_row_with_reference_metrics, _persist_family_results,
        _persist_scenario_result, _summarize_direct_baseline, _write_json, _write_pickle,
    )

    rows, direct_rows = [], []
    for item in manifest["cohort"]:
        family, sid = item["family"], item["scenario_id"]
        pair = json.loads((out / "cases" / family / f"{sid}.json").read_text(encoding="utf-8"))
        sample = UCLabelDataset(roots=[str(root / "uc_jepa/dataset_eval" / family / f"{sid}_uc.npz")])[0]
        if sample.scenario_id != sid or _family_alias(sample.family) != _family_alias(family):
            raise ValueError(f"Label identity mismatch: {family}/{sid}")
        fallback = asdict(PipelineResult(scenario_id=sid, family=_family_alias(family)))
        fallback.update(success=False, lp_objective=math.inf, n_zones=sample.n_zones,
                        n_timesteps=sample.n_timesteps, criticality_index=sample.criticality)
        row = result_row(pair["pipeline"], fallback, method="pipeline", k=manifest["k"])
        row = _augment_result_row_with_reference_metrics(row, sample.objective)
        rows.append(row)
        _persist_scenario_result(out, row)
        direct_fallback = dict(
            scenario_id=sid, family=_family_alias(family), dr_formulation=FORMULATION,
            criticality_index=sample.criticality, milp_objective=sample.objective,
            success=False, lp_objective=math.inf, gap_pct=math.nan, dr_metrics={},
            solve_time=0.0, total_time_s=0.0, stage_used="", stage_label="", stage_reached="",
        )
        direct_rows.append(result_row(pair["jepa"], direct_fallback, method="jepa", k=1))
    _write_pickle(out / "pipeline_eval_all_families.pkl", rows)
    _write_json(out / "pipeline_eval_all_families.json", rows)
    for family in sorted({r["family"] for r in rows}):
        _persist_family_results(out, family, [r for r in rows if r["family"] == family],
                                [r for r in direct_rows if r["family"] == family])
    pd.DataFrame(direct_rows).to_csv(out / "direct_baseline.csv", index=False)
    _write_json(out / "direct_baseline_summary.json", _summarize_direct_baseline(direct_rows))


def run_campaign(args):
    root, out = Path(args.root).resolve(), Path(args.output_dir).resolve()
    # Reuse input/checkpoint/source inventory, with this protocol's own settings.
    inventory = make_manifest(root, args.families, args.threads, args.budget, [args.k])
    manifest = {key: inventory[key] for key in (
        "root", "families", "cohort", "inputs_sha256", "source_sha256",
        "python", "platform", "logical_cpus", "processor", "machine", "versions",
    )}
    manifest["source_sha256"]["scripts/run_strict_standard_inference.py"] = sha256(__file__)
    manifest.update(
        schema_version=1, protocol=PROTOCOL, dr_formulation=FORMULATION,
        k=args.k, highs_threads=args.threads, budget_seconds=args.budget,
        jepa_budget_seconds=args.jepa_budget, selection="energy",
        device="cuda if available, otherwise cpu",
        timing="separate cold-process deadlines; offline HTE excluded; cleanup separately reported",
    )
    if (out / "manifest.json").exists():
        raise ValueError("Use fresh outputs; the PowerShell runner archives previous stage-10 results")
    atomic_json(out / "manifest.json", manifest)
    for i, item in enumerate(manifest["cohort"], 1):
        family, sid = item["family"], item["scenario_id"]
        print(f"[{i}/{len(manifest['cohort'])}] {family}/{sid}", flush=True)
        seed = int(hashlib.sha256(f"{family}/{sid}".encode()).hexdigest()[:8], 16) % 1_000_000_000
        common = dict(
            root=str(root), scenario_id=sid, scenario_root=str(root / family),
            label_file=str(root / "uc_jepa/dataset_eval" / family / f"{sid}_uc.npz"),
            threads=args.threads, seed=seed, k=args.k,
        )
        pair = run_pair(common, out / "jobs" / family / sid, args.budget, args.jepa_budget)
        atomic_json(out / "cases" / family / f"{sid}.json", pair)
    export_results(root, out, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--root")
    parser.add_argument("--output-dir")
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--budget", type=float, default=120.0)
    parser.add_argument("--jepa-budget", type=float, default=120.0)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if not args.root or not args.output_dir or not args.families:
        parser.error("--root, --output-dir and --families are required")
    if args.threads < 1 or args.k < 1 or any(
        not math.isfinite(x) or x <= 0 for x in (args.budget, args.jepa_budget)
    ):
        parser.error("Thread count, K and both budgets must be positive and finite")
    run_campaign(args)


if __name__ == "__main__":
    main()
