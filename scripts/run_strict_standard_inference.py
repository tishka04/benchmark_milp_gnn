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
# Windows/OneDrive can briefly deny opening even an atomically published file.
# Retries use the existing polling loop and never extend the scenario deadline.
FILE_ACCESS_RETRY_SECONDS = 2.0
# Audited predecessor: same inference, seeding, selection and timing protocol.
# Only orchestration paths/checkpoint reuse change in this compatibility update.
COMPATIBLE_RUNNER_HASHES = {
    "8e2c07ffdd17f1f73f47314d4e584fd2d550fefd57a6660f7690184a4514f825",
}
RUNNER_SOURCE = "scripts/run_strict_standard_inference.py"


def validate_resume_manifest(previous, current):
    """Reject changed models, data, physics, settings or unknown source edits."""
    previous, current = dict(previous), dict(current)
    old_sources = dict(previous.pop("source_sha256"))
    new_sources = dict(current.pop("source_sha256"))
    old_runner = old_sources.pop(RUNNER_SOURCE)
    new_runner = new_sources.pop(RUNNER_SOURCE)
    if previous != current or old_sources != new_sources:
        raise ValueError("Incompatible stage-10 resume: inputs, sources or settings changed. "
                         "Use -StartStep 10 for an explicit fresh evaluation.")
    if old_runner != new_runner and old_runner not in COMPATIBLE_RUNNER_HASHES:
        raise ValueError("Incompatible stage-10 runner revision; use -StartStep 10.")


def validate_outcome(outcome, common, budget):
    """Check checkpoint identity/timing, without excluding experimental failures."""
    if not isinstance(outcome, dict) or "row" not in outcome:
        raise ValueError("Invalid completed inference checkpoint")
    if outcome.get("budget_seconds") != budget or type(outcome.get("hard_timeout")) is not bool:
        raise ValueError("Checkpoint budget/timeout mismatch")
    for key in ("decision_wall_s", "cleanup_wall_s", "deadline_detection_lag_s"):
        value = outcome.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid checkpoint timing: {key}")
    if outcome["decision_wall_s"] > budget:
        raise ValueError("Checkpoint decision exceeds its budget")
    row, received = outcome["row"], outcome.get("snapshot_available_after_s")
    if row is None:
        if received is not None:
            raise ValueError("Checkpoint reception time without a result")
    else:
        family = Path(common["scenario_root"]).name.removesuffix("_criticality_scenarios")
        if (not isinstance(row, dict) or row.get("scenario_id") != common["scenario_id"]
                or row.get("dr_formulation") != FORMULATION or row.get("family") != family):
            raise ValueError("Checkpoint scenario/family/formulation mismatch")
        if not isinstance(received, (int, float)) or not 0 <= received < budget:
            raise ValueError("Checkpoint result was not received within its budget")


def fresh_attempt(directory):
    """Keep interrupted attempts untouched; every retry has fresh snapshots."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        path = directory / f"a{attempt:03d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            attempt += 1


def validate_worker_path(directory, family):
    # Longest family export currently written by evaluate(), plus both methods.
    path = (Path(directory).resolve() / "pipeline/a001/artifacts/family_results"
            / f"direct_baseline_family_{family}_summary.json")
    if os.name == "nt" and len(str(path)) >= 260:
        raise ValueError(f"Worker output path exceeds Windows MAX_PATH; use a shorter "
                         f"output directory: {path}")


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
    denied_since = {}
    file_access_retries = 0
    log = (directory / "worker.log").open("wb")

    def read_worker_json(path):
        nonlocal file_access_retries
        try:
            text = path.read_text(encoding="utf-8")
        except PermissionError as exc:
            now = clock()
            first_attempt = path not in denied_since
            first_denied = denied_since.setdefault(path, now)
            file_access_retries += 1
            if now - first_denied >= FILE_ACCESS_RETRY_SECONDS:
                raise PermissionError(
                    f"Worker output remains inaccessible after {FILE_ACCESS_RETRY_SECONDS:g}s "
                    f"of retries: {path}"
                ) from exc
            if first_attempt:
                print(f"  Temporarily locked worker output; retrying within budget: {path.name}", flush=True)
            return None
        denied_since.pop(path, None)
        # Malformed JSON and identity errors remain fatal, not retryable I/O.
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Worker output must be a JSON object: {path}")
        return payload

    def read_snapshot():
        """Read the newest accessible snapshot; report whether the latest is read."""
        nonlocal snapshot, received_at
        if clock() >= deadline:
            return False
        paths = sorted(directory.glob("snapshot-*.json"))
        # Immutable snapshots avoid replacing a file being read on Windows.
        # If the newest file is locked, retain/read an older result while waiting.
        for index, path in enumerate(reversed(paths)):
            if clock() >= deadline:
                return False
            if snapshot is not None and path.stem <= f"snapshot-{snapshot['sequence']:06d}":
                return index == 0
            payload = read_worker_json(path)
            if payload is None:
                continue
            now = clock()
            if now >= deadline:
                return False
            if payload.get("method") != job["method"]:
                raise ValueError("Worker snapshot method mismatch")
            row = payload["row"]
            if row.get("dr_formulation") != FORMULATION or row.get("scenario_id") != job["scenario_id"]:
                raise ValueError("Worker snapshot scenario/formulation mismatch")
            if path.stem != f"snapshot-{payload['sequence']:06d}":
                raise ValueError("Worker snapshot sequence mismatch")
            # Only change reception time when a new snapshot is received.
            snapshot, received_at = payload, now - started
            return index == 0
        return not paths

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
                if code != 0:
                    raise RuntimeError(f"Worker failed ({job['method']}); see {directory / 'worker.log'}")
                try:
                    done = read_worker_json(done_path)
                except FileNotFoundError as exc:
                    raise RuntimeError(f"Worker failed ({job['method']}); see {directory / 'worker.log'}") from exc
                if done is not None:
                    if clock() >= deadline:
                        break
                    if done.get("status") != "complete":
                        raise RuntimeError(f"Worker failed ({job['method']}): {done.get('error', done)}")
                    # Do not finish with an older result while the final one is locked.
                    latest_read = read_snapshot()
                    if clock() >= deadline:
                        break
                    if latest_read:
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
        "file_access_retries": file_access_retries,
    }


def run_pair(common, directory, pipeline_budget, jepa_budget, *, runner=None, checkpoint=False):
    """JEPA always gets a fresh deadline, including after a pipeline timeout."""
    runner = runner or run_method
    results = {}
    for method, budget in (("pipeline", pipeline_budget), ("jepa", jepa_budget)):
        job = dict(common, method=method)
        saved = Path(directory) / f"{method}.json"
        if checkpoint and saved.exists():
            payload = json.loads(saved.read_text(encoding="utf-8"))
            if payload.get("job") != job:
                raise ValueError(f"Method checkpoint identity mismatch: {saved}")
            validate_outcome(payload["outcome"], common, budget)
            results[method] = payload["outcome"]
            print(f"  {method}: reused completed result", flush=True)
            continue
        print(f"  {method}: {budget:g}s", flush=True)
        work = Path(directory) / method
        if checkpoint:
            work = fresh_attempt(work)
        results[method] = runner(job, work, budget)
        if checkpoint:
            validate_outcome(results[method], common, budget)
            atomic_json(saved, {"job": job, "outcome": results[method]})
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
        file_access_retries=outcome.get("file_access_retries", 0),
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


def build_manifest(args):
    root = Path(args.root).resolve()
    # Reuse input/checkpoint/source inventory, with this protocol's own settings.
    inventory = make_manifest(root, args.families, args.threads, args.budget, [args.k])
    manifest = {key: inventory[key] for key in (
        "root", "families", "cohort", "inputs_sha256", "source_sha256",
        "python", "platform", "logical_cpus", "processor", "machine", "versions",
    )}
    manifest["source_sha256"][RUNNER_SOURCE] = sha256(__file__)
    manifest.update(
        schema_version=1, protocol=PROTOCOL, dr_formulation=FORMULATION,
        k=args.k, highs_threads=args.threads, budget_seconds=args.budget,
        jepa_budget_seconds=args.jepa_budget, selection="energy",
        device="cuda if available, otherwise cpu",
        timing="separate cold-process deadlines; offline HTE excluded; cleanup separately reported",
    )
    return manifest


def common_job(root, item, args):
    family, sid = item["family"], item["scenario_id"]
    seed = int(hashlib.sha256(f"{family}/{sid}".encode()).hexdigest()[:8], 16) % 1_000_000_000
    return dict(
        root=str(root), scenario_id=sid, scenario_root=str(root / family),
        label_file=str(root / "uc_jepa/dataset_eval" / family / f"{sid}_uc.npz"),
        threads=args.threads, seed=seed, k=args.k,
    )


def completed_cases(out, manifest, args):
    """Read-only preflight; a malformed completed record is never silently skipped."""
    completed = set()
    for item in manifest["cohort"]:
        family, sid = item["family"], item["scenario_id"]
        path = Path(out) / "cases" / family / f"{sid}.json"
        if not path.exists():
            continue
        try:
            pair = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(pair, dict) or set(pair) != {"pipeline", "jepa"}:
                raise ValueError("Incomplete pair")
            common = common_job(Path(manifest["root"]), item, args)
            validate_outcome(pair["pipeline"], common, args.budget)
            validate_outcome(pair["jepa"], common, args.jepa_budget)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid completed case {path}: {exc}") from exc
        completed.add((family, sid))
    return completed


def run_campaign(args):
    root, out = Path(args.root).resolve(), Path(args.output_dir).resolve()
    manifest = build_manifest(args)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise ValueError("Existing stage-10 outputs; use --resume or a fresh output directory")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_resume_manifest(previous, manifest)
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Cannot resume nonempty stage-10 outputs without a manifest")
    completed = completed_cases(out, manifest, args)
    for i, item in enumerate(manifest["cohort"], 1):
        validate_worker_path(out / "jobs" / f"{i:04d}", item["family"])
    if not manifest_path.exists():
        atomic_json(manifest_path, manifest)
    elif args.resume:
        # Preserve the original manifest and log the exact compatible runner update.
        atomic_json(out / "resumes" / f"{time.time_ns()}.json", {
            "manifest": manifest, "reused_completed_cases": len(completed),
        })
    print(f"Stage 10: {len(completed)}/{len(manifest['cohort'])} completed pairs reused", flush=True)
    for i, item in enumerate(manifest["cohort"], 1):
        family, sid = item["family"], item["scenario_id"]
        print(f"[{i}/{len(manifest['cohort'])}] {family}/{sid}", flush=True)
        if (family, sid) in completed:
            print("  reused completed pair", flush=True)
            continue
        common = common_job(root, item, args)
        # The manifest maps this short index to the full family/scenario identity.
        # Avoid nested long family names that exceeded Windows MAX_PATH (261 chars).
        pair = run_pair(common, out / "jobs" / f"{i:04d}", args.budget, args.jepa_budget,
                        checkpoint=True)
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
    parser.add_argument("--resume", action="store_true")
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
