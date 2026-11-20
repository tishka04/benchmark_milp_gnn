"""Run each day independently at each deadline and resume verified terminal outcomes."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import uuid

from research.paper1_weights.data import config as loss_config, dispatch_carbon
from src.operational_exit.common import digest, io_path, load_record, save_record, verify
from src.operational_exit.supervisor import run_experiment


REFERENCE_WALL_MARGIN = 60


@contextmanager
def campaign_lock(run_dir):
    """Single process for prepare/run in a directory. The OS releases it on exit."""
    lock_path = io_path(Path(run_dir) / ".writer.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        if not lock_path.stat().st_size:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Another process is preparing or running {run_dir}.") from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another process is preparing or running {run_dir}.") from exc
        # Closing the file descriptor releases either platform's lock.
        yield


def iter_jobs(manifest, kind=None):
    if kind not in (None, "all", "reference", "trial"):
        raise ValueError(f"Unknown job kind: {kind!r}")
    jobs = []
    for day in manifest["days"]:
        if kind != "trial":
            jobs.append((day, "reference", manifest["reference_seconds"]))
        if kind != "reference":
            jobs.extend((day, "trial", budget) for budget in manifest["deadlines"])
    return sorted(jobs, key=lambda j: digest([manifest["order_seed"], j[0]["day_id"], j[1], j[2]]))


def result_path(run_dir, day_id, kind, budget):
    return Path(run_dir) / "results" / f"{day_id}_{kind}_{budget}.json"


def _identity(manifest, day_id, kind, budget):
    return {"manifest_sha256": manifest["sha256"], "day_id": day_id, "kind": kind, "budget": budget}


def read_job(run_dir, manifest, day_id, kind, budget):
    """Read a terminal result with matching protocol identity; never retry corrupt results."""
    path = io_path(result_path(run_dir, day_id, kind, budget))
    if not (path.exists() or path.with_suffix(".json.bak").exists()):
        if next(path.parent.glob(path.name + "*.corrupt-*"), None) is not None:
            raise ValueError(f"Quarantined terminal result. Inspect the original attempt: {path}")
        return None
    record = load_record(path, _identity(manifest, day_id, kind, budget))
    if record is None or not record.get("complete"):
        raise ValueError(f"Damaged terminal result; inspect the backup and attempt before resuming: {path}")
    return record


def _run_one(run_dir, manifest, day, kind, budget, scoring):
    day_id = day["day_id"]
    identity = _identity(manifest, day_id, kind, budget)
    wall_budget = budget + REFERENCE_WALL_MARGIN if kind == "reference" else budget
    settings = {"threads": manifest["threads"], "solver_seconds": budget, "reserve_s": min(2.0, wall_budget * 0.2)}
    request = {
        "identity": identity, "anchor": manifest["anchor"], "anchor_sha256": manifest["anchor_sha256"],
        "day_seed": day["seed"], "solver_seed": manifest["solver_seed"], "budget": wall_budget,
        "profiles_sha256": day["features"]["profiles_sha256"],
        "static_sha256": day["features"]["static_sha256"],
    }
    attempt = Path("attempts") / (f"{day_id}_{kind}_{budget}_" + uuid.uuid4().hex[:12])
    attempt_dir = io_path(Path(run_dir) / attempt)
    try:
        outcome = run_experiment(settings, request, attempt_dir, worker_module="research.paper1_fixed_days.worker")
    except Exception as exc:
        raise RuntimeError(f"Failed to start or supervise {day_id}/{kind}/{budget}. See {attempt_dir / 'worker.log'}") from exc
    if not outcome["complete"] or not outcome["all_processes_exited"]:
        raise RuntimeError(f"Unconfirmed process exit. Inspect {attempt_dir / 'worker.log'}")
    expected = {"success", "forced_timeout", "late_exit", "cooperative_deadline", "no_incumbent", "invalid_dispatch"}
    if outcome["status"] not in expected:
        raise RuntimeError(f"Worker infrastructure failure ({outcome['status']}). Inspect {attempt_dir / 'worker.log'}")
    payload = outcome.get("result") or {}
    valid = outcome["success"]
    carbon = {}
    if valid:
        check = payload["validation"]
        carbon = dispatch_carbon(payload["dispatch"], payload["dt_hours"], payload["demand_mwh"], check["ens_mwh"], scoring)
    else:
        check = {}
    record = dict(
        identity=identity, day_id=day_id, kind=kind, budget=budget, wall_budget=wall_budget,
        status=outcome["status"], complete=True, feasible=valid, attempt=attempt.as_posix(),
        preload_s=outcome["preload_s"], exit_s=outcome["exit_s"], compute_s=outcome["compute_s"],
        finalization_s=outcome["finalization_s"], stage_seconds=payload.get("stage_seconds", {}),
        preload_metadata=outcome["preload_metadata"], all_processes_exited=True,
        forced=outcome["forced"], solver_status=payload.get("solver_status") or payload.get("mip", {}).get("termination"),
        proven_optimal=bool(valid and payload.get("proven_optimal")), bound=payload.get("bound"),
        gap=payload.get("gap"), cost=check.get("cost"), ens_mwh=check.get("ens_mwh"),
        demand_mwh=payload.get("demand_mwh"), thermal_mwh=carbon.get("thermal_mwh"),
        served_mwh=carbon.get("served_mwh"), co2_direct_t=carbon.get("co2_direct_t"),
        co2_t_per_served_mwh=carbon.get("co2_t_per_served_mwh"),
        imports_mwh=carbon.get("imports_mwh"), exports_mwh=carbon.get("exports_mwh"),
    )
    save_record(result_path(run_dir, day_id, kind, budget), record)
    return record


def run_jobs(run_dir, manifest, *, max_jobs=None, kind=None):
    """Run at most max_jobs new measurements. All terminal observations include failures."""
    if max_jobs is not None and (not isinstance(max_jobs, int) or max_jobs < 1):
        raise ValueError("max_jobs must be a positive integer")
    verify(manifest)
    jobs = iter_jobs(manifest, kind)
    scoring = loss_config()
    with campaign_lock(run_dir):
        pending = [(d, k, b) for d, k, b in jobs if read_job(run_dir, manifest, d["day_id"], k, b) is None]
        selected = pending[:max_jobs]
        for index, (day, role, budget) in enumerate(selected, 1):
            label = f"{day['day_id']} {role} {budget}s"
            print(f"[{index}/{len(selected)}] {label}", flush=True)
            result = _run_one(run_dir, manifest, day, role, budget, scoring)
            print(f"  {result['status']} (exit {result['exit_s']:.2f}s)", flush=True)
    return {"planned": len(jobs), "already_completed": len(jobs) - len(pending),
            "completed_now": len(selected), "remaining": len(pending) - len(selected)}
