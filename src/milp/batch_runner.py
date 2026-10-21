from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.demand_response import DR_FORMULATION, validate_dr_report
from .solve import DEFAULT_TIME_LIMIT_SECONDS, solve_scenario, SolveSummary

LOGGER = logging.getLogger("milp.batch")


@dataclass
class BatchResult:
    scenario: Path
    success: bool
    message: str
    report_path: Optional[Path] = None
    dispatch_prefix: Optional[Path] = None
    hdf_path: Optional[Path] = None
    cost_components: Optional[Dict[str, float]] = None
    objective: Optional[float] = None
    solve_seconds: Optional[float] = None
    lp_solve_seconds: Optional[float] = None
    wall_seconds: Optional[float] = None
    reused: bool = False


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _summary_to_dict(summary: SolveSummary) -> Dict[str, Any]:
    return summary.as_dict()


def _serialize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    serializable = {
        "scenario_id": report["scenario_id"],
        "mip": _summary_to_dict(report["mip"]),
        "lp": _summary_to_dict(report["lp"]),
        "cost_components": report["cost_components"],
        "lp_duals": {
            name: {f"{k[0]}|{k[1]}": float(v) for k, v in duals.items()}
            for name, duals in report["lp_duals"].items()
        },
        "periods": report["periods"],
        "zones": report["zones"],
    }
    if report["detail"] is not None:
        serializable["detail"] = report["detail"]
    if "dr_formulation" in report:
        serializable["dr_formulation"] = report["dr_formulation"]
    if "solver_resources" in report:
        serializable["solver_resources"] = report["solver_resources"]
    return serializable


@lru_cache(maxsize=1)
def _solve_code_hash() -> str:
    """Fingerprint the formulation and input/solver code, not launcher/log edits."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in ("demand_response.py", "milp/model.py", "milp/solve.py",
                 "milp/scenario_loader.py", "milp/defaults.py", "milp/profiles.py",
                 "milp/solver_resources.py"):
        digest.update(name.encode("utf-8"))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _provenance(scenario_path: Path, solver: str) -> Dict[str, Any]:
    raw_threads = os.environ.get("BENCHMARK_HIGHS_THREADS")
    return {
        "schema_version": 1,
        "scenario_sha256": hashlib.sha256(scenario_path.read_bytes()).hexdigest(),
        "solve_code_sha256": _solve_code_hash(),
        "dr_formulation": DR_FORMULATION,
        "solver": solver.lower(),
        "highs_threads": int(raw_threads) if raw_threads else None,
        "time_limit_seconds": DEFAULT_TIME_LIMIT_SECONDS,
    }


def _report_hash(payload: Dict[str, Any]) -> str:
    # Also rejects NaN/Infinity anywhere in the report. The checkpoint itself
    # is excluded so its digest can detect damaged or manually edited reports.
    content = {k: v for k, v in payload.items() if k != "batch_checkpoint"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _validate_completed_report(payload: Dict[str, Any]) -> None:
    accepted = {"optimal", "feasible", "maxTimeLimit", "maxIterations"}
    for kind in ("mip", "lp"):
        result = payload[kind]
        if (not math.isfinite(float(result["objective"]))
                or result["termination"] not in accepted
                or result["status"] not in {"ok", "warning", "aborted"}):
            raise ValueError(f"No usable {kind.upper()} solution")
    periods = payload["periods"]
    if not periods or payload["detail"]["time_steps"] != periods:
        raise ValueError("Missing or incomplete dispatch horizon")
    if payload["detail"]["zones"] != payload["zones"]:
        raise ValueError("Dispatch zones do not match the report")
    validate_dr_report(payload, n_timesteps=len(periods))


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_completed_result(scenario_path: Path, report_path: Path, solver: str):
    """A report, not the batch summary, is the per-scenario completion record."""
    if not report_path.is_file():
        return None, "missing report"
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        checkpoint = payload["batch_checkpoint"]
        if checkpoint["complete"] is not True:
            raise ValueError("Incomplete checkpoint")
        if checkpoint["provenance"] != _provenance(scenario_path, solver):
            raise ValueError("Scenario, formulation or solver settings changed")
        if checkpoint["report_sha256"] != _report_hash(payload):
            raise ValueError("Report checksum mismatch")
        _validate_completed_report(payload)
        return BatchResult(
            scenario_path, True, "reused", report_path=report_path,
            cost_components=payload["cost_components"],
            objective=float(payload["mip"]["objective"]),
            solve_seconds=payload["mip"].get("solve_seconds"),
            lp_solve_seconds=payload["lp"].get("solve_seconds"),
            reused=True,
        ), None
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
        return None, f"invalid/incompatible report: {exc}"


def _extract_scenario_number(path: Path) -> int:
    """Extract scenario number from path like 'scenario_00292.json' -> 292."""
    suffix = path.stem[len("scenario_"):]
    try:
        return int(suffix)
    except ValueError:
        return -1


def _collect_scenarios(inputs: Iterable[Path], start_from: int = 1) -> List[Path]:
    """Collect scenarios, optionally filtering to start from a specific number.
    
    Args:
        inputs: Input paths (directories, files, or globs)
        start_from: Scenario number to start from (e.g., 292 to start from scenario_00292)
    """
    scenarios: List[Path] = []
    for inp in inputs:
        if inp.is_dir():
            for path in sorted(inp.glob("scenario_*.json")):
                suffix = path.stem[len("scenario_"):]
                if suffix.isdigit():
                    scenarios.append(path)
        elif any(ch in str(inp) for ch in "*?["):
            scenarios.extend(sorted(inp.parent.glob(inp.name)))
        else:
            if inp.suffix.lower() == ".json" and inp.exists():
                scenarios.append(inp)
    
    # Filter scenarios to start from specified number
    if start_from > 1:
        scenarios = [s for s in scenarios if _extract_scenario_number(s) >= start_from]
        scenarios = sorted(scenarios, key=_extract_scenario_number)
    
    return scenarios


def _run_single(
    scenario_path: Path,
    solver: str,
    tee: bool,
    capture_detail: bool,
    report_path: Optional[Path],
    export_prefix: Optional[Path],
    export_hdf: Optional[Path],
    plot: bool,
    plot_dir: Optional[Path],
) -> BatchResult:
    started = time.perf_counter()
    try:
        provenance = _provenance(scenario_path, solver) if report_path else None
        report = solve_scenario(
            scenario_path,
            solver_name=solver,
            tee=tee,
            capture_detail=capture_detail,
            export_csv_prefix=export_prefix,
            export_hdf=export_hdf,
        )
        if not math.isfinite(float(report["mip"].objective)):
            raise ValueError("No finite MIP incumbent returned")

        if plot:
            from .visualize import plot_scenario_and_dispatch

            out_dir = plot_dir or scenario_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_scenario_and_dispatch(scenario_path, report, out_dir)

        if report_path:
            payload = _serialize_report(report)
            _validate_completed_report(payload)
            if provenance != _provenance(scenario_path, solver):
                raise ValueError("Scenario changed during the solve")
            payload["batch_checkpoint"] = {
                "complete": True, "provenance": provenance,
                "report_sha256": _report_hash(payload),
            }
            # Commit only after MIP, LP and all requested exports succeeded.
            # A killed worker can leave a .tmp file, never a completed report.
            _write_json_atomic(report_path, payload)

        return BatchResult(
            scenario=scenario_path,
            success=True,
            message="ok",
            report_path=report_path,
            dispatch_prefix=export_prefix,
            hdf_path=export_hdf,
            cost_components=report.get("cost_components"),
            objective=float(report["mip"].objective),
            solve_seconds=report["mip"].solve_seconds,
            lp_solve_seconds=report["lp"].solve_seconds,
            wall_seconds=time.perf_counter() - started,
        )
    except Exception as exc:  # pylint: disable=broad-except
        return BatchResult(
            scenario=scenario_path,
            success=False,
            message=str(exc),
            report_path=report_path,
            dispatch_prefix=export_prefix,
            hdf_path=export_hdf,
            wall_seconds=time.perf_counter() - started,
        )


def _prepare_paths(
    scenario_path: Path,
    reports_dir: Optional[Path],
    dispatch_dir: Optional[Path],
    hdf_dir: Optional[Path],
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    stem = scenario_path.stem
    report_path = reports_dir / f"{stem}.json" if reports_dir else None
    export_prefix = (dispatch_dir / stem) if dispatch_dir else None
    hdf_path = (hdf_dir / f"{stem}.h5") if hdf_dir else None
    return report_path, export_prefix, hdf_path


def _configure_logging(log_file: Optional[Path], verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    if log_file:
        _ensure_parent(log_file)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _aggregate_summary(results: List[BatchResult]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "scenarios_total": len(results),
        "success_count": sum(1 for r in results if r.success),
        "failure_count": sum(1 for r in results if not r.success),
        "reused_count": sum(1 for r in results if r.reused),
        "executed_count": sum(1 for r in results if not r.reused),
        "new_success_count": sum(1 for r in results if r.success and not r.reused),
    }
    summary["executed_solve_seconds_total"] = round(sum(
        r.solve_seconds or 0.0 for r in results if not r.reused), 2)
    summary["executed_lp_solve_seconds_total"] = round(sum(
        r.lp_solve_seconds or 0.0 for r in results if not r.reused), 2)
    # These are sums across scenarios, never the elapsed duration of the batch.
    for field in ("lp_solve_seconds", "wall_seconds"):
        values = [getattr(r, field) for r in results if getattr(r, field) is not None]
        if values:
            key = "scenario_wall_seconds_total" if field == "wall_seconds" else f"{field}_total"
            summary[key] = round(float(sum(values)), 2)

    success_results = [r for r in results if r.success and r.objective is not None]
    if success_results:
        objectives = [r.objective for r in success_results if r.objective is not None]
        summary["objective_avg"] = round(float(mean(objectives)), 2)
        summary["objective_min"] = round(float(min(objectives)), 2)
        summary["objective_max"] = round(float(max(objectives)), 2)

        solve_times = [r.solve_seconds for r in success_results if r.solve_seconds is not None]
        if solve_times:
            total_time = sum(solve_times)
            summary["solve_seconds_total"] = round(float(total_time), 2)
            summary["solve_seconds_avg"] = round(float(total_time / len(solve_times)), 2)
            summary["solve_seconds_min"] = round(float(min(solve_times)), 2)
            summary["solve_seconds_max"] = round(float(max(solve_times)), 2)

        cost_totals: Dict[str, float] = {}
        for r in success_results:
            if not r.cost_components:
                continue
            for key, value in r.cost_components.items():
                cost_totals[key] = cost_totals.get(key, 0.0) + float(value)
        if cost_totals:
            summary["cost_totals"] = {k: round(v, 2) for k, v in cost_totals.items()}
            summary["cost_totals_avg"] = {
                k: round(v / len(success_results), 2) for k, v in cost_totals.items()
            }

    failures = [r for r in results if not r.success]
    if failures:
        summary["failed_scenarios"] = {
            str(r.scenario): r.message for r in failures
        }
    return summary


def _stop_executor(executor: ProcessPoolExecutor) -> None:
    """Stop in-flight solves too, rather than waiting for their time limits."""
    terminate_workers = getattr(executor, "terminate_workers", None)
    if terminate_workers is not None:  # Public API on Python 3.14+.
        terminate_workers()
        return
    # Python 3.11 (the campaign environment) has no public hard-stop API.
    # Snapshot before terminating: the executor's manager may clear this map
    # after the first worker exits. HiGHS runs inside these worker processes.
    processes = list((executor._processes or {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    executor.shutdown(wait=True, cancel_futures=True)


def _run_parallel(scenarios, workers, make_job, record_result, fail_fast, started):
    executor = ProcessPoolExecutor(max_workers=workers)
    remaining = iter(scenarios)
    pending = {}
    completed_normally = False

    def fill_available_slots():
        while len(pending) < workers:
            path = next(remaining, None)
            if path is None:
                break
            pending[executor.submit(_run_single, *make_job(path))] = path

    try:
        fill_available_slots()
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            if not done:
                LOGGER.info("%d scenarios in flight; batch elapsed %.1f s",
                            len(pending), time.perf_counter() - started)
                continue
            failed = False
            for future in done:
                path = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = BatchResult(path, False, f"Worker error: {exc}")
                record_result(result)
                failed = failed or not result.success
            # Inspect all completed results before submitting any replacement.
            if failed and fail_fast:
                LOGGER.error("Fail-fast: stopping active workers; no more scenarios will be submitted")
                return
            fill_available_slots()
        completed_normally = True
    finally:
        if completed_normally:
            executor.shutdown(wait=True)
        else:
            for future in pending:
                future.cancel()
            _stop_executor(executor)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch MILP scenario runner.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Scenario files, directories, or globs")
    parser.add_argument("--solver", default="highs", help="Solver name (default: highs)")
    parser.add_argument("--reports-dir", type=Path, help="Directory to store JSON reports")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed compatible JSON reports; retry missing/invalid/failed scenarios")
    parser.add_argument("--dispatch-dir", type=Path, help="Directory for CSV dispatch exports")
    parser.add_argument("--hdf-dir", type=Path, help="Directory for HDF5 exports")
    parser.add_argument("--plot", action="store_true", help="Generate plots for each scenario")
    parser.add_argument("--plots-dir", type=Path, help="Directory for plot images")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes")
    parser.add_argument("--log-file", type=Path, help="Optional log file path")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--tee", action="store_true", help="Stream solver output (per scenario)")
    parser.add_argument("--summary-json", type=Path, help="Optional JSON file for aggregated summary stats")
    parser.add_argument("--start-from", type=int, default=1, 
                        help="Scenario number to start from (e.g., 292 to resume from scenario_00292)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.resume and (not args.reports_dir or args.dispatch_dir or args.hdf_dir or args.plot or args.plots_dir):
        parser.error("--resume requires --reports-dir and supports JSON reports only (no CSV/HDF/plots)")
    started = time.perf_counter()

    _configure_logging(args.log_file, args.verbose)

    scenarios = _collect_scenarios(args.inputs, start_from=args.start_from)
    if not scenarios:
        LOGGER.error("No scenarios found for inputs: %s", ", ".join(str(p) for p in args.inputs))
        raise SystemExit(1)
    if args.reports_dir and len({p.stem for p in scenarios}) != len(scenarios):
        parser.error("Scenario names must be unique within a reports directory")

    if args.start_from > 1:
        LOGGER.info("Resuming from scenario %d - found %d scenarios to process", args.start_from, len(scenarios))
    else:
        LOGGER.info("Found %d scenarios to process", len(scenarios))

    capture_detail = any([args.reports_dir, args.dispatch_dir, args.hdf_dir, args.plot, args.plots_dir])
    reports_dir = args.reports_dir
    dispatch_dir = args.dispatch_dir
    hdf_dir = args.hdf_dir

    if reports_dir:
        reports_dir.mkdir(parents=True, exist_ok=True)
    if dispatch_dir:
        dispatch_dir.mkdir(parents=True, exist_ok=True)
    if hdf_dir:
        hdf_dir.mkdir(parents=True, exist_ok=True)

    results: List[BatchResult] = []
    total = len(scenarios)

    def make_job(scenario_path: Path):
        report_path, export_prefix, hdf_path = _prepare_paths(scenario_path, reports_dir, dispatch_dir, hdf_dir)
        return (
            scenario_path,
            args.solver,
            args.tee,
            capture_detail,
            report_path,
            export_prefix,
            hdf_path,
            args.plot,
            args.plots_dir,
        )

    def record_result(result):
        results.append(result)
        elapsed = time.perf_counter() - started
        if result.success:
            LOGGER.info("[%d/%d] %s OK (objective=%.2f); batch elapsed %.1f s",
                        len(results), total, result.scenario,
                        result.objective if result.objective is not None else float("nan"), elapsed)
        else:
            LOGGER.error("[%d/%d] %s FAILED: %s; batch elapsed %.1f s",
                         len(results), total, result.scenario, result.message, elapsed)

    batch_error = None
    pending_scenarios = []
    try:
        for scenario_path in scenarios:
            if args.resume:
                report_path, _, _ = _prepare_paths(scenario_path, reports_dir, None, None)
                cached, reason = _read_completed_result(scenario_path, report_path, args.solver)
                if cached is not None:
                    results.append(cached)
                    continue
                LOGGER.info("Retry %s: %s", scenario_path.name, reason)
            pending_scenarios.append(scenario_path)
        LOGGER.info("[%d/%d] completed reports reused; %d scenarios to solve",
                    len(results), total, len(pending_scenarios))
        if pending_scenarios and args.workers > 1:
            LOGGER.info("Running with %d parallel workers", args.workers)
            _run_parallel(pending_scenarios, args.workers, make_job, record_result, args.fail_fast, started)
        elif pending_scenarios:
            LOGGER.info("Running sequentially")
            for scenario_path in pending_scenarios:
                result = _run_single(*make_job(scenario_path))
                record_result(result)
                if not result.success and args.fail_fast:
                    break
    except (Exception, KeyboardInterrupt) as exc:
        batch_error = exc
        LOGGER.error("Batch interrupted: %s", exc)

    summary = _aggregate_summary(results)
    summary.update(
        scenarios_requested=total,
        scenarios_unfinished=total - len(results),
        workers=args.workers,
        highs_threads=int(os.environ["BENCHMARK_HIGHS_THREADS"]) if os.environ.get("BENCHMARK_HIGHS_THREADS") else None,
        elapsed_wall_seconds=round(time.perf_counter() - started, 3),
        status="interrupted" if batch_error is not None else ("failed" if summary["failure_count"] else "completed"),
        timing_note="elapsed_wall_seconds covers this invocation only, including report validation and worker shutdown; solve_seconds_total and lp_solve_seconds_total include reused solutions; executed_* totals and scenario_wall_seconds_total cover collected results from this invocation only, not interrupted workers",
    )
    if batch_error is not None:
        summary["error"] = f"{type(batch_error).__name__}: {batch_error}"
    LOGGER.info("Batch complete: %d success, %d failure", summary.get("success_count", 0), summary.get("failure_count", 0))
    LOGGER.info("Batch wall time: %.1f s (%.2f h); %d/%d results collected",
                summary["elapsed_wall_seconds"], summary["elapsed_wall_seconds"] / 3600, len(results), total)
    if "objective_avg" in summary:
        LOGGER.info(
            "Objective avg/min/max: %.2f / %.2f / %.2f",
            summary["objective_avg"], summary["objective_min"], summary["objective_max"],
        )
    if "cost_totals" in summary:
        LOGGER.info("Aggregate cost components: %s", summary["cost_totals"])
        LOGGER.info("Average cost components per scenario: %s", summary["cost_totals_avg"])
    if summary.get("failure_count"):
        LOGGER.warning("Failures: %s", summary.get("failed_scenarios"))

    if args.summary_json:
        _write_json_atomic(args.summary_json, summary)
        LOGGER.info("Summary written to %s", args.summary_json)

    if batch_error is not None:
        raise batch_error
    if summary.get("failure_count") and args.fail_fast:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
