from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .solve import solve_scenario, SolveSummary

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
    return serializable


def _collect_scenarios(inputs: Iterable[Path]) -> List[Path]:
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
    try:
        report = solve_scenario(
            scenario_path,
            solver_name=solver,
            tee=tee,
            capture_detail=capture_detail,
            export_csv_prefix=export_prefix,
            export_hdf=export_hdf,
        )
        if report_path:
            _ensure_parent(report_path)
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(_serialize_report(report), f, indent=2)

        if plot:
            from .visualize import plot_scenario_and_dispatch

            out_dir = plot_dir or scenario_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_scenario_and_dispatch(scenario_path, report, out_dir)

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
        )
    except Exception as exc:  # pylint: disable=broad-except
        return BatchResult(
            scenario=scenario_path,
            success=False,
            message=str(exc),
            report_path=report_path,
            dispatch_prefix=export_prefix,
            hdf_path=export_hdf,
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
    }

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch MILP scenario runner.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Scenario files, directories, or globs")
    parser.add_argument("--solver", default="highs", help="Solver name (default: highs)")
    parser.add_argument("--reports-dir", type=Path, help="Directory to store JSON reports")
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
    args = parser.parse_args()

    _configure_logging(args.log_file, args.verbose)

    scenarios = _collect_scenarios(args.inputs)
    if not scenarios:
        LOGGER.error("No scenarios found for inputs: %s", ", ".join(str(p) for p in args.inputs))
        raise SystemExit(1)

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

    def submit(executor: ProcessPoolExecutor, scenario_path: Path):
        report_path, export_prefix, hdf_path = _prepare_paths(scenario_path, reports_dir, dispatch_dir, hdf_dir)
        return executor.submit(
            _run_single,
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

    if args.workers > 1:
        LOGGER.info("Running with %d parallel workers", args.workers)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {submit(executor, scenario): scenario for scenario in scenarios}
            completed = 0
            for future in as_completed(future_map):
                result = future.result()
                results.append(result)
                completed += 1
                if result.success:
                    LOGGER.info(
                        "[%d/%d] %s OK (objective=%.2f)",
                        completed,
                        total,
                        result.scenario,
                        result.objective if result.objective is not None else float("nan"),
                    )
                else:
                    LOGGER.error("[%d/%d] %s FAILED: %s", completed, total, result.scenario, result.message)
                    if args.fail_fast:
                        break
    else:
        LOGGER.info("Running sequentially")
        for idx, scenario_path in enumerate(scenarios, start=1):
            report_path, export_prefix, hdf_path = _prepare_paths(scenario_path, reports_dir, dispatch_dir, hdf_dir)
            result = _run_single(
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
            results.append(result)
            if result.success:
                LOGGER.info(
                    "[%d/%d] %s OK (objective=%.2f)",
                    idx,
                    total,
                    scenario_path,
                    result.objective if result.objective is not None else float("nan"),
                )
            else:
                LOGGER.error("[%d/%d] %s FAILED: %s", idx, total, scenario_path, result.message)
                if args.fail_fast:
                    break

    summary = _aggregate_summary(results)
    LOGGER.info("Batch complete: %d success, %d failure", summary.get("success_count", 0), summary.get("failure_count", 0))
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
        _ensure_parent(args.summary_json)
        with args.summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        LOGGER.info("Summary written to %s", args.summary_json)

    if summary.get("failure_count") and args.fail_fast:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
