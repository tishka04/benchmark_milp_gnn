from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

from src.milp.solve import solve_scenario


def _payload_from_solver_exception(
    scenario_path: Path,
    budget_seconds: float,
    elapsed_seconds: float,
    exc: BaseException,
) -> Optional[Dict[str, Any]]:
    message = str(exc)
    lowered = message.lower()
    if (
        "feasible solution was not found" in lowered
        or "no solution can be loaded" in lowered
    ):
        return {
            "scenario_id": scenario_path.stem,
            "mip": SimpleNamespace(
                objective=float("inf"),
                termination_condition="no_feasible_solution_loaded",
                solver_status="error",
                solve_seconds=float(elapsed_seconds),
            ),
            "lp": SimpleNamespace(
                objective=float("nan"),
                termination_condition="skipped",
                solver_status="skipped",
                solve_seconds=float("nan"),
            ),
            "cost_components": {},
            "detail": None,
            "feasibility_status": "no_feasible_solution_loaded",
            "scenario_classification": "time_constrained_failure",
            "exception_message": message,
            "time_limit_seconds": float(budget_seconds),
        }
    return None


def _hard_timeout_payload(
    scenario_path: Path,
    budget_seconds: float,
    elapsed_seconds: float,
    *,
    termination_condition: str = "hard_time_limit_killed",
    solver_status: str = "killed",
    exception_message: str = "",
) -> Dict[str, Any]:
    return {
        "scenario_id": scenario_path.stem,
        "mip": SimpleNamespace(
            objective=float("inf"),
            termination_condition=str(termination_condition),
            solver_status=str(solver_status),
            solve_seconds=float(elapsed_seconds),
        ),
        "lp": SimpleNamespace(
            objective=float("nan"),
            termination_condition="skipped",
            solver_status="skipped",
            solve_seconds=float("nan"),
        ),
        "cost_components": {},
        "detail": None,
        "feasibility_status": "hard_timeout",
        "scenario_classification": "time_constrained_failure",
        "exception_message": str(exception_message),
        "time_limit_seconds": float(budget_seconds),
    }


def run_constrained_milp_with_guard(
    scenario_path: Path,
    budget_seconds: float,
    *,
    solver_name: str = "appsi_highs",
    tee: bool = False,
    capture_detail: bool = False,
    warm_start_data: Optional[Dict[str, Any]] = None,
    hard_timeout_enabled: bool = False,
    hard_timeout_grace_s: float = 0.5,
) -> Tuple[Dict[str, Any], float]:
    scenario_path = Path(scenario_path)
    budget_seconds = float(budget_seconds)
    t0 = time.perf_counter()

    if not hard_timeout_enabled:
        try:
            payload = solve_scenario(
                scenario_path=scenario_path,
                solver_name=solver_name,
                tee=tee,
                capture_detail=capture_detail,
                warm_start_data=warm_start_data,
                time_limit_seconds=budget_seconds,
            )
        except RuntimeError as exc:
            elapsed = time.perf_counter() - t0
            fallback = _payload_from_solver_exception(scenario_path, budget_seconds, elapsed, exc)
            if fallback is not None:
                return fallback, elapsed
            raise
        elapsed = time.perf_counter() - t0
        return payload, elapsed

    with tempfile.TemporaryDirectory(prefix="strict_tc_milp_") as tmp_dir:
        input_path = Path(tmp_dir) / "input.pkl"
        output_path = Path(tmp_dir) / "output.pkl"
        worker_payload = {
            "scenario_path": str(scenario_path),
            "solver_name": str(solver_name),
            "tee": bool(tee),
            "capture_detail": bool(capture_detail),
            "warm_start_data": warm_start_data,
            "time_limit_seconds": budget_seconds,
        }
        with input_path.open("wb") as handle:
            pickle.dump(worker_payload, handle)

        cmd = [
            sys.executable,
            "-m",
            "src.eval.time_constrained_milp_worker",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=None if tee else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _, stderr_text = proc.communicate(timeout=budget_seconds)
            elapsed = time.perf_counter() - t0
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                _, stderr_text = proc.communicate(timeout=max(0.1, float(hard_timeout_grace_s)))
            except subprocess.TimeoutExpired:
                stderr_text = ""
            elapsed = time.perf_counter() - t0
            return _hard_timeout_payload(
                scenario_path,
                budget_seconds,
                elapsed,
                exception_message=f"Hard MILP timeout after {budget_seconds:.2f}s. {stderr_text.strip()}".strip(),
            ), elapsed

        if not output_path.exists():
            elapsed = time.perf_counter() - t0
            stderr_suffix = f" stderr={stderr_text.strip()}" if stderr_text else ""
            return _hard_timeout_payload(
                scenario_path,
                budget_seconds,
                elapsed,
                termination_condition="subprocess_no_output",
                solver_status="error",
                exception_message=f"MILP subprocess exited without output.{stderr_suffix}",
            ), elapsed

        with output_path.open("rb") as handle:
            message = pickle.load(handle)

        if not bool(message.get("ok", False)):
            elapsed = time.perf_counter() - t0
            error_text = str(message.get("error", "Unknown subprocess error"))
            fallback = _payload_from_solver_exception(
                scenario_path,
                budget_seconds,
                elapsed,
                RuntimeError(error_text),
            )
            if fallback is not None:
                return fallback, elapsed
            return _hard_timeout_payload(
                scenario_path,
                budget_seconds,
                elapsed,
                termination_condition="subprocess_error",
                solver_status="error",
                exception_message=error_text,
            ), elapsed

        elapsed = time.perf_counter() - t0
        payload = dict(message.get("payload", {}) or {})
        return payload, elapsed
