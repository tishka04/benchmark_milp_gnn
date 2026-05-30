"""
Time-constrained RH-MO+LP runner with hard wall-clock enforcement.

This module mirrors :mod:`src.eval.time_constrained_milp` but applies
the budget to the *full* RH-MO+LP path (rolling-horizon merit-order
schedule + pass-through warm-start decoder + cascading LP worker).

Public entry point :func:`run_constrained_heuristic_with_guard` returns
a payload + elapsed seconds pair that summarises one budget-constrained
run on a single scenario, in a schema parallel to
``run_constrained_milp_with_guard`` so that downstream comparison code
can hold MILP, RH-MO+LP and Latent-pipeline results in a single
dataframe (see ``scripts/run_time_constrained_heuristic.py``).

The budget can be enforced either:

* **Soft** -- the LP worker is given ``candidate_time_budget_s =
  budget_seconds``; the heuristic itself is cheap (milliseconds) so the
  LP cascade is the only blocking stage and stops gracefully when the
  budget is exhausted. This is the default.
* **Hard** -- the entire scenario evaluation is forked into a
  subprocess via :mod:`src.eval.time_constrained_heuristic_worker` and
  the parent ``kill()``s it after ``budget_seconds``. This guarantees
  the parent observes at most ``budget_seconds + hard_timeout_grace_s``
  of wall-clock time per scenario, no matter what HiGHS is doing.
"""

from __future__ import annotations

import math
import pickle
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _result_to_payload(
    result_dict: Dict[str, Any],
    *,
    scenario_path: Path,
    family: str,
    budget_seconds: float,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    """Normalise a HeuristicResult dict into the time-constrained schema.

    The returned payload mirrors the MILP cache row but with ``heur_*``
    fields so a downstream merge can keep MILP, heuristic and pipeline
    columns side by side.
    """
    objective = float(result_dict.get("lp_objective", float("nan")))
    feasible = bool(
        result_dict.get("success", False)
        and math.isfinite(objective)
        and not math.isinf(objective)
    )
    within_budget = bool(
        math.isfinite(elapsed_seconds) and elapsed_seconds <= budget_seconds + 1e-9
    )
    feasible_within_budget = bool(feasible and within_budget)

    return {
        "scenario_id": str(scenario_path.stem),
        "family": str(family),
        "criticality_index": float(result_dict.get("criticality_index", 0.0) or 0.0),
        "n_zones": int(result_dict.get("n_zones", 0) or 0),
        "n_timesteps": int(result_dict.get("n_timesteps", 0) or 0),
        "budget_seconds": float(budget_seconds),
        # --- heuristic outcome ----------------------------------------
        "heur_found_feasible": feasible,
        "heur_within_budget": within_budget,
        "heur_feasible_within_budget": feasible_within_budget,
        "heur_objective": objective if feasible else float("nan"),
        "heur_lp_status": str(result_dict.get("lp_status", "")),
        "heur_lp_stage_used": str(result_dict.get("lp_stage_used", "")),
        "heur_lp_stage_reached": str(result_dict.get("lp_stage_reached", "")),
        "heur_slack_mwh": float(result_dict.get("lp_slack", 0.0) or 0.0),
        "heur_n_flips": int(result_dict.get("lp_n_flips", 0) or 0),
        "heur_time_heuristic_s": float(result_dict.get("time_heuristic", 0.0) or 0.0),
        "heur_time_decoder_s": float(result_dict.get("time_decoder", 0.0) or 0.0),
        "heur_time_lp_solve_s": float(result_dict.get("time_lp_solve", 0.0) or 0.0),
        "heur_time_total_s": float(result_dict.get("time_total", 0.0) or 0.0),
        "heur_end_to_end_time_s": float(elapsed_seconds),
        "heur_error_message": str(result_dict.get("error_message", "") or ""),
    }


def _hard_timeout_payload(
    scenario_path: Path,
    family: str,
    budget_seconds: float,
    elapsed_seconds: float,
    *,
    error_message: str = "hard_timeout",
) -> Dict[str, Any]:
    return {
        "scenario_id": str(scenario_path.stem),
        "family": str(family),
        "criticality_index": 0.0,
        "n_zones": 0,
        "n_timesteps": 0,
        "budget_seconds": float(budget_seconds),
        "heur_found_feasible": False,
        "heur_within_budget": False,
        "heur_feasible_within_budget": False,
        "heur_objective": float("nan"),
        "heur_lp_status": "hard_timeout",
        "heur_lp_stage_used": "",
        "heur_lp_stage_reached": "",
        "heur_slack_mwh": float("nan"),
        "heur_n_flips": 0,
        "heur_time_heuristic_s": float("nan"),
        "heur_time_decoder_s": float("nan"),
        "heur_time_lp_solve_s": float("nan"),
        "heur_time_total_s": float(elapsed_seconds),
        "heur_end_to_end_time_s": float(elapsed_seconds),
        "heur_error_message": str(error_message),
    }


def _run_in_process(
    scenario_path: Path,
    family: str,
    budget_seconds: float,
    *,
    window: int,
    solver_name: str,
    slack_tol_mwh: float,
    deviation_penalty: float,
    warm_start: bool,
    max_stages: int,
) -> Tuple[Dict[str, Any], float]:
    """Soft-budget path. Passes ``candidate_time_budget_s`` to the LP
    worker so the cascade stops early when the budget is exhausted."""
    from src.ebm.feasibility import (
        HierarchicalFeasibilityDecoder,
        load_physics_from_scenario,
    )
    from src.heuristics.rolling_horizon import HeuristicConfig, rolling_horizon_heuristic
    from src.heuristics.runner import (
        HeuristicResult,
        _binary_stats,
        _deepest_stage_reached,
        _display_stage_name,
        _extract_stage_flags,
    )
    from src.milp.lp_worker_two_stage import LPWorkerTwoStage
    import json
    import numpy as np

    t_total = time.perf_counter()
    sc_id = scenario_path.stem
    result = HeuristicResult(scenario_id=sc_id, family=family)

    try:
        with scenario_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        result.criticality_index = float(meta.get("criticality_index", 0.0))
    except Exception:
        pass

    try:
        # ── Stage A: rolling-horizon merit-order heuristic ────────────
        t0 = time.perf_counter()
        physics = load_physics_from_scenario(sc_id, str(scenario_path.parent))
        cfg = HeuristicConfig(window=int(window))
        u_heur = rolling_horizon_heuristic(physics, cfg)
        result.time_heuristic = time.perf_counter() - t0
        result.n_zones = int(physics.n_zones)
        result.n_timesteps = int(physics.n_timesteps)
        result.binary_active_fractions = _binary_stats(u_heur)

        # ── Pass-through decoder for LP warm-start ───────────────────
        feasible_plan = None
        if warm_start:
            t0 = time.perf_counter()
            decoder = HierarchicalFeasibilityDecoder(physics)
            feasible_plan = decoder.decode_passthrough(u_heur)
            result.time_decoder = time.perf_counter() - t0

        # ── Stage B: cascading LP worker with soft budget cap ────────
        worker = LPWorkerTwoStage(
            scenarios_dir=str(scenario_path.parent),
            solver_name=solver_name,
            slack_tol_mwh=slack_tol_mwh,
            deviation_penalty=deviation_penalty,
            verbose=False,
        )
        t0 = time.perf_counter()
        # Leave a small slice of the budget for bookkeeping after the LP.
        remaining = max(
            0.05,
            float(budget_seconds) - (time.perf_counter() - t_total),
        )
        lp_res = worker.solve(
            sc_id,
            u_heur,
            feasible_plan=feasible_plan,
            max_stages=int(max_stages),
            candidate_time_budget_s=float(remaining),
        )
        result.time_lp_solve = time.perf_counter() - t0

        stage_used = lp_res.stage_used
        result.lp_status = str(lp_res.status)
        stage_key = (
            stage_used.value if hasattr(stage_used, "value") else str(stage_used)
        )
        result.lp_stage_used = _display_stage_name(lp_res, stage_key)
        result.lp_stage_reached = _deepest_stage_reached(lp_res)
        result.lp_objective = float(lp_res.objective_value)
        result.lp_slack = float(getattr(lp_res, "slack_used", 0.0))
        result.lp_n_flips = int(getattr(lp_res, "n_flips", 0) or 0)
        result.lp_slack_hard_fix = float(getattr(lp_res, "slack_hard_fix", 0.0))
        result.lp_slack_repair_20 = float(getattr(lp_res, "slack_repair_20", 0.0))
        result.lp_slack_repair_100 = float(getattr(lp_res, "slack_repair_100", 0.0))
        result.lp_slack_full_soft = float(getattr(lp_res, "slack_full_soft", 0.0))
        result.lp_slack_round_refix = float(getattr(lp_res, "slack_round_refix", 0.0))
        stage_flags = _extract_stage_flags(lp_res)
        result.lp_reached_hard_fix = stage_flags["hard_fix"]
        result.lp_reached_repair_20 = stage_flags["repair_20"]
        result.lp_reached_repair_100 = stage_flags["repair_100"]
        result.lp_reached_full_soft = stage_flags["full_soft"]
        result.lp_reached_round_refix = stage_flags["round_refix"]
        result.success = bool(np.isfinite(result.lp_objective)) and result.lp_status not in {
            "infeasible", "error", "pending"
        }
    except Exception as exc:
        result.success = False
        result.error_message = f"{type(exc).__name__}: {exc}"

    result.time_total = time.perf_counter() - t_total
    elapsed = result.time_total
    payload = _result_to_payload(
        asdict(result),
        scenario_path=scenario_path,
        family=family,
        budget_seconds=budget_seconds,
        elapsed_seconds=elapsed,
    )
    return payload, elapsed


def run_constrained_heuristic_with_guard(
    scenario_path: Path,
    budget_seconds: float,
    *,
    family: str = "",
    window: int = 6,
    solver_name: str = "appsi_highs",
    slack_tol_mwh: float = 1.0,
    deviation_penalty: float = 10000.0,
    warm_start: bool = True,
    max_stages: int = 3,
    hard_timeout_enabled: bool = False,
    hard_timeout_grace_s: float = 0.5,
) -> Tuple[Dict[str, Any], float]:
    """Run RH-MO+LP on one scenario under a wall-clock budget.

    Returns a ``(payload, elapsed_seconds)`` pair where ``payload`` is
    a flat dict with ``heur_*`` fields plus scenario metadata
    (``family``, ``scenario_id``, ``criticality_index``,
    ``budget_seconds``). The function never raises -- failures are
    reported via ``heur_lp_status`` / ``heur_error_message``.
    """
    scenario_path = Path(scenario_path)
    budget_seconds = float(budget_seconds)

    if not hard_timeout_enabled:
        return _run_in_process(
            scenario_path,
            family,
            budget_seconds,
            window=window,
            solver_name=solver_name,
            slack_tol_mwh=slack_tol_mwh,
            deviation_penalty=deviation_penalty,
            warm_start=warm_start,
            max_stages=max_stages,
        )

    t0 = time.perf_counter()
    # The subprocess pays a fixed startup cost (Python + pyomo + torch
    # imports) that the in-process LP cascade does not. We grant the
    # child a generous absolute floor on top of ``budget_seconds`` so
    # short budgets (1--2 s) are not killed before the LP even starts;
    # the in-process LP worker still observes the original budget via
    # ``candidate_time_budget_s`` so the wall-clock semantics of the
    # heuristic itself remain correct.
    subprocess_floor_s = 15.0
    subprocess_grace = max(float(budget_seconds) + subprocess_floor_s, 2.0 * float(budget_seconds))
    with tempfile.TemporaryDirectory(prefix="strict_tc_heur_") as tmp_dir:
        input_path = Path(tmp_dir) / "input.pkl"
        output_path = Path(tmp_dir) / "output.pkl"
        worker_payload = {
            "scenario_path": str(scenario_path),
            "family": str(family),
            "budget_seconds": float(budget_seconds),
            "window": int(window),
            "solver_name": str(solver_name),
            "slack_tol_mwh": float(slack_tol_mwh),
            "deviation_penalty": float(deviation_penalty),
            "warm_start": bool(warm_start),
            "max_stages": int(max_stages),
        }
        with input_path.open("wb") as handle:
            pickle.dump(worker_payload, handle)

        cmd = [
            sys.executable,
            "-m",
            "src.eval.time_constrained_heuristic_worker",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _, stderr_text = proc.communicate(timeout=subprocess_grace)
            elapsed = time.perf_counter() - t0
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                _, stderr_text = proc.communicate(
                    timeout=max(0.1, float(hard_timeout_grace_s))
                )
            except subprocess.TimeoutExpired:
                stderr_text = ""
            elapsed = time.perf_counter() - t0
            return _hard_timeout_payload(
                scenario_path,
                family,
                budget_seconds,
                elapsed,
                error_message=(
                    f"Hard RH-MO+LP timeout after {budget_seconds:.2f}s. "
                    f"{(stderr_text or '').strip()}"
                ).strip(),
            ), elapsed

        if not output_path.exists():
            elapsed = time.perf_counter() - t0
            return _hard_timeout_payload(
                scenario_path,
                family,
                budget_seconds,
                elapsed,
                error_message=(
                    f"RH-MO+LP subprocess exited without output. "
                    f"stderr={(stderr_text or '').strip()}"
                ).strip(),
            ), elapsed

        with output_path.open("rb") as handle:
            message = pickle.load(handle)

        elapsed = time.perf_counter() - t0
        if not bool(message.get("ok", False)):
            return _hard_timeout_payload(
                scenario_path,
                family,
                budget_seconds,
                elapsed,
                error_message=str(message.get("error", "Unknown subprocess error")),
            ), elapsed

        # The subprocess now returns a fully-formed payload via the
        # ``_run_in_process`` helper, so we just patch the elapsed time
        # to the parent's wall-clock view (which includes subprocess
        # startup overhead).
        payload = dict(message.get("payload", {}) or {})
        if payload:
            payload["budget_seconds"] = float(budget_seconds)
            payload["heur_end_to_end_time_s"] = float(elapsed)
            within = bool(elapsed <= budget_seconds + 1e-9)
            payload["heur_within_budget"] = within
            payload["heur_feasible_within_budget"] = bool(
                within and payload.get("heur_found_feasible", False)
            )
            return payload, elapsed
        # Fallback: malformed subprocess output, treat as hard timeout.
        return _hard_timeout_payload(
            scenario_path,
            family,
            budget_seconds,
            elapsed,
            error_message="Subprocess returned an empty payload.",
        ), elapsed
