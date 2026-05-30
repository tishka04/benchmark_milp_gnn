"""
Subprocess worker for hard time-limit enforcement on RH-MO+LP runs.

Mirrors :mod:`src.eval.time_constrained_milp_worker` but runs the
rolling-horizon merit-order heuristic followed by the cascading LP
worker (Stages 1--3). Invoked by
:func:`src.eval.time_constrained_heuristic.run_constrained_heuristic_with_guard`
when ``hard_timeout_enabled=True`` so that the parent process can
``proc.kill()`` after ``budget_seconds`` and recover a partial / failed
payload, exactly like the MILP variant.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single RH-MO+LP heuristic in a subprocess for hard timeout enforcement.",
    )
    parser.add_argument("--input", required=True, help="Pickle input payload path.")
    parser.add_argument("--output", required=True, help="Pickle output payload path.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        with input_path.open("rb") as handle:
            payload: Dict[str, Any] = pickle.load(handle)

        # We reuse the in-process function from
        # ``src.eval.time_constrained_heuristic`` because it already
        # threads ``candidate_time_budget_s`` into the LP worker so the
        # cascade stops gracefully when the budget is exhausted. The
        # parent's ``proc.kill()`` then only fires as a safety net.
        from src.eval.time_constrained_heuristic import _run_in_process

        scenario_path = Path(payload["scenario_path"])
        family = str(payload.get("family", ""))
        budget_seconds = float(payload.get("budget_seconds", float("inf")))

        result_payload, elapsed = _run_in_process(
            scenario_path,
            family,
            budget_seconds,
            window=int(payload.get("window", 6)),
            solver_name=str(payload.get("solver_name", "appsi_highs")),
            slack_tol_mwh=float(payload.get("slack_tol_mwh", 1.0)),
            deviation_penalty=float(payload.get("deviation_penalty", 10000.0)),
            warm_start=bool(payload.get("warm_start", True)),
            max_stages=int(payload.get("max_stages", 3)),
        )
        output = {"ok": True, "payload": result_payload, "elapsed": float(elapsed)}
    except BaseException as exc:  # pragma: no cover - defensive subprocess wrapper
        output = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(output, handle)


if __name__ == "__main__":
    main()
