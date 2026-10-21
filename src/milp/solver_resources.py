"""Explicit HiGHS thread allocation for reproducible benchmark processes."""
from __future__ import annotations

import os


def configure_highs_threads(solver, solver_name: str) -> dict:
    """Leave existing callers unchanged unless the benchmark requests a cap.

    This configures an allowed thread count; it does not measure effective
    parallelism inside HiGHS. Set the environment before creating a solver.
    """
    raw = os.environ.get("BENCHMARK_HIGHS_THREADS")
    if raw is None:
        return {}
    threads = int(raw)
    if threads < 1:
        raise ValueError("BENCHMARK_HIGHS_THREADS must be a positive integer")
    if "highs" not in solver_name.lower():
        raise ValueError("An explicit HiGHS thread allocation requires HiGHS")
    options = getattr(solver, "highs_options", None)
    if options is None:
        options = solver.options
    options["threads"] = threads
    options["parallel"] = "on" if threads > 1 else "off"
    return {"highs_threads": threads, "highs_parallel": options["parallel"]}
