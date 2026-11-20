"""One cold scenario request after static imports, under the paper's EXIT supervisor."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time


def main():
    def event(name, **fields):
        print(json.dumps({"event": name, **fields}), flush=True)

    event("BOOT")
    from src.operational_exit.supervisor import enroll_current_process

    enroll_current_process(json.loads(sys.stdin.readline())["job_name"])
    event("ENROLLED")
    settings = json.loads(sys.stdin.readline())

    import highspy
    from pyomo.contrib.appsi.solvers.highs import Highs  # noqa: F401 - static preload
    from src.milp.model import build_uc_model  # noqa: F401 - static preload
    from src.operational_exit.common import atomic_json, sealed, sha
    from src.operational_exit.solving import solve_data
    from .design import load_day, profiles_sha256, static_sha256

    event("READY", metadata={"highs": highspy.Highs().version(), "threads": settings["threads"]})
    request = json.loads(sys.stdin.readline())
    start = time.perf_counter()
    if sha(request["anchor"]) != request["anchor_sha256"]:
        raise ValueError("Anchor file differs from the prepared pilot.")
    data = load_day(Path(request["anchor"]), request["day_seed"])
    if (profiles_sha256(data) != request["profiles_sha256"]
            or static_sha256(data) != request["static_sha256"]):
        raise ValueError("Loaded day or infrastructure differs from the prepared pilot.")
    demand = float(sum(data.demand.values()) * data.dt_hours)
    loaded = time.perf_counter()
    solve_end = request["deadline"] - settings["reserve_s"]
    if loaded >= solve_end:
        result = {"status": "cooperative_deadline", "valid": False}
    else:
        result = solve_data(data, threads=settings["threads"], seconds=settings["solver_seconds"],
                            seed=request["solver_seed"], deadline=solve_end)
        if result.get("valid"):
            result["status"] = "solution"
        elif "dispatch" in result:
            result["status"] = "invalid_dispatch"
    finished = time.perf_counter()
    # The supervisor accepts only a fully serialized, validated dispatch followed by
    # complete process-tree exit. The model's generic report duplicates the dispatch.
    for field in ("detail", "lp", "lp_duals"):
        result.pop(field, None)
    result.update(
        identity=request["identity"], demand_mwh=demand, dt_hours=data.dt_hours,
        compute_end=finished,
        stage_seconds={
            "load_and_check_day": loaded - start,
            "model_solve_and_validate": finished - loaded,
            "native_solver": result.get("solver_seconds"),
        },
    )
    atomic_json(request["result_file"], sealed(result), deadline=request["deadline"])


if __name__ == "__main__":
    main()
