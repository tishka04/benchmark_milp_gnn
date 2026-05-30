from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

from src.milp.lp_worker_two_stage import LPWorkerTwoStage
from src.uc_jepa.workers.milp_warm_worker import _serialize_lp_result


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single LP warm-start candidate solve in a subprocess.",
    )
    parser.add_argument("--input", required=True, help="Pickle payload path.")
    parser.add_argument("--output", required=True, help="Pickle result path.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        with input_path.open("rb") as handle:
            payload: Dict[str, Any] = pickle.load(handle)

        worker = LPWorkerTwoStage(
            scenarios_dir=str(payload["scenario_root"]),
            solver_name=str(payload["solver_name"]),
            verbose=bool(payload.get("verbose", False)),
            **dict(payload.get("worker_kwargs", {}) or {}),
        )
        decoder_output = torch.as_tensor(payload["decoder_output"], dtype=torch.float32)
        result = worker.solve(
            scenario_id=str(payload["scenario_id"]),
            decoder_output=decoder_output,
            max_stages=int(payload["max_stages"]),
            force_repair_20=bool(payload.get("force_repair_20", False)),
            candidate_time_budget_s=payload.get("candidate_time_budget_s"),
        )
        output = {"ok": True, "payload": _serialize_lp_result(result)}
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
