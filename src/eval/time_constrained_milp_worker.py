from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from src.milp.solve import solve_scenario


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single exact MILP solve in a subprocess for hard timeout enforcement.",
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

        result = solve_scenario(
            scenario_path=Path(payload["scenario_path"]),
            solver_name=str(payload.get("solver_name", "appsi_highs")),
            tee=bool(payload.get("tee", False)),
            capture_detail=bool(payload.get("capture_detail", False)),
            warm_start_data=payload.get("warm_start_data"),
            time_limit_seconds=payload.get("time_limit_seconds"),
        )
        output = {"ok": True, "payload": result}
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
