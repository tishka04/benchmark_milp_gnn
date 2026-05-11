from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "evaluation_final_local.ipynb"


class RawExpression(str):
    """Marker for notebook source overrides that should be injected verbatim."""


def _render_value(value: Any) -> str:
    if isinstance(value, RawExpression):
        return str(value)
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    return repr(value)


def _apply_overrides(source: str, overrides: Dict[str, Any]) -> str:
    lines = source.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        for name, value in overrides.items():
            prefix = f"{name} ="
            if stripped.startswith(prefix):
                indent = line[: len(line) - len(line.lstrip())]
                lines[idx] = f"{indent}{name} = {_render_value(value)}"
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute evaluation_final_local.ipynb as a plain Python script."
    )
    parser.add_argument("--run-pipeline", action="store_true", help="Force RUN_PIPELINE = True")
    parser.add_argument(
        "--max-scenarios-per-family",
        type=int,
        default=None,
        help="Override MAX_SCENARIOS_PER_FAMILY in the notebook.",
    )
    parser.add_argument(
        "--output-dir-name",
        type=str,
        default=None,
        help="Override the final outputs subdirectory under REPO_PATH / outputs.",
    )
    args = parser.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")

    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]

    overrides = {
        "RUN_PIPELINE": True if args.run_pipeline else False,
        "MAX_SCENARIOS_PER_FAMILY": args.max_scenarios_per_family,
    }
    if args.output_dir_name:
        overrides["OUTPUT_DIR"] = RawExpression(
            f"REPO_PATH / 'outputs' / {args.output_dir_name!r}"
        )

    ns: Dict[str, Any] = {
        "__name__": "__main__",
    }
    try:
        from IPython.display import display  # type: ignore
    except Exception:  # pragma: no cover - best effort only
        def display(*objects: Any, **_: Any) -> None:
            for obj in objects:
                print(obj)

    ns["display"] = display

    for cell_idx, cell in enumerate(code_cells):
        source = "".join(cell.get("source", []))
        source = _apply_overrides(source, overrides)
        print(f"[runner] Executing code cell {cell_idx + 1}/{len(code_cells)}")
        exec(compile(source, f"{NOTEBOOK_PATH.name}:cell_{cell_idx}", "exec"), ns, ns)


if __name__ == "__main__":
    main()
