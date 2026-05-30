from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "time_constrained_eval.ipynb"


class RawExpression(str):
    """Marker for notebook overrides that should be injected verbatim."""


def _render_value(value: Any) -> str:
    if isinstance(value, RawExpression):
        return str(value)
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    return repr(value)


_BRACKET_PAIRS = {"{": "}", "[": "]", "(": ")"}


def _block_end(lines: list, start: int) -> int:
    """If line `start` opens a bracketed block, return the (inclusive) line
    index where it closes; otherwise return start.
    Tracks nesting across {, [, (. Ignores brackets inside string literals.

    IMPORTANT: only scans for an opening bracket within the FIRST line of the
    segment. If line ``start`` is a plain single-line assignment (no opening
    bracket), this returns ``start`` immediately so the caller does not
    accidentally consume subsequent unrelated statements (which may themselves
    contain parentheses, e.g. ``if foo.exists():``).
    """
    first_line = lines[start]
    # Locate an opening bracket on `first_line`, ignoring string literals.
    in_str = None
    open_col = -1
    j = 0
    while j < len(first_line):
        ch = first_line[j]
        if in_str is not None:
            if ch == "\\":
                j += 2
                continue
            if ch == in_str:
                in_str = None
            j += 1
            continue
        if ch in ("'", '"'):
            in_str = ch
            j += 1
            continue
        if ch in _BRACKET_PAIRS:
            open_col = j
            break
        j += 1
    if open_col < 0:
        return start  # plain single-line statement, no block to track

    text = "\n".join(lines[start:])
    i = open_col
    stack = []
    in_str = None  # '\'' or '"' if inside a string literal
    while i < len(text):
        ch = text[i]
        if in_str is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in ("'", '"'):
            in_str = ch
            i += 1
            continue
        if ch in _BRACKET_PAIRS:
            stack.append(_BRACKET_PAIRS[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
            if not stack:
                # Map character offset back to line index.
                consumed = 0
                for offset, ln in enumerate(lines[start:]):
                    consumed += len(ln)
                    if consumed >= i:
                        return start + offset
                    consumed += 1  # newline separator
                return start
        i += 1
    return start


def _apply_overrides(source: str, overrides: Dict[str, Any]) -> str:
    lines = source.splitlines()
    out: list = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        matched = False
        for name, value in overrides.items():
            prefix = f"{name} ="
            if stripped.startswith(prefix):
                indent = line[: len(line) - len(line.lstrip())]
                out.append(f"{indent}{name} = {_render_value(value)}")
                end = _block_end(lines, idx)
                idx = end + 1
                matched = True
                break
        if not matched:
            out.append(line)
            idx += 1
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute time_constrained_eval.ipynb as a plain Python script."
    )
    parser.add_argument("--rerun-milp", action="store_true", help="Force RERUN_TIME_CONSTRAINED_MILP = True")
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
        help="Override the outputs subdirectory under REPO_PATH / outputs.",
    )
    parser.add_argument(
        "--pipeline-output-dir-name",
        type=str,
        default=None,
        help="Override the pipeline outputs subdirectory under REPO_PATH / outputs.",
    )
    parser.add_argument(
        "--hard-timeout",
        action="store_true",
        help="Enable strict subprocess-based hard timeout for constrained MILP runs.",
    )
    parser.add_argument(
        "--hard-timeout-grace-s",
        type=float,
        default=None,
        help="Cleanup grace in seconds after killing a hard-timed-out MILP subprocess.",
    )
    parser.add_argument(
        "--fixed-budgets",
        type=str,
        default=None,
        help="Optional comma-separated override for FIXED_BUDGETS_SECONDS, e.g. '120' or '1,2,5,10,30,60,120'.",
    )
    parser.add_argument(
        "--pipeline-results-filename",
        type=str,
        default="pipeline_eval_all_families_exact_projection.pkl",
        help=(
            "Filename of the pipeline results pickle inside the pipeline output dir. "
            "Use 'pipeline_eval_all_families.pkl' for runs produced by "
            "`src.uc_jepa.evaluation.evaluate_latent_pipeline`."
        ),
    )
    parser.add_argument(
        "--families-json",
        type=str,
        default=None,
        help=(
            "Path to a JSON file with the FAMILIES dict, e.g. "
            "{\"vre_battery\": {\"scenarios_dir\": \"outputs/persona_vre_battery_scenarios\", "
            "\"reports_dir\": \"outputs/persona_vre_battery_scenarios/reports\"}}. "
            "Paths may be absolute, relative to repo root, or use a leading '~'."
        ),
    )
    args = parser.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")

    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]

    overrides: Dict[str, Any] = {
        "RERUN_TIME_CONSTRAINED_MILP": True if args.rerun_milp else False,
        "MAX_SCENARIOS_PER_FAMILY": args.max_scenarios_per_family,
    }
    if args.hard_timeout:
        overrides["MILP_HARD_TIMEOUT_ENABLED"] = True
    if args.hard_timeout_grace_s is not None:
        overrides["MILP_HARD_TIMEOUT_GRACE_SECONDS"] = float(args.hard_timeout_grace_s)
    if args.fixed_budgets:
        values = [float(token.strip()) for token in str(args.fixed_budgets).split(",") if token.strip()]
        overrides["FIXED_BUDGETS_SECONDS"] = values
    if args.families_json:
        spec_path = Path(args.families_json).expanduser()
        if not spec_path.is_absolute():
            spec_path = (REPO_ROOT / spec_path).resolve()
        if not spec_path.exists():
            raise FileNotFoundError(f"--families-json path not found: {spec_path}")
        with spec_path.open("r", encoding="utf-8") as fh:
            families_spec = json.load(fh)
        if not isinstance(families_spec, dict) or not families_spec:
            raise ValueError(f"{spec_path} must contain a non-empty JSON object")
        parts: list = []
        for family, paths in families_spec.items():
            if not isinstance(paths, dict) or "scenarios_dir" not in paths or "reports_dir" not in paths:
                raise ValueError(
                    f"Family '{family}' must specify both 'scenarios_dir' and 'reports_dir'"
                )
            parts.append(
                f"    {family!r}: {{\n"
                f"        'scenarios_dir': Path({str(paths['scenarios_dir'])!r}),\n"
                f"        'reports_dir': Path({str(paths['reports_dir'])!r}),\n"
                f"    }},"
            )
        overrides["FAMILIES"] = RawExpression("{\n" + "\n".join(parts) + "\n}")
    if args.output_dir_name:
        overrides["OUTPUT_DIR"] = RawExpression(f"REPO_PATH / 'outputs' / {args.output_dir_name!r}")
        overrides["TIME_CONSTRAINED_CACHE_PATH"] = RawExpression(
            f"OUTPUT_DIR / 'time_constrained_milp_runs.pkl'"
        )
    if args.pipeline_output_dir_name:
        overrides["PIPELINE_RESULTS_DIR"] = RawExpression(
            f"REPO_PATH / 'outputs' / {args.pipeline_output_dir_name!r}"
        )
        overrides["PIPELINE_RESULTS_PATH"] = RawExpression(
            f"PIPELINE_RESULTS_DIR / {args.pipeline_results_filename!r}"
        )

    ns: Dict[str, Any] = {"__name__": "__main__"}
    try:
        from IPython.display import display  # type: ignore
    except Exception:
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
