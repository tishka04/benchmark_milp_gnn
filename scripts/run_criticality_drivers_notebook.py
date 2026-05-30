"""Execute notebooks/criticality_drivers_analysis.ipynb as a script with overrides.

This produces the analysis artifacts that the CLI tools don't generate:
- ``fixed_budget_headline.csv``, ``fixed_budget_driver_ranking.csv``
- ``fig_advantage_vs_all_stress.png`` / ``fig_winrate_vs_all_stress.png``
- ``fig_advantage_vs_all_hardness.png`` / ``fig_winrate_vs_all_hardness.png``
- ``fig_advantage_vs_budget.png`` / ``fig_winrate_vs_budget.png``
- ``fig_driver_heatmap_vs_budget.png``

Usage::

    python scripts/run_criticality_drivers_notebook.py \\
        --eval-dir outputs/time_constrained_eval_latent_personas \\
        --family-colors persona_vre_battery_scenarios:#2ecc71 \\
                        persona_network_operator_scenarios:#f39c12 \\
                        persona_mathematician_scenarios:#e74c3c

The ``--eval-dir`` is patched into cell 0's ``EVAL_DIR = ...`` assignment, and
``--family-colors`` overrides the ``fam_colors`` dict in cell 10 (the
per-family win-rate / advantage vs budget plot).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "criticality_drivers_analysis.ipynb"


def _patch_eval_dir(source: str, eval_dir_abs: Path) -> str:
    """Replace the ``EVAL_DIR = REPO_PATH / 'outputs' / ...`` line.

    We rewrite the assignment to use an absolute Path so the result is
    independent of REPO_PATH discovery quirks. Idempotent: matches any
    existing ``EVAL_DIR = ...`` single-line assignment.
    """
    pattern = re.compile(r"^EVAL_DIR\s*=.*$", flags=re.MULTILINE)
    replacement = f"EVAL_DIR = Path(r{str(eval_dir_abs)!r})"
    new, n = pattern.subn(replacement, source, count=1)
    if n == 0:
        raise RuntimeError("Did not find an `EVAL_DIR = ...` line to patch.")
    # Make sure `Path` is imported in scope. The notebook already imports Path
    # at the top of cell 0, so this should be a no-op.
    if "from pathlib import Path" not in new and "import pathlib" not in new:
        new = "from pathlib import Path\n" + new
    return new


def _patch_fam_colors(source: str, fam_colors: Dict[str, str]) -> str:
    """Replace the ``fam_colors = {...}`` literal in cell 10."""
    pattern = re.compile(r"fam_colors\s*=\s*\{[^}]*\}")
    rendered = ", ".join(f"{k!r}: {v!r}" for k, v in fam_colors.items())
    replacement = f"fam_colors = {{{rendered}}}"
    new, n = pattern.subn(replacement, source, count=1)
    return new if n else source


def _parse_color_pairs(items):
    out: Dict[str, str] = {}
    if not items:
        return out
    for item in items:
        if ":" not in item:
            raise ValueError(f"--family-colors expects 'family:#hex' pairs, got {item!r}")
        family, color = item.split(":", 1)
        out[family.strip()] = color.strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--eval-dir", required=True, type=Path,
                        help="Directory with the per-scenario CSVs (matched_budget, fixed_budget, components, drivers).")
    parser.add_argument("--family-colors", nargs="*", default=None,
                        help="Optional family:#hex color pairs for the per-family budget plot.")
    args = parser.parse_args()

    eval_dir_abs = args.eval_dir.expanduser()
    if not eval_dir_abs.is_absolute():
        eval_dir_abs = (REPO_ROOT / eval_dir_abs).resolve()
    if not eval_dir_abs.is_dir():
        raise FileNotFoundError(f"--eval-dir does not exist: {eval_dir_abs}")

    fam_colors = _parse_color_pairs(args.family_colors)

    os.environ.setdefault("MPLBACKEND", "Agg")

    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]

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
        if cell_idx == 0:
            source = _patch_eval_dir(source, eval_dir_abs)
        elif "fam_colors" in source and fam_colors:
            source = _patch_fam_colors(source, fam_colors)
        print(f"[runner] Executing code cell {cell_idx + 1}/{len(code_cells)}")
        exec(compile(source, f"{NOTEBOOK_PATH.name}:cell_{cell_idx}", "exec"), ns, ns)

    print(f"\n[runner] Done. Artifacts in: {eval_dir_abs}")


if __name__ == "__main__":
    main()
