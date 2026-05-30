"""
Time-constrained RH-MO+LP evaluation driver.

For each scenario in the configured families and for each fixed /
matched budget, run the rolling-horizon merit-order heuristic followed
by the cascading LP worker under a hard wall-clock budget, and dump a
single pickle / CSV cache compatible with the comparison code already
used for the MILP + Latent-pipeline analyses
(:mod:`src.uc_jepa.evaluation.compare_time_constrained_latent`).

Output schema is row-per-(scenario, budget) with two prefixes:

* ``budget_*`` -- the budget that was applied.
* ``heur_*``   -- the RH-MO+LP outcome under that budget.

Plus scenario metadata (``family``, ``scenario_id``,
``criticality_index``, ``n_zones``, ``n_timesteps``) and the reference
objective taken from the per-scenario MILP report under
``<scenarios_dir>/reports/``.

Example commands are listed at the bottom of this docstring; see also
``--help``.

Reference run on the three evaluation families:

    python -m scripts.run_time_constrained_heuristic \\
        --families eval \\
        --pipeline-results outputs/uc_jepa/latent_time_budget_120_caps_k1/pipeline_eval_all_families.pkl \\
        --output-dir outputs/uc_jepa/latent_vs_grid_milp_120_caps_k1/heuristic \\
        --hard-timeout

Reference run on the three persona families:

    python -m scripts.run_time_constrained_heuristic \\
        --families personas \\
        --pipeline-results outputs/uc_jepa/latent_time_budget_120_personas_k1/pipeline_eval_all_families.pkl \\
        --output-dir outputs/uc_jepa/latent_vs_grid_milp_120_personas_k1/heuristic \\
        --hard-timeout
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# --------------------------------------------------------------------------
# Default family layouts (mirrors ``scripts/build_time_constrained_eval_notebook.py``).
# --------------------------------------------------------------------------
def _eval_families(repo_root: Path) -> Dict[str, Dict[str, Path]]:
    return {
        "low": {
            "scenarios_dir": repo_root / "outputs" / "low_criticality_scenarios",
            "reports_dir": repo_root / "outputs" / "low_criticality_scenarios" / "reports",
        },
        "medium": {
            "scenarios_dir": repo_root / "outputs" / "medium_criticality_scenarios",
            "reports_dir": repo_root / "outputs" / "medium_criticality_scenarios" / "reports",
        },
        "high": {
            "scenarios_dir": repo_root / "outputs" / "high_criticality_scenarios",
            "reports_dir": repo_root / "outputs" / "high_criticality_scenarios" / "reports",
        },
    }


def _persona_families(repo_root: Path) -> Dict[str, Dict[str, Path]]:
    return {
        "vre_battery": {
            "scenarios_dir": repo_root / "outputs" / "persona_vre_battery_scenarios",
            "reports_dir": repo_root / "outputs" / "persona_vre_battery_scenarios" / "reports",
        },
        "network_operator": {
            "scenarios_dir": repo_root / "outputs" / "persona_network_operator_scenarios",
            "reports_dir": repo_root / "outputs" / "persona_network_operator_scenarios" / "reports",
        },
        "mathematician": {
            "scenarios_dir": repo_root / "outputs" / "persona_mathematician_scenarios",
            "reports_dir": repo_root / "outputs" / "persona_mathematician_scenarios" / "reports",
        },
    }


def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if (cand / "src").exists() and (cand / "outputs").exists():
            return cand
    raise RuntimeError("Could not locate benchmark_milp_gnn repository root.")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _load_milp_reports(reports_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not reports_dir.exists():
        return out
    for report_file in sorted(reports_dir.glob("scenario_*.json")):
        try:
            with report_file.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        mip = (report or {}).get("mip", {}) or {}
        out[report_file.stem] = {
            "objective": _safe_float(mip.get("objective", float("nan"))),
            "solve_seconds": _safe_float(mip.get("solve_seconds", float("nan"))),
            "termination": str(mip.get("termination", mip.get("termination_condition", ""))),
        }
    return out


def _load_pipeline_times(pipeline_results_path: Optional[Path]) -> Dict[tuple, float]:
    """Return ``{(family, scenario_id): pipeline_solve_time_seconds}``."""
    if pipeline_results_path is None or not pipeline_results_path.exists():
        return {}
    with pipeline_results_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list):
        return {}
    out: Dict[tuple, float] = {}
    for record in payload:
        if not isinstance(record, dict):
            continue
        family = str(record.get("family", "") or "")
        scenario_id = str(record.get("scenario_id", "") or "")
        if not scenario_id:
            continue
        # `pipeline_eval_all_families` records expose `time_total` (full
        # best-of-N pipeline) as the wall-clock figure used as the
        # "pipeline solve time" in the matched-budget protocol.
        total = _safe_float(record.get("time_total", record.get("pipeline_solve_time", float("nan"))))
        if math.isfinite(total):
            out[(family, scenario_id)] = float(total)
    return out


def _parse_families_arg(
    families_arg: str,
    families_json_path: Optional[Path],
    repo_root: Path,
) -> Dict[str, Dict[str, Path]]:
    if families_json_path is not None:
        with families_json_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return {
            family: {
                "scenarios_dir": Path(entry["scenarios_dir"]),
                "reports_dir": Path(entry.get("reports_dir", Path(entry["scenarios_dir"]) / "reports")),
            }
            for family, entry in raw.items()
        }
    families_arg = (families_arg or "eval").lower()
    if families_arg == "eval":
        return _eval_families(repo_root)
    if families_arg == "personas":
        return _persona_families(repo_root)
    raise ValueError(
        f"Unknown --families value '{families_arg}' "
        "(expected 'eval', 'personas', or pass --families-json)."
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the RH-MO+LP heuristic under a list of wall-clock budgets "
            "on a set of scenario families. Outputs a per-(scenario, budget) "
            "pickle + CSV cache."
        )
    )
    p.add_argument(
        "--families",
        type=str,
        default="eval",
        choices=["eval", "personas"],
        help="Preset family layout. Use --families-json for a custom layout.",
    )
    p.add_argument(
        "--families-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file mapping family name to "
            "{'scenarios_dir': ..., 'reports_dir': ...}. Overrides --families."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the heuristic cache pickle + CSV.",
    )
    p.add_argument(
        "--pipeline-results",
        type=Path,
        default=None,
        help=(
            "Optional path to a pipeline_eval_all_families.pkl. Its per-scenario "
            "wall-clock time is used as the matched budget. If omitted, no "
            "matched-budget row is produced."
        ),
    )
    p.add_argument(
        "--fixed-budgets",
        type=str,
        default="1,2,5,10,30,60,120",
        help="Comma-separated list of fixed budgets in seconds.",
    )
    p.add_argument(
        "--min-matched-budget-seconds",
        type=float,
        default=0.25,
        help="Floor applied to per-scenario matched budgets.",
    )
    p.add_argument("--window", type=int, default=6)
    p.add_argument("--solver", type=str, default="appsi_highs")
    p.add_argument("--slack-tol", type=float, default=1.0)
    p.add_argument("--deviation-penalty", type=float, default=10000.0)
    p.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Skip the pass-through decoder warm-start (debug only).",
    )
    p.add_argument(
        "--max-stages",
        type=int,
        default=3,
        help="LP worker cascade depth. Defaults to the 3-stage cascade.",
    )
    p.add_argument(
        "--hard-timeout",
        action="store_true",
        help="Enforce the budget by killing a subprocess (recommended).",
    )
    p.add_argument(
        "--hard-timeout-grace-s",
        type=float,
        default=0.5,
        help="Extra seconds to wait for subprocess cleanup after SIGKILL.",
    )
    p.add_argument(
        "--max-scenarios-per-family",
        type=int,
        default=None,
        help="Smoke-test cap on scenarios per family.",
    )
    p.add_argument(
        "--families-filter",
        type=str,
        default=None,
        help="Comma-separated subset of family names to run (defaults to all).",
    )
    p.add_argument(
        "--budgets-filter",
        type=str,
        default=None,
        help="Comma-separated list of budget_mode values ('fixed', 'matched_pipeline').",
    )
    return p


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)

    repo_root = _resolve_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    families = _parse_families_arg(args.families, args.families_json, repo_root)
    if args.families_filter:
        keep = {f.strip() for f in args.families_filter.split(",") if f.strip()}
        families = {k: v for k, v in families.items() if k in keep}
        if not families:
            raise SystemExit(f"--families-filter eliminated all families.")

    fixed_budgets = [float(x) for x in args.fixed_budgets.split(",") if x.strip()]

    pipeline_times = _load_pipeline_times(args.pipeline_results)
    if pipeline_times:
        print(f"[info] Loaded {len(pipeline_times)} pipeline wall-clock times "
              f"for matched-budget computation.")
    elif args.pipeline_results is not None:
        print(f"[warn] Could not load pipeline results from {args.pipeline_results}; "
              "no matched_pipeline rows will be produced.")

    budgets_filter = (
        {b.strip() for b in args.budgets_filter.split(",") if b.strip()}
        if args.budgets_filter
        else None
    )
    do_fixed = budgets_filter is None or "fixed" in budgets_filter
    do_matched = (
        budgets_filter is None or "matched_pipeline" in budgets_filter
    ) and bool(pipeline_times)

    output_dir = args.output_dir or (repo_root / "outputs" / "time_constrained_heuristic")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Defer the heavy import until after the family / budget validation
    # so --help is fast.
    from src.eval.time_constrained_heuristic import run_constrained_heuristic_with_guard

    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(x, **_kw):
            return x

    runs: List[Dict[str, Any]] = []

    for family, layout in families.items():
        scenarios_dir = Path(layout["scenarios_dir"])
        reports_dir = Path(layout.get("reports_dir", scenarios_dir / "reports"))
        scenario_paths = sorted(scenarios_dir.glob("scenario_*.json"))
        if args.max_scenarios_per_family is not None:
            scenario_paths = scenario_paths[: int(args.max_scenarios_per_family)]
        if not scenario_paths:
            print(f"[warn] No scenarios found in {scenarios_dir} (skipping {family}).")
            continue

        reports = _load_milp_reports(reports_dir)
        print(
            f"[run] family={family} | scenarios={len(scenario_paths)} "
            f"| budgets fixed={fixed_budgets} | matched={do_matched}"
        )

        for scenario_path in tqdm(scenario_paths, desc=f"RH-MO+LP [{family}]"):
            sc_id = scenario_path.stem
            ref = reports.get(sc_id, {})
            ref_obj = _safe_float(ref.get("objective", float("nan")))
            ref_time = _safe_float(ref.get("solve_seconds", float("nan")))

            budget_jobs: List[tuple] = []
            if do_fixed:
                for budget in fixed_budgets:
                    budget_jobs.append(
                        ("fixed", f"fixed_{budget:g}s", float(budget), "")
                    )
            if do_matched:
                pipe_time = pipeline_times.get((family, sc_id), float("nan"))
                if math.isfinite(pipe_time):
                    matched = max(float(args.min_matched_budget_seconds), float(pipe_time))
                    budget_jobs.append(
                        (
                            "matched_pipeline",
                            "matched_pipeline_solve_time",
                            float(matched),
                            "pipeline_solve_time",
                        )
                    )

            for budget_mode, budget_label, budget_seconds, matched_source in budget_jobs:
                t_wall = time.perf_counter()
                payload, elapsed = run_constrained_heuristic_with_guard(
                    scenario_path,
                    budget_seconds=budget_seconds,
                    family=family,
                    window=int(args.window),
                    solver_name=str(args.solver),
                    slack_tol_mwh=float(args.slack_tol),
                    deviation_penalty=float(args.deviation_penalty),
                    warm_start=not bool(args.no_warm_start),
                    max_stages=int(args.max_stages),
                    hard_timeout_enabled=bool(args.hard_timeout),
                    hard_timeout_grace_s=float(args.hard_timeout_grace_s),
                )
                row = dict(payload)
                row["budget_mode"] = budget_mode
                row["budget_label"] = budget_label
                row["matched_budget_source"] = matched_source
                row["reference_milp_objective"] = ref_obj
                row["reference_milp_solve_time"] = ref_time
                # Gap vs reference MILP (signed: negative = heuristic cheaper)
                obj = row.get("heur_objective", float("nan"))
                gap = float("nan")
                if math.isfinite(obj) and math.isfinite(ref_obj) and abs(ref_obj) > 1e-9:
                    gap = 100.0 * (obj - ref_obj) / abs(ref_obj)
                row["heur_gap_vs_reference_pct"] = gap
                row["heur_abs_gap_vs_reference_pct"] = (
                    abs(gap) if math.isfinite(gap) else float("nan")
                )
                runs.append(row)
                _ = time.perf_counter() - t_wall  # for future diagnostics

    if not runs:
        print("[warn] No runs produced; nothing to save.")
        return

    cache_pkl = output_dir / "time_constrained_heuristic_runs.pkl"
    cache_csv = output_dir / "time_constrained_heuristic_runs.csv"
    with cache_pkl.open("wb") as handle:
        pickle.dump(runs, handle)
    pd.DataFrame(runs).to_csv(cache_csv, index=False)

    summary = {
        "n_rows": len(runs),
        "families": sorted({r["family"] for r in runs}),
        "budgets_used": sorted({r["budget_seconds"] for r in runs}),
        "fixed_budgets_seconds": fixed_budgets,
        "matched_pipeline_used": bool(pipeline_times),
        "hard_timeout": bool(args.hard_timeout),
        "max_stages": int(args.max_stages),
        "warm_start": not bool(args.no_warm_start),
        "cache_pkl": str(cache_pkl),
        "cache_csv": str(cache_csv),
    }
    with (output_dir / "time_constrained_heuristic_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(f"[done] Wrote {len(runs)} rows -> {cache_pkl}")


if __name__ == "__main__":
    main()
