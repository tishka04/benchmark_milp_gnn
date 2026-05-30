from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


OUT_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "time_constrained_eval.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells = [
        md(
            """
            # Time-Constrained Evaluation

            This notebook compares a compute-bounded pipeline against time-limited
            MILP baselines under two complementary protocols:

            1. **Fixed budgets**: both methods are evaluated under budget caps
               such as `1s`, `2s`, `5s`, `10s`, ...
            2. **Matched budgets**: each scenario's MILP budget is set equal to
               the observed pipeline wall-clock time on that scenario.

            The main question is not only objective quality, but **time-constrained
            feasibility**:

            - Does the method return a feasible solution within the budget?
            - If yes, how good is that solution relative to the reference MILP run?
            """
        ),
        code(
            """
            import json
            import math
            import os
            import pickle
            import shutil
            import sys
            import time
            from pathlib import Path
            from types import SimpleNamespace
            from typing import Any, Dict, List, Optional, Sequence, Tuple

            _cwd = Path.cwd().resolve()
            _repo_candidates = [_cwd, *_cwd.parents]
            for _cand in list(_repo_candidates):
                _nested = _cand / "benchmark_milp_gnn"
                if (_nested / "src").exists():
                    _repo_candidates.append(_nested)
            for _cand in _repo_candidates:
                if (_cand / "src").exists():
                    if str(_cand) not in sys.path:
                        sys.path.insert(0, str(_cand))
                    break

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from tqdm.auto import tqdm

            from src.eval.metrics import build_comparison_dataframe, load_milp_reports
            from src.eval.time_constrained_milp import (
                run_constrained_milp_with_guard as run_constrained_milp_with_guard_helper,
            )

            sns.set_theme(style="whitegrid")
            sns.set_context("talk", font_scale=0.85)

            print(f"Python cwd: {Path.cwd().resolve()}")
            """
        ),
        md(
            """
            ## 1. Configuration
            """
        ),
        code(
            """
            def find_repo_path(start: Path) -> Path:
                start = start.resolve()
                candidates = [start, *start.parents]
                for cand in candidates:
                    if (cand / "src").exists() and (cand / "outputs").exists():
                        return cand
                    nested = cand / "benchmark_milp_gnn"
                    if (nested / "src").exists() and (nested / "outputs").exists():
                        return nested
                raise RuntimeError("Could not locate benchmark_milp_gnn repository root from current working directory.")


            REPO_PATH = find_repo_path(Path.cwd())
            OUTPUT_DIR = REPO_PATH / "outputs" / "time_constrained_eval"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            PIPELINE_RESULTS_DIR = REPO_PATH / "outputs" / "compute_bounded_pipeline_eval"
            PIPELINE_RESULTS_PATH = PIPELINE_RESULTS_DIR / "pipeline_eval_all_families_exact_projection.pkl"
            if not PIPELINE_RESULTS_PATH.exists():
                alt = REPO_PATH / "outputs" / "compute_bounded_pipeline_eval_smoke" / "pipeline_eval_all_families_exact_projection.pkl"
                if alt.exists():
                    PIPELINE_RESULTS_PATH = alt
                    PIPELINE_RESULTS_DIR = alt.parent
                    print(f"[warn] Falling back to smoke pipeline results: {PIPELINE_RESULTS_PATH}")

            TIME_CONSTRAINED_CACHE_PATH = OUTPUT_DIR / "time_constrained_milp_runs.pkl"
            RERUN_TIME_CONSTRAINED_MILP = False
            CLEAR_TIME_CONSTRAINED_CACHE = False
            MAX_SCENARIOS_PER_FAMILY = None

            FIXED_BUDGETS_SECONDS = [1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
            MATCHED_BUDGET_SOURCE = "pipeline_solve_time"  # or "pipeline_first_candidate_time"
            MIN_MATCHED_BUDGET_SECONDS = 0.25

            SOLVER_NAME = "appsi_highs"
            TEE_SOLVER = False
            MILP_HARD_TIMEOUT_ENABLED = True
            MILP_HARD_TIMEOUT_GRACE_SECONDS = 0.5

            FAMILIES = {
                "low": {
                    "scenarios_dir": REPO_PATH / "outputs" / "low_criticality_scenarios",
                    "reports_dir": REPO_PATH / "outputs" / "low_criticality_scenarios" / "reports",
                },
                "medium": {
                    "scenarios_dir": REPO_PATH / "outputs" / "medium_criticality_scenarios",
                    "reports_dir": REPO_PATH / "outputs" / "medium_criticality_scenarios" / "reports",
                },
                "high": {
                    "scenarios_dir": REPO_PATH / "outputs" / "high_criticality_scenarios",
                    "reports_dir": REPO_PATH / "outputs" / "high_criticality_scenarios" / "reports",
                },
            }

            print(f"REPO_PATH = {REPO_PATH}")
            print(f"PIPELINE_RESULTS_PATH exists: {PIPELINE_RESULTS_PATH.exists()}")
            print(f"OUTPUT_DIR = {OUTPUT_DIR}")
            print(f"FIXED_BUDGETS_SECONDS = {FIXED_BUDGETS_SECONDS}")
            print(f"MATCHED_BUDGET_SOURCE = {MATCHED_BUDGET_SOURCE}")
            print(f"MILP_HARD_TIMEOUT_ENABLED = {MILP_HARD_TIMEOUT_ENABLED}")
            """
        ),
        md(
            """
            ## 2. Helper Functions
            """
        ),
        code(
            """
            def _scenario_key(family: str, scenario_id: str) -> Tuple[str, str]:
                return (str(family), str(scenario_id))


            def _is_finite_number(value: Any) -> bool:
                try:
                    value_f = float(value)
                except (TypeError, ValueError):
                    return False
                return np.isfinite(value_f)


            def _safe_float(value: Any, default: float = float("nan")) -> float:
                try:
                    value_f = float(value)
                except (TypeError, ValueError):
                    return default
                return value_f if np.isfinite(value_f) else default


            def _safe_gap_pct(objective: Any, reference: Any) -> float:
                obj = _safe_float(objective)
                ref = _safe_float(reference)
                if not np.isfinite(obj) or not np.isfinite(ref):
                    return float("nan")
                denom = max(abs(ref), 1e-9)
                return 100.0 * (obj - ref) / denom


            def _termination_str(term: Any) -> str:
                if term is None:
                    return ""
                if hasattr(term, "name"):
                    return str(term.name)
                return str(term)


            def _is_timeout_term(term: Any) -> bool:
                label = _termination_str(term).strip().lower()
                normalized = "".join(ch for ch in label if ch.isalnum())
                return normalized in {
                    "maxtimelimit",
                    "timelimit",
                    "maxtime",
                    "timelimitreached",
                    "maxruntime",
                    "maxiterations",
                    "hardtimelimitkilled",
                }


            def load_pipeline_results(path: Path) -> List[Dict[str, Any]]:
                with Path(path).open("rb") as f:
                    payload = pickle.load(f)
                if not isinstance(payload, list):
                    raise TypeError(f"Expected list payload in {path}, got {type(payload)!r}")
                return payload


            def load_all_reference_milp_reports(families: Dict[str, Dict[str, Path]]) -> Dict[str, Dict[str, Any]]:
                all_reports: Dict[str, Dict[str, Any]] = {}
                for family, paths in families.items():
                    family_reports = load_milp_reports(paths["reports_dir"])
                    for scenario_id, report in family_reports.items():
                        all_reports[f"{family}/{scenario_id}"] = report
                return all_reports


            def select_scenarios(families: Dict[str, Dict[str, Path]], max_per_family: Optional[int] = None) -> List[Dict[str, Any]]:
                selected: List[Dict[str, Any]] = []
                for family, paths in families.items():
                    scenario_paths = sorted(paths["scenarios_dir"].glob("scenario_*.json"))
                    if max_per_family is not None:
                        scenario_paths = scenario_paths[: int(max_per_family)]
                    for scenario_path in scenario_paths:
                        selected.append(
                            {
                                "family": family,
                                "scenario_id": scenario_path.stem,
                                "scenario_path": scenario_path,
                                "reports_dir": paths["reports_dir"],
                            }
                        )
                return selected


            def summarize_milp_run(
                payload: Dict[str, Any],
                *,
                family: str,
                scenario_id: str,
                criticality_index: float,
                reference_objective: float,
                reference_solve_time: float,
                budget_mode: str,
                budget_label: str,
                budget_seconds: float,
                matched_budget_source: str,
                end_to_end_seconds: float,
                pipeline_row: Optional[pd.Series],
            ) -> Dict[str, Any]:
                mip_summary = payload.get("mip")
                lp_summary = payload.get("lp")

                mip_obj_raw = _safe_float(getattr(mip_summary, "objective", float("nan")))
                milp_found_feasible = np.isfinite(mip_obj_raw) and not math.isinf(mip_obj_raw)
                milp_objective = mip_obj_raw if milp_found_feasible else float("nan")
                mip_term = _termination_str(getattr(mip_summary, "termination_condition", ""))
                mip_status = _termination_str(getattr(mip_summary, "solver_status", ""))
                milp_within_budget = bool(np.isfinite(float(end_to_end_seconds)) and float(end_to_end_seconds) <= float(budget_seconds) + 1e-9)
                milp_feasible_within_budget = bool(milp_found_feasible and milp_within_budget)

                row: Dict[str, Any] = {
                    "scenario_id": scenario_id,
                    "family": family,
                    "criticality_index": criticality_index,
                    "budget_mode": budget_mode,
                    "budget_label": budget_label,
                    "budget_seconds": float(budget_seconds),
                    "matched_budget_source": matched_budget_source,
                    "reference_milp_objective": reference_objective,
                    "reference_milp_solve_time": reference_solve_time,
                    "milp_found_feasible": bool(milp_found_feasible),
                    "milp_within_budget": milp_within_budget,
                    "milp_feasible_within_budget": milp_feasible_within_budget,
                    "milp_optimal": mip_term.lower() == "optimal",
                    "milp_timed_out": _is_timeout_term(mip_term),
                    "milp_termination": mip_term,
                    "milp_status": mip_status,
                    "milp_objective": milp_objective,
                    "milp_gap_vs_reference_pct": _safe_gap_pct(milp_objective, reference_objective),
                    "milp_abs_gap_vs_reference_pct": abs(_safe_gap_pct(milp_objective, reference_objective))
                    if np.isfinite(_safe_gap_pct(milp_objective, reference_objective))
                    else float("nan"),
                    "milp_mip_solve_time_s": _safe_float(getattr(mip_summary, "solve_seconds", float("nan"))),
                    "milp_lp_relax_time_s": _safe_float(getattr(lp_summary, "solve_seconds", float("nan"))),
                    "milp_end_to_end_time_s": float(end_to_end_seconds),
                    "milp_feasibility_status": str(payload.get("feasibility_status", "")),
                    "milp_scenario_classification": str(payload.get("scenario_classification", "")),
                    "milp_exception_message": str(payload.get("exception_message", "")),
                }

                if pipeline_row is not None:
                    pipeline_objective = _safe_float(pipeline_row.get("pipeline_objective", float("nan")))
                    pipeline_gap = _safe_gap_pct(pipeline_objective, reference_objective)
                    pipeline_time = _safe_float(pipeline_row.get("pipeline_solve_time", float("nan")))
                    row.update(
                        {
                            "pipeline_found_feasible": bool(np.isfinite(pipeline_objective)),
                            "pipeline_success_flag": bool(pipeline_row.get("success", False)),
                            "pipeline_within_budget": bool(np.isfinite(pipeline_time) and pipeline_time <= float(budget_seconds) + 1e-9),
                            "pipeline_feasible_within_budget": bool(np.isfinite(pipeline_objective) and np.isfinite(pipeline_time) and pipeline_time <= float(budget_seconds) + 1e-9),
                            "pipeline_objective": pipeline_objective,
                            "pipeline_gap_vs_reference_pct": pipeline_gap,
                            "pipeline_abs_gap_vs_reference_pct": abs(pipeline_gap) if np.isfinite(pipeline_gap) else float("nan"),
                            "pipeline_solve_time": pipeline_time,
                            "pipeline_first_candidate_time": _safe_float(pipeline_row.get("pipeline_first_candidate_time", float("nan"))),
                            "pipeline_best_status": str(pipeline_row.get("pipeline_best_status", "")),
                            "pipeline_stage_reached": str(pipeline_row.get("pipeline_stage_reached", "")),
                            "pipeline_n_candidates_evaluated": int(pipeline_row.get("pipeline_n_candidates_evaluated", 0) or 0),
                            "pipeline_n_candidates_projected": int(pipeline_row.get("pipeline_n_candidates_projected", 0) or 0),
                            "pipeline_fallback_used": bool(pipeline_row.get("pipeline_fallback_used", False)),
                        }
                    )
                return row


            def build_budget_summary(
                df: pd.DataFrame,
                budget_mode: str,
            ) -> pd.DataFrame:
                scope = df[df["budget_mode"] == budget_mode].copy()
                if len(scope) == 0:
                    return pd.DataFrame()

                rows: List[Dict[str, Any]] = []
                for family in ["low", "medium", "high", "ALL"]:
                    fam_df = scope if family == "ALL" else scope[scope["family"] == family]
                    if len(fam_df) == 0:
                        continue
                    for budget_seconds, budget_df in fam_df.groupby("budget_seconds", sort=True):
                        both_feasible = budget_df["pipeline_feasible_within_budget"] & budget_df["milp_feasible_within_budget"]
                        pipeline_better = (
                            both_feasible
                            & (budget_df["pipeline_abs_gap_vs_reference_pct"] <= budget_df["milp_abs_gap_vs_reference_pct"])
                        )
                        milp_better = (
                            both_feasible
                            & (budget_df["milp_abs_gap_vs_reference_pct"] < budget_df["pipeline_abs_gap_vs_reference_pct"])
                        )
                        rows.append(
                            {
                                "family": family,
                                "budget_mode": budget_mode,
                                "budget_seconds": float(budget_seconds),
                                "n_scenarios": int(len(budget_df)),
                                "pipeline_feasible_rate_pct": float(100.0 * budget_df["pipeline_feasible_within_budget"].mean()),
                                "milp_feasible_rate_pct": float(100.0 * budget_df["milp_feasible_within_budget"].mean()),
                                "both_feasible_rate_pct": float(100.0 * both_feasible.mean()),
                                "pipeline_only_rate_pct": float(100.0 * (budget_df["pipeline_feasible_within_budget"] & ~budget_df["milp_feasible_within_budget"]).mean()),
                                "milp_only_rate_pct": float(100.0 * (~budget_df["pipeline_feasible_within_budget"] & budget_df["milp_feasible_within_budget"]).mean()),
                                "neither_feasible_rate_pct": float(100.0 * (~budget_df["pipeline_feasible_within_budget"] & ~budget_df["milp_feasible_within_budget"]).mean()),
                                "pipeline_median_abs_gap_pct": float(budget_df.loc[budget_df["pipeline_feasible_within_budget"], "pipeline_abs_gap_vs_reference_pct"].median()),
                                "milp_median_abs_gap_pct": float(budget_df.loc[budget_df["milp_feasible_within_budget"], "milp_abs_gap_vs_reference_pct"].median()),
                                "pipeline_better_gap_rate_pct": float(100.0 * pipeline_better.mean()),
                                "milp_better_gap_rate_pct": float(100.0 * milp_better.mean()),
                                "median_pipeline_time_s": float(budget_df["pipeline_solve_time"].median()),
                                "median_milp_end_to_end_time_s": float(budget_df["milp_end_to_end_time_s"].median()),
                            }
                        )
                return pd.DataFrame(rows).sort_values(["family", "budget_seconds"]).reset_index(drop=True)


            def run_constrained_milp_with_guard(
                scenario_path: Path,
                budget_seconds: float,
            ) -> Tuple[Dict[str, Any], float]:
                return run_constrained_milp_with_guard_helper(
                    scenario_path=scenario_path,
                    budget_seconds=float(budget_seconds),
                    solver_name=SOLVER_NAME,
                    tee=TEE_SOLVER,
                    capture_detail=False,
                    warm_start_data=None,
                    hard_timeout_enabled=MILP_HARD_TIMEOUT_ENABLED,
                    hard_timeout_grace_s=float(MILP_HARD_TIMEOUT_GRACE_SECONDS),
                )
            """
        ),
        md(
            """
            ## 3. Load the Pipeline Results and Reference MILP Reports
            """
        ),
        code(
            """
            if not PIPELINE_RESULTS_PATH.exists():
                raise FileNotFoundError(
                    f"Pipeline results not found: {PIPELINE_RESULTS_PATH}. "
                    "Run the compute-bounded pipeline experiment first."
                )

            pipeline_results = load_pipeline_results(PIPELINE_RESULTS_PATH)
            reference_milp_reports = load_all_reference_milp_reports(FAMILIES)
            pipeline_df = build_comparison_dataframe(pipeline_results, reference_milp_reports)
            pipeline_df["pipeline_found_feasible"] = pipeline_df["pipeline_objective"].replace([np.inf, -np.inf], np.nan).notna()
            pipeline_df["pipeline_abs_gap_vs_reference_pct"] = pipeline_df["cost_gap_pct"].abs()

            pipeline_keyed = {
                _scenario_key(row.family, row.scenario_id): row
                for row in pipeline_df.itertuples(index=False)
            }

            print(f"Loaded pipeline results: {len(pipeline_results)} scenarios")
            print(f"Pipeline result families: {pipeline_df['family'].value_counts().to_dict()}")
            print(f"Reference MILP reports: {len(reference_milp_reports)} family-scenario pairs")

            display(
                pipeline_df[
                    [
                        "family",
                        "scenario_id",
                        "pipeline_solve_time",
                        "pipeline_first_candidate_time",
                        "pipeline_best_status",
                        "pipeline_objective",
                        "cost_gap_pct",
                    ]
                ].head(12)
            )
            """
        ),
        md(
            """
            ## 4. Run or Load the Time-Constrained MILP Experiments
            """
        ),
        code(
            """
            if CLEAR_TIME_CONSTRAINED_CACHE and TIME_CONSTRAINED_CACHE_PATH.exists():
                TIME_CONSTRAINED_CACHE_PATH.unlink()
                print(f"Deleted cache: {TIME_CONSTRAINED_CACHE_PATH}")

            scenario_records = select_scenarios(FAMILIES, max_per_family=MAX_SCENARIOS_PER_FAMILY)
            print(f"Scenario records selected: {len(scenario_records)}")

            if RERUN_TIME_CONSTRAINED_MILP or not TIME_CONSTRAINED_CACHE_PATH.exists():
                constrained_runs: List[Dict[str, Any]] = []

                for record in tqdm(scenario_records, desc="Time-constrained MILP"):
                    family = record["family"]
                    scenario_id = record["scenario_id"]
                    scenario_path = record["scenario_path"]
                    pipeline_row = pipeline_keyed.get(_scenario_key(family, scenario_id))
                    if pipeline_row is None:
                        raise KeyError(f"Missing pipeline result for {family}/{scenario_id}")

                    reference_objective = _safe_float(getattr(pipeline_row, "milp_objective", float("nan")))
                    reference_solve_time = _safe_float(getattr(pipeline_row, "milp_solve_time", float("nan")))
                    criticality_index = _safe_float(getattr(pipeline_row, "criticality_index", float("nan")))

                    budgets: List[Tuple[str, str, float, str]] = []
                    for budget_seconds in FIXED_BUDGETS_SECONDS:
                        budgets.append(("fixed", f"fixed_{budget_seconds:g}s", float(budget_seconds), ""))

                    matched_budget_seconds = max(
                        MIN_MATCHED_BUDGET_SECONDS,
                        _safe_float(getattr(pipeline_row, MATCHED_BUDGET_SOURCE, float("nan")), MIN_MATCHED_BUDGET_SECONDS),
                    )
                    budgets.append(
                        (
                            "matched_pipeline",
                            f"matched_{MATCHED_BUDGET_SOURCE}",
                            float(matched_budget_seconds),
                            MATCHED_BUDGET_SOURCE,
                        )
                    )

                    for budget_mode, budget_label, budget_seconds, matched_budget_source in budgets:
                        payload, end_to_end_seconds = run_constrained_milp_with_guard(
                            scenario_path=scenario_path,
                            budget_seconds=float(budget_seconds),
                        )
                        constrained_runs.append(
                            summarize_milp_run(
                                payload,
                                family=family,
                                scenario_id=scenario_id,
                                criticality_index=criticality_index,
                                reference_objective=reference_objective,
                                reference_solve_time=reference_solve_time,
                                budget_mode=budget_mode,
                                budget_label=budget_label,
                                budget_seconds=budget_seconds,
                                matched_budget_source=matched_budget_source,
                                end_to_end_seconds=end_to_end_seconds,
                                pipeline_row=pd.Series(pipeline_row._asdict()),
                            )
                        )

                with TIME_CONSTRAINED_CACHE_PATH.open("wb") as f:
                    pickle.dump(constrained_runs, f)
                print(f"Saved constrained MILP cache: {TIME_CONSTRAINED_CACHE_PATH}")
            else:
                with TIME_CONSTRAINED_CACHE_PATH.open("rb") as f:
                    constrained_runs = pickle.load(f)
                print(f"Loaded constrained MILP cache: {TIME_CONSTRAINED_CACHE_PATH}")

            tc_df = pd.DataFrame(constrained_runs)
            tc_df.to_csv(OUTPUT_DIR / "time_constrained_milp_runs.csv", index=False)

            print(f"time-constrained rows: {len(tc_df)}")
            print(tc_df["budget_mode"].value_counts().to_dict())
            display(tc_df.head(12))
            """
        ),
        md(
            """
            ## 5. Fixed-Budget Comparison
            """
        ),
        code(
            """
            fixed_summary = build_budget_summary(tc_df, budget_mode="fixed")
            display(fixed_summary.round(3))

            fixed_all = fixed_summary[fixed_summary["family"] == "ALL"].copy()
            fixed_family = fixed_summary[fixed_summary["family"] != "ALL"].copy()

            fig, axes = plt.subplots(2, 2, figsize=(16, 10))

            ax = axes[0, 0]
            ax.plot(fixed_all["budget_seconds"], fixed_all["pipeline_feasible_rate_pct"], marker="o", lw=2.5, label="Pipeline")
            ax.plot(fixed_all["budget_seconds"], fixed_all["milp_feasible_rate_pct"], marker="s", lw=2.5, label="MILP")
            ax.set_xscale("log")
            ax.set_xlabel("Fixed budget (s)", fontweight="bold")
            ax.set_ylabel("Feasible rate (%)", fontweight="bold")
            ax.set_title("(A) Overall Feasible Rate vs Budget", fontweight="bold")
            ax.legend(fontsize=9)

            ax = axes[0, 1]
            for family, fam_color in {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}.items():
                fam_df = fixed_family[fixed_family["family"] == family]
                if len(fam_df) == 0:
                    continue
                ax.plot(
                    fam_df["budget_seconds"],
                    fam_df["milp_feasible_rate_pct"],
                    marker="o",
                    lw=2,
                    color=fam_color,
                    label=f"MILP {family}",
                )
                ax.plot(
                    fam_df["budget_seconds"],
                    fam_df["pipeline_feasible_rate_pct"],
                    marker="s",
                    lw=2,
                    linestyle="--",
                    color=fam_color,
                    label=f"Pipeline {family}",
                )
            ax.set_xscale("log")
            ax.set_xlabel("Fixed budget (s)", fontweight="bold")
            ax.set_ylabel("Feasible rate (%)", fontweight="bold")
            ax.set_title("(B) Feasible Rate by Family", fontweight="bold")
            ax.legend(fontsize=8, ncol=2)

            ax = axes[1, 0]
            ax.plot(fixed_all["budget_seconds"], fixed_all["pipeline_median_abs_gap_pct"], marker="o", lw=2.5, label="Pipeline")
            ax.plot(fixed_all["budget_seconds"], fixed_all["milp_median_abs_gap_pct"], marker="s", lw=2.5, label="MILP")
            ax.set_xscale("log")
            ax.set_xlabel("Fixed budget (s)", fontweight="bold")
            ax.set_ylabel("Median absolute gap vs reference (%)", fontweight="bold")
            ax.set_title("(C) Quality vs Budget on Feasible Instances", fontweight="bold")
            ax.legend(fontsize=9)

            ax = axes[1, 1]
            heatmap_df = (
                fixed_family.assign(feasible_rate_delta_pct=lambda x: x["milp_feasible_rate_pct"] - x["pipeline_feasible_rate_pct"])
                .pivot(index="family", columns="budget_seconds", values="feasible_rate_delta_pct")
                .reindex(index=["low", "medium", "high"])
            )
            sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="coolwarm", center=0.0, ax=ax)
            ax.set_title("(D) MILP - Pipeline Feasible Rate Delta (pp)", fontweight="bold")
            ax.set_xlabel("Fixed budget (s)")
            ax.set_ylabel("")

            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "fig_fixed_budget_comparison.png", dpi=300, bbox_inches="tight")
            plt.show()
            """
        ),
        md(
            """
            ## 6. Matched-Budget Comparison
            """
        ),
        code(
            """
            matched_summary = build_budget_summary(tc_df, budget_mode="matched_pipeline")
            display(matched_summary.round(3))

            matched_df = tc_df[tc_df["budget_mode"] == "matched_pipeline"].copy()

            def _matched_outcome(row: pd.Series) -> str:
                pipe_ok = bool(row["pipeline_feasible_within_budget"])
                milp_ok = bool(row["milp_feasible_within_budget"])
                if pipe_ok and milp_ok:
                    return "both_feasible"
                if pipe_ok and not milp_ok:
                    return "pipeline_only"
                if milp_ok and not pipe_ok:
                    return "milp_only"
                return "neither"


            matched_df["matched_outcome"] = matched_df.apply(_matched_outcome, axis=1)
            matched_counts = (
                matched_df.groupby(["family", "matched_outcome"])
                .size()
                .rename("count")
                .reset_index()
            )
            display(matched_counts)

            fig, axes = plt.subplots(2, 2, figsize=(16, 10))

            ax = axes[0, 0]
            bars_df = matched_summary[matched_summary["family"] != "ALL"].copy()
            x = np.arange(len(bars_df))
            width = 0.36
            ax.bar(x - width / 2, bars_df["pipeline_feasible_rate_pct"], width=width, label="Pipeline", color="#3498db", alpha=0.85)
            ax.bar(x + width / 2, bars_df["milp_feasible_rate_pct"], width=width, label="MILP", color="#e67e22", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(bars_df["family"].str.capitalize())
            ax.set_ylabel("Feasible rate (%)", fontweight="bold")
            ax.set_title("(A) Feasible Rate at Matched Budgets", fontweight="bold")
            ax.legend(fontsize=9)

            ax = axes[0, 1]
            ax.bar(x - width / 2, bars_df["pipeline_median_abs_gap_pct"], width=width, label="Pipeline", color="#3498db", alpha=0.85)
            ax.bar(x + width / 2, bars_df["milp_median_abs_gap_pct"], width=width, label="MILP", color="#e67e22", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(bars_df["family"].str.capitalize())
            ax.set_ylabel("Median absolute gap (%)", fontweight="bold")
            ax.set_title("(B) Quality at Matched Budgets", fontweight="bold")
            ax.legend(fontsize=9)

            ax = axes[1, 0]
            outcome_order = ["both_feasible", "pipeline_only", "milp_only", "neither"]
            outcome_colors = {
                "both_feasible": "#2ecc71",
                "pipeline_only": "#3498db",
                "milp_only": "#e67e22",
                "neither": "#95a5a6",
            }
            stacked = (
                matched_counts.pivot(index="family", columns="matched_outcome", values="count")
                .fillna(0.0)
                .reindex(index=["low", "medium", "high"])
            )
            bottom = np.zeros(len(stacked))
            for outcome in outcome_order:
                vals = stacked.get(outcome, pd.Series(0.0, index=stacked.index)).to_numpy(dtype=float)
                ax.bar(
                    np.arange(len(stacked)),
                    vals,
                    bottom=bottom,
                    color=outcome_colors[outcome],
                    alpha=0.9,
                    label=outcome.replace("_", " "),
                )
                bottom += vals
            ax.set_xticks(np.arange(len(stacked)))
            ax.set_xticklabels([idx.capitalize() for idx in stacked.index])
            ax.set_ylabel("Scenario count", fontweight="bold")
            ax.set_title("(C) Matched-Budget Outcome Breakdown", fontweight="bold")
            ax.legend(fontsize=8)

            ax = axes[1, 1]
            scatter_palette = {
                "both_feasible": "#2ecc71",
                "pipeline_only": "#3498db",
                "milp_only": "#e67e22",
                "neither": "#95a5a6",
            }
            for outcome, sub in matched_df.groupby("matched_outcome"):
                ax.scatter(
                    sub["pipeline_solve_time"],
                    sub["milp_abs_gap_vs_reference_pct"],
                    s=60,
                    alpha=0.8,
                    color=scatter_palette[outcome],
                    edgecolors="white",
                    linewidth=0.6,
                    label=outcome.replace("_", " "),
                )
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_xlabel("Pipeline wall-clock time / matched MILP budget (s)", fontweight="bold")
            ax.set_ylabel("MILP absolute gap vs reference (%)", fontweight="bold")
            ax.set_title("(D) MILP Quality at Pipeline-Matched Budgets", fontweight="bold")
            ax.legend(fontsize=8)

            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "fig_matched_budget_comparison.png", dpi=300, bbox_inches="tight")
            plt.show()
            """
        ),
        md(
            """
            ## 7. Scenario-Level Diagnostics
            """
        ),
        code(
            """
            matched_df = tc_df[tc_df["budget_mode"] == "matched_pipeline"].copy()
            matched_df["matched_outcome"] = matched_df.apply(
                lambda row: (
                    "both_feasible" if row["pipeline_feasible_within_budget"] and row["milp_feasible_within_budget"]
                    else "pipeline_only" if row["pipeline_feasible_within_budget"] and not row["milp_feasible_within_budget"]
                    else "milp_only" if row["milp_feasible_within_budget"] and not row["pipeline_feasible_within_budget"]
                    else "neither"
                ),
                axis=1,
            )

            pipeline_only_cases = matched_df[matched_df["matched_outcome"] == "pipeline_only"].copy()
            milp_only_cases = matched_df[matched_df["matched_outcome"] == "milp_only"].copy()
            both_feasible_cases = matched_df[matched_df["matched_outcome"] == "both_feasible"].copy()
            if len(both_feasible_cases):
                both_feasible_cases["pipeline_minus_milp_abs_gap_pct"] = (
                    both_feasible_cases["pipeline_abs_gap_vs_reference_pct"]
                    - both_feasible_cases["milp_abs_gap_vs_reference_pct"]
                )

            display_cols = [
                "family",
                "scenario_id",
                "criticality_index",
                "budget_seconds",
                "pipeline_solve_time",
                "pipeline_best_status",
                "pipeline_stage_reached",
                "pipeline_abs_gap_vs_reference_pct",
                "milp_feasible_within_budget",
                "milp_termination",
                "milp_abs_gap_vs_reference_pct",
            ]

            print("Pipeline feasible, MILP infeasible at matched budget:")
            display(pipeline_only_cases[display_cols].sort_values(["family", "budget_seconds"], ascending=[True, False]).head(25).round(3))

            print("MILP feasible, pipeline infeasible at matched budget:")
            display(milp_only_cases[display_cols].sort_values(["family", "budget_seconds"], ascending=[True, False]).head(25).round(3))

            if len(both_feasible_cases):
                print("Largest gap disagreements when both are feasible:")
                display(
                    both_feasible_cases[
                        display_cols + ["pipeline_minus_milp_abs_gap_pct"]
                    ]
                    .sort_values("pipeline_minus_milp_abs_gap_pct", ascending=False)
                    .head(25)
                    .round(3)
                )
            """
        ),
        md(
            """
            ## 8. Save Outputs
            """
        ),
        code(
            """
            fixed_summary.to_csv(OUTPUT_DIR / "fixed_budget_summary.csv", index=False)
            matched_summary.to_csv(OUTPUT_DIR / "matched_budget_summary.csv", index=False)
            matched_df.to_csv(OUTPUT_DIR / "matched_budget_per_scenario.csv", index=False)

            pipeline_only_cases.to_csv(OUTPUT_DIR / "matched_budget_pipeline_only_cases.csv", index=False)
            milp_only_cases.to_csv(OUTPUT_DIR / "matched_budget_milp_only_cases.csv", index=False)
            if "both_feasible_cases" in globals():
                both_feasible_cases.to_csv(OUTPUT_DIR / "matched_budget_both_feasible_cases.csv", index=False)

            notebook_summary = {
                "pipeline_results_path": str(PIPELINE_RESULTS_PATH),
                "time_constrained_cache_path": str(TIME_CONSTRAINED_CACHE_PATH),
                "fixed_budgets_seconds": [float(x) for x in FIXED_BUDGETS_SECONDS],
                "matched_budget_source": MATCHED_BUDGET_SOURCE,
                "n_pipeline_results": int(len(pipeline_results)),
                "n_time_constrained_rows": int(len(tc_df)),
            }
            with open(OUTPUT_DIR / "time_constrained_eval_summary.json", "w", encoding="utf-8") as f:
                json.dump(notebook_summary, f, indent=2)

            print(f"Saved outputs to: {OUTPUT_DIR}")
            for path in sorted(OUTPUT_DIR.iterdir()):
                size_kb = path.stat().st_size / 1024
                print(f"  {path.name:45s} {size_kb:9.1f} KB")
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT_PATH.write_text(json.dumps(build_notebook(), indent=2), encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
