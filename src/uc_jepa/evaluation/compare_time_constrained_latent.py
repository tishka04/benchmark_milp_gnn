from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.eval.metrics import build_comparison_dataframe
from src.eval.pipeline_runner import backfill_projection_diagnostics

FAMILY_ORDER = ["low", "medium", "high"]
FAMILY_COLORS = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}
TTF_FAMILY_COLORS = {"low": "#2ecc71", "medium": "#f1c40f", "high": "#e74c3c"}

# Default color palette used when families fall outside the canonical
# low/medium/high layout (e.g. persona buckets).
_DEFAULT_FAMILY_PALETTE = (
    "#2ecc71", "#f39c12", "#e74c3c", "#3498db", "#9b59b6",
    "#1abc9c", "#34495e", "#e67e22", "#16a085", "#c0392b",
)


def _configure_families_from_rows(rows: List[Dict[str, Any]]) -> None:
    """Update the module-level FAMILY_ORDER / FAMILY_COLORS based on the
    families actually observed in the cached MILP rows.

    Known criticality families (low/medium/high) keep their canonical order
    and palette; any unknown families (e.g. ``persona_*``) are appended in
    sorted order with colors drawn from ``_DEFAULT_FAMILY_PALETTE``.
    """
    global FAMILY_ORDER, FAMILY_COLORS, TTF_FAMILY_COLORS
    observed: List[str] = []
    seen = set()
    for row in rows:
        fam = row.get("family") if isinstance(row, dict) else None
        if fam is None or fam in seen:
            continue
        seen.add(fam)
        observed.append(str(fam))
    if not observed:
        return
    canonical = [f for f in ["low", "medium", "high"] if f in seen]
    extras = sorted(f for f in observed if f not in canonical)
    FAMILY_ORDER = canonical + extras
    palette = list(_DEFAULT_FAMILY_PALETTE)
    FAMILY_COLORS = {fam: palette[i % len(palette)] for i, fam in enumerate(FAMILY_ORDER)}
    TTF_FAMILY_COLORS = dict(FAMILY_COLORS)


MATCHED_OUTCOME_ORDER = ["both_feasible", "pipeline_only", "milp_only", "neither"]
MATCHED_OUTCOME_COLORS = {
    "both_feasible": "#2ecc71",
    "pipeline_only": "#3498db",
    "milp_only": "#e67e22",
    "neither": "#95a5a6",
}


def _scenario_key(family: str, scenario_id: str) -> Tuple[str, str]:
    return str(family), str(scenario_id)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _safe_gap_pct(objective: Any, reference: Any) -> float:
    objective_f = _safe_float(objective)
    reference_f = _safe_float(reference)
    if not np.isfinite(objective_f) or not np.isfinite(reference_f):
        return float("nan")
    denom = max(abs(reference_f), 1e-9)
    return 100.0 * (objective_f - reference_f) / denom


def _milp_feasible_within_budget_mask(df: pd.DataFrame) -> pd.Series:
    if "milp_feasible_within_budget" in df.columns:
        return df["milp_feasible_within_budget"].fillna(False).astype(bool)
    return df["milp_found_feasible"].fillna(False).astype(bool)


def _load_cache(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"Expected list in {path}, got {type(data).__name__}")
    return data


def _build_milp_lookup_from_cache(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        family = str(row.get("family", ""))
        scenario_id = str(row.get("scenario_id", ""))
        key = f"{family}/{scenario_id}"
        lookup[key] = {
            "milp_objective": _safe_float(row.get("reference_milp_objective", float("nan"))),
            "milp_solve_time": _safe_float(row.get("reference_milp_solve_time", float("nan"))),
            "milp_status": "reference",
            "milp_termination": "reference",
        }
    return lookup


def _augment_with_pipeline(
    constrained_rows: Sequence[Dict[str, Any]],
    pipeline_df: pd.DataFrame,
) -> pd.DataFrame:
    pipeline_keyed = {
        _scenario_key(row.family, row.scenario_id): row
        for row in pipeline_df.itertuples(index=False)
    }

    augmented: List[Dict[str, Any]] = []
    for row in constrained_rows:
        out = dict(row)
        family = str(out.get("family", ""))
        scenario_id = str(out.get("scenario_id", ""))
        pipeline_row = pipeline_keyed.get(_scenario_key(family, scenario_id))

        reference_objective = _safe_float(out.get("reference_milp_objective", float("nan")))
        budget_seconds = _safe_float(out.get("budget_seconds", float("nan")))

        if pipeline_row is None:
            out.update(
                {
                    "pipeline_found_feasible": False,
                    "pipeline_success_flag": False,
                    "pipeline_within_budget": False,
                    "pipeline_feasible_within_budget": False,
                    "pipeline_objective": float("nan"),
                    "pipeline_gap_vs_reference_pct": float("nan"),
                    "pipeline_abs_gap_vs_reference_pct": float("nan"),
                    "pipeline_solve_time": float("nan"),
                    "pipeline_first_candidate_time": float("nan"),
                    "pipeline_best_status": "",
                    "pipeline_stage_reached": "",
                    "pipeline_n_candidates_evaluated": 0,
                    "pipeline_n_candidates_projected": 0,
                    "pipeline_fallback_used": False,
                }
            )
            augmented.append(out)
            continue

        pipeline_objective = _safe_float(getattr(pipeline_row, "pipeline_objective", float("nan")))
        pipeline_time = _safe_float(getattr(pipeline_row, "pipeline_solve_time", float("nan")))
        pipeline_gap = _safe_gap_pct(pipeline_objective, reference_objective)
        out.update(
            {
                "pipeline_found_feasible": bool(np.isfinite(pipeline_objective)),
                "pipeline_success_flag": bool(getattr(pipeline_row, "success", False)),
                "pipeline_within_budget": bool(np.isfinite(pipeline_time) and pipeline_time <= budget_seconds + 1e-9),
                "pipeline_feasible_within_budget": bool(
                    np.isfinite(pipeline_objective)
                    and np.isfinite(pipeline_time)
                    and pipeline_time <= budget_seconds + 1e-9
                ),
                "pipeline_objective": pipeline_objective,
                "pipeline_gap_vs_reference_pct": pipeline_gap,
                "pipeline_abs_gap_vs_reference_pct": abs(pipeline_gap) if np.isfinite(pipeline_gap) else float("nan"),
                "pipeline_solve_time": pipeline_time,
                "pipeline_first_candidate_time": _safe_float(getattr(pipeline_row, "pipeline_first_candidate_time", float("nan"))),
                "pipeline_best_status": str(getattr(pipeline_row, "pipeline_best_status", "")),
                "pipeline_stage_reached": str(getattr(pipeline_row, "pipeline_stage_reached", "")),
                "pipeline_n_candidates_evaluated": int(getattr(pipeline_row, "pipeline_n_candidates_evaluated", 0) or 0),
                "pipeline_n_candidates_projected": int(getattr(pipeline_row, "pipeline_n_candidates_projected", 0) or 0),
                "pipeline_fallback_used": bool(getattr(pipeline_row, "pipeline_fallback_used", False)),
            }
        )
        augmented.append(out)

    return pd.DataFrame(augmented)


def build_budget_summary(df: pd.DataFrame, budget_mode: str) -> pd.DataFrame:
    scope = df[df["budget_mode"] == budget_mode].copy()
    if len(scope) == 0:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    # Use the dynamically-set FAMILY_ORDER (configured in `compare()` from the
    # actual data) so this works for any family layout, not just low/medium/high.
    for family in FAMILY_ORDER + ["ALL"]:
        fam_df = scope if family == "ALL" else scope[scope["family"] == family]
        if len(fam_df) == 0:
            continue
        for budget_seconds, budget_df in fam_df.groupby("budget_seconds", sort=True):
            milp_ok = _milp_feasible_within_budget_mask(budget_df)
            both_feasible = budget_df["pipeline_feasible_within_budget"] & milp_ok
            pipeline_better = both_feasible & (
                budget_df["pipeline_abs_gap_vs_reference_pct"] <= budget_df["milp_abs_gap_vs_reference_pct"]
            )
            milp_better = both_feasible & (
                budget_df["milp_abs_gap_vs_reference_pct"] < budget_df["pipeline_abs_gap_vs_reference_pct"]
            )
            rows.append(
                {
                    "family": family,
                    "budget_mode": budget_mode,
                    "budget_seconds": float(budget_seconds),
                    "n_scenarios": int(len(budget_df)),
                    "pipeline_feasible_rate_pct": float(100.0 * budget_df["pipeline_feasible_within_budget"].mean()),
                    "milp_feasible_rate_pct": float(100.0 * milp_ok.mean()),
                    "both_feasible_rate_pct": float(100.0 * both_feasible.mean()),
                    "pipeline_only_rate_pct": float(100.0 * (budget_df["pipeline_feasible_within_budget"] & ~milp_ok).mean()),
                    "milp_only_rate_pct": float(100.0 * (~budget_df["pipeline_feasible_within_budget"] & milp_ok).mean()),
                    "neither_feasible_rate_pct": float(100.0 * (~budget_df["pipeline_feasible_within_budget"] & ~milp_ok).mean()),
                    "pipeline_median_abs_gap_pct": float(budget_df.loc[budget_df["pipeline_feasible_within_budget"], "pipeline_abs_gap_vs_reference_pct"].median()),
                    "milp_median_abs_gap_pct": float(budget_df.loc[milp_ok, "milp_abs_gap_vs_reference_pct"].median()),
                    "pipeline_better_gap_rate_pct": float(100.0 * pipeline_better.mean()),
                    "milp_better_gap_rate_pct": float(100.0 * milp_better.mean()),
                    "median_pipeline_time_s": float(budget_df["pipeline_solve_time"].median()),
                    "median_milp_end_to_end_time_s": float(budget_df["milp_end_to_end_time_s"].median()),
                }
            )
    return pd.DataFrame(rows).sort_values(["family", "budget_seconds"]).reset_index(drop=True)


def _matched_outcome(row: pd.Series) -> str:
    pipe_ok = bool(row.get("pipeline_feasible_within_budget", False))
    milp_ok = bool(row.get("milp_feasible_within_budget", row.get("milp_found_feasible", False)))
    if pipe_ok and milp_ok:
        return "both_feasible"
    if pipe_ok and not milp_ok:
        return "pipeline_only"
    if milp_ok and not pipe_ok:
        return "milp_only"
    return "neither"


def _aggregate_matched_by_family(matched_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for family in FAMILY_ORDER + ["ALL"]:
        fam_df = matched_df if family == "ALL" else matched_df[matched_df["family"] == family]
        if len(fam_df) == 0:
            continue
        pipe_ok = fam_df["pipeline_feasible_within_budget"].fillna(False)
        milp_ok = _milp_feasible_within_budget_mask(fam_df)
        rows.append(
            {
                "family": family,
                "n_scenarios": int(len(fam_df)),
                "pipeline_feasible_rate_pct": float(100.0 * pipe_ok.mean()),
                "milp_feasible_rate_pct": float(100.0 * milp_ok.mean()),
                "both_feasible_rate_pct": float(100.0 * (pipe_ok & milp_ok).mean()),
                "pipeline_only_rate_pct": float(100.0 * (pipe_ok & ~milp_ok).mean()),
                "milp_only_rate_pct": float(100.0 * (~pipe_ok & milp_ok).mean()),
                "neither_feasible_rate_pct": float(100.0 * (~pipe_ok & ~milp_ok).mean()),
                "pipeline_median_abs_gap_pct": float(
                    fam_df.loc[pipe_ok, "pipeline_abs_gap_vs_reference_pct"].median()
                ),
                "milp_median_abs_gap_pct": float(
                    fam_df.loc[milp_ok, "milp_abs_gap_vs_reference_pct"].median()
                ),
                "median_pipeline_time_s": float(fam_df["pipeline_solve_time"].median()),
                "median_milp_end_to_end_time_s": float(fam_df["milp_end_to_end_time_s"].median()),
            }
        )
    return pd.DataFrame(rows)


def _plot_fixed_budget_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    fixed_all = summary[summary["family"] == "ALL"].copy()
    fixed_family = summary[summary["family"] != "ALL"].copy()
    if len(fixed_all) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    ax.plot(
        fixed_all["budget_seconds"],
        fixed_all["pipeline_feasible_rate_pct"],
        marker="o",
        lw=2.5,
        label="Pipeline",
    )
    ax.plot(
        fixed_all["budget_seconds"],
        fixed_all["milp_feasible_rate_pct"],
        marker="s",
        lw=2.5,
        label="MILP",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Fixed budget (s)", fontweight="bold")
    ax.set_ylabel("Feasible rate (%)", fontweight="bold")
    ax.set_title("(A) Overall Feasible Rate vs Budget", fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    for family in FAMILY_ORDER:
        fam_df = fixed_family[fixed_family["family"] == family]
        if len(fam_df) == 0:
            continue
        color = FAMILY_COLORS[family]
        ax.plot(
            fam_df["budget_seconds"],
            fam_df["milp_feasible_rate_pct"],
            marker="o",
            lw=2,
            color=color,
            label=f"MILP {family}",
        )
        ax.plot(
            fam_df["budget_seconds"],
            fam_df["pipeline_feasible_rate_pct"],
            marker="s",
            lw=2,
            linestyle="--",
            color=color,
            label=f"Pipeline {family}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Fixed budget (s)", fontweight="bold")
    ax.set_ylabel("Feasible rate (%)", fontweight="bold")
    ax.set_title("(B) Feasible Rate by Family", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 0]
    ax.plot(
        fixed_all["budget_seconds"],
        fixed_all["pipeline_median_abs_gap_pct"],
        marker="o",
        lw=2.5,
        label="Pipeline",
    )
    ax.plot(
        fixed_all["budget_seconds"],
        fixed_all["milp_median_abs_gap_pct"],
        marker="s",
        lw=2.5,
        label="MILP",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Fixed budget (s)", fontweight="bold")
    ax.set_ylabel("Median absolute gap vs reference (%)", fontweight="bold")
    ax.set_title("(C) Quality vs Budget on Feasible Instances", fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    heatmap_df = (
        fixed_family.assign(
            feasible_rate_delta_pct=lambda x: x["milp_feasible_rate_pct"] - x["pipeline_feasible_rate_pct"]
        )
        .pivot(index="family", columns="budget_seconds", values="feasible_rate_delta_pct")
        .reindex(index=FAMILY_ORDER)
    )
    if len(heatmap_df) and np.isfinite(heatmap_df.to_numpy(dtype=float)).any():
        sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="coolwarm", center=0.0, ax=ax)
    else:
        ax.text(0.5, 0.5, "No family heatmap data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("(D) MILP - Pipeline Feasible Rate Delta (pp)", fontweight="bold")
    ax.set_xlabel("Fixed budget (s)")
    ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(output_path / "fig_fixed_budget_comparison_latent.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_matched_budget_comparison(matched_family_summary: pd.DataFrame, matched_df: pd.DataFrame, output_path: Path) -> None:
    if len(matched_family_summary) == 0 or len(matched_df) == 0:
        return

    matched_counts = (
        matched_df.groupby(["family", "matched_outcome"]).size().rename("count").reset_index()
    )
    bars_df = matched_family_summary[matched_family_summary["family"] != "ALL"].copy()
    if len(bars_df) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    x = np.arange(len(bars_df))
    width = 0.36
    ax.bar(
        x - width / 2,
        bars_df["pipeline_feasible_rate_pct"],
        width=width,
        label="Pipeline",
        color="#3498db",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        bars_df["milp_feasible_rate_pct"],
        width=width,
        label="MILP",
        color="#e67e22",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(bars_df["family"].str.capitalize())
    ax.set_ylabel("Feasible rate (%)", fontweight="bold")
    ax.set_title("(A) Feasible Rate at Matched Budgets", fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.bar(
        x - width / 2,
        bars_df["pipeline_median_abs_gap_pct"],
        width=width,
        label="Pipeline",
        color="#3498db",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        bars_df["milp_median_abs_gap_pct"],
        width=width,
        label="MILP",
        color="#e67e22",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(bars_df["family"].str.capitalize())
    ax.set_ylabel("Median absolute gap (%)", fontweight="bold")
    ax.set_title("(B) Quality at Matched Budgets", fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    stacked = (
        matched_counts.pivot(index="family", columns="matched_outcome", values="count")
        .fillna(0.0)
        .reindex(index=FAMILY_ORDER)
    )
    bottom = np.zeros(len(stacked))
    for outcome in MATCHED_OUTCOME_ORDER:
        vals = stacked.get(outcome, pd.Series(0.0, index=stacked.index)).to_numpy(dtype=float)
        ax.bar(
            np.arange(len(stacked)),
            vals,
            bottom=bottom,
            color=MATCHED_OUTCOME_COLORS[outcome],
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
    for outcome, sub in matched_df.groupby("matched_outcome"):
        ax.scatter(
            sub["pipeline_solve_time"],
            sub["milp_abs_gap_vs_reference_pct"],
            s=60,
            alpha=0.8,
            color=MATCHED_OUTCOME_COLORS.get(outcome, "#95a5a6"),
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
    plt.savefig(output_path / "fig_matched_budget_comparison_latent.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_ttf_frame(tc_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    fixed_df = tc_df[tc_df["budget_mode"] == "fixed"].copy()
    matched_df = tc_df[tc_df["budget_mode"] == "matched_pipeline"].copy()
    if len(fixed_df) == 0 or len(matched_df) == 0:
        return pd.DataFrame(), pd.DataFrame(), float("nan")

    def _scenario_ttf(scenario_runs: pd.DataFrame) -> float:
        feas = scenario_runs[_milp_feasible_within_budget_mask(scenario_runs)]
        if len(feas) == 0:
            return float("nan")
        return float(feas["budget_seconds"].min())

    milp_ttf_rows: List[Dict[str, Any]] = []
    for (fam, sid), grp in fixed_df.groupby(["family", "scenario_id"]):
        milp_ttf_rows.append(
            {
                "family": fam,
                "scenario_id": sid,
                "milp_first_feasible_budget_s": _scenario_ttf(grp),
            }
        )
    milp_ttf = pd.DataFrame(milp_ttf_rows)

    matched_merge_df = matched_df.copy()
    matched_merge_df["milp_feasible_at_matched"] = _milp_feasible_within_budget_mask(matched_merge_df)
    ttf_df = milp_ttf.merge(
        matched_merge_df[
            [
                "family",
                "scenario_id",
                "criticality_index",
                "pipeline_solve_time",
                "pipeline_feasible_within_budget",
                "milp_feasible_at_matched",
                "milp_abs_gap_vs_reference_pct",
            ]
        ].rename(
            columns={
                "milp_abs_gap_vs_reference_pct": "milp_abs_gap_at_matched_pct",
            }
        ),
        on=["family", "scenario_id"],
        how="left",
    )

    max_fixed = float(fixed_df["budget_seconds"].max())
    needs_fallback = ttf_df["milp_first_feasible_budget_s"].isna() & ttf_df["milp_feasible_at_matched"].fillna(False)
    ttf_df["milp_ttf_is_right_censored"] = needs_fallback
    ttf_df.loc[needs_fallback, "milp_first_feasible_budget_s"] = (
        ttf_df.loc[needs_fallback, "pipeline_solve_time"].clip(lower=max_fixed)
    )

    ttf_df["pipeline_finishes_before_milp_feasible"] = (
        ttf_df["milp_first_feasible_budget_s"].isna()
        | (ttf_df["pipeline_solve_time"] < ttf_df["milp_first_feasible_budget_s"])
    )
    ttf_df["milp_ttf_to_pipeline_ratio"] = (
        ttf_df["milp_first_feasible_budget_s"] / ttf_df["pipeline_solve_time"].replace(0, np.nan)
    )

    summary = (
        ttf_df.groupby("family")
        .agg(
            n=("scenario_id", "count"),
            milp_never_feasible_at_fixed=("milp_first_feasible_budget_s", lambda s: int(s.isna().sum())),
            right_censored=("milp_ttf_is_right_censored", "sum"),
            milp_ttf_median_s=("milp_first_feasible_budget_s", "median"),
            milp_ttf_p90_s=("milp_first_feasible_budget_s", lambda s: float(s.quantile(0.90))),
            pipeline_time_median_s=("pipeline_solve_time", "median"),
            ratio_median=("milp_ttf_to_pipeline_ratio", "median"),
            pipe_finishes_first_pct=("pipeline_finishes_before_milp_feasible", lambda s: 100.0 * s.mean()),
        )
        .reset_index()
    )
    return ttf_df, summary, max_fixed


def _plot_ttf_comparison(ttf_df: pd.DataFrame, max_fixed: float, output_path: Path) -> None:
    if len(ttf_df) == 0:
        return

    if not np.isfinite(max_fixed):
        max_fixed = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for fam in FAMILY_ORDER:
        sub = ttf_df[ttf_df["family"] == fam].dropna(subset=["milp_first_feasible_budget_s"])
        ax.scatter(
            sub["pipeline_solve_time"],
            sub["milp_first_feasible_budget_s"],
            s=55,
            alpha=0.75,
            color=TTF_FAMILY_COLORS[fam],
            edgecolors="white",
            linewidth=0.6,
            label=f"{fam} (n={len(sub)})",
        )
    finite_x = ttf_df["pipeline_solve_time"].replace([np.inf, -np.inf], np.nan).dropna()
    finite_y = ttf_df["milp_first_feasible_budget_s"].replace([np.inf, -np.inf], np.nan).dropna()
    lim_lo = max(0.5, min(finite_x.min() if len(finite_x) else 1.0, finite_y.min() if len(finite_y) else 1.0))
    lim_hi = max(finite_x.max() if len(finite_x) else 1.0, finite_y.max() if len(finite_y) else 1.0)
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, alpha=0.5, label="x = y (tie)")
    ax.axhline(max_fixed, color="gray", linestyle=":", alpha=0.5, label=f"Right-censor floor ({max_fixed:.0f}s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Pipeline wall-clock time (s)", fontweight="bold")
    ax.set_ylabel("MILP time-to-first-feasible (s, right-censored)", fontweight="bold")
    ax.set_title("(A) MILP TTF vs Pipeline Time -- below diagonal = MILP wins on speed", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1]
    finite = ttf_df.dropna(subset=["milp_ttf_to_pipeline_ratio"])
    for fam in FAMILY_ORDER:
        sub = finite[finite["family"] == fam]
        if len(sub) == 0:
            continue
        ax.hist(
            np.log10(sub["milp_ttf_to_pipeline_ratio"].clip(lower=1e-3)),
            bins=20,
            alpha=0.55,
            color=TTF_FAMILY_COLORS[fam],
            label=fam,
            edgecolor="white",
        )
    ax.axvline(0.0, color="black", linestyle="--", alpha=0.7, label="ratio = 1")
    ax.set_xlabel("log10(MILP TTF / pipeline wall-clock)", fontweight="bold")
    ax.set_ylabel("Scenario count", fontweight="bold")
    ax.set_title("(B) Where the pipeline could in principle win (ratio > 1)", fontweight="bold")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path / "fig_milp_ttf_vs_pipeline_latent.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def compare(
    latent_results_pkl: str,
    time_constrained_cache: str,
    output_dir: str,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    constrained_rows = _load_cache(Path(time_constrained_cache))
    _configure_families_from_rows(constrained_rows)
    milp_lookup = _build_milp_lookup_from_cache(constrained_rows)

    with Path(latent_results_pkl).open("rb") as handle:
        pipeline_results = pickle.load(handle)
    if not isinstance(pipeline_results, list):
        raise TypeError(f"Expected list in {latent_results_pkl}, got {type(pipeline_results).__name__}")
    backfill_projection_diagnostics(pipeline_results)

    pipeline_df = build_comparison_dataframe(pipeline_results, milp_lookup)
    pipeline_df["pipeline_found_feasible"] = pipeline_df["pipeline_objective"].replace([np.inf, -np.inf], np.nan).notna()
    pipeline_df["pipeline_abs_gap_vs_reference_pct"] = pipeline_df["cost_gap_pct"].abs()

    tc_df = _augment_with_pipeline(constrained_rows, pipeline_df)
    tc_df.to_csv(output_path / "time_constrained_latent_runs.csv", index=False)

    written: Dict[str, Any] = {
        "n_pipeline_results": int(len(pipeline_results)),
        "n_time_constrained_rows": int(len(tc_df)),
        "latent_results_pkl": str(Path(latent_results_pkl).resolve()),
        "time_constrained_cache": str(Path(time_constrained_cache).resolve()),
    }

    fixed_summary = pd.DataFrame()
    matched_summary = pd.DataFrame()
    for budget_mode in sorted(tc_df["budget_mode"].dropna().unique().tolist()):
        summary = build_budget_summary(tc_df, budget_mode=budget_mode)
        if len(summary):
            summary.to_csv(output_path / f"{budget_mode}_budget_summary.csv", index=False)
            written[f"{budget_mode}_rows"] = int(len(summary))
            if budget_mode == "fixed":
                fixed_summary = summary
            elif budget_mode == "matched_pipeline":
                matched_summary = summary

    matched_df = tc_df[tc_df["budget_mode"] == "matched_pipeline"].copy()
    if len(matched_df):
        matched_df["matched_outcome"] = matched_df.apply(_matched_outcome, axis=1)
        matched_family_summary = _aggregate_matched_by_family(matched_df)
        matched_df.to_csv(output_path / "matched_budget_per_scenario.csv", index=False)
        matched_summary.to_csv(output_path / "matched_budget_summary.csv", index=False)
        matched_family_summary.to_csv(output_path / "matched_budget_family_summary.csv", index=False)

        pipeline_only_cases = matched_df[matched_df["matched_outcome"] == "pipeline_only"].copy()
        milp_only_cases = matched_df[matched_df["matched_outcome"] == "milp_only"].copy()
        both_feasible_cases = matched_df[matched_df["matched_outcome"] == "both_feasible"].copy()
        if len(both_feasible_cases):
            both_feasible_cases["pipeline_minus_milp_abs_gap_pct"] = (
                both_feasible_cases["pipeline_abs_gap_vs_reference_pct"]
                - both_feasible_cases["milp_abs_gap_vs_reference_pct"]
            )
        pipeline_only_cases.to_csv(output_path / "matched_budget_pipeline_only_cases.csv", index=False)
        milp_only_cases.to_csv(output_path / "matched_budget_milp_only_cases.csv", index=False)
        both_feasible_cases.to_csv(output_path / "matched_budget_both_feasible_cases.csv", index=False)

        _plot_matched_budget_comparison(matched_family_summary, matched_df, output_path)

    fixed_all = tc_df[tc_df["budget_mode"] == "fixed"].copy()
    if len(fixed_all):
        fixed_all.to_csv(output_path / "fixed_budget_per_scenario.csv", index=False)
        fixed_summary.to_csv(output_path / "fixed_budget_summary.csv", index=False)
        _plot_fixed_budget_comparison(fixed_summary, output_path)

    ttf_df, ttf_summary, max_fixed = _build_ttf_frame(tc_df)
    if len(ttf_df):
        ttf_df.to_csv(output_path / "milp_time_to_first_feasible_latent.csv", index=False)
        ttf_summary.to_csv(output_path / "milp_time_to_first_feasible_latent_summary.csv", index=False)
        _plot_ttf_comparison(ttf_df, max_fixed, output_path)
        written["ttf_rows"] = int(len(ttf_df))

    with (output_path / "time_constrained_latent_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(written, handle, indent=2)
    return written


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare latent pipeline results against cached time-constrained MILP runs.")
    parser.add_argument("--latent-results-pkl", required=True, help="pipeline_eval_all_families.pkl from evaluate_latent_pipeline.")
    parser.add_argument("--time-constrained-cache", required=True, help="time_constrained_milp_runs.pkl or .csv from the notebook.")
    parser.add_argument("--output-dir", required=True, help="Directory for comparison CSV/JSON outputs.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    summary = compare(
        latent_results_pkl=args.latent_results_pkl,
        time_constrained_cache=args.time_constrained_cache,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
