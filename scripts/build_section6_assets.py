"""Build the figures + JSON statistics referenced from Section 6 of the paper.

Outputs land in
``outputs/uc_jepa/latent_vs_grid_milp_120_caps_k1/section6/``:

* ``fig_fixed_budget_comparison_3way.png``
* ``fig_matched_budget_comparison_3way.png``
* ``fig_milp_ttf_vs_pipeline_3way.png``
* ``fig_compute_bounded_k_scaling.png``
* ``fig_criticality_scaling.png``
* ``section6_stats.json`` (all scalar facts cited in main.tex)

The script makes no I/O outside these output paths and does not modify any
existing artefact. It only reads the pre-computed pickles / CSVs in
``outputs/uc_jepa/...``.
"""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "outputs" / "uc_jepa" / "latent_vs_grid_milp_120_caps_k1"
HEUR_PKL = EVAL_DIR / "heuristic" / "time_constrained_heuristic_runs.pkl"
PERSONA_HEUR_PKL = REPO / "outputs" / "uc_jepa" / "personas_grid_milp_120_k1" / "heuristic" / "time_constrained_heuristic_runs.pkl"
TC_LATENT_CSV = EVAL_DIR / "time_constrained_latent_runs.csv"
FIXED_HEAD_CSV = EVAL_DIR / "fixed_budget_headline.csv"
FIXED_SUMMARY_CSV = EVAL_DIR / "fixed_budget_summary.csv"
MATCHED_FAMILY_CSV = EVAL_DIR / "matched_budget_family_summary.csv"
MATCHED_PER_SCEN_CSV = EVAL_DIR / "matched_budget_per_scenario.csv"
TTF_CSV = EVAL_DIR / "milp_time_to_first_feasible_latent.csv"
DRIVER_SUMMARY = EVAL_DIR / "driver_summary.json"
PARTIAL_CORR_SUMMARY = EVAL_DIR / "partial_correlation_summary.csv"
K1_PKL = REPO / "outputs" / "uc_jepa" / "latent_time_budget_120_caps_k1" / "pipeline_eval_all_families.pkl"
K5_PKL = REPO / "outputs" / "uc_jepa" / "latent_time_budget_120_caps_k5" / "pipeline_eval_all_families.pkl"
PERSONA_FAMILY_SUM = REPO / "outputs" / "uc_jepa" / "personas_grid_milp_120_k1" / "fixed_budget_summary.csv"
PERSONA_DIRS = {
    "vre_battery": REPO / "outputs" / "persona_vre_battery_scenarios",
    "network_operator": REPO / "outputs" / "persona_network_operator_scenarios",
    "mathematician": REPO / "outputs" / "persona_mathematician_scenarios",
}
PERSONA_COLOURS = {
    "vre_battery": "#1F77B4",
    "network_operator": "#FF7F0E",
    "mathematician": "#9467BD",
}

OUT_DIR = EVAL_DIR / "section6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAMILIES = ["low", "medium", "high"]
FAMILY_COLOURS = {"low": "#4C72B0", "medium": "#DD8452", "high": "#C44E52"}
METHOD_COLOURS = {
    "pipeline": "#2E7D32",   # green
    "milp": "#1565C0",       # blue
    "heuristic": "#C62828",  # red
}
METHOD_LABELS = {
    "pipeline": "Latent pipeline (k=1)",
    "milp": "MILP (HiGHS)",
    "heuristic": "RH-MO+LP",
}
BUDGETS = [1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _abs_gap_pct(obj: float, ref: float) -> float:
    if not (_is_finite(obj) and _is_finite(ref)) or abs(ref) < 1e-9:
        return float("nan")
    return 100.0 * abs(float(obj) - float(ref)) / abs(float(ref))


def _signed_gap_pct(obj: float, ref: float) -> float:
    if not (_is_finite(obj) and _is_finite(ref)) or abs(ref) < 1e-9:
        return float("nan")
    return 100.0 * (float(obj) - float(ref)) / abs(float(ref))


# --------------------------------------------------------------------------
# Aggregation: heuristic per (family, budget)
# --------------------------------------------------------------------------
def aggregate_heuristic(df: pd.DataFrame, families: List[str] | None = None) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    df_fixed = df[df["budget_mode"] == "fixed"].copy()
    fam_list = families if families is not None else FAMILIES
    for family in fam_list:
        sub_fam = df_fixed[df_fixed["family"] == family]
        for budget in BUDGETS:
            sub = sub_fam[np.isclose(sub_fam["budget_seconds"], budget)]
            if sub.empty:
                continue
            n = len(sub)
            feas = sub["heur_feasible_within_budget"].astype(bool)
            feas_rate = float(feas.mean())
            abs_gaps = sub.loc[feas, "heur_abs_gap_vs_reference_pct"].dropna().values
            med_abs_gap = float(np.median(abs_gaps)) if len(abs_gaps) else float("nan")
            mean_abs_gap = float(np.mean(abs_gaps)) if len(abs_gaps) else float("nan")
            elapsed = sub.loc[feas, "heur_end_to_end_time_s"].dropna().values
            med_elapsed = float(np.median(elapsed)) if len(elapsed) else float("nan")
            rows.append(
                {
                    "family": family,
                    "budget_seconds": float(budget),
                    "n": int(n),
                    "n_feasible": int(feas.sum()),
                    "heuristic_feasible_rate": feas_rate,
                    "heuristic_median_abs_gap_pct": med_abs_gap,
                    "heuristic_mean_abs_gap_pct": mean_abs_gap,
                    "heuristic_median_solve_time_s": med_elapsed,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figure 1: 3-method fixed-budget comparison (feasibility + abs gap)
# --------------------------------------------------------------------------
def figure_fixed_budget(
    fixed_summary: pd.DataFrame,
    heur_agg: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), sharex=True)
    for col, family in enumerate(FAMILIES):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        fs = fixed_summary[fixed_summary["family"] == family].sort_values("budget_seconds")
        hs = heur_agg[heur_agg["family"] == family].sort_values("budget_seconds")

        # Feasibility rate panel (CSVs already in percent)
        ax_top.plot(
            fs["budget_seconds"],
            fs["pipeline_feasible_rate_pct"],
            "-o",
            color=METHOD_COLOURS["pipeline"],
            label=METHOD_LABELS["pipeline"],
            linewidth=2.0,
        )
        ax_top.plot(
            fs["budget_seconds"],
            fs["milp_feasible_rate_pct"],
            "-s",
            color=METHOD_COLOURS["milp"],
            label=METHOD_LABELS["milp"],
            linewidth=2.0,
        )
        ax_top.plot(
            hs["budget_seconds"],
            hs["heuristic_feasible_rate"] * 100.0,
            "-^",
            color=METHOD_COLOURS["heuristic"],
            label=METHOD_LABELS["heuristic"],
            linewidth=2.0,
        )
        ax_top.set_xscale("log")
        ax_top.set_ylim(-3, 103)
        ax_top.grid(True, alpha=0.3)
        if col == 0:
            ax_top.set_ylabel("Feasibility rate (%)")

        # Median abs-gap panel (only feasible scenarios contribute)
        # MILP gap is the median |MILP_obj_at_budget - MILP_120s_obj| / |ref|
        # which we read from fixed_budget_summary's 'milp_median_abs_gap_pct'.
        if "milp_median_abs_gap_pct" in fs.columns:
            ax_bot.plot(
                fs["budget_seconds"],
                fs["milp_median_abs_gap_pct"],
                "-s",
                color=METHOD_COLOURS["milp"],
                label=METHOD_LABELS["milp"],
                linewidth=2.0,
            )
        if "pipeline_median_abs_gap_pct" in fs.columns:
            ax_bot.plot(
                fs["budget_seconds"],
                fs["pipeline_median_abs_gap_pct"],
                "-o",
                color=METHOD_COLOURS["pipeline"],
                label=METHOD_LABELS["pipeline"],
                linewidth=2.0,
            )
        ax_bot.plot(
            hs["budget_seconds"],
            hs["heuristic_median_abs_gap_pct"],
            "-^",
            color=METHOD_COLOURS["heuristic"],
            label=METHOD_LABELS["heuristic"],
            linewidth=2.0,
        )
        ax_bot.set_xscale("log")
        ax_bot.set_yscale("symlog", linthresh=0.1)
        ax_bot.grid(True, which="both", alpha=0.3)
        ax_bot.set_xlabel("Budget (s)")
        if col == 0:
            ax_bot.set_ylabel("Median |gap to MILP$_{120s}$| (%)")

    # Single legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2: matched-budget comparison
# --------------------------------------------------------------------------
def figure_matched_budget(
    matched_per_scen: pd.DataFrame,
    heur_df: pd.DataFrame,
    out_path: Path,
) -> Dict[str, Any]:
    """Compare pipeline / MILP / heuristic at the same per-scenario matched budget.

    Returns scalar facts used in the LaTeX prose.
    """
    heur_matched = heur_df[heur_df["budget_mode"] == "matched_pipeline"][
        [
            "family",
            "scenario_id",
            "budget_seconds",
            "heur_feasible_within_budget",
            "heur_objective",
            "heur_end_to_end_time_s",
            "reference_milp_objective",
        ]
    ].rename(columns={"budget_seconds": "matched_budget_s"})

    merged = matched_per_scen.merge(
        heur_matched,
        on=["family", "scenario_id"],
        how="left",
        suffixes=("", "_h"),
    )

    facts: Dict[str, Any] = {"by_family": {}}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # --- Panel 1: feasibility rates by family × method --------------------
    ax = axes[0]
    width = 0.27
    x = np.arange(len(FAMILIES))
    feas = {"pipeline": [], "milp": [], "heuristic": []}
    for fam in FAMILIES:
        sub = merged[merged["family"] == fam]
        n = len(sub)
        pipe_feas = float((sub.get("pipeline_feasible_within_budget", pd.Series(dtype=bool)).fillna(False).astype(bool)).mean()) if n else 0.0
        milp_feas = float((sub.get("milp_feasible_within_budget", pd.Series(dtype=bool)).fillna(False).astype(bool)).mean()) if n else 0.0
        heur_feas = float(sub["heur_feasible_within_budget"].fillna(False).astype(bool).mean()) if n else 0.0
        feas["pipeline"].append(pipe_feas * 100.0)
        feas["milp"].append(milp_feas * 100.0)
        feas["heuristic"].append(heur_feas * 100.0)
        facts["by_family"][fam] = {
            "n": int(n),
            "pipeline_feasible_pct": pipe_feas * 100.0,
            "milp_feasible_pct": milp_feas * 100.0,
            "heuristic_feasible_pct": heur_feas * 100.0,
            "matched_budget_median_s": float(sub["matched_budget_s"].median()) if n else float("nan"),
        }
    for i, m in enumerate(["pipeline", "milp", "heuristic"]):
        ax.bar(
            x + (i - 1) * width,
            feas[m],
            width=width,
            color=METHOD_COLOURS[m],
            label=METHOD_LABELS[m],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in FAMILIES])
    ax.set_ylabel("Feasibility rate at matched budget (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="upper right")

    # --- Panel 2: median |gap to MILP_120s| on scenarios where the method is feasible
    ax2 = axes[1]
    per_method_gap = {"pipeline": [], "milp": [], "heuristic": []}
    for fam in FAMILIES:
        sub = merged[merged["family"] == fam]
        # gaps already in matched_budget_per_scenario for pipeline / milp
        if "pipeline_abs_gap_vs_reference_pct" in sub.columns:
            g_p = sub.loc[sub.get("pipeline_feasible_within_budget", False).astype(bool), "pipeline_abs_gap_vs_reference_pct"].dropna()
        else:
            g_p = pd.Series(dtype=float)
        if "milp_abs_gap_vs_reference_pct" in sub.columns:
            g_m = sub.loc[sub.get("milp_feasible_within_budget", False).astype(bool), "milp_abs_gap_vs_reference_pct"].dropna()
        else:
            g_m = pd.Series(dtype=float)
        feas_h = sub["heur_feasible_within_budget"].fillna(False).astype(bool)
        gap_h_vals = []
        for _, r in sub[feas_h].iterrows():
            gap_h_vals.append(_abs_gap_pct(r["heur_objective"], r["reference_milp_objective"]))
        gap_h_vals = [v for v in gap_h_vals if _is_finite(v)]
        per_method_gap["pipeline"].append(float(np.median(g_p)) if len(g_p) else float("nan"))
        per_method_gap["milp"].append(float(np.median(g_m)) if len(g_m) else float("nan"))
        per_method_gap["heuristic"].append(float(np.median(gap_h_vals)) if gap_h_vals else float("nan"))
        facts["by_family"][fam].update(
            {
                "pipeline_median_abs_gap_pct": per_method_gap["pipeline"][-1],
                "milp_median_abs_gap_pct": per_method_gap["milp"][-1],
                "heuristic_median_abs_gap_pct": per_method_gap["heuristic"][-1],
            }
        )
    for i, m in enumerate(["pipeline", "milp", "heuristic"]):
        vals = per_method_gap[m]
        ax2.bar(
            x + (i - 1) * width,
            [v if _is_finite(v) else 0 for v in vals],
            width=width,
            color=METHOD_COLOURS[m],
            label=METHOD_LABELS[m],
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels([f.capitalize() for f in FAMILIES])
    ax2.set_ylabel("Median |gap to MILP$_{120s}$| (%)")
    ax2.set_yscale("symlog", linthresh=0.01)
    ax2.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return facts


# --------------------------------------------------------------------------
# Figure 3: TTF MILP vs pipeline + heuristic median-time overlay
# --------------------------------------------------------------------------
def figure_ttf(
    ttf: pd.DataFrame,
    heur_df: pd.DataFrame,
    out_path: Path,
) -> Dict[str, Any]:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    facts: Dict[str, Any] = {"by_family": {}}
    for fam in FAMILIES:
        sub = ttf[ttf["family"] == fam]
        ax.scatter(
            sub["pipeline_solve_time"],
            sub["milp_first_feasible_budget_s"],
            s=22,
            alpha=0.55,
            color=FAMILY_COLOURS[fam],
            label=f"{fam.capitalize()} (n={len(sub)})",
            edgecolors="white",
            linewidth=0.4,
        )
    # Diagonal y = x reference
    lo = max(1e-2, float(np.nanmin(ttf["pipeline_solve_time"])))
    hi = float(np.nanmax(ttf["milp_first_feasible_budget_s"]))
    diag = np.linspace(lo, hi, 100)
    ax.plot(diag, diag, "--", color="grey", alpha=0.7, label="$y=x$")

    # Median-time overlay for the heuristic at the matched_pipeline budget
    matched_h = heur_df[heur_df["budget_mode"] == "matched_pipeline"]
    matched_h_feas = matched_h[matched_h["heur_feasible_within_budget"].astype(bool)]
    for fam in FAMILIES:
        med_t = float(matched_h_feas.loc[matched_h_feas["family"] == fam, "heur_end_to_end_time_s"].median())
        if _is_finite(med_t):
            ax.axhline(med_t, color=FAMILY_COLOURS[fam], alpha=0.55, linestyle=":", linewidth=1.6)
            ax.text(
                hi * 0.92,
                med_t,
                f"  RH-MO+LP {fam[0].upper()}: {med_t:.1f}s",
                color=FAMILY_COLOURS[fam],
                fontsize=8,
                va="bottom",
                ha="right",
            )
        facts["by_family"][fam] = {
            "rhmolp_median_feas_time_s": med_t if _is_finite(med_t) else None,
            "rhmolp_feasibility_rate_at_matched_pct": float(
                matched_h.loc[matched_h["family"] == fam, "heur_feasible_within_budget"].mean()
            ) * 100.0,
        }
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Pipeline wall-clock time (s)")
    ax.set_ylabel("MILP time to first feasible (s)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return facts


# --------------------------------------------------------------------------
# Figure 4: compute-bounded exploration k=1 vs k=5
# --------------------------------------------------------------------------
def figure_compute_bounded(
    k1: pd.DataFrame,
    k5: pd.DataFrame,
    out_path: Path,
) -> Dict[str, Any]:
    """Paired k=1 vs k=5 analysis at matched 120 s total budget."""
    from scipy.stats import wilcoxon

    facts: Dict[str, Any] = {"by_family": {}, "best_of_k": {}}

    # --- Best-of-K curve from k=5 records --------------------------------
    # Each k=5 record exposes ``all_objectives`` (length up to k); compute
    # the running best (most negative cost = best) for K=1..k.
    K_VALS = [1, 2, 3, 4, 5]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    ax_curve = axes[0]
    for fam in FAMILIES:
        sub = k5[k5["family"] == fam]
        per_k_means: Dict[int, List[float]] = {k: [] for k in K_VALS}
        per_k_feas: Dict[int, List[bool]] = {k: [] for k in K_VALS}
        for _, row in sub.iterrows():
            objs = list(row.get("all_objectives", []) or [])
            if not objs:
                continue
            # Replace non-finite with +inf so a feasible later sample still wins
            cleaned = [float(o) if _is_finite(o) else float("inf") for o in objs]
            running_best = []
            cur = float("inf")
            for v in cleaned:
                cur = min(cur, v)
                running_best.append(cur)
            for k in K_VALS:
                idx = min(k - 1, len(running_best) - 1)
                if idx >= 0:
                    per_k_means[k].append(running_best[idx])
                    per_k_feas[k].append(_is_finite(running_best[idx]))
        feas_rates = [
            100.0 * float(np.mean(per_k_feas[k])) if per_k_feas[k] else float("nan")
            for k in K_VALS
        ]
        ax_curve.plot(
            K_VALS, feas_rates, "-o",
            color=FAMILY_COLOURS[fam],
            label=fam.capitalize(),
            linewidth=2.0,
        )
        facts["best_of_k"][fam] = {
            "K_values": K_VALS,
            "feasibility_rate_pct": [v if _is_finite(v) else None for v in feas_rates],
        }
    ax_curve.set_xticks(K_VALS)
    ax_curve.set_xlabel("Best-of-K candidates (K)")
    ax_curve.set_ylabel("Feasibility rate (%)")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(frameon=False)

    # --- Paired k=1 vs k=5 selected-objective comparison ---------------
    ax_p = axes[1]
    boxes = []
    labels = []
    for fam in FAMILIES:
        m = k1[k1["family"] == fam][["family", "scenario_id", "lp_objective"]].merge(
            k5[k5["family"] == fam][["family", "scenario_id", "lp_objective"]],
            on=["family", "scenario_id"],
            suffixes=("_k1", "_k5"),
        )
        # Restrict to scenarios where BOTH are feasible to make Wilcoxon
        # well-defined. Report the share of scenarios excluded for transparency.
        both_finite = m[
            m["lp_objective_k1"].apply(_is_finite) & m["lp_objective_k5"].apply(_is_finite)
        ].copy()
        n_paired = len(both_finite)
        if n_paired:
            delta = both_finite["lp_objective_k5"] - both_finite["lp_objective_k1"]
            # Negative delta = k=5 cheaper.
            # Express as % of |MILP-120 reference|: not available here cleanly,
            # so use signed % of |k=1 obj|.
            denom = both_finite["lp_objective_k1"].abs().replace(0, np.nan)
            delta_pct = (100.0 * delta / denom).dropna()
            try:
                w = wilcoxon(
                    both_finite["lp_objective_k1"], both_finite["lp_objective_k5"]
                )
                p_val = float(w.pvalue)
            except Exception:
                p_val = float("nan")
            facts["by_family"][fam] = {
                "n_paired_both_feasible": int(n_paired),
                "median_delta_obj_pct": float(np.median(delta_pct)) if len(delta_pct) else float("nan"),
                "mean_delta_obj_pct": float(np.mean(delta_pct)) if len(delta_pct) else float("nan"),
                "wilcoxon_p_value": p_val,
                "share_k5_better_pct": 100.0 * float((delta < -1e-9).mean()),
                "share_k5_equal_pct": 100.0 * float((delta.abs() <= 1e-9).mean()),
                "share_k1_better_pct": 100.0 * float((delta > 1e-9).mean()),
                "k1_feasibility_rate_pct": 100.0 * float(k1[k1["family"] == fam]["lp_objective"].apply(_is_finite).mean()),
                "k5_feasibility_rate_pct": 100.0 * float(k5[k5["family"] == fam]["lp_objective"].apply(_is_finite).mean()),
            }
            boxes.append(delta_pct.values)
            labels.append(f"{fam.capitalize()}\n(n={n_paired})")
        else:
            facts["by_family"][fam] = {
                "n_paired_both_feasible": 0,
                "k1_feasibility_rate_pct": 100.0 * float(k1[k1["family"] == fam]["lp_objective"].apply(_is_finite).mean()),
                "k5_feasibility_rate_pct": 100.0 * float(k5[k5["family"] == fam]["lp_objective"].apply(_is_finite).mean()),
            }
            boxes.append([0.0])
            labels.append(f"{fam.capitalize()}\n(n=0)")
    bp = ax_p.boxplot(boxes, labels=labels, showfliers=False, patch_artist=True)
    for patch, fam in zip(bp["boxes"], FAMILIES):
        patch.set_facecolor(FAMILY_COLOURS[fam])
        patch.set_alpha(0.55)
    ax_p.axhline(0, color="grey", linestyle="--", alpha=0.7)
    ax_p.set_ylabel(r"$(\mathrm{cost}_{k=5} - \mathrm{cost}_{k=1}) / |\mathrm{cost}_{k=1}|$ (%)")
    ax_p.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return facts


# --------------------------------------------------------------------------
# Figure 5: criticality scaling — pipeline advantage vs criticality bucket
# --------------------------------------------------------------------------
def figure_criticality_scaling(
    matched_per_scen: pd.DataFrame,
    out_path: Path,
) -> Dict[str, Any]:
    facts: Dict[str, Any] = {"by_family": {}}
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    width = 0.33
    x = np.arange(len(FAMILIES))
    pipe_only = []
    milp_only = []
    both = []
    for fam in FAMILIES:
        sub = matched_per_scen[matched_per_scen["family"] == fam]
        pipe_f = sub.get("pipeline_feasible_within_budget", pd.Series(dtype=bool)).fillna(False).astype(bool)
        milp_f = sub.get("milp_feasible_within_budget", pd.Series(dtype=bool)).fillna(False).astype(bool)
        n = len(sub)
        pipe_only_pct = 100.0 * float(((pipe_f) & (~milp_f)).mean()) if n else 0.0
        milp_only_pct = 100.0 * float(((~pipe_f) & (milp_f)).mean()) if n else 0.0
        both_pct = 100.0 * float((pipe_f & milp_f).mean()) if n else 0.0
        pipe_only.append(pipe_only_pct)
        milp_only.append(milp_only_pct)
        both.append(both_pct)
        facts["by_family"][fam] = {
            "n": int(n),
            "pipeline_only_feasible_pct": pipe_only_pct,
            "milp_only_feasible_pct": milp_only_pct,
            "both_feasible_pct": both_pct,
        }
    ax.bar(x - width, pipe_only, width=width, color=METHOD_COLOURS["pipeline"], label="Pipeline-only feasible")
    ax.bar(x, both, width=width, color="#7E57C2", label="Both feasible")
    ax.bar(x + width, milp_only, width=width, color=METHOD_COLOURS["milp"], label="MILP-only feasible")
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in FAMILIES])
    ax.set_ylabel("Share of scenarios at matched budget (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return facts


# --------------------------------------------------------------------------
# Persona 3-way fixed-budget and matched-budget comparison
# --------------------------------------------------------------------------
PERSONA_SHORT_FAMILIES = ["vre_battery", "network_operator", "mathematician"]
PERSONA_LONG_FAMILIES = {
    "vre_battery": "persona_vre_battery_scenarios",
    "network_operator": "persona_network_operator_scenarios",
    "mathematician": "persona_mathematician_scenarios",
}
PERSONA_DISPLAY = {
    "vre_battery": "VRE & battery",
    "network_operator": "Network operator",
    "mathematician": "Mathematician",
}
# Persona-identity palette: must match the radar of Section 6.4.1
# (figure_persona_radar) so that each persona is identifiable across all
# Section 6.4 figures by the same colour.
PERSONA_IDENTITY_COLOURS = {
    "vre_battery": "#1F77B4",
    "network_operator": "#FF7F0E",
    "mathematician": "#9467BD",
}


def figure_persona_fixed_budget_3way(
    persona_fixed: pd.DataFrame,
    persona_heur_agg: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.0), sharex=True)
    for col, short in enumerate(PERSONA_SHORT_FAMILIES):
        long = PERSONA_LONG_FAMILIES[short]
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        persona_colour = PERSONA_IDENTITY_COLOURS[short]
        for spine_ax in (ax_top, ax_bot):
            for spine in spine_ax.spines.values():
                spine.set_edgecolor(persona_colour)
                spine.set_linewidth(2.2)
        fs = persona_fixed[persona_fixed["family"] == long].sort_values("budget_seconds")
        hs = persona_heur_agg[persona_heur_agg["family"] == short].sort_values("budget_seconds")

        ax_top.plot(fs["budget_seconds"], fs["pipeline_feasible_rate_pct"], "-o",
                    color=METHOD_COLOURS["pipeline"], label=METHOD_LABELS["pipeline"], linewidth=2.0)
        ax_top.plot(fs["budget_seconds"], fs["milp_feasible_rate_pct"], "-s",
                    color=METHOD_COLOURS["milp"], label=METHOD_LABELS["milp"], linewidth=2.0)
        ax_top.plot(hs["budget_seconds"], hs["heuristic_feasible_rate"] * 100.0, "-^",
                    color=METHOD_COLOURS["heuristic"], label=METHOD_LABELS["heuristic"], linewidth=2.0)
        ax_top.set_xscale("log")
        ax_top.set_ylim(-3, 103)
        ax_top.grid(True, alpha=0.3)
        if col == 0:
            ax_top.set_ylabel("Feasibility rate (%)")

        ax_bot.plot(fs["budget_seconds"], fs["milp_median_abs_gap_pct"], "-s",
                    color=METHOD_COLOURS["milp"], linewidth=2.0)
        ax_bot.plot(fs["budget_seconds"], fs["pipeline_median_abs_gap_pct"], "-o",
                    color=METHOD_COLOURS["pipeline"], linewidth=2.0)
        ax_bot.plot(hs["budget_seconds"], hs["heuristic_median_abs_gap_pct"], "-^",
                    color=METHOD_COLOURS["heuristic"], linewidth=2.0)
        ax_bot.set_xscale("log")
        ax_bot.set_yscale("symlog", linthresh=0.1)
        ax_bot.grid(True, which="both", alpha=0.3)
        ax_bot.set_xlabel("Budget (s)")
        if col == 0:
            ax_bot.set_ylabel("Median |gap to MILP$_{120s}$| (%)")

    # Colour-coded persona identifier under each column, matching the radar palette.
    for col, short in enumerate(PERSONA_SHORT_FAMILIES):
        ax_bot = axes[1, col]
        ax_bot.annotate(
            PERSONA_DISPLAY[short],
            xy=(0.5, -0.30),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=PERSONA_IDENTITY_COLOURS[short],
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def figure_persona_matched_3way(
    persona_matched_per_scen: pd.DataFrame,
    persona_heur_df: pd.DataFrame,
    out_path: Path,
) -> Dict[str, Any]:
    """Bar chart comparing the three solvers at the per-scenario matched budget.

    The persona RH-MO+LP run only exposes fixed-budget rows, so the heuristic
    feasibility at the matched budget is approximated by the smallest fixed
    grid budget that is >= the scenario's pipeline solve time. When no grid
    budget reaches that bound (i.e. ``matched_budget > 120 s``) the heuristic
    is reported as feasible only if it was feasible at the 120 s grid point.
    """
    facts: Dict[str, Any] = {"by_persona": {}}
    persona_heur_fixed = persona_heur_df[persona_heur_df["budget_mode"] == "fixed"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    ax_feas = axes[0]
    ax_gap = axes[1]
    width = 0.27
    x = np.arange(len(PERSONA_SHORT_FAMILIES))
    feas = {"pipeline": [], "milp": [], "heuristic": []}
    gaps = {"pipeline": [], "milp": [], "heuristic": []}

    for short in PERSONA_SHORT_FAMILIES:
        long = PERSONA_LONG_FAMILIES[short]
        sub = persona_matched_per_scen[persona_matched_per_scen["family"] == long]
        n = len(sub)
        pipe_f = sub["pipeline_feasible_within_budget"].fillna(False).astype(bool)
        milp_f = sub["milp_feasible_within_budget"].fillna(False).astype(bool)

        # Per-scenario heuristic feasibility at the smallest grid budget >= matched.
        heur_sub = persona_heur_fixed[persona_heur_fixed["family"] == short]
        # Pivot: scenario_id x budget -> feasible flag
        heur_pivot = heur_sub.pivot_table(
            index="scenario_id",
            columns="budget_seconds",
            values="heur_feasible_within_budget",
            aggfunc="first",
        )
        budgets_sorted = sorted(heur_pivot.columns)
        heur_feas_at_matched: List[bool] = []
        heur_obj_at_matched: List[float] = []
        for _, row in sub.iterrows():
            sid = row["scenario_id"]
            mb = float(row["budget_seconds"])
            chosen = None
            for b in budgets_sorted:
                if b >= mb - 1e-9:
                    chosen = b
                    break
            if chosen is None:
                chosen = budgets_sorted[-1]
            if sid in heur_pivot.index and chosen in heur_pivot.columns:
                feas_val = bool(heur_pivot.loc[sid, chosen])
            else:
                feas_val = False
            heur_feas_at_matched.append(feas_val)
            if feas_val:
                r2 = heur_sub[(heur_sub["scenario_id"] == sid)
                              & np.isclose(heur_sub["budget_seconds"], chosen)]
                obj = float(r2["heur_objective"].iloc[0]) if len(r2) else float("nan")
                ref = float(r2["reference_milp_objective"].iloc[0]) if len(r2) and "reference_milp_objective" in r2.columns else float("nan")
                heur_obj_at_matched.append(_abs_gap_pct(obj, ref))
            else:
                heur_obj_at_matched.append(float("nan"))

        heur_feas_arr = np.array(heur_feas_at_matched, dtype=bool)
        feas["pipeline"].append(100.0 * float(pipe_f.mean()) if n else 0.0)
        feas["milp"].append(100.0 * float(milp_f.mean()) if n else 0.0)
        feas["heuristic"].append(100.0 * float(heur_feas_arr.mean()) if n else 0.0)

        g_p = sub.loc[pipe_f, "pipeline_abs_gap_vs_reference_pct"].dropna()
        g_m = sub.loc[milp_f, "milp_abs_gap_vs_reference_pct"].dropna()
        g_h = [v for v in heur_obj_at_matched if _is_finite(v)]
        gaps["pipeline"].append(float(np.median(g_p)) if len(g_p) else float("nan"))
        gaps["milp"].append(float(np.median(g_m)) if len(g_m) else float("nan"))
        gaps["heuristic"].append(float(np.median(g_h)) if g_h else float("nan"))

        facts["by_persona"][short] = {
            "n": int(n),
            "matched_budget_median_s": float(sub["budget_seconds"].median()) if n else float("nan"),
            "pipeline_feasible_pct": feas["pipeline"][-1],
            "milp_feasible_pct": feas["milp"][-1],
            "heuristic_feasible_pct_upper_bound": feas["heuristic"][-1],
            "pipeline_median_abs_gap_pct": gaps["pipeline"][-1],
            "milp_median_abs_gap_pct": gaps["milp"][-1],
            "heuristic_median_abs_gap_pct_upper_bound": gaps["heuristic"][-1],
        }

    for i, m in enumerate(["pipeline", "milp", "heuristic"]):
        ax_feas.bar(x + (i - 1) * width, feas[m], width=width,
                    color=METHOD_COLOURS[m], label=METHOD_LABELS[m])
    ax_feas.set_xticks(x)
    ax_feas.set_xticklabels([PERSONA_DISPLAY[s] for s in PERSONA_SHORT_FAMILIES])
    for tick_label, short in zip(ax_feas.get_xticklabels(), PERSONA_SHORT_FAMILIES):
        tick_label.set_color(PERSONA_IDENTITY_COLOURS[short])
        tick_label.set_fontweight("bold")
    ax_feas.set_ylabel("Feasibility rate at matched budget (%)")
    ax_feas.set_ylim(0, 105)
    ax_feas.grid(True, axis="y", alpha=0.3)
    ax_feas.legend(frameon=False, loc="upper right")

    for i, m in enumerate(["pipeline", "milp", "heuristic"]):
        vals = [v if _is_finite(v) else 0.0 for v in gaps[m]]
        ax_gap.bar(x + (i - 1) * width, vals, width=width,
                   color=METHOD_COLOURS[m], label=METHOD_LABELS[m])
    ax_gap.set_xticks(x)
    ax_gap.set_xticklabels([PERSONA_DISPLAY[s] for s in PERSONA_SHORT_FAMILIES])
    for tick_label, short in zip(ax_gap.get_xticklabels(), PERSONA_SHORT_FAMILIES):
        tick_label.set_color(PERSONA_IDENTITY_COLOURS[short])
        tick_label.set_fontweight("bold")
    ax_gap.set_ylabel("Median |gap to MILP$_{120s}$| (%)")
    ax_gap.set_yscale("symlog", linthresh=0.01)
    ax_gap.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return facts


# --------------------------------------------------------------------------
# Figure 6: persona radar of criticality drivers
# --------------------------------------------------------------------------
PERSONA_CRITICALITY_CSV = (
    REPO / "outputs" / "personas_criticality_radar" / "persona_criticality_means.csv"
)
PERSONA_LABEL_FROM_CSV = {
    "persona_vre_battery_scenarios": "VRE & battery",
    "persona_network_operator_scenarios": "Network operator",
    "persona_mathematician_scenarios": "Mathematician",
}


def figure_persona_radar(out_path: Path) -> Dict[str, Any]:
    """22-axis criticality radar (A1-A12 stress + B1-B10 hardness).

    Consumes the per-persona normalised means produced by the standalone
    persona-radar script (one row per persona). Each axis is on a [0, 1]
    scale identical to the one used in the criticality decomposition of
    Appendix C, so the radar is directly comparable to the driver table.
    No in-figure title: the figure caption is sufficient.
    """
    facts: Dict[str, Any] = {"by_persona": {}}
    df = pd.read_csv(PERSONA_CRITICALITY_CSV)

    stress_keys = [f"A{i}" for i in range(1, 13)]
    hardness_keys = [f"B{i}" for i in range(1, 11)]
    all_keys = stress_keys + hardness_keys

    # CSV uses long-form column names (e.g. "A1_vre_penetration_norm");
    # build a key -> column lookup.
    col_for: Dict[str, str] = {}
    for short in all_keys:
        match = [c for c in df.columns if c.startswith(f"{short}_") and c.endswith("_norm")]
        if not match:
            raise KeyError(f"Could not find normalised column for axis {short}")
        col_for[short] = match[0]

    angles = np.linspace(0.0, 2.0 * np.pi, len(all_keys), endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(9.0, 9.0))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(all_keys, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=8, color="gray")
    ax.set_ylim(0.0, 1.0)
    ax.set_rlabel_position(180.0 / max(len(all_keys), 1))

    palette = ["#1F77B4", "#FF7F0E", "#9467BD", "#2CA02C", "#D62728"]
    for i, (_, row) in enumerate(df.iterrows()):
        raw_label = str(row["label"])
        display_label = PERSONA_LABEL_FROM_CSV.get(raw_label, raw_label)
        values = np.array([float(row[col_for[k]]) for k in all_keys], dtype=float)
        values_closed = np.concatenate([values, values[:1]])
        colour = palette[i % len(palette)]
        ax.plot(angles_closed, values_closed, "-", color=colour, linewidth=2.4, label=display_label)
        ax.fill(angles_closed, values_closed, color=colour, alpha=0.12)
        facts["by_persona"][raw_label] = {
            "display_label": display_label,
            "n_scenarios": int(row["n_scenarios"]),
            "stress_means": {k: float(row[col_for[k]]) for k in stress_keys},
            "hardness_means": {k: float(row[col_for[k]]) for k in hardness_keys},
        }
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.05), frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return facts


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------
def main() -> None:
    print(f"[load] {HEUR_PKL}")
    heur_df = pd.DataFrame(pickle.load(HEUR_PKL.open("rb")))
    fixed_summary = pd.read_csv(FIXED_SUMMARY_CSV)
    matched_per_scen = pd.read_csv(MATCHED_PER_SCEN_CSV)
    fixed_head = pd.read_csv(FIXED_HEAD_CSV)
    matched_fam = pd.read_csv(MATCHED_FAMILY_CSV)
    ttf = pd.read_csv(TTF_CSV)
    driver_summary = json.load(DRIVER_SUMMARY.open("r", encoding="utf-8"))
    partial_corr = pd.read_csv(PARTIAL_CORR_SUMMARY)
    k1 = pd.DataFrame(pickle.load(K1_PKL.open("rb")))
    k5 = pd.DataFrame(pickle.load(K5_PKL.open("rb")))
    persona_fixed = pd.read_csv(PERSONA_FAMILY_SUM)

    heur_agg = aggregate_heuristic(heur_df)

    figure_fixed_budget(
        fixed_summary,
        heur_agg,
        OUT_DIR / "fig_fixed_budget_comparison_3way.png",
    )
    matched_facts = figure_matched_budget(
        matched_per_scen,
        heur_df,
        OUT_DIR / "fig_matched_budget_comparison_3way.png",
    )
    ttf_facts = figure_ttf(
        ttf,
        heur_df,
        OUT_DIR / "fig_milp_ttf_vs_pipeline_3way.png",
    )
    cb_facts = figure_compute_bounded(
        k1, k5, OUT_DIR / "fig_compute_bounded_k_scaling.png"
    )
    crit_facts = figure_criticality_scaling(
        matched_per_scen, OUT_DIR / "fig_criticality_scaling.png"
    )
    persona_facts = figure_persona_radar(OUT_DIR / "fig_persona_radar.png")

    # Persona 3-way comparison using the new persona heuristic pkl
    persona_heur_df = pd.DataFrame(pickle.load(PERSONA_HEUR_PKL.open("rb")))
    persona_heur_agg = aggregate_heuristic(persona_heur_df, families=PERSONA_SHORT_FAMILIES)
    figure_persona_fixed_budget_3way(
        persona_fixed, persona_heur_agg,
        OUT_DIR / "fig_persona_fixed_budget_comparison_3way.png",
    )
    persona_matched_per_scen = pd.read_csv(
        REPO / "outputs" / "uc_jepa" / "personas_grid_milp_120_k1" / "matched_budget_per_scenario.csv"
    )
    persona_matched_facts = figure_persona_matched_3way(
        persona_matched_per_scen, persona_heur_df,
        OUT_DIR / "fig_persona_matched_budget_comparison_3way.png",
    )

    # ---- Aggregate scalar facts cited from main.tex ----------------------
    all_only_pct = float(
        100.0
        * (
            matched_per_scen.get("pipeline_feasible_within_budget", pd.Series(dtype=bool)).fillna(False).astype(bool)
            & ~matched_per_scen.get("milp_feasible_within_budget", pd.Series(dtype=bool)).fillna(False).astype(bool)
        ).mean()
    )
    out_stats = {
        "n_scenarios_per_family": 100,
        "n_scenarios_total": 300,
        "fixed_budgets_seconds": BUDGETS,
        "fixed_summary_pipeline_vs_milp": fixed_head.to_dict(orient="records"),
        "matched_budget_family_summary": matched_fam.to_dict(orient="records"),
        "heuristic_per_family_per_budget": heur_agg.to_dict(orient="records"),
        "matched_budget_3way": matched_facts,
        "ttf_3way": ttf_facts,
        "compute_bounded_k_scaling": cb_facts,
        "criticality_scaling": crit_facts,
        "persona_radar": persona_facts,
        "persona_heuristic_per_budget": persona_heur_agg.to_dict(orient="records"),
        "persona_matched_3way": persona_matched_facts,
        "driver_summary": driver_summary,
        "partial_correlation_summary": partial_corr.head(15).to_dict(orient="records"),
        "persona_fixed_budget_summary": persona_fixed.to_dict(orient="records"),
        "all_pipeline_only_feasible_pct": all_only_pct,
        "figures": {
            "fixed_budget_3way": "fig_fixed_budget_comparison_3way.png",
            "matched_budget_3way": "fig_matched_budget_comparison_3way.png",
            "ttf_3way": "fig_milp_ttf_vs_pipeline_3way.png",
            "compute_bounded": "fig_compute_bounded_k_scaling.png",
            "criticality_scaling": "fig_criticality_scaling.png",
            "persona_radar": "fig_persona_radar.png",
            "persona_fixed_3way": "fig_persona_fixed_budget_comparison_3way.png",
            "persona_matched_3way": "fig_persona_matched_budget_comparison_3way.png",
        },
    }
    (OUT_DIR / "section6_stats.json").write_text(
        json.dumps(out_stats, indent=2, default=lambda o: None if isinstance(o, float) and not _is_finite(o) else o),
        encoding="utf-8",
    )
    print(f"[done] Figures + stats in {OUT_DIR}")


if __name__ == "__main__":
    main()
