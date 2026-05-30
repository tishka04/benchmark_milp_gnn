"""Build the feasibility-first headline figures for the partial-grid recovery
subsection (6.5).

All per-scenario/per-arm outcomes are recomputed from ``per_contingency.csv`` --
the source of truth that retains *infeasible* contingencies. The earlier
``per_scenario_arm.csv`` conditioned load shed on feasibility, which hid the
dominant failure mode (learned/heuristic commitments that are infeasible to
dispatch). Here a contingency counts as **recovered** only if it is both
feasible *and* within the load-shed tolerance ``tau``; infeasible contingencies
are counted as non-recovered (the worst outcome).

Produces:

* ``fig_partial_recovery_feasibility.png`` -- per arm x family: %% of scenarios
  whose commitment is dispatch-feasible for *every* contingency, and the mean
  per-scenario feasible fraction.
* ``fig_partial_recovery_recovery.png`` -- per arm x family: %% of scenarios
  fully recovered (all contingencies feasible and within tau), and -- as a
  secondary, explicitly *conditional* panel -- the median worst-case load shed
  over the feasible contingencies only.
* ``partial_recovery_summary.csv`` -- the per-(arm, family) table used in 6.5.

Usage:
    python -m scripts.build_partial_recovery_figures --in outputs/uc_jepa/partial_recovery
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ARM_LABELS = {
    "A": "A: Full MILP",
    "B": "B: Partial MILP + fill",
    "C": "C: Pipeline (full)",
    "D": "D: Pipeline-on-$S^*$ + fill",
}
ARM_COLORS = {"A": "#e74c3c", "B": "#f39c12", "C": "#3498db", "D": "#1abc9c"}
FAMILY_ORDER = ["low", "medium", "high"]
FAMILY_LABELS = {"low": "Low", "medium": "Medium", "high": "High"}


def _scenario_arm_table(cont: pd.DataFrame, tau_mwh: float) -> pd.DataFrame:
    """Recompute per-(family, arm, scenario) outcomes from raw contingency rows.

    Feasibility-first: a contingency is *recovered* iff feasible AND lol <= tau.
    Infeasible contingencies count as non-recovered (worst outcome).
    """
    df = cont.copy()
    df["feasible"] = df["feasible"].astype(bool)
    df["recovered"] = df["feasible"] & (df["lol_mwh"].fillna(np.inf) <= tau_mwh)

    arms_present = sorted(df["arm"].unique())
    rows = []
    for (family, arm, sc), g in df.groupby(["family", "arm", "scenario_id"], dropna=False):
        n = len(g)
        feas = g[g["feasible"]]
        base = g[g["outage"] == "__base__"]
        rows.append({
            "family": family, "arm": arm, "scenario_id": sc,
            "n_contingencies": n,
            "n_feasible": int(g["feasible"].sum()),
            "feasible_fraction": float(g["feasible"].mean()) if n else np.nan,
            "recovered_fraction": float(g["recovered"].mean()) if n else np.nan,
            "all_feasible": bool(g["feasible"].all()) if n else False,
            "fully_recovered": bool(g["recovered"].all()) if n else False,
            "base_feasible": bool(base["feasible"].all()) if len(base) else False,
            # conditional worst-case shed (over feasible contingencies only)
            "worst_lol_feasible": float(feas["lol_mwh"].max()) if len(feas) else np.nan,
            "max_cascade_risk": float(feas["cascade_risk_score"].max()) if len(feas) else np.nan,
        })
    sa = pd.DataFrame(rows)

    # Fair accounting for commitment-level failures: if an arm's commitment MILP
    # returns infeasible, the launcher writes no contingency rows, so the scenario
    # would silently vanish from that arm's denominator. Use arm A (always
    # co-optimized and feasible) as the per-family reference set of *attempted*
    # scenarios; any reference scenario missing for another arm is recorded as a
    # commitment-level failure (not dropped).
    ref_arm = "A" if "A" in arms_present else arms_present[0]
    ref = sa[sa["arm"] == ref_arm].groupby("family")["scenario_id"].agg(set).to_dict()
    extra = []
    for arm in arms_present:
        if arm == ref_arm:
            continue
        have = sa[sa["arm"] == arm].groupby("family")["scenario_id"].agg(set).to_dict()
        for family, ref_scs in ref.items():
            missing = ref_scs - have.get(family, set())
            for sc in missing:
                extra.append({
                    "family": family, "arm": arm, "scenario_id": sc,
                    "n_contingencies": 0, "n_feasible": 0,
                    "feasible_fraction": 0.0, "recovered_fraction": 0.0,
                    "all_feasible": False, "fully_recovered": False,
                    "base_feasible": False,
                    "worst_lol_feasible": np.nan, "max_cascade_risk": np.nan,
                })
    if extra:
        sa = pd.concat([sa, pd.DataFrame(extra)], ignore_index=True)
    return sa


def _headline(sa: pd.DataFrame) -> pd.DataFrame:
    return (
        sa.groupby(["arm", "family"])
        .agg(
            n_scenarios=("scenario_id", "nunique"),
            pct_all_feasible=("all_feasible", lambda s: 100.0 * s.mean()),
            mean_feasible_fraction=("feasible_fraction", "mean"),
            pct_fully_recovered=("fully_recovered", lambda s: 100.0 * s.mean()),
            mean_recovered_fraction=("recovered_fraction", "mean"),
            median_worst_lol_feasible=("worst_lol_feasible", "median"),
            mean_worst_lol_feasible=("worst_lol_feasible", "mean"),
            median_max_cascade_risk=("max_cascade_risk", "median"),
        )
        .reset_index()
    )


def _grouped_bars(ax, sa_metric: pd.DataFrame, arms, fams, ylabel, title, pct=False):
    width = 0.8 / max(len(arms), 1)
    xpos = np.arange(len(fams))
    for i, arm in enumerate(arms):
        offs = (i - (len(arms) - 1) / 2) * width
        yvals = [
            sa_metric.loc[arm, f] if (arm in sa_metric.index and f in sa_metric.columns) else np.nan
            for f in fams
        ]
        ax.bar(xpos + offs, yvals, width=width, color=ARM_COLORS[arm],
               edgecolor="black", linewidth=0.4, label=ARM_LABELS[arm])
    ax.set_xticks(xpos)
    ax.set_xticklabels([FAMILY_LABELS[f] for f in fams])
    ax.set_xlabel("Criticality family")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    if pct:
        ax.set_ylim(0, 105)


def build_feasibility_fig(sa: pd.DataFrame, out_path: Path):
    arms = [a for a in ["A", "B", "C", "D"] if a in set(sa["arm"])]
    fams = [f for f in FAMILY_ORDER if f in set(sa["family"])]
    pct_all = (sa.groupby(["arm", "family"])["all_feasible"].mean().mul(100.0).unstack("family"))
    mean_ff = (sa.groupby(["arm", "family"])["feasible_fraction"].mean().unstack("family"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    _grouped_bars(axes[0], pct_all, arms, fams,
                  "% scenarios dispatch-feasible (all N-1)",
                  "Commitment feasibility under contingencies", pct=True)
    _grouped_bars(axes[1], mean_ff, arms, fams,
                  "Mean feasible fraction of contingencies",
                  "Per-scenario feasible fraction", pct=False)
    axes[1].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def build_recovery_fig(sa: pd.DataFrame, out_path: Path, tau_mwh: float):
    arms = [a for a in ["A", "B", "C", "D"] if a in set(sa["arm"])]
    fams = [f for f in FAMILY_ORDER if f in set(sa["family"])]
    pct_rec = (sa.groupby(["arm", "family"])["fully_recovered"].mean().mul(100.0).unstack("family"))
    med_lol = (sa.groupby(["arm", "family"])["worst_lol_feasible"].median().unstack("family"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    _grouped_bars(axes[0], pct_rec, arms, fams,
                  "% scenarios fully recovered",
                  rf"Fully recovered (feasible & LOL $\leq$ {tau_mwh:g} MWh, all N-1)", pct=True)
    _grouped_bars(axes[1], med_lol, arms, fams,
                  "Median worst-case load shed (MWh)",
                  "Worst-case load shed (feasible contingencies only)", pct=False)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description="Build partial-grid recovery figures (feasibility-first).")
    p.add_argument("--in", dest="in_dir", default="outputs/uc_jepa/partial_recovery")
    p.add_argument("--tau-mwh", type=float, default=1.0)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    cpath = in_dir / "per_contingency.csv"
    if not cpath.exists():
        raise SystemExit(f"Missing source of truth: {cpath}")
    cont = pd.read_csv(cpath)
    if cont.empty:
        raise SystemExit(f"No rows in {cpath}")

    sa = _scenario_arm_table(cont, args.tau_mwh)
    sa.to_csv(in_dir / "per_scenario_arm_feasibility.csv", index=False)

    headline = _headline(sa)
    headline.to_csv(in_dir / "partial_recovery_summary.csv", index=False)
    print(f"wrote {in_dir / 'partial_recovery_summary.csv'}")
    print(headline.to_string(index=False))

    build_feasibility_fig(sa, in_dir / "fig_partial_recovery_feasibility.png")
    build_recovery_fig(sa, in_dir / "fig_partial_recovery_recovery.png", args.tau_mwh)


if __name__ == "__main__":
    main()
