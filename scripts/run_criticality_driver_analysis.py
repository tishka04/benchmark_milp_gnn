"""Run the criticality-driver analysis end-to-end (CLI mirror of
``notebooks/criticality_drivers_analysis.ipynb``).

For a given evaluation directory containing
``matched_budget_per_scenario.csv`` and
``criticality_components_per_scenario.csv``, produces:

- ``driver_ranking_pooled.csv``
- ``driver_ranking_within_family.csv``
- ``driver_multivariate_importance.csv``
- ``fig_driver_ranking.png``
- ``fig_advantage_vs_top_drivers.png``
- ``fig_winrate_vs_top_drivers.png``
- ``fig_driver_multivariate.png``
- ``driver_summary.json`` (top-N drivers, model fit stats, sign flips).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler


INFEASIBLE_GAP = 1000.0  # %; sentinel for the continuous advantage metric.


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _spearman_safe(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    mask = x.notna() & y.notna()
    if mask.sum() < 5 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return float("nan"), float("nan")
    rho, p = spearmanr(x[mask], y[mask])
    return float(rho), float(p)


def driver_table(frame: pd.DataFrame, features: List[str], label: str) -> pd.DataFrame:
    rows = []
    for feat in features:
        rho_w, p_w = _spearman_safe(frame[feat], frame["win"])
        rho_a, p_a = _spearman_safe(frame[feat], frame["advantage_pct"])
        rows.append(
            {
                "scope": label,
                "feature": feat,
                "spearman_win": rho_w,
                "p_win": p_w,
                "spearman_advantage": rho_a,
                "p_advantage": p_a,
            }
        )
    out = pd.DataFrame(rows)
    out["abs_spearman_advantage"] = out["spearman_advantage"].abs()
    return out.sort_values("abs_spearman_advantage", ascending=False).reset_index(drop=True)


def _decile_summary(frame: pd.DataFrame, feature: str, n_bins: int = 10) -> pd.DataFrame:
    sub = frame[[feature, "advantage_pct", "win"]].dropna(subset=[feature]).copy()
    nunique = sub[feature].nunique()
    if nunique < n_bins:
        n_bins = max(2, nunique)
    sub["bin"] = pd.qcut(sub[feature], q=n_bins, duplicates="drop")
    grouped = (
        sub.groupby("bin", observed=True)
        .agg(
            x=(feature, "mean"),
            n=("advantage_pct", "size"),
            mean_advantage=("advantage_pct", "mean"),
            win_rate=("win", "mean"),
            sem_advantage=(
                "advantage_pct",
                lambda s: s.std(ddof=1) / max(np.sqrt(len(s)), 1),
            ),
        )
        .reset_index(drop=True)
    )
    return grouped


def run_analysis(eval_dir: Path, top_k: int = 6, top_n: int = 5) -> None:
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", font_scale=0.85)

    per_scn_csv = eval_dir / "matched_budget_per_scenario.csv"
    components_csv = eval_dir / "criticality_components_per_scenario.csv"
    if not per_scn_csv.exists():
        raise FileNotFoundError(per_scn_csv)
    if not components_csv.exists():
        raise FileNotFoundError(components_csv)

    per_scn = pd.read_csv(per_scn_csv)
    components = pd.read_csv(components_csv)
    df = per_scn.merge(
        components,
        on=["family", "scenario_id"],
        how="left",
        validate="one_to_one",
    )

    for col in [
        "pipeline_feasible_within_budget",
        "milp_feasible_within_budget",
        "pipeline_found_feasible",
        "milp_found_feasible",
    ]:
        if col in df.columns:
            df[col] = _coerce_bool(df[col])

    pipe_ok = df["pipeline_feasible_within_budget"].astype(bool)
    milp_ok = df["milp_feasible_within_budget"].astype(bool)
    pipe_gap = df["pipeline_abs_gap_vs_reference_pct"].astype(float)
    milp_gap = df["milp_abs_gap_vs_reference_pct"].astype(float)

    pipe_eff = np.where(pipe_ok, pipe_gap.fillna(INFEASIBLE_GAP), INFEASIBLE_GAP)
    milp_eff = np.where(milp_ok, milp_gap.fillna(INFEASIBLE_GAP), INFEASIBLE_GAP)

    df["advantage_pct"] = milp_eff - pipe_eff
    df["win"] = (
        (pipe_ok & ~milp_ok)
        | (pipe_ok & milp_ok & (pipe_gap < milp_gap))
    ).astype(int)

    stress_keys = sorted(c for c in df.columns if c.endswith("_norm") and c.startswith("A"))
    hardness_keys = sorted(c for c in df.columns if c.endswith("_norm") and c.startswith("B"))
    features: List[str] = stress_keys + hardness_keys
    if not features:
        raise RuntimeError("No normalized A/B components found in components CSV.")

    pooled = driver_table(df, features, label="pooled")
    within_family = pd.concat(
        [driver_table(sub, features, label=f"family={fam}") for fam, sub in df.groupby("family")],
        ignore_index=True,
    )
    pooled.to_csv(eval_dir / "driver_ranking_pooled.csv", index=False)
    within_family.to_csv(eval_dir / "driver_ranking_within_family.csv", index=False)

    # Univariate ranking bar.
    fig, ax = plt.subplots(figsize=(10, 9))
    rank_df = pooled.copy()
    colors = ["#3498db" if f.startswith("A") else "#e67e22" for f in rank_df["feature"]]
    ax.barh(
        rank_df["feature"][::-1],
        rank_df["spearman_advantage"][::-1],
        color=list(reversed(colors)),
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Spearman ρ between component and pipeline advantage (%)", fontweight="bold")
    ax.set_title("Univariate driver ranking (pooled, matched 120 s)", fontweight="bold")
    ax.legend(
        handles=[
            Patch(facecolor="#3498db", label="Stress (A1–A12)"),
            Patch(facecolor="#e67e22", label="Hardness (B1–B10)"),
        ],
        loc="lower right",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(eval_dir / "fig_driver_ranking.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Decile small multiples for top-K.
    top_features = pooled.head(top_k)["feature"].tolist()
    ncols = 3
    nrows = int(np.ceil(top_k / ncols))
    fig_adv, axes_adv = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), sharey=True)
    fig_win, axes_win = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), sharey=True)
    axes_adv = np.atleast_2d(axes_adv).reshape(nrows, ncols)
    axes_win = np.atleast_2d(axes_win).reshape(nrows, ncols)

    for i, feat in enumerate(top_features):
        r, c = divmod(i, ncols)
        summary = _decile_summary(df, feat)
        short = feat.replace("_norm", "")

        ax = axes_adv[r, c]
        ax.errorbar(
            summary["x"],
            summary["mean_advantage"],
            yerr=summary["sem_advantage"],
            marker="o",
            lw=2,
            capsize=3,
            color="#2c3e50",
        )
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_title(short, fontweight="bold", fontsize=11)
        ax.set_xlabel("component (normalized)")
        if c == 0:
            ax.set_ylabel("mean advantage (%)")

        ax = axes_win[r, c]
        ax.plot(summary["x"], 100.0 * summary["win_rate"], marker="o", lw=2, color="#27ae60")
        ax.axhline(50, color="gray", ls="--", lw=0.8)
        ax.set_title(short, fontweight="bold", fontsize=11)
        ax.set_xlabel("component (normalized)")
        if c == 0:
            ax.set_ylabel("pipeline win rate (%)")
        ax.set_ylim(0, 100)

    for fig_, axes_ in [(fig_adv, axes_adv), (fig_win, axes_win)]:
        for i in range(top_k, nrows * ncols):
            r, c = divmod(i, ncols)
            axes_[r, c].axis("off")
        fig_.tight_layout()

    fig_adv.savefig(eval_dir / "fig_advantage_vs_top_drivers.png", dpi=200, bbox_inches="tight")
    fig_win.savefig(eval_dir / "fig_winrate_vs_top_drivers.png", dpi=200, bbox_inches="tight")
    plt.close(fig_adv)
    plt.close(fig_win)

    # Multivariate attribution.
    X = df[features].astype(float).values
    y_win = df["win"].astype(int).values
    y_adv = df["advantage_pct"].astype(float).values

    Xz = StandardScaler().fit_transform(X)
    logreg = LogisticRegression(max_iter=2000, C=1.0)
    logreg.fit(Xz, y_win)
    logit_coef = pd.Series(logreg.coef_[0], index=features, name="logit_coef")
    logit_acc = float(logreg.score(Xz, y_win))
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    logit_cv_acc = float(np.mean(cross_val_score(
        LogisticRegression(max_iter=2000, C=1.0), Xz, y_win, cv=cv, scoring="accuracy"
    )))

    gbm = GradientBoostingRegressor(
        random_state=0, n_estimators=400, max_depth=3, learning_rate=0.05
    )
    gbm.fit(X, y_adv)
    perm = permutation_importance(gbm, X, y_adv, n_repeats=20, random_state=0, n_jobs=1)
    perm_imp = pd.Series(perm.importances_mean, index=features, name="gbm_perm_importance")
    perm_std = pd.Series(perm.importances_std, index=features, name="gbm_perm_std")
    gbm_r2 = float(gbm.score(X, y_adv))
    gbm_cv_r2 = float(np.mean(cross_val_score(
        GradientBoostingRegressor(random_state=0, n_estimators=400, max_depth=3, learning_rate=0.05),
        X, y_adv, cv=cv, scoring="r2",
    )))

    multi = (
        pd.concat([logit_coef, perm_imp, perm_std], axis=1)
        .reset_index(names="feature")
        .assign(abs_logit=lambda x: x["logit_coef"].abs())
        .sort_values("gbm_perm_importance", ascending=False)
        .reset_index(drop=True)
    )
    multi.to_csv(eval_dir / "driver_multivariate_importance.csv", index=False)

    # Multivariate figure.
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    logit_sorted = multi.sort_values("logit_coef")
    colors = ["#3498db" if f.startswith("A") else "#e67e22" for f in logit_sorted["feature"]]
    axes[0].barh(logit_sorted["feature"], logit_sorted["logit_coef"], color=colors)
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_title("Logistic regression on `win` (signed)", fontweight="bold")
    axes[0].set_xlabel("coefficient (standardized)")

    gbm_sorted = multi.sort_values("gbm_perm_importance")
    colors2 = ["#3498db" if f.startswith("A") else "#e67e22" for f in gbm_sorted["feature"]]
    axes[1].barh(
        gbm_sorted["feature"],
        gbm_sorted["gbm_perm_importance"],
        xerr=gbm_sorted["gbm_perm_std"],
        color=colors2,
        error_kw={"lw": 0.7},
    )
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_title("GBM permutation importance (advantage_pct)", fontweight="bold")
    axes[1].set_xlabel("mean drop in score when shuffled")

    plt.tight_layout()
    fig.savefig(eval_dir / "fig_driver_multivariate.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Headline summary.
    pooled_by_win = (
        pooled.assign(abs_w=lambda x: x["spearman_win"].abs())
        .sort_values("abs_w", ascending=False)
        .reset_index(drop=True)
    )

    wf_mean = (
        within_family.groupby("feature")["spearman_advantage"].mean().rename("wf_mean_rho")
    )
    compare = pooled[["feature", "spearman_advantage"]].merge(wf_mean, on="feature")
    compare["sign_flip"] = np.sign(compare["spearman_advantage"]) != np.sign(compare["wf_mean_rho"])
    sign_flips = compare[compare["sign_flip"]][["feature", "spearman_advantage", "wf_mean_rho"]]

    summary_payload = {
        "n_scenarios": int(len(df)),
        "family_breakdown": df["family"].value_counts().to_dict(),
        "win_rate_overall": float(df["win"].mean()),
        "mean_advantage_pct_overall": float(df["advantage_pct"].mean()),
        "median_advantage_pct_overall": float(df["advantage_pct"].median()),
        "by_family": (
            df.groupby("family")
            .agg(
                n=("win", "size"),
                win_rate=("win", "mean"),
                mean_advantage_pct=("advantage_pct", "mean"),
                median_advantage_pct=("advantage_pct", "median"),
            )
            .round(4)
            .reset_index()
            .to_dict(orient="records")
        ),
        "logit_train_accuracy": logit_acc,
        "logit_cv5_accuracy": logit_cv_acc,
        "logit_base_rate": float(y_win.mean()),
        "gbm_train_r2": gbm_r2,
        "gbm_cv5_r2": gbm_cv_r2,
        "top_drivers_by_advantage": pooled.head(top_n)[
            ["feature", "spearman_advantage", "p_advantage", "spearman_win", "p_win"]
        ].round(4).to_dict(orient="records"),
        "top_drivers_by_win": pooled_by_win.head(top_n)[
            ["feature", "spearman_win", "p_win", "spearman_advantage", "p_advantage"]
        ].round(4).to_dict(orient="records"),
        "top_gbm_features": multi.head(top_n)[
            ["feature", "gbm_perm_importance", "gbm_perm_std", "logit_coef"]
        ].round(4).to_dict(orient="records"),
        "pooled_vs_within_family_sign_flips": sign_flips.round(4).to_dict(orient="records"),
    }
    with (eval_dir / "driver_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2)

    # Console digest.
    print(f"== Criticality Driver Analysis @ {eval_dir} ==")
    print(f"n_scenarios={len(df)}  win_rate={summary_payload['win_rate_overall']:.3f}  "
          f"mean_adv={summary_payload['mean_advantage_pct_overall']:.2f}%  "
          f"median_adv={summary_payload['median_advantage_pct_overall']:.2f}%")
    print(
        f"LogReg train_acc={logit_acc:.3f} cv5_acc={logit_cv_acc:.3f} "
        f"(base rate={y_win.mean():.3f})  GBM train_R2={gbm_r2:.3f} cv5_R2={gbm_cv_r2:.3f}"
    )
    print("\nTop drivers by |Spearman vs advantage_pct|:")
    print(pooled.head(top_n)[["feature", "spearman_advantage", "p_advantage"]].to_string(index=False))
    print("\nTop drivers by GBM permutation importance:")
    print(multi.head(top_n)[["feature", "gbm_perm_importance", "logit_coef"]].to_string(index=False))
    if len(sign_flips):
        print("\nPooled vs within-family sign flips (possible family confounding):")
        print(sign_flips.to_string(index=False))
    else:
        print("\nNo sign flips between pooled and within-family rankings.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Directory containing matched_budget_per_scenario.csv and criticality_components_per_scenario.csv",
    )
    parser.add_argument("--top-k", type=int, default=6, help="Top-K features for decile plots.")
    parser.add_argument("--top-n", type=int, default=5, help="Top-N features in console / JSON summary.")
    args = parser.parse_args()
    run_analysis(args.eval_dir, top_k=args.top_k, top_n=args.top_n)


if __name__ == "__main__":
    main()
