"""Full partial-correlation analysis of criticality drivers vs pipeline advantage.

For each of the 22 normalized A/B components, compute the partial Spearman
correlation with `advantage_pct` (and with `win`) under three control sets:

1. **None** -- raw Spearman ρ (sanity baseline; matches `driver_ranking_pooled.csv`).
2. **`B7_ramp_tightness` only** -- isolates "what survives after stripping the
   dominant driver?".
3. **All other 21 components** -- full partial correlation; this is the
   standard "everything else held constant" view.
4. **B7 + family** -- single-control variant that also strips the
   low/medium/high family bin (controls for the obvious confound).

Implementation uses rank-transformed data and the precision-matrix identity
``ρ_ij | rest = -P_ij / sqrt(P_ii * P_jj)`` for view (3); for the smaller
control sets we use the regress-residuals method.

Outputs (in --eval-dir):
- ``partial_correlation_summary.csv`` -- one row per (target, feature) with
  raw and partial ρ under all four control sets, plus z-tests.
- ``fig_partial_corr_advantage.png`` -- side-by-side raw vs full-partial ρ
  bar chart for `advantage_pct`.
- ``fig_partial_corr_win.png`` -- same for the binary win flag.
- ``fig_partial_corr_full_heatmap.png`` -- 24x24 partial-correlation
  heatmap of all features + advantage + win (for completeness).

Example
-------
python scripts/run_partial_correlation_analysis.py \
    --eval-dir outputs/uc_jepa/latent_vs_grid_milp_120_caps_k1
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import spearmanr

sns.set_theme(style="whitegrid")
sns.set_context("talk", font_scale=0.85)

INFEASIBLE_GAP = 1000.0
PRIMARY_CONTROL = "B7_ramp_tightness_norm"
FAMILY_ORDER = ["low", "medium", "high"]


# ---------------------------------------------------------------------------
# I/O + advantage/win definitions (kept identical to run_criticality_driver_analysis.py)
# ---------------------------------------------------------------------------

def _coerce_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def _load(eval_dir: Path) -> pd.DataFrame:
    per = pd.read_csv(eval_dir / "matched_budget_per_scenario.csv")
    comps = pd.read_csv(eval_dir / "criticality_components_per_scenario.csv")
    df = per.merge(comps, on=["family", "scenario_id"], how="left", validate="one_to_one")
    for col in ["pipeline_feasible_within_budget", "milp_feasible_within_budget"]:
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
    return df


# ---------------------------------------------------------------------------
# Partial-correlation primitives
# ---------------------------------------------------------------------------

def _rank_standardize(arr: np.ndarray) -> np.ndarray:
    """Column-wise rank-transform then z-score. Constant columns become zeros."""
    out = np.empty_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        # Average rank for ties.
        order = np.argsort(col, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(col), dtype=float)
        # Tie correction (mean rank).
        unique_vals, inv, counts = np.unique(col, return_inverse=True, return_counts=True)
        if (counts > 1).any():
            sums = np.zeros_like(unique_vals, dtype=float)
            np.add.at(sums, inv, ranks)
            mean_rank = sums / counts
            ranks = mean_rank[inv]
        std = ranks.std(ddof=0)
        if std == 0:
            out[:, j] = 0.0
        else:
            out[:, j] = (ranks - ranks.mean()) / std
    return out


def _partial_corr_via_precision(Z: np.ndarray) -> np.ndarray:
    """Full partial-correlation matrix from rank-standardized data Z (n, p).

    ρ_ij | rest = -P_ij / sqrt(P_ii * P_jj) where P = (Z^T Z / (n-1))^{-1}.

    Uses pseudo-inverse if Σ is rank-deficient.
    """
    n = Z.shape[0]
    sigma = (Z.T @ Z) / max(n - 1, 1)
    try:
        precision = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(sigma)
    diag = np.sqrt(np.clip(np.diag(precision), 1e-12, None))
    out = -precision / np.outer(diag, diag)
    np.fill_diagonal(out, 1.0)
    return out


def _residualize(Z: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Return Z - C (C^T C)^{-1} C^T Z, residuals of Z columns against controls.

    Adds an implicit intercept column to controls.
    """
    n = Z.shape[0]
    intercept = np.ones((n, 1))
    if controls.size == 0:
        C = intercept
    else:
        C = np.hstack([intercept, controls])
    # Least-squares projection.
    beta, *_ = np.linalg.lstsq(C, Z, rcond=None)
    return Z - C @ beta


def _partial_corr_subset(
    z_target: np.ndarray, z_features: np.ndarray, z_controls: np.ndarray
) -> np.ndarray:
    """Partial Spearman of each feature vs target, controlling for `controls`.

    All inputs are rank-standardized. Returns one ρ per feature column.
    """
    target_resid = _residualize(z_target.reshape(-1, 1), z_controls).ravel()
    features_resid = _residualize(z_features, z_controls)
    out = np.empty(z_features.shape[1])
    sd_t = target_resid.std(ddof=0)
    for j in range(z_features.shape[1]):
        col = features_resid[:, j]
        sd_c = col.std(ddof=0)
        if sd_t == 0 or sd_c == 0:
            out[j] = np.nan
        else:
            out[j] = float(np.mean(col * target_resid) / (sd_t * sd_c))
    return out


def _t_pvalue(rho: float, n: int, k_controls: int) -> float:
    """Two-sided t-test for a partial correlation."""
    df = n - 2 - k_controls
    if df <= 0 or not np.isfinite(rho) or abs(rho) >= 1.0:
        return float("nan")
    t = rho * np.sqrt(df / max(1.0 - rho ** 2, 1e-12))
    # 2-sided p from t-distribution; small-table approximation via scipy.
    from scipy.stats import t as student_t

    return float(2.0 * (1.0 - student_t.cdf(abs(t), df)))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _identify_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    stress = sorted(c for c in df.columns if c.endswith("_norm") and c.startswith("A"))
    hardness = sorted(c for c in df.columns if c.endswith("_norm") and c.startswith("B"))
    return stress, hardness


def _drop_constant(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    keep, dropped = [], []
    for c in cols:
        if df[c].nunique(dropna=True) > 1:
            keep.append(c)
        else:
            dropped.append(c)
    return keep, dropped


def _bar_compare(
    summary: pd.DataFrame,
    target: str,
    output_path: Path,
    title: str,
) -> None:
    sub = summary[summary["target"] == target].copy()
    # Order by |full_partial|.
    sub["abs_full"] = sub["partial_full"].abs()
    sub = sub.sort_values("abs_full", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(sub) + 2))
    y = np.arange(len(sub))
    ax.barh(y - 0.20, sub["raw_spearman"], height=0.36, color="#cccccc",
            edgecolor="black", lw=0.4, label="raw ρ")
    colors = ["#3498db" if f.startswith("A") else "#e67e22" for f in sub["feature"]]
    ax.barh(y + 0.20, sub["partial_full"], height=0.36, color=colors,
            edgecolor="black", lw=0.4, label="partial ρ | all others")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace("_norm", "") for f in sub["feature"]])
    ax.set_xlabel(f"Spearman ρ vs {target}", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(handles=[
        Patch(facecolor="#cccccc", edgecolor="black", label="raw ρ"),
        Patch(facecolor="#3498db", edgecolor="black", label="partial ρ (stress driver)"),
        Patch(facecolor="#e67e22", edgecolor="black", label="partial ρ (hardness driver)"),
    ], loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _heatmap_full(P: np.ndarray, names: Sequence[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(0.45 * len(names) + 2, 0.45 * len(names) + 2))
    sns.heatmap(
        pd.DataFrame(P, index=names, columns=names),
        cmap="coolwarm", center=0.0, vmin=-1.0, vmax=1.0,
        annot=True, fmt=".2f", annot_kws={"size": 6},
        square=True, cbar_kws={"label": "partial ρ | all others"},
        ax=ax,
    )
    ax.set_title("Full partial-correlation matrix (rank-based)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(eval_dir: Path) -> pd.DataFrame:
    df = _load(eval_dir)
    stress, hardness = _identify_features(df)
    feat_all = stress + hardness
    feat_kept, feat_dropped = _drop_constant(df, feat_all)
    if feat_dropped:
        print(f"[info] dropping constant features: {feat_dropped}")

    n = len(df)
    targets = ["advantage_pct", "win"]

    # Build the full augmented matrix (features + targets) for the precision-based
    # partial-corr view. Rank-transform every column.
    cols_full = feat_kept + targets
    Z_full = _rank_standardize(df[cols_full].astype(float).to_numpy())
    P_full = _partial_corr_via_precision(Z_full)
    P_full_df = pd.DataFrame(P_full, index=cols_full, columns=cols_full)
    P_full_df.to_csv(eval_dir / "partial_correlation_full_matrix.csv")

    # Family encoded as ordered int (low=0, medium=1, high=2).
    family_int = df["family"].astype(str).map({f: i for i, f in enumerate(FAMILY_ORDER)})
    if family_int.isna().any():
        # Unknown families → label-encode them after the known ones.
        unknown = sorted(df["family"][family_int.isna()].unique())
        for k, fam in enumerate(unknown):
            family_int = family_int.fillna({i: len(FAMILY_ORDER) + k for i in df.index[df["family"] == fam]})
        family_int = family_int.astype(float)
    Z_family = _rank_standardize(family_int.to_numpy().reshape(-1, 1))

    # Per-target rows.
    rows = []
    for target in targets:
        Z_target = _rank_standardize(df[[target]].astype(float).to_numpy()).ravel()
        Z_features = _rank_standardize(df[feat_kept].astype(float).to_numpy())

        # Raw Spearman from the target column.
        raw_rho = []
        raw_p = []
        for f in feat_kept:
            x = df[f].astype(float)
            y = df[target].astype(float)
            mask = x.notna() & y.notna()
            if mask.sum() < 5 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
                raw_rho.append(float("nan"))
                raw_p.append(float("nan"))
            else:
                rho, p = spearmanr(x[mask], y[mask])
                raw_rho.append(float(rho))
                raw_p.append(float(p))

        # Partial ρ controlling for B7 alone.
        if PRIMARY_CONTROL in feat_kept:
            Z_b7 = Z_features[:, feat_kept.index(PRIMARY_CONTROL)].reshape(-1, 1)
            partial_b7 = _partial_corr_subset(Z_target, Z_features, Z_b7)
        else:
            partial_b7 = np.full(len(feat_kept), np.nan)

        # Partial ρ controlling for family + B7.
        if PRIMARY_CONTROL in feat_kept:
            Z_fb7 = np.hstack([Z_family, Z_b7])
        else:
            Z_fb7 = Z_family
        partial_fb7 = _partial_corr_subset(Z_target, Z_features, Z_fb7)

        # Full partial ρ (control = all other 21 features).
        target_idx = cols_full.index(target)
        partial_full = []
        for f in feat_kept:
            j = cols_full.index(f)
            partial_full.append(float(P_full[target_idx, j]))
        partial_full = np.array(partial_full)

        for i, f in enumerate(feat_kept):
            rows.append({
                "target": target,
                "feature": f,
                "raw_spearman": raw_rho[i],
                "raw_p": raw_p[i],
                "partial_b7_only": float(partial_b7[i]) if PRIMARY_CONTROL in feat_kept else float("nan"),
                "partial_b7_only_p": _t_pvalue(float(partial_b7[i]), n, 1)
                if PRIMARY_CONTROL in feat_kept else float("nan"),
                "partial_family_b7": float(partial_fb7[i]),
                "partial_family_b7_p": _t_pvalue(
                    float(partial_fb7[i]), n, 2 if PRIMARY_CONTROL in feat_kept else 1
                ),
                "partial_full": float(partial_full[i]),
                "partial_full_p": _t_pvalue(float(partial_full[i]), n, len(feat_kept) - 1),
                # Self-control case (e.g. B7 vs advantage controlling for B7) is degenerate;
                # we leave the value but flag it with the existing partial_b7_only being NaN/0.
            })

        if f == feat_kept[-1]:
            pass

    summary = pd.DataFrame(rows)
    # Mark B7's own "controlling for B7" entry as NaN to avoid confusion.
    if PRIMARY_CONTROL in feat_kept:
        mask = summary["feature"] == PRIMARY_CONTROL
        summary.loc[mask, ["partial_b7_only", "partial_b7_only_p"]] = float("nan")
    summary.sort_values(["target", "feature"], inplace=True)
    summary.to_csv(eval_dir / "partial_correlation_summary.csv", index=False)

    # Bar charts.
    _bar_compare(
        summary, "advantage_pct",
        eval_dir / "fig_partial_corr_advantage.png",
        "Raw vs full-partial Spearman ρ (advantage_pct)",
    )
    _bar_compare(
        summary, "win",
        eval_dir / "fig_partial_corr_win.png",
        "Raw vs full-partial Spearman ρ (win)",
    )

    # Full heatmap.
    short_names = [c.replace("_norm", "") for c in cols_full]
    _heatmap_full(P_full, short_names, eval_dir / "fig_partial_corr_full_heatmap.png")

    # Console digest.
    print("=== Partial-correlation analysis ===")
    print(f"n_scenarios={n}, n_features={len(feat_kept)} (dropped {feat_dropped})")
    for target in targets:
        sub = summary[summary["target"] == target].copy()
        sub["abs_full"] = sub["partial_full"].abs()
        top = sub.sort_values("abs_full", ascending=False).head(5)
        print(f"\nTop 5 features by |partial ρ vs {target}| (control = all 21 others):")
        for _, row in top.iterrows():
            stars = (
                "***" if row["partial_full_p"] < 0.001
                else "**" if row["partial_full_p"] < 0.01
                else "*" if row["partial_full_p"] < 0.05
                else ""
            )
            print(
                f"  {row['feature']:35s} raw={row['raw_spearman']:+.2f}  "
                f"|B7={row['partial_b7_only']:+.2f}  |fam+B7={row['partial_family_b7']:+.2f}  "
                f"|all={row['partial_full']:+.2f}{stars}"
            )

    # Highlight features where the dominant raw effect SURVIVES vs DISAPPEARS once
    # we control for B7 alone -- the supervisor's direct question.
    print("\n--- Features whose raw effect on advantage_pct survives vs collapses after controlling for B7 ---")
    sub = summary[(summary["target"] == "advantage_pct") & (summary["feature"] != PRIMARY_CONTROL)].copy()
    sub["raw_abs"] = sub["raw_spearman"].abs()
    sub["b7_abs"] = sub["partial_b7_only"].abs()
    sub["delta_abs"] = sub["raw_abs"] - sub["b7_abs"]
    sub = sub.sort_values("raw_abs", ascending=False).head(10)
    for _, row in sub.iterrows():
        verdict = (
            "SURVIVES" if row["b7_abs"] >= 0.5 * row["raw_abs"] and row["b7_abs"] >= 0.1
            else "WEAKENED"
            if row["b7_abs"] >= 0.1
            else "ABSORBED by B7"
        )
        print(
            f"  {row['feature']:35s} raw={row['raw_spearman']:+.2f} -> |B7={row['partial_b7_only']:+.2f}  [{verdict}]"
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.eval_dir)


if __name__ == "__main__":
    main()
