"""
Applied Energy - Statistical analysis and central figure.
Generates:
  fig14 - Central "killer" figure: solve time vs complexity percentile (MILP vs pipeline)
  fig15 - Robustness: cost gap containment analysis
  
Prints:
  - Full statistical tables (mean +/- std, quantiles, win-rate)
  - Mann-Whitney U tests (per-family and overall)
  - Spearman/Pearson correlations (complexity components vs metrics)
  - LaTeX-ready tables
"""

import pickle, json, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats as sp_stats

# ---- paths ----
REPO = Path(r"C:\Users\Dell\projects\multilayer_milp_gnn\benchmark")
PKL  = REPO / "outputs" / "pipeline_eval_criticality" / "pipeline_eval_all_families.pkl"
OUT  = REPO / "outputs" / "pipeline_eval_criticality"
FAMILY_DIRS = {
    "low":    REPO / "outputs" / "low_criticality_scenarios",
    "medium": REPO / "outputs" / "medium_criticality_scenarios",
    "high":   REPO / "outputs" / "high_criticality_scenarios",
}

# ---- load & merge ----
with open(PKL, "rb") as f:
    raw = pickle.load(f)
df = pd.DataFrame(raw)

milp_rows = []
for family, fdir in FAMILY_DIRS.items():
    reports_dir = fdir / "reports"
    for rp in sorted(reports_dir.glob("*.json")):
        with open(rp) as f:
            r = json.load(f)
        sid = rp.stem
        mip = r["mip"]
        scenario_path = fdir / f"{sid}.json"
        meta = {}
        if scenario_path.exists():
            with open(scenario_path) as f:
                s = json.load(f)
            diff = s.get("difficulty_indicators", {})
            meta["vre_penetration_pct"] = diff.get("vre_penetration_pct", 0)
            meta["net_demand_volatility"] = diff.get("net_demand_volatility", 0)
            meta["peak_to_valley_ratio"] = diff.get("peak_to_valley_ratio", 0)
            meta["n_binary_variables"] = diff.get("n_binary_variables", 0)
            meta["complexity_score"] = diff.get("complexity_score", 0)
            meta["weather_profile"] = s.get("meta", {}).get("weather_profile", "unknown")
            meta["demand_scale_factor"] = s.get("meta", {}).get("demand_scale_factor", 1.0)
        milp_rows.append({
            "scenario_id": sid, "family": family,
            "milp_objective": mip["objective"],
            "milp_termination": mip["termination"],
            "milp_solve_seconds": mip["solve_seconds"],
            **meta,
        })

df_milp = pd.DataFrame(milp_rows)
df = df.merge(df_milp, on=["scenario_id", "family"], how="left")

# ---- derived ----
df["cost_gap_pct"] = (df["lp_objective"] - df["milp_objective"]) / df["milp_objective"].abs() * 100
df["speedup"] = df["milp_solve_seconds"] / df["time_total"]
df["problem_size"] = df["n_zones"] * df["n_timesteps"]
df["pipeline_better"] = df["cost_gap_pct"] < 0
df["is_timelimit"] = df["milp_termination"] == "maxTimeLimit"
df["abs_cost_gap_pct"] = df["cost_gap_pct"].abs()

family_order = ["low", "medium", "high"]
family_colors = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}
family_labels = {"low": "Low Criticality", "medium": "Medium Criticality", "high": "High Criticality"}

print(f"Loaded {len(df)} scenarios")

# =====================================================================
# FIG 14 - THE KILLER FIGURE: Solve time vs complexity percentile
# =====================================================================
print("\n=== FIG 14: Central figure ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle("", fontsize=1)  # no suptitle, let panels speak

# --- Panel A: Solve time vs criticality percentile (THE key plot) ---
ax = axes[0]

# Sort ALL scenarios by criticality index, compute percentile rank
df_sorted = df.sort_values("criticality_index").reset_index(drop=True)
df_sorted["crit_percentile"] = np.linspace(0, 100, len(df_sorted))

# Rolling median (window=15) for smooth curves
window = 15
milp_rolling = df_sorted["milp_solve_seconds"].rolling(window, center=True, min_periods=3).median()
pipe_rolling = df_sorted["time_total"].rolling(window, center=True, min_periods=3).median()

# Plot individual points with low alpha
for fam in family_order:
    mask = df_sorted["family"] == fam
    ax.scatter(df_sorted.loc[mask, "crit_percentile"],
               df_sorted.loc[mask, "milp_solve_seconds"],
               c=family_colors[fam], alpha=0.15, s=12, marker="^", zorder=1)
    ax.scatter(df_sorted.loc[mask, "crit_percentile"],
               df_sorted.loc[mask, "time_total"],
               c=family_colors[fam], alpha=0.15, s=12, marker="o", zorder=1)

# Smooth trend lines
ax.plot(df_sorted["crit_percentile"], milp_rolling, color="#c0392b", linewidth=2.5,
        label="MILP (rolling median)", zorder=3)
ax.plot(df_sorted["crit_percentile"], pipe_rolling, color="#27ae60", linewidth=2.5,
        label="Pipeline (rolling median)", zorder=3)

# Fill the divergence zone
ax.fill_between(df_sorted["crit_percentile"], pipe_rolling, milp_rolling,
                where=(milp_rolling > pipe_rolling),
                alpha=0.15, color="#27ae60", zorder=2, label="Pipeline advantage")
ax.fill_between(df_sorted["crit_percentile"], pipe_rolling, milp_rolling,
                where=(milp_rolling <= pipe_rolling),
                alpha=0.15, color="#c0392b", zorder=2, label="MILP advantage")

ax.set_yscale("log")
ax.set_xlabel("Complexity Percentile (%)", fontsize=11)
ax.set_ylabel("Solve Time (s, log scale)", fontsize=11)
ax.set_title("(A) Solve Time vs. Complexity Percentile", fontweight="bold", fontsize=12)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)
ax.set_xlim(0, 100)

# Annotate the crossover
crossover_idx = None
for i in range(len(df_sorted) - 1):
    m = milp_rolling.iloc[i]
    p = pipe_rolling.iloc[i]
    m_next = milp_rolling.iloc[i+1]
    p_next = pipe_rolling.iloc[i+1]
    if pd.notna(m) and pd.notna(p) and pd.notna(m_next) and pd.notna(p_next):
        if p > m and p_next <= m_next:
            crossover_idx = i
            break
        elif p <= m and p_next > m_next:
            crossover_idx = i
            break

if crossover_idx is not None:
    cx = df_sorted["crit_percentile"].iloc[crossover_idx]
    cy = milp_rolling.iloc[crossover_idx]
    ax.annotate(f"Crossover\n({cx:.0f}th pctile)",
                xy=(cx, cy), xytext=(cx + 10, cy * 3),
                fontsize=9, fontweight="bold", color="#2c3e50",
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2c3e50", alpha=0.9))

# --- Panel B: Speedup vs criticality index (scatter + regression) ---
ax = axes[1]
for fam in family_order:
    fdf = df[df["family"] == fam]
    ax.scatter(fdf["criticality_index"], fdf["speedup"], c=family_colors[fam],
               alpha=0.5, s=25, label=family_labels[fam], edgecolors="white", linewidths=0.3)

ax.axhline(1, color="red", linestyle="--", linewidth=1.2, label="Break-even (1x)")
ax.set_yscale("log")

# Spearman correlation
rho_s, p_s = sp_stats.spearmanr(df["criticality_index"], df["speedup"])
ax.text(0.05, 0.95, f"Spearman $\\rho$ = {rho_s:.3f}\n$p$ = {p_s:.1e}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

ax.set_xlabel("Criticality Index", fontsize=11)
ax.set_ylabel("Speedup (x, log scale)", fontsize=11)
ax.set_title("(B) Speedup vs. Criticality Index", fontweight="bold", fontsize=12)
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

# --- Panel C: Cost gap bounded envelope (clipped to readable range) ---
ax = axes[2]

# Clip cost gap to [-200, 200] for visualization (outliers noted in caption)
gap_clipped = df["cost_gap_pct"].clip(-200, 200)

# Compute percentile-based envelope on clipped data
pctiles = np.arange(0, 101, 2)
crit_bins = np.percentile(df["criticality_index"], pctiles)

medians, p10s, p90s, p25s, p75s, xs = [], [], [], [], [], []
for i in range(len(crit_bins) - 1):
    mask = (df["criticality_index"] >= crit_bins[i]) & (df["criticality_index"] < crit_bins[i+1])
    if mask.sum() >= 3:
        gaps = gap_clipped[mask]
        medians.append(gaps.median())
        p10s.append(gaps.quantile(0.10))
        p90s.append(gaps.quantile(0.90))
        p25s.append(gaps.quantile(0.25))
        p75s.append(gaps.quantile(0.75))
        xs.append((pctiles[i] + pctiles[i+1]) / 2)

xs = np.array(xs)
medians = np.array(medians)
p10s = np.array(p10s)
p90s = np.array(p90s)
p25s = np.array(p25s)
p75s = np.array(p75s)

ax.fill_between(xs, p10s, p90s, alpha=0.15, color="#3498db", label="P10--P90 envelope")
ax.fill_between(xs, p25s, p75s, alpha=0.3, color="#3498db", label="IQR envelope")
ax.plot(xs, medians, color="#2c3e50", linewidth=2.5, label="Median cost gap", zorder=3)
ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax.axhline(20, color="#e74c3c", linestyle=":", linewidth=1, alpha=0.7, label="+20% threshold")
ax.axhline(-20, color="#e74c3c", linestyle=":", linewidth=1, alpha=0.7, label="-20% threshold")

ax.set_xlabel("Complexity Percentile (%)", fontsize=11)
ax.set_ylabel("Cost Gap (%, clipped to [-200, 200])", fontsize=11)
ax.set_title("(C) Cost Gap Envelope vs. Complexity", fontweight="bold", fontsize=12)
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(-220, 220)

plt.tight_layout()
fig.savefig(OUT / "fig14_central_divergence.png", dpi=300, bbox_inches="tight")
plt.close()
print("  -> Saved fig14_central_divergence.png")


# =====================================================================
# FIG 15 - ROBUSTNESS: cost gap containment + bounded degradation
# =====================================================================
print("\n=== FIG 15: Robustness analysis ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# --- Panel A: Empirical CDF of |cost gap| per family ---
ax = axes[0]
for fam in family_order:
    fdf = df[df["family"] == fam]
    sorted_gaps = np.sort(fdf["abs_cost_gap_pct"].values)
    ecdf_y = np.arange(1, len(sorted_gaps) + 1) / len(sorted_gaps) * 100
    ax.plot(sorted_gaps, ecdf_y, color=family_colors[fam], linewidth=2, label=family_labels[fam])

# All
sorted_all = np.sort(df["abs_cost_gap_pct"].values)
ecdf_all = np.arange(1, len(sorted_all) + 1) / len(sorted_all) * 100
ax.plot(sorted_all, ecdf_all, color="#2c3e50", linewidth=2.5, linestyle="--", label="All (n=300)")

ax.axvline(5, color="gray", linestyle=":", alpha=0.5)
ax.axvline(20, color="gray", linestyle=":", alpha=0.5)
ax.axvline(50, color="gray", linestyle=":", alpha=0.5)
ax.text(5, 102, "5%", ha="center", fontsize=8, color="gray")
ax.text(20, 102, "20%", ha="center", fontsize=8, color="gray")
ax.text(50, 102, "50%", ha="center", fontsize=8, color="gray")

ax.set_xlabel("|Cost Gap| (%)", fontsize=11)
ax.set_ylabel("Cumulative % of Scenarios", fontsize=11)
ax.set_title("(A) Empirical CDF of |Cost Gap|", fontweight="bold", fontsize=12)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_xlim(0, 200)
ax.set_ylim(0, 105)

# --- Panel B: Cost gap vs MILP termination (optimal vs time-limited) ---
ax = axes[1]
df_opt = df[df["milp_termination"] == "optimal"]
df_tl = df[df["is_timelimit"]]

bp = ax.boxplot(
    [df_opt["cost_gap_pct"].clip(-300, 300).values, df_tl["cost_gap_pct"].clip(-300, 300).values],
    positions=[0, 1], widths=0.5, patch_artist=True,
    showfliers=True, flierprops=dict(marker="o", markersize=3, alpha=0.4)
)
bp["boxes"][0].set_facecolor("#3498db")
bp["boxes"][0].set_alpha(0.7)
bp["boxes"][1].set_facecolor("#e67e22")
bp["boxes"][1].set_alpha(0.7)

ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax.set_xticks([0, 1])
ax.set_xticklabels([f"MILP Optimal\n(n={len(df_opt)})", f"MILP Time-Limited\n(n={len(df_tl)})"])
ax.set_ylabel("Cost Gap (%)", fontsize=11)
ax.set_title("(B) Cost Gap by MILP Termination", fontweight="bold", fontsize=12)
ax.grid(axis="y", alpha=0.3)

# Mann-Whitney between the two groups
u_stat, u_p = sp_stats.mannwhitneyu(df_opt["cost_gap_pct"].dropna(),
                                      df_tl["cost_gap_pct"].dropna(),
                                      alternative="two-sided")
ax.text(0.5, 0.95, f"Mann-Whitney U = {u_stat:.0f}\n$p$ = {u_p:.2e}",
        transform=ax.transAxes, fontsize=9, va="top", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

# --- Panel C: Win-rate curve as a function of complexity ---
ax = axes[2]

# For each complexity percentile bucket, compute win-rate (pipeline < MILP)
win_rates = []
speedup_above_1 = []
xs_wr = []
bucket_size = 20  # scenarios per bucket

df_by_crit = df.sort_values("criticality_index").reset_index(drop=True)
for start in range(0, len(df_by_crit) - bucket_size + 1, 5):
    chunk = df_by_crit.iloc[start:start + bucket_size]
    wr = (chunk["pipeline_better"].sum() / len(chunk)) * 100
    sr = ((chunk["speedup"] > 1).sum() / len(chunk)) * 100
    pct = (start + bucket_size / 2) / len(df_by_crit) * 100
    win_rates.append(wr)
    speedup_above_1.append(sr)
    xs_wr.append(pct)

ax.plot(xs_wr, win_rates, color="#27ae60", linewidth=2.5, label="Cost win-rate (%)")
ax.plot(xs_wr, speedup_above_1, color="#2980b9", linewidth=2.5, label="Speed win-rate (%)")
ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.fill_between(xs_wr, 0, win_rates, alpha=0.1, color="#27ae60")
ax.fill_between(xs_wr, 0, speedup_above_1, alpha=0.1, color="#2980b9")

ax.set_xlabel("Complexity Percentile (%)", fontsize=11)
ax.set_ylabel("Win-Rate (%)", fontsize=11)
ax.set_title("(C) Pipeline Win-Rate vs. Complexity", fontweight="bold", fontsize=12)
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(0, 105)

plt.tight_layout()
fig.savefig(OUT / "fig15_robustness_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
print("  -> Saved fig15_robustness_analysis.png")


# =====================================================================
# STATISTICAL TABLES - LaTeX output
# =====================================================================
print("\n" + "="*80)
print("STATISTICAL ANALYSIS FOR LATEX")
print("="*80)

# --- Table: Comprehensive statistics (mean +/- std, quantiles, win-rate) ---
print("\n% --- Comprehensive per-family statistics table ---")
print("\\begin{table}[htbp]")
print("\\centering")
print("\\small")
print("\\setlength{\\tabcolsep}{4pt}")
print("\\begin{tabular}{lrllllrr}")
print("\\toprule")
print("\\textbf{Family} & \\textbf{$n$} & \\textbf{Gap (mean$\\pm$std)} & \\textbf{Gap median} & \\textbf{$S$ (mean$\\pm$std)} & \\textbf{$S$ median} & \\textbf{Cost WR} & \\textbf{Speed WR} \\\\")
print(" & & (\\%) & (\\%) & ($\\times$) & ($\\times$) & (\\%) & (\\%) \\\\")
print("\\midrule")

for fam in family_order + ["all"]:
    fdf = df if fam == "all" else df[df["family"] == fam]
    label = "\\textbf{All}" if fam == "all" else fam.capitalize()
    n = len(fdf)
    gap = fdf["cost_gap_pct"]
    spd = fdf["speedup"]
    cost_wr = (fdf["pipeline_better"].sum() / n * 100)
    speed_wr = ((spd > 1).sum() / n * 100)

    gap_str = f"${gap.mean():.1f} \\pm {gap.std():.1f}$"
    spd_str = f"${spd.mean():.2f} \\pm {spd.std():.2f}$"
    med_gap = f"${gap.median():.1f}$"
    med_spd = f"${spd.median():.2f}$"

    if fam == "all":
        print("\\midrule")
    print(f"{label} & {n} & {gap_str} & {med_gap} & {spd_str} & {med_spd} & {cost_wr:.0f} & {speed_wr:.0f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Comprehensive evaluation statistics. Gap = cost gap (\\%); $S$ = speedup; Cost WR = cost win-rate (pipeline better than MILP); Speed WR = speed win-rate (pipeline faster than MILP).}")
print("\\label{tab:comprehensive_stats}")
print("\\end{table}")


# --- Mann-Whitney U tests ---
print("\n\n% --- Mann-Whitney U tests ---")
print("\\begin{table}[htbp]")
print("\\centering")
print("\\small")
print("\\begin{tabular}{llrrp{4cm}}")
print("\\toprule")
print("\\textbf{Comparison} & \\textbf{Metric} & \\textbf{$U$ statistic} & \\textbf{$p$-value} & \\textbf{Interpretation} \\\\")
print("\\midrule")

# Test: cost gap differs between families
for fam_a, fam_b in [("low", "medium"), ("low", "high"), ("medium", "high")]:
    ga = df[df["family"] == fam_a]["cost_gap_pct"].dropna()
    gb = df[df["family"] == fam_b]["cost_gap_pct"].dropna()
    u, p = sp_stats.mannwhitneyu(ga, gb, alternative="two-sided")
    sig = "$^{***}$" if p < 0.001 else ("$^{**}$" if p < 0.01 else ("$^{*}$" if p < 0.05 else "n.s."))
    print(f"{fam_a.capitalize()} vs {fam_b.capitalize()} & Cost gap & {u:.0f} & {p:.2e} & {sig} \\\\")

# Test: speedup differs between families
for fam_a, fam_b in [("low", "medium"), ("low", "high"), ("medium", "high")]:
    sa = df[df["family"] == fam_a]["speedup"].dropna()
    sb = df[df["family"] == fam_b]["speedup"].dropna()
    u, p = sp_stats.mannwhitneyu(sa, sb, alternative="two-sided")
    sig = "$^{***}$" if p < 0.001 else ("$^{**}$" if p < 0.01 else ("$^{*}$" if p < 0.05 else "n.s."))
    print(f"{fam_a.capitalize()} vs {fam_b.capitalize()} & Speedup & {u:.0f} & {p:.2e} & {sig} \\\\")

# Test: cost gap vs 0 (Wilcoxon signed-rank)
print("\\midrule")
for fam in family_order + ["all"]:
    fdf = df if fam == "all" else df[df["family"] == fam]
    label = "All" if fam == "all" else fam.capitalize()
    gap = fdf["cost_gap_pct"].dropna()
    w, p = sp_stats.wilcoxon(gap)
    sig = "$^{***}$" if p < 0.001 else ("$^{**}$" if p < 0.01 else ("$^{*}$" if p < 0.05 else "n.s."))
    print(f"{label} & Gap $\\neq$ 0 (Wilcoxon) & {w:.0f} & {p:.2e} & {sig} \\\\")

# Test: speedup vs 1 (Wilcoxon signed-rank on speedup - 1)
for fam in family_order + ["all"]:
    fdf = df if fam == "all" else df[df["family"] == fam]
    label = "All" if fam == "all" else fam.capitalize()
    spd = fdf["speedup"].dropna() - 1
    w, p = sp_stats.wilcoxon(spd)
    sig = "$^{***}$" if p < 0.001 else ("$^{**}$" if p < 0.01 else ("$^{*}$" if p < 0.05 else "n.s."))
    print(f"{label} & $S \\neq 1$ (Wilcoxon) & {w:.0f} & {p:.2e} & {sig} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Non-parametric hypothesis tests. Mann-Whitney $U$ tests assess whether distributions differ between families. Wilcoxon signed-rank tests assess whether medians differ from the null (gap $= 0$\\%, speedup $= 1\\times$). $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$.}")
print("\\label{tab:stat_tests}")
print("\\end{table}")


# --- Correlation table ---
print("\n\n% --- Spearman correlation table ---")
print("\\begin{table}[htbp]")
print("\\centering")
print("\\small")
print("\\begin{tabular}{llrr}")
print("\\toprule")
print("\\textbf{Complexity Indicator} & \\textbf{Outcome Metric} & \\textbf{Spearman $\\rho$} & \\textbf{$p$-value} \\\\")
print("\\midrule")

complexity_vars = [
    ("criticality_index", "Criticality index"),
    ("n_zones", "Number of zones"),
    ("problem_size", "Problem size ($Z \\times T$)"),
    ("n_binary_variables", "Binary variables"),
]
outcome_vars = [
    ("milp_solve_seconds", "MILP solve time"),
    ("time_total", "Pipeline solve time"),
    ("speedup", "Speedup"),
    ("cost_gap_pct", "Cost gap (\\%)"),
]

for cv, cv_label in complexity_vars:
    for ov, ov_label in outcome_vars:
        vals_c = df[cv].dropna()
        vals_o = df[ov].dropna()
        common = vals_c.index.intersection(vals_o.index)
        rho, p = sp_stats.spearmanr(df.loc[common, cv], df.loc[common, ov])
        sig = "$^{***}$" if p < 0.001 else ("$^{**}$" if p < 0.01 else ("$^{*}$" if p < 0.05 else ""))
        print(f"{cv_label} & {ov_label} & {rho:+.3f}{sig} & {p:.1e} \\\\")
    print("\\addlinespace")

print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Spearman rank correlations between complexity indicators and outcome metrics across all 300 evaluation scenarios. All correlations are computed on the full sample. $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$.}")
print("\\label{tab:correlations}")
print("\\end{table}")


# --- Key numbers for formal claim ---
print("\n\n% --- Key numbers for formal claim ---")
rho_crit_spd, p_crit_spd = sp_stats.spearmanr(df["criticality_index"], df["speedup"])
rho_crit_milp, p_crit_milp = sp_stats.spearmanr(df["criticality_index"], df["milp_solve_seconds"])
rho_size_spd, p_size_spd = sp_stats.spearmanr(df["problem_size"], df["speedup"])
print(f"Spearman(criticality, speedup) = {rho_crit_spd:.3f}, p = {p_crit_spd:.1e}")
print(f"Spearman(criticality, milp_time) = {rho_crit_milp:.3f}, p = {p_crit_milp:.1e}")
print(f"Spearman(problem_size, speedup) = {rho_size_spd:.3f}, p = {p_size_spd:.1e}")

# Win-rate by family
for fam in family_order:
    fdf = df[df["family"] == fam]
    cost_wr = fdf["pipeline_better"].sum() / len(fdf) * 100
    speed_wr = (fdf["speedup"] > 1).sum() / len(fdf) * 100
    print(f"{fam}: cost_win_rate={cost_wr:.0f}%, speed_win_rate={speed_wr:.0f}%")

# Overall
cost_wr_all = df["pipeline_better"].sum() / len(df) * 100
speed_wr_all = (df["speedup"] > 1).sum() / len(df) * 100
print(f"ALL: cost_win_rate={cost_wr_all:.0f}%, speed_win_rate={speed_wr_all:.0f}%")

# Robustness: what fraction have |gap| < 50% across all families?
for thresh in [5, 10, 20, 50, 100]:
    frac = (df["abs_cost_gap_pct"] <= thresh).sum() / len(df) * 100
    print(f"|gap| <= {thresh}%: {frac:.0f}% of all scenarios")

# High-crit robustness
hdf = df[df["family"] == "high"]
for thresh in [5, 10, 20, 50, 100]:
    frac = (hdf["abs_cost_gap_pct"] <= thresh).sum() / len(hdf) * 100
    print(f"HIGH |gap| <= {thresh}%: {frac:.0f}% of high-crit scenarios")

print("\n=== Done ===")
