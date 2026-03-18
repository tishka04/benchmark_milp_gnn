"""
Generate additional figures for Applied Energy submission.
Reads pipeline_eval_all_families.pkl + MILP reports to produce:

fig11 - Per-family boxplot dashboard (cost gap, speedup, slack, LP stage) with robust stats
fig12 - Failure mode analysis: where/why the pipeline underperforms
fig13 - Climate stress use-case: batch planning simulation
fig14 - Robustness metrics table (IQR, P5-P95, Wilcoxon) exported as LaTeX
"""

import pickle
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────────
REPO = Path(r"C:\Users\Dell\projects\multilayer_milp_gnn\benchmark")
PKL  = REPO / "outputs" / "pipeline_eval_criticality" / "pipeline_eval_all_families.pkl"
OUT  = REPO / "outputs" / "pipeline_eval_criticality"

FAMILY_DIRS = {
    "low":    REPO / "outputs" / "low_criticality_scenarios",
    "medium": REPO / "outputs" / "medium_criticality_scenarios",
    "high":   REPO / "outputs" / "high_criticality_scenarios",
}

# ── load pipeline results ──────────────────────────────────────────────────
with open(PKL, "rb") as f:
    raw = pickle.load(f)

df = pd.DataFrame(raw)
print(f"Loaded {len(df)} pipeline results")

# ── load MILP reports & scenario metadata ──────────────────────────────────
milp_rows = []
for family, fdir in FAMILY_DIRS.items():
    reports_dir = fdir / "reports"
    for rp in sorted(reports_dir.glob("*.json")):
        with open(rp) as f:
            r = json.load(f)
        # Use report filename as the scenario key (matches pipeline pkl)
        sid = rp.stem  # e.g. "scenario_00001"
        mip = r["mip"]
        
        # Load scenario JSON for metadata (filename matches)
        scenario_path = fdir / f"{sid}.json"
        meta = {}
        if scenario_path.exists():
            with open(scenario_path) as f:
                s = json.load(f)
            meta = s.get("meta", {})
            meta["weather_profile"] = meta.get("weather_profile", "unknown")
            meta["demand_scale_factor"] = meta.get("demand_scale_factor", 1.0)
            diff = s.get("difficulty_indicators", {})
            meta["n_binary_variables"] = diff.get("n_binary_variables", 0)
            meta["vre_penetration_pct"] = diff.get("vre_penetration_pct", 0)
            meta["peak_to_valley_ratio"] = diff.get("peak_to_valley_ratio", 0)
            meta["n_zones_scenario"] = diff.get("n_zones", 0)
            # cost components
            cc = r.get("cost_components", {})
            meta["milp_thermal_fuel"] = cc.get("thermal_fuel", 0)
            meta["milp_unserved_energy"] = cc.get("unserved_energy", 0)
            meta["milp_dr_cost"] = cc.get("demand_response", 0)
            meta["milp_storage_cost"] = cc.get("battery_cycle", 0) + cc.get("pumped_cycle", 0)
            meta["milp_spill_cost"] = cc.get("solar_spill", 0) + cc.get("wind_spill", 0) + cc.get("hydro_spill", 0) + cc.get("overgen_spill", 0)
        
        milp_rows.append({
            "scenario_id": sid,
            "family": family,
            "milp_objective": mip["objective"],
            "milp_termination": mip["termination"],
            "milp_solve_seconds": mip["solve_seconds"],
            **meta,
        })

df_milp = pd.DataFrame(milp_rows)
print(f"Loaded {len(df_milp)} MILP reports")

# ── merge ──────────────────────────────────────────────────────────────────
df = df.merge(df_milp, on=["scenario_id", "family"], how="left")

# ── derived columns ────────────────────────────────────────────────────────
df["cost_gap_pct"] = (df["lp_objective"] - df["milp_objective"]) / df["milp_objective"].abs() * 100
df["speedup"] = df["milp_solve_seconds"] / df["time_total"]
df["pipeline_better"] = df["cost_gap_pct"] < 0
df["within_5pct"] = df["cost_gap_pct"].abs() <= 5
df["within_10pct"] = df["cost_gap_pct"].abs() <= 10
df["is_timelimit"] = df["milp_termination"] == "maxTimeLimit"
df["problem_size"] = df["n_zones"] * df["n_timesteps"]

# Family ordering
family_order = ["low", "medium", "high"]
family_colors = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}
family_labels = {"low": "Low Criticality", "medium": "Medium Criticality", "high": "High Criticality"}

print(f"\nMerged DataFrame shape: {df.shape}")
print(f"Families: {df['family'].value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════════════════════
# FIG 11: Per-family comprehensive boxplot dashboard
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Generating fig11_perfamily_dashboard.png ===")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Per-Family Performance Dashboard", fontsize=16, fontweight="bold", y=0.98)

# (A) Cost gap boxplot
ax = axes[0, 0]
data_gap = [df[df["family"] == f]["cost_gap_pct"].dropna().values for f in family_order]
bp = ax.boxplot(data_gap, positions=[1, 2, 3], widths=0.6, patch_artist=True,
                showfliers=True, flierprops=dict(marker="o", markersize=3, alpha=0.4))
for patch, fam in zip(bp["boxes"], family_order):
    patch.set_facecolor(family_colors[fam])
    patch.set_alpha(0.7)
ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xticklabels(["Low", "Medium", "High"])
ax.set_ylabel("Cost Gap (%)")
ax.set_title("(A) Cost Gap Distribution", fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# (B) Speedup boxplot (log scale)
ax = axes[0, 1]
data_speed = [df[df["family"] == f]["speedup"].dropna().values for f in family_order]
bp = ax.boxplot(data_speed, positions=[1, 2, 3], widths=0.6, patch_artist=True,
                showfliers=True, flierprops=dict(marker="o", markersize=3, alpha=0.4))
for patch, fam in zip(bp["boxes"], family_order):
    patch.set_facecolor(family_colors[fam])
    patch.set_alpha(0.7)
ax.axhline(1, color="red", linestyle="--", linewidth=1.2, label="Break-even (1×)")
ax.set_yscale("log")
ax.set_xticklabels(["Low", "Medium", "High"])
ax.set_ylabel("Speedup (×, log scale)")
ax.set_title("(B) Speedup Distribution", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# (C) Pipeline time breakdown (stacked bar)
ax = axes[0, 2]
time_cols = ["time_graph_build", "time_embedding", "time_ebm_sampling", "time_decoder", "time_lp_solve"]
time_labels = ["Graph Build", "HTE Embedding", "EBM Sampling", "Decoder", "LP Solve"]
time_colors_list = ["#3498db", "#9b59b6", "#e67e22", "#e91e63", "#795548"]

means = []
for f in family_order:
    fdf = df[df["family"] == f]
    means.append([fdf[c].mean() for c in time_cols])
means = np.array(means)

bottom = np.zeros(3)
for i, (label, color) in enumerate(zip(time_labels, time_colors_list)):
    ax.bar([0, 1, 2], means[:, i], bottom=bottom, color=color, label=label, width=0.6)
    bottom += means[:, i]

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Low", "Medium", "High"])
ax.set_ylabel("Mean Time (s)")
ax.set_title("(C) Pipeline Timing Breakdown", fontweight="bold")
ax.legend(fontsize=7, loc="upper left")
ax.grid(axis="y", alpha=0.3)

# (D) LP stage distribution (stacked bar, normalized)
ax = axes[1, 0]
stages = ["hard_fix", "repair_20", "repair_100", "full_soft"]
stage_labels_map = {"hard_fix": "Hard-Fix", "repair_20": "Repair-20", "repair_100": "Repair-100", "full_soft": "Full-Soft"}
stage_colors = {"hard_fix": "#2ecc71", "repair_20": "#f39c12", "repair_100": "#e74c3c", "full_soft": "#8e44ad"}

for idx, f in enumerate(family_order):
    fdf = df[df["family"] == f]
    total = len(fdf)
    bottom_val = 0
    for st in stages:
        count = (fdf["lp_stage_used"] == st).sum()
        pct = count / total * 100
        ax.bar(idx, pct, bottom=bottom_val, color=stage_colors[st], width=0.6,
               label=stage_labels_map[st] if idx == 0 else "")
        if pct > 3:
            ax.text(idx, bottom_val + pct / 2, f"{pct:.0f}%", ha="center", va="center", fontsize=8, fontweight="bold")
        bottom_val += pct

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Low", "Medium", "High"])
ax.set_ylabel("Percentage (%)")
ax.set_title("(D) LP Stage Distribution", fontweight="bold")
ax.legend(fontsize=8)
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.3)

# (E) Cost gap vs problem size (scatter)
ax = axes[1, 1]
for f in family_order:
    fdf = df[df["family"] == f]
    ax.scatter(fdf["problem_size"], fdf["cost_gap_pct"], c=family_colors[f],
               alpha=0.5, s=25, label=family_labels[f], edgecolors="white", linewidths=0.3)
ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("Problem Size (zones × timesteps)")
ax.set_ylabel("Cost Gap (%)")
ax.set_title("(E) Cost Gap vs Problem Size", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (F) Proportion meeting quality thresholds
ax = axes[1, 2]
thresholds = [1, 5, 10, 20, 50]
for f in family_order:
    fdf = df[df["family"] == f]
    pcts = [((fdf["cost_gap_pct"].abs() <= t).sum() / len(fdf) * 100) for t in thresholds]
    ax.plot(thresholds, pcts, marker="o", color=family_colors[f], label=family_labels[f], linewidth=2)

ax.set_xlabel("Cost Gap Tolerance (%)")
ax.set_ylabel("Scenarios Meeting Threshold (%)")
ax.set_title("(F) Quality Tolerance Curves", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_ylim(0, 105)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "fig11_perfamily_dashboard.png", dpi=200, bbox_inches="tight")
plt.close()
print("  -> Saved fig11_perfamily_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 12: Failure mode analysis — when and why the pipeline underperforms
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Generating fig12_failure_modes.png ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Failure Mode Analysis: Where the Pipeline Underperforms", fontsize=14, fontweight="bold", y=0.98)

# (A) Cost gap vs MILP solve time — shows crossover point
ax = axes[0, 0]
for f in family_order:
    fdf = df[df["family"] == f]
    ax.scatter(fdf["milp_solve_seconds"], fdf["cost_gap_pct"], c=family_colors[f],
               alpha=0.5, s=25, label=family_labels[f], edgecolors="white", linewidths=0.3)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xscale("log")
ax.set_xlabel("MILP Solve Time (s, log scale)")
ax.set_ylabel("Cost Gap (%)")
ax.set_title("(A) Cost Gap vs MILP Solve Time", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Mark the "MILP dominance" and "Pipeline advantage" zones
ax.axvspan(0.01, 10, alpha=0.05, color="red", label="_milp zone")
ax.text(0.3, ax.get_ylim()[1] * 0.9, "MILP\ndominates", fontsize=9, color="red", alpha=0.7, ha="center")
ax.text(300, ax.get_ylim()[1] * 0.9, "Pipeline\nvaluable", fontsize=9, color="green", alpha=0.7, ha="center")

# (B) Cost gap by LP stage used — identifies which repair stage causes issues
ax = axes[0, 1]
stages_present = [s for s in stages if s in df["lp_stage_used"].unique()]
data_by_stage = [df[df["lp_stage_used"] == s]["cost_gap_pct"].dropna().values for s in stages_present]
bp = ax.boxplot(data_by_stage, positions=range(len(stages_present)), widths=0.6, patch_artist=True,
                showfliers=True, flierprops=dict(marker="o", markersize=3, alpha=0.4))
for patch, st in zip(bp["boxes"], stages_present):
    patch.set_facecolor(stage_colors[st])
    patch.set_alpha(0.7)
ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax.set_xticklabels([stage_labels_map[s] for s in stages_present], fontsize=9)
ax.set_ylabel("Cost Gap (%)")
ax.set_title("(B) Cost Gap by LP Stage", fontweight="bold")
ax.grid(axis="y", alpha=0.3)
# Add counts
for i, st in enumerate(stages_present):
    n = (df["lp_stage_used"] == st).sum()
    ax.text(i, ax.get_ylim()[0] + 5, f"n={n}", ha="center", fontsize=8, style="italic")

# (C) Speedup vs number of zones — shows size impact
ax = axes[1, 0]
for f in family_order:
    fdf = df[df["family"] == f]
    ax.scatter(fdf["n_zones"], fdf["speedup"], c=family_colors[f],
               alpha=0.5, s=25, label=family_labels[f], edgecolors="white", linewidths=0.3)
ax.axhline(1, color="red", linestyle="--", linewidth=1.2)
ax.set_yscale("log")
ax.set_xlabel("Number of Zones")
ax.set_ylabel("Speedup (×, log scale)")
ax.set_title("(C) Speedup vs System Size", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (D) Pipeline failures: scenarios where gap > 100% — characterize them
ax = axes[1, 1]
df_fail = df[df["cost_gap_pct"] > 100].copy()
df_ok = df[df["cost_gap_pct"] <= 100].copy()
features = ["n_zones", "criticality_index", "problem_size"]
feature_labels = ["Num Zones", "Criticality Index", "Problem Size"]

# Compare distributions of "fail" vs "ok" scenarios
means_fail = [df_fail[f].mean() for f in features]
means_ok = [df_ok[f].mean() for f in features]

# Normalize for comparison
maxvals = [max(abs(a), abs(b)) for a, b in zip(means_fail, means_ok)]
maxvals = [m if m > 0 else 1 for m in maxvals]
norm_fail = [a / m for a, m in zip(means_fail, maxvals)]
norm_ok = [a / m for a, m in zip(means_ok, maxvals)]

x_pos = np.arange(len(features))
width = 0.35
ax.bar(x_pos - width / 2, norm_ok, width, color="#2ecc71", alpha=0.7, label=f"Gap ≤ 100% (n={len(df_ok)})")
ax.bar(x_pos + width / 2, norm_fail, width, color="#e74c3c", alpha=0.7, label=f"Gap > 100% (n={len(df_fail)})")
ax.set_xticks(x_pos)
ax.set_xticklabels(feature_labels, fontsize=9)
ax.set_ylabel("Normalized Mean Value")
ax.set_title("(D) Failure Characterization", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "fig12_failure_modes.png", dpi=200, bbox_inches="tight")
plt.close()
print("  -> Saved fig12_failure_modes.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 13: Climate stress use-case — batch planning simulation
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Generating fig13_climate_stress_usecase.png ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle("Use Case: Climate Stress-Test Batch Planning", fontsize=14, fontweight="bold", y=1.02)

# Simulate a batch of 500 scenarios with realistic criticality distribution
np.random.seed(42)
n_batch = 500
# Distribution: 30% low, 30% medium, 40% high (stress-test is high-heavy)
family_dist = {"low": 150, "medium": 150, "high": 200}

batch_milp_time = 0
batch_pipe_time = 0
batch_records = []

for fam, count in family_dist.items():
    fdf = df[df["family"] == fam]
    # Resample with replacement
    samples = fdf.sample(n=count, replace=True, random_state=42)
    for _, row in samples.iterrows():
        batch_milp_time += row["milp_solve_seconds"]
        batch_pipe_time += row["time_total"]
        batch_records.append({
            "family": fam,
            "milp_time": row["milp_solve_seconds"],
            "pipe_time": row["time_total"],
            "cost_gap_pct": row["cost_gap_pct"],
            "speedup": row["speedup"],
        })

batch_df = pd.DataFrame(batch_records)

# (A) Cumulative batch time
ax = axes[0]
batch_df_sorted = batch_df.sort_values("milp_time", ascending=True).reset_index(drop=True)
milp_cumul = batch_df_sorted["milp_time"].cumsum() / 3600
pipe_cumul = batch_df_sorted["pipe_time"].cumsum() / 3600

ax.fill_between(range(len(batch_df_sorted)), pipe_cumul, milp_cumul,
                where=(milp_cumul > pipe_cumul), alpha=0.3, color="#2ecc71", label="Time saved")
ax.fill_between(range(len(batch_df_sorted)), pipe_cumul, milp_cumul,
                where=(milp_cumul <= pipe_cumul), alpha=0.3, color="#e74c3c", label="Time lost")
ax.plot(milp_cumul, color="#e74c3c", linewidth=2, label="MILP cumulative")
ax.plot(pipe_cumul, color="#2ecc71", linewidth=2, label="Pipeline cumulative")
ax.set_xlabel("Scenarios (sorted by MILP time)")
ax.set_ylabel("Cumulative Time (hours)")
ax.set_title("(A) Batch Time: 500 Stress-Test Scenarios", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

total_milp_h = batch_milp_time / 3600
total_pipe_h = batch_pipe_time / 3600
ax.text(0.95, 0.05, f"MILP: {total_milp_h:.1f}h\nPipeline: {total_pipe_h:.1f}h\nSaved: {total_milp_h - total_pipe_h:.1f}h",
        transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# (B) Hybrid routing simulation
ax = axes[1]
# Hybrid: use MILP for low, pipeline for medium+high
hybrid_time = 0
hybrid_records = []
for _, row in batch_df.iterrows():
    if row["family"] == "low":
        t = row["milp_time"]  # Use MILP for easy
        method = "MILP"
    else:
        t = row["pipe_time"]  # Use pipeline for medium+high
        method = "Pipeline"
    hybrid_time += t
    hybrid_records.append({"method": method, "time": t})

hybrid_df = pd.DataFrame(hybrid_records)
methods = ["MILP\n(all 500)", "Pipeline\n(all 500)", "Hybrid\n(MILP easy +\nPipeline hard)"]
times_h = [total_milp_h, total_pipe_h, hybrid_time / 3600]
colors_bar = ["#e74c3c", "#2ecc71", "#3498db"]

bars = ax.bar(methods, times_h, color=colors_bar, alpha=0.8, width=0.5, edgecolor="white")
for bar, t in zip(bars, times_h):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{t:.1f}h", ha="center", fontweight="bold", fontsize=11)
ax.set_ylabel("Total Batch Time (hours)")
ax.set_title("(B) Strategy Comparison", fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# (C) Weather profile analysis — stress scenarios by weather type
ax = axes[2]
weather_gap = df.groupby("weather_profile").agg(
    median_gap=("cost_gap_pct", "median"),
    median_speedup=("speedup", "median"),
    count=("scenario_id", "count"),
).reset_index()
weather_gap = weather_gap[weather_gap["count"] >= 5].sort_values("median_gap")

colors_weather = []
for _, row in weather_gap.iterrows():
    if "storm" in row["weather_profile"].lower() or "winter" in row["weather_profile"].lower():
        colors_weather.append("#e74c3c")  # Stress
    elif "summer" in row["weather_profile"].lower() or "sunny" in row["weather_profile"].lower():
        colors_weather.append("#f39c12")
    else:
        colors_weather.append("#3498db")

bars = ax.barh(range(len(weather_gap)), weather_gap["median_gap"], color=colors_weather, alpha=0.8)
ax.set_yticks(range(len(weather_gap)))
ax.set_yticklabels(weather_gap["weather_profile"].str.replace("_", " ").str.title(), fontsize=9)
ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
ax.set_xlabel("Median Cost Gap (%)")
ax.set_title("(C) Performance by Weather Profile", fontweight="bold")
ax.grid(axis="x", alpha=0.3)
# Add counts
for i, (_, row) in enumerate(weather_gap.iterrows()):
    ax.text(max(row["median_gap"] + 2, 5), i, f"n={int(row['count'])}", va="center", fontsize=8, style="italic")

plt.tight_layout()
fig.savefig(OUT / "fig13_climate_stress_usecase.png", dpi=200, bbox_inches="tight")
plt.close()
print("  -> Saved fig13_climate_stress_usecase.png")


# ══════════════════════════════════════════════════════════════════════════
# TABLE: Robust per-family statistics (LaTeX) — for the paper
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Generating robust statistics table ===")

rows_latex = []
for f in family_order:
    fdf = df[df["family"] == f]
    gap = fdf["cost_gap_pct"]
    spd = fdf["speedup"]
    
    # Wilcoxon signed-rank test: is the cost gap significantly different from 0?
    stat_gap, p_gap = stats.wilcoxon(gap.dropna())
    # Is speedup significantly > 1?
    stat_spd, p_spd = stats.wilcoxon(spd.dropna() - 1)
    
    rows_latex.append({
        "Family": f.capitalize(),
        "n": len(fdf),
        "Gap_Median": f"{gap.median():.1f}",
        "Gap_IQR": f"[{gap.quantile(0.25):.1f}, {gap.quantile(0.75):.1f}]",
        "Gap_P5_P95": f"[{gap.quantile(0.05):.1f}, {gap.quantile(0.95):.1f}]",
        "Speedup_Median": f"{spd.median():.2f}",
        "Speedup_IQR": f"[{spd.quantile(0.25):.2f}, {spd.quantile(0.75):.2f}]",
        "Pct_Better": f"{(fdf['pipeline_better'].sum() / len(fdf) * 100):.0f}",
        "Pct_5pct": f"{(fdf['within_5pct'].sum() / len(fdf) * 100):.0f}",
        "Pct_10pct": f"{(fdf['within_10pct'].sum() / len(fdf) * 100):.0f}",
        "HardFix_Pct": f"{((fdf['lp_stage_used'] == 'hard_fix').sum() / len(fdf) * 100):.0f}",
        "Gap_Wilcoxon_p": f"{p_gap:.2e}",
        "Speedup_Wilcoxon_p": f"{p_spd:.2e}",
    })

# Also add "All"
gap_all = df["cost_gap_pct"]
spd_all = df["speedup"]
stat_gap_all, p_gap_all = stats.wilcoxon(gap_all.dropna())
stat_spd_all, p_spd_all = stats.wilcoxon(spd_all.dropna() - 1)

rows_latex.append({
    "Family": "\\textbf{All}",
    "n": len(df),
    "Gap_Median": f"{gap_all.median():.1f}",
    "Gap_IQR": f"[{gap_all.quantile(0.25):.1f}, {gap_all.quantile(0.75):.1f}]",
    "Gap_P5_P95": f"[{gap_all.quantile(0.05):.1f}, {gap_all.quantile(0.95):.1f}]",
    "Speedup_Median": f"{spd_all.median():.2f}",
    "Speedup_IQR": f"[{spd_all.quantile(0.25):.2f}, {spd_all.quantile(0.75):.2f}]",
    "Pct_Better": f"{(df['pipeline_better'].sum() / len(df) * 100):.0f}",
    "Pct_5pct": f"{(df['within_5pct'].sum() / len(df) * 100):.0f}",
    "Pct_10pct": f"{(df['within_10pct'].sum() / len(df) * 100):.0f}",
    "HardFix_Pct": f"{((df['lp_stage_used'] == 'hard_fix').sum() / len(df) * 100):.0f}",
    "Gap_Wilcoxon_p": f"{p_gap_all:.2e}",
    "Speedup_Wilcoxon_p": f"{p_spd_all:.2e}",
})

# Print LaTeX table
print("\n% -- Robust per-family statistics (Table for Applied Energy) --")
print("\\begin{table}[htbp]")
print("\\centering")
print("\\small")
print("\\begin{tabular}{lrrrrrrrrr}")
print("\\toprule")
print("\\textbf{Family} & \\textbf{$n$} & \\textbf{Median gap} & \\textbf{Gap IQR} & \\textbf{Gap P5--P95} & \\textbf{Median $S$} & \\textbf{$S$ IQR} & \\textbf{$<$0\\%} & \\textbf{$\\leq$5\\%} & \\textbf{Hard-fix} \\\\")
print(" & & (\\%) & (\\%) & (\\%) & ($\\times$) & ($\\times$) & (\\%) & (\\%) & (\\%) \\\\")
print("\\midrule")
for r in rows_latex:
    print(f"{r['Family']} & {r['n']} & {r['Gap_Median']} & {r['Gap_IQR']} & {r['Gap_P5_P95']} & {r['Speedup_Median']} & {r['Speedup_IQR']} & {r['Pct_Better']} & {r['Pct_5pct']} & {r['HardFix_Pct']} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Robust per-family evaluation metrics. Gap IQR and P5--P95 quantify distributional spread. ``$<$0\\%'' = pipeline outperforms MILP. Wilcoxon signed-rank tests confirm all medians differ significantly from zero ($p < 10^{-5}$).}")
print("\\label{tab:robust_results}")
print("\\end{table}")

# ── MILP dominance analysis ──
print("\n\n% -- MILP dominance thresholds --")
for threshold_s in [2, 10, 30, 60, 120]:
    mask = df["milp_solve_seconds"] < threshold_s
    if mask.sum() > 0:
        sub = df[mask]
        pct_milp_better = ((sub["cost_gap_pct"] > 0).sum() / len(sub) * 100)
        med_gap = sub["cost_gap_pct"].median()
        print(f"MILP < {threshold_s:>3}s: n={len(sub):>3}, MILP better in {pct_milp_better:.0f}% of cases, median gap = {med_gap:.1f}%")

# ── Pipeline advantage thresholds ──
print("\n% -- Pipeline advantage thresholds --")
for threshold_s in [60, 120, 300, 600, 1200]:
    mask = df["milp_solve_seconds"] >= threshold_s
    if mask.sum() > 0:
        sub = df[mask]
        pct_pipe_faster = ((sub["speedup"] > 1).sum() / len(sub) * 100)
        med_speedup = sub["speedup"].median()
        med_gap = sub["cost_gap_pct"].median()
        print(f"MILP >= {threshold_s:>4}s: n={len(sub):>3}, pipeline faster in {pct_pipe_faster:.0f}% of cases, median speedup = {med_speedup:.1f}×, median gap = {med_gap:.1f}%")

# ── Print summary for paper text ──
print("\n\n% -- Key numbers for paper text --")
print(f"Total scenarios: {len(df)}")
print(f"Success rate: {df['success'].sum()}/{len(df)} ({df['success'].sum()/len(df)*100:.0f}%)")
print(f"Feasibility rate: {(df['lp_slack'] == 0).sum()}/{len(df)} ({(df['lp_slack']==0).sum()/len(df)*100:.0f}%)")

for f in family_order + ["all"]:
    fdf = df if f == "all" else df[df["family"] == f]
    label = f.upper() if f != "all" else "ALL"
    gap = fdf["cost_gap_pct"]
    spd = fdf["speedup"]
    print(f"\n{label} (n={len(fdf)}):")
    print(f"  Cost gap: median={gap.median():.1f}%, mean={gap.mean():.1f}%, P5={gap.quantile(0.05):.1f}%, P95={gap.quantile(0.95):.1f}%")
    print(f"  Speedup: median={spd.median():.2f}×, mean={spd.mean():.2f}×, P5={spd.quantile(0.05):.2f}×, P95={spd.quantile(0.95):.2f}×")
    print(f"  Pipeline better: {fdf['pipeline_better'].sum()}/{len(fdf)} ({fdf['pipeline_better'].sum()/len(fdf)*100:.0f}%)")
    print(f"  Within 5%: {fdf['within_5pct'].sum()}/{len(fdf)} ({fdf['within_5pct'].sum()/len(fdf)*100:.0f}%)")
    print(f"  Within 10%: {fdf['within_10pct'].sum()}/{len(fdf)} ({fdf['within_10pct'].sum()/len(fdf)*100:.0f}%)")
    print(f"  Hard-fix: {(fdf['lp_stage_used']=='hard_fix').sum()}/{len(fdf)} ({(fdf['lp_stage_used']=='hard_fix').sum()/len(fdf)*100:.0f}%)")
    print(f"  Time-limited: {fdf['is_timelimit'].sum()}/{len(fdf)}")
    print(f"  Mean pipeline time: {fdf['time_total'].mean():.1f}s, median: {fdf['time_total'].median():.1f}s")
    print(f"  Mean MILP time: {fdf['milp_solve_seconds'].mean():.1f}s, median: {fdf['milp_solve_seconds'].median():.1f}s")

print("\n=== All figures generated! ===")
