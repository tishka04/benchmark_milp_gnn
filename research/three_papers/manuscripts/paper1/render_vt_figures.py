"""Render the decision figures for main_VT.tex from its archived runs."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import StrMethodFormatter


MANUSCRIPTS = Path(__file__).resolve().parents[1]
RUN = MANUSCRIPTS.parents[1] / "paper1_revision" / "runs" / "revision_v1"
FINAL_TEST = RUN / "diagnostics" / "final_external_v1_20261119T085611Z"
FIXED_DAYS = MANUSCRIPTS.parents[1] / "paper1_fixed_days" / "runs" / "pilot_v1"
FIGURES = MANUSCRIPTS / "figures"
DEADLINES = [5, 10, 30, 60, 120]
INK = "#263744"
BLUE = "#1D6181"
ORANGE = "#AD501F"
TEAL = "#157668"
GRID = "#DAE1E4"
PRIORITIES = {
    "service4_co2": "Service",
    "cost_priority": "Cost",
    "time_priority": "Time",
    "co2_priority": "Emissions",
    "equal4": "Balanced",
}


def strengthened_development():
    keys = ["group", "budget"]
    fields = ["F", "E", "ES", *PRIORITIES,
              "ens_mwh", "cost", "reference_cost"]
    frame = pd.read_parquet(RUN / "analysis.parquet", columns=keys + fields)
    frame = frame.set_index(keys).sort_index()
    changes = pd.read_csv(RUN / "physical_campaign" / "reference_target_changes.csv")
    by_target = changes.pivot(index=keys, columns="target", values="after")
    references = changes.groupby(keys).best_reference
    assert frame.index.is_unique and len(frame) == 600 * 5
    assert len(by_target) == 217 * 5
    assert by_target.index.isin(frame.index).all()
    assert references.nunique().eq(1).all()
    revised = ["E", "ES", *PRIORITIES]
    before = changes.pivot(index=keys, columns="target", values="before")[revised]
    np.testing.assert_allclose(frame.loc[before.index, revised], before, atol=1e-9)
    frame.update(by_target[revised])
    frame.update(references.first().to_frame("reference_cost"))
    return frame


def save(figure, name, description):
    figure.savefig(FIGURES / name, metadata={
        "Title": description, "Creator": "render_vt_figures.py",
        "CreationDate": None, "ModDate": None,
    })
    plt.close(figure)


def plot_deadlines(frame):
    counts = frame.groupby("budget")[["F", "E", "ES"]].sum().loc[DEADLINES]
    assert (counts.ES <= counts.E).all() and (counts.E <= counts.F).all()
    assert frame.groupby("budget").size().eq(600).all()
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(6.25, 2.38))
    fig.subplots_adjust(left=.095, right=.992, top=.78, bottom=.27, wspace=.15)
    y = np.arange(len(counts))
    panels = [
        ("F", "Valid schedule\ndelivered", BLUE),
        ("E", "Also met the\ncost threshold", TEAL),
        ("ES", "Also met the\nservice threshold", ORANGE),
    ]
    for ax, (field, title, color) in zip(axes, panels):
        ax.set_title(title, pad=7, fontsize=10.5, linespacing=1.15)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, linewidth=.65)
        values = counts[field].to_numpy()
        ax.barh(y, values, height=.52, color=color)
        for row, value in zip(y, values):
            ax.text(value + 10, row, str(value), va="center", fontsize=8.2)
        ax.set(xlim=(0, 800), xticks=[0, 300, 600], yticks=y,
               yticklabels=[f"{b} s" for b in DEADLINES])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(length=0, pad=2)
    axes[0].invert_yaxis()
    fig.supxlabel("Number of cases (out of 600)", y=.035, fontsize=10)
    save(fig, "vt_deadline_outcomes.pdf", "Schedule delivery, cost and service by deadline")


def plot_shortlist(frame):
    day = frame.xs(120, level="budget").copy()
    day["gap"] = 100 * (day.cost - day.reference_cost) / day.reference_cost.abs()
    service = day.service4_co2.nlargest(61)
    cost = day.cost_priority.nlargest(61)
    assert service.iloc[-2] > service.iloc[-1] and cost.iloc[-2] > cost.iloc[-1]
    service_list = service.index[:60]
    cost_list = cost.index[:60]
    assert len(service_list.intersection(cost_list)) == 51

    fig, ax = plt.subplots(figsize=(6.25, 2.94))
    fig.subplots_adjust(left=.124, right=.98, bottom=.21, top=.975)
    for ids, label, color, marker in [
        (service_list.difference(cost_list), "Service priority only", BLUE, "o"),
        (cost_list.difference(service_list), "Cost priority only", ORANGE, "D"),
    ]:
        selected = day.loc[ids]
        ax.scatter(selected.gap, selected.ens_mwh, marker=marker, s=42, c=color,
                   edgecolor="white", linewidth=.6, label=label, zorder=3)
    ax.set(xlim=(0, 90), ylim=(0, 1875), xticks=range(0, 91, 10),
           yticks=[0, 400, 800, 1200, 1600],
           xlabel="Operating cost above the case reference (%)",
           ylabel="Unserved energy (MWh)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(color=GRID, linewidth=.6)
    ax.legend(loc="upper right", frameon=False, handletextpad=.35)
    save(fig, "vt_priority_shortlist.pdf", "Outcomes of the eighteen cases selected under only one priority")


def plot_profile_overlap(frame):
    scores = frame.xs(120, level="budget")[list(PRIORITIES)]
    assert len(scores) == 600 and scores.notna().all().all()
    quota = 60
    membership = pd.DataFrame(False, index=scores.index, columns=scores.columns)
    for profile, values in scores.items():
        ranked = values.nlargest(quota + 1)
        assert ranked.iloc[quota - 1] > ranked.iloc[quota], f"Tie at the cutoff for {profile}"
        membership.loc[ranked.index[:quota], profile] = True
    selected = membership.astype(int)
    replacements = (quota - selected.T @ selected).to_numpy()
    common = membership.all(axis=1).sum()
    union = membership.any(axis=1).sum()

    fig, ax = plt.subplots(figsize=(6.25, 2.95))
    fig.subplots_adjust(left=.149, right=.985, bottom=.025, top=.745)
    n = len(PRIORITIES)
    positions = np.arange(n)
    labels = list(PRIORITIES.values())
    ax.set(xlim=(-.5, n-.5), ylim=(n-.5, -.5), xticks=positions, xticklabels=labels,
           yticks=positions, yticklabels=labels)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=6, labelsize=10)
    ax.spines[:].set_visible(False)
    ax.set_title(f"Cases replaced in a list of {quota}", loc="left", pad=27,
                 fontsize=12, fontweight="semibold")
    max_change = max(1, replacements.max())
    for row, col in zip(*np.tril_indices(n)):
        value = replacements[row, col]
        # Map zero and the largest observed count to the two ends of a single-hue scale.
        shade = plt.cm.Blues(.055 + .82 * value / max_change)
        ax.add_patch(Rectangle((col-.5, row-.5), 1, 1, facecolor=shade,
                               edgecolor="white", linewidth=1.8))
        ax.text(col, row, str(value), ha="center", va="center", fontsize=12,
                fontweight="semibold", color="white" if value / max_change >= .75 else INK)
    ax.text(2.38, .22, f"{common} cases appear on every list.\n"
            f"{union} appear on at least one.", ha="left", va="center",
            fontsize=10.5, color=INK, linespacing=1.4)
    save(fig, "vt_profile_overlap.pdf",
         "How many of the 60 cases change when each of five review priorities is used")


def plot_fixed_grid_days():
    frame = pd.read_csv(FIXED_DAYS / "analysis" / "shortlist.csv")
    assert frame.days.eq(30).all() and frame.quota.eq(6).all()
    reference = frame[frame.strategy.isin(["uniform_random", "model_size", "realized_compact"])]
    assert len(reference) == 3 * 2 * len(PRIORITIES)
    assert reference.distinct_priority_values.eq(1).all()
    chance = 6 * 6 / 30
    np.testing.assert_allclose(reference.expected_hits, chance)
    methods = {
        "residual_peak_mw": "Peak residual\nload",
        "upward_ramp_mw_per_hour": "Largest\nhourly rise",
        "realized": "22-input\nforecast",
    }
    counts = frame.pivot(index="target", columns=["budget", "strategy"], values="expected_hits")
    fig, axes = plt.subplots(1, 2, figsize=(6.25, 2.56))
    fig.subplots_adjust(left=.17, right=.995, bottom=.15, top=.72, wspace=.12)
    for ax, budget in zip(axes, [30, 120]):
        values = counts[budget].loc[list(PRIORITIES), list(methods)].to_numpy()
        assert np.all((values >= 0) & (values <= 6) & (values == np.floor(values)))
        for (row, col), value in np.ndenumerate(values):
            shade = plt.cm.Blues(.055 + .86 * value / 6)
            ax.add_patch(Rectangle((col-.5, row-.5), 1, 1, facecolor=shade,
                                   edgecolor="white", linewidth=1.5))
            ax.text(col, row, str(int(value)), va="center", ha="center",
                    fontsize=11, fontweight="semibold", color="white" if value >= 4 else INK)
        ax.set(xlim=(-.5, len(methods)-.5), ylim=(len(PRIORITIES)-.5, -.5),
               xticks=range(len(methods)), xticklabels=list(methods.values()),
               yticks=range(len(PRIORITIES)), yticklabels=list(PRIORITIES.values()))
        ax.xaxis.tick_top()
        ax.tick_params(axis="both", length=0, pad=3)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.tick_params(axis="y", labelleft=ax == axes[0], labelsize=9.2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{budget}-second deadline", fontsize=11, fontweight="semibold", pad=8)
    fig.text(.5, .039, f"A random draw finds {chance:.1f} priority days on average. "
             "Grid structure gives all 30 days the same rank.",
             ha="center", va="bottom", fontsize=8.4)
    save(fig, "vt_fixed_grid_days.pdf", "Priority days found among six selected on the same grid")


def plot_transfer():
    columns = ["group", "target", "batch", "budget", "b9", "outside", "milp", "y"]
    frame = pd.read_parquet(FINAL_TEST / "b9_paired_diagnostic.parquet", columns=columns)
    batch = frame.query("target == 'equal4' and batch == 'topology' and budget == 5")
    assert len(batch) == batch.group.nunique() == 100
    assert batch.outside.sum() == 75
    historical = pd.read_parquet(RUN / "scenarios.parquet", columns=["n_lines", "n_zones"])
    training_ratio = historical.n_lines / historical.n_zones
    fig, ax = plt.subplots(figsize=(6.25, 2.95))
    fig.subplots_adjust(left=.092, right=.977, top=.955, bottom=.205)
    ax.axvspan(training_ratio.min(), training_ratio.max(), color="#DEF0EB", zorder=0)
    ax.axvline(training_ratio.max(), color=TEAL, linestyle=(0, (2, 3)), linewidth=.9)
    ax.annotate("Upper end of training range", xy=(training_ratio.max(), 5.9),
                xytext=(1.16, 5.9), color=TEAL, fontsize=9, va="center",
                arrowprops={"arrowstyle": "-", "color": TEAL, "lw": .8})
    ax.axhline(1, linewidth=.9, linestyle=(0, (4, 3)), color="#67717A")
    ax.scatter(batch.b9, batch.milp, s=23, marker="o", facecolor="none",
               linewidth=.9, edgecolor=ORANGE, label="Model prediction", zorder=2)
    ax.scatter(batch.b9, batch.y, s=18, marker="x", linewidth=.9, color=BLUE,
               label="Observed score", zorder=3)
    ax.text(2.99, 1.15, "Maximum possible score", ha="right", va="bottom", fontsize=8.4)
    ax.set(xlim=(.72, 3.07), ylim=(-.16, 6.9), yticks=[0, 1, 2, 3, 4, 5, 6],
           xticks=[.75, 1, 1.5, 2, 2.5, 3],
           xticklabels=["0.75", "1", "1.5", "2", "2.5", "3"],
           xlabel="Transmission lines per zone", ylabel="Balanced score")
    ax.grid(axis="y", color=GRID, linewidth=.5, zorder=0)
    ax.legend(loc="center right", bbox_to_anchor=(.99, .53), frameon=True,
              edgecolor="none", framealpha=.94, handletextpad=.25, borderpad=.3)
    save(fig, "vt_topology_transfer.pdf", "Extrapolation of the frozen model under altered network topology")


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#8A959C",
        "xtick.color": INK, "ytick.color": INK, "axes.titlesize": 10.5,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "xtick.major.width": .7, "ytick.major.width": .7,
    })
    FIGURES.mkdir(exist_ok=True)
    data = strengthened_development()
    plot_deadlines(data)
    plot_profile_overlap(data)
    plot_shortlist(data)
    plot_transfer()
    plot_fixed_grid_days()
