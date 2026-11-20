"""Print manuscript appendix tables from the initial and strengthened development references."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import RUN, read
from ..future_analysis import economic_threshold


PROFILES = ["service4_co2", "cost_priority", "time_priority", "co2_priority", "equal4"]
FAMILIES = {
    "high_criticality_scenarios": "H",
    "persona_mathematician_scenarios": "D",
    "persona_network_operator_scenarios": "N",
}


def development_data():
    keys = ["group", "budget"]
    original = pd.read_parquet(RUN / "analysis.parquet").set_index(keys)
    if not original.index.is_unique:
        raise ValueError("Expected one original observation per case and deadline")
    changes = pd.read_csv(RUN / "physical_campaign" / "reference_target_changes.csv")
    strengthened = original.copy()
    for target in [*PROFILES, "E", "ES"]:
        rows = changes.loc[changes.target == target].set_index(keys)
        if not rows.index.is_unique:
            raise ValueError(f"Duplicate strengthened references for {target}")
        np.testing.assert_allclose(original.loc[rows.index, target], rows.before)
        strengthened.loc[rows.index, target] = rows.after.astype(original[target].dtype)
        if target == "equal4":
            np.testing.assert_allclose(original.loc[rows.index, "reference_cost"], rows.old_reference)
            strengthened.loc[rows.index, "reference_cost"] = rows.best_reference
    return original, strengthened


def shortlist_table(original, strengthened):
    deadline = 120
    quota = 60
    cases = strengthened.xs(deadline, level="budget").copy()
    initial = original.xs(deadline, level="budget")
    for label, profile in zip(("service", "cost"), PROFILES[:2]):
        ranked = cases[profile].sort_values(ascending=False)
        cutoff = ranked.iloc[quota - 1]
        if cutoff == ranked.iloc[quota]:
            raise ValueError(f"Observed tie at the {label} cutoff")
        np.testing.assert_array_equal(
            np.sort(ranked.index[:quota]), np.sort(initial[profile].nlargest(quota).index)
        )
        cases[label + "_rank"] = cases[profile].rank(ascending=False, method="min")
        cases[label + "_margin"] = 1000 * (cases[profile] - cutoff)
    parts = [
        cases[(cases.service_rank <= quota) & (cases.cost_rank > quota)].sort_values("service_rank"),
        cases[(cases.cost_rank <= quota) & (cases.service_rank > quota)].sort_values("cost_rank"),
    ]
    lines = [
        r"\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{3.2pt}",
        r"\caption{Development cases selected exclusively under the service or cost priority "
        r"at 120 seconds, using the strengthened references.}\label{tab:exclusive_cases}",
        r"\begin{tabular}{@{}lrrrrrrrrrr@{}}\toprule",
        r" & \multicolumn{2}{c}{Rank} & \multicolumn{2}{c}{Score margin}"
        r" & \multicolumn{2}{c}{Unserved} & \multicolumn{2}{c}{Cost excess}"
        r" & Time & Thermal\\",
        r"Case & Service & Cost & Service & Cost & MWh & \% & M EUR & \% & s & \%\\\midrule",
    ]
    for caption, part in zip(("Service only", "Cost only"), parts):
        lines.append(r"\multicolumn{11}{@{}l}{\emph{" + caption + r"}}\\")
        for case in part.itertuples():
            code = FAMILIES[case.family] + "-" + case.Index.rsplit("_", 1)[1]
            gap = case.cost - case.reference_cost
            lines.append(
                f"{code} & {case.service_rank:.0f} & {case.cost_rank:.0f} & "
                f"{case.service_margin:+.1f} & {case.cost_margin:+.1f} & {case.ens_mwh:,.0f} & "
                f"{100 * case.ens_mwh / case.demand_mwh:.3f} & {gap / 1e6:.2f} & "
                f"{100 * gap / abs(case.reference_cost):.2f} & {case.exit_s:.1f} & "
                f"{100 * case.thermal_mwh / case.served_mwh:.2f}" + r"\\"
            )
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule\end{tabular}"
    lines.extend([
        r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize",
        "H denotes the high-criticality family, D the discrete-decision family and N the "
        "network family. The number is the case identifier within its family. Rank 1 has the "
        "largest loss. Each margin is the case's score minus its profile's 60th score, "
        "multiplied by 1,000. A positive margin places the case inside the shortlist. "
        "Unserved energy is also expressed as a percentage of demand. Cost excess is the "
        "dispatch cost minus its reference, expressed in millions of euros and as a percentage "
        "of the absolute reference cost. Thermal is internal thermal generation as a percentage "
        "of served energy.",
        r"\end{minipage}\end{table}",
    ])
    return lines


def delivered_table(original, strengthened):
    def medians(data):
        return data.loc[data.F == 1, PROFILES].groupby(level="budget").median()

    values = medians(strengthened)
    pd.testing.assert_frame_equal(values.round(3), medians(original).round(3))
    counts = strengthened.F.groupby(level="budget").sum()
    lines = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Median profile loss among development cases that delivered a valid dispatch.}"
        r"\label{tab:conditional_loss}",
        r"\begin{tabular}{@{}rrrrrrr@{}}\toprule",
        r"Deadline (s) & Delivered & Service & Cost & Time & Emissions & Balanced\\\midrule",
    ]
    for deadline, row in values.iterrows():
        lines.append(
            f"{deadline} & {int(counts[deadline])} & "
            + " & ".join(f"{v:.3f}" for v in row) + r"\\"
        )
    lines.extend([
        r"\bottomrule\end{tabular}",
        r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize",
        "All five profiles evaluate the same deliveries at each deadline. The set of "
        "delivered cases changes with the deadline. Each row uses the strengthened references. "
        "The initial references give the same medians to the precision shown.",
        r"\end{minipage}\end{table}",
    ])
    return lines


def service_witnesses(original, strengthened):
    """Check feasible archived alternatives for cases that missed ES at every deadline."""
    original_cases = original.xs(120, level="budget")
    final_cases = strengthened.xs(120, level="budget")
    persistent = strengthened.ES.groupby(level="group").max().eq(0)
    cases = final_cases.loc[persistent].copy()
    limits = cases.reference_cost.map(economic_threshold)
    original_refs = pd.read_parquet(RUN / "references.parquet").set_index("group").loc[cases.index]
    np.testing.assert_allclose(original_refs.cost, original_cases.loc[cases.index, "reference_cost"])
    initial_pass = original_refs.valid & (original_refs.cost <= limits) & (cases.reference_ens_mwh <= 0.001)
    resolved = set(cases.index[initial_pass])
    initial_count = len(resolved)

    campaign = RUN / "physical_campaign"
    candidates = set(cases.index) - resolved
    for job in read(campaign / "manifest.json")["jobs"]:
        if job["kind"] != "reference_upgrade" or job["key"] not in candidates:
            continue
        receipt = read(campaign / "results" / (job["job_id"] + ".json"))
        if receipt["job"] != job:
            raise ValueError("Reference-upgrade receipt does not match the planned job")
        run = receipt["result"]
        result = run.get("result") or {}
        check = result.get("validation") or {}
        if (run.get("success") and result.get("valid") and check.get("valid")
                and check["ens_mwh"] <= 0.001 and check["cost"] <= limits[job["key"]]):
            resolved.add(job["key"])
    reference_count = len(resolved)
    columns = ["group", "method", "condition", "feasible", "cost", "ens_mwh", "budget", "repetition"]
    archive = pd.read_parquet(RUN / "runs.parquet", columns=columns)
    alternatives = archive[
        archive.group.isin(set(cases.index) - resolved) & archive.condition.eq("historical")
        & archive.feasible.eq(True) & archive.ens_mwh.le(0.001) & archive.method.ne("highs")
        & archive.cost.le(archive.group.map(limits))
    ].sort_values(["group", "cost"]).drop_duplicates("group")
    lines = [
        r"\paragraph{Feasible alternatives to the delivered schedules.}",
        f"Direct optimization missed ES at all five deadlines in {len(cases)} cases. "
        f"The initial reference dispatch met the strengthened cost and service criteria for {initial_count}. "
        f"Including the longer reference runs raised this count to {reference_count}.",
    ]
    for trial in alternatives.itertuples():
        name = FAMILIES[final_cases.loc[trial.group, "family"]] + "-" + trial.group.rsplit("_", 1)[1]
        reference = cases.loc[trial.group, "reference_cost"]
        gap = 100 * (trial.cost - reference) / abs(reference)
        lines.append(
            f"For {name}, the archived {trial.method} schedule at {trial.budget} seconds "
            f"left {trial.ens_mwh:g} MWh unserved with a {gap:.2f}" + r"\% cost excess."
        )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the LaTeX tables here instead of stdout.")
    options = parser.parse_args()
    original, strengthened = development_data()
    lines = shortlist_table(original, strengthened) + [""] + delivered_table(original, strengthened)
    lines += [""] + service_witnesses(original, strengthened)
    result = "\n".join(lines) + "\n"
    if options.output:
        options.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
