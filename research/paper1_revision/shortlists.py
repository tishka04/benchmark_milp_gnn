"""Post-hoc evaluation of 120-second review lists. Run as python -m research.paper1_revision.shortlists."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HOME = Path(__file__).resolve().parent
RUN = HOME / "runs" / "revision_v1"
WEIGHTS = HOME.parent / "paper1_weights/runs/weights_v2_variants/report"
OUTPUT = RUN / "shortlists_120s"
PROFILES = ("service4_co2", "cost_priority", "time_priority", "co2_priority", "equal4")
QUOTAS = (5, 10, 20)
SEED = 20261119
N_BOOT = 2000


def historical():
    pred = pd.read_csv(WEIGHTS / "predictions.csv")
    pred = pred.query("budget == 120 and model == 'separate_deadlines' and policy in @PROFILES")
    observations = pd.read_parquet(RUN / "analysis.parquet").query("budget == 120")
    upgrades = pd.read_csv(RUN / "physical_campaign/reference_target_changes.csv")
    upgrades = upgrades.query("budget == 120 and target in @PROFILES")
    equal = pred.query("policy == 'equal4'").set_index("group").predicted
    data = pred.rename(columns={"policy": "target", "fold": "portfolio",
                                "predicted": "tailored", "observed": "y"})
    data["balanced"] = data.group.map(equal)
    columns = ["group", "n_zones", "cost", "reference_cost", "ens_mwh", "exit_s",
               "feasible", "co2_t_per_served_mwh", "demand_mwh", "thermal_mwh", "served_mwh"]
    data = data.merge(observations[columns], on="group", validate="many_to_one")
    data = data.merge(upgrades[["group", "target", "before", "after", "best_reference"]],
                      on=["group", "target"], how="left", validate="one_to_one")
    known = data["before"].notna()
    np.testing.assert_allclose(data.loc[known, "before"], data.loc[known, "y"], atol=1e-9)
    strengthened = data.copy()
    strengthened["y"] = data.after.fillna(data.y)
    strengthened["reference_cost"] = data.best_reference.fillna(data.reference_cost)
    return data, strengthened


def independent():
    data = pd.read_parquet(RUN / "realized_milp_v1/external_evaluation.parquet")
    data = data.query("budget == 120 and kind == 'positive' and target in @PROFILES")
    scores = data.pivot(index=["group", "batch", "target", "y"], columns="representation",
                        values="p").reset_index()
    scores = scores.rename(columns={"batch": "portfolio", "legacy_raw": "tailored",
                                    "realized_compact": "trio", "model_size": "dimensions"})
    equal = scores.query("target == 'equal4'").set_index("group").tailored
    scores["balanced"] = scores.group.map(equal)
    observations = pd.read_parquet(RUN / "physical_campaign/external_observations.parquet")
    measures = ["group", "cost", "reference_cost", "ens_mwh", "exit_s", "feasible",
                "co2_t_per_served_mwh", "demand_mwh"]
    observations = observations.query("budget == 120")[measures]
    features = pd.read_parquet(RUN / "realized_milp_v1/features.parquet",
                               columns=["group", "audit_zones"])
    observations = observations.merge(features, on="group", validate="one_to_one")
    scores = scores.merge(observations, on="group", validate="many_to_one")
    return scores.rename(columns={"audit_zones": "n_zones"})


def order(values, ids, salt):
    # An immutable, outcome-blind key avoids using a family-coded identifier to break ties.
    tie = [hashlib.sha256(f"{salt}:{case}".encode()).digest() for case in ids]
    indices = np.lexsort((tie, -values))
    return np.argsort(indices)


def evaluate(frame, methods, n_boot=0):
    cohort_rows, severity_rows, intervals = [], [], []
    for target, profile in frame.groupby("target"):
        by_quota = {quota: [] for quota in QUOTAS}
        draws = np.random.default_rng(SEED)
        for portfolio, g in profile.groupby("portfolio"):
            g = g.sort_values("group")
            n = len(g)
            assert g.group.is_unique and g[["y", *methods]].notna().all().all()
            y = g.y.to_numpy()
            rows = np.arange(n)[None, :]
            if n_boot:
                rows = np.concatenate([rows, draws.integers(n, size=(n_boot, n))])
            observed = y[rows]
            truth_rank = order(y, g.group, "observed")
            truth = np.argsort(truth_rank[rows], axis=1, kind="stable")
            rankings = {key: np.argsort(order(g[key].to_numpy(), g.group, "predicted")[rows],
                                        axis=1, kind="stable") for key in methods}
            for quota in QUOTAS:
                k = n * quota // 100
                assert 100 * k == n * quota
                best = truth[:, :k]
                ideal = np.take_along_axis(observed, best, axis=1).mean(axis=1)
                truth_mask = np.zeros_like(rows, dtype=bool)
                np.put_along_axis(truth_mask, best, True, axis=1)
                metrics = {}
                for method in [*methods, "uniform"]:
                    if method == "uniform":
                        recall = np.full(len(rows), k/n)
                        regret = ideal - observed.mean(axis=1)
                    else:
                        choice = rankings[method][:, :k]
                        recall = np.take_along_axis(truth_mask, choice, axis=1).mean(axis=1)
                        regret = ideal - np.take_along_axis(observed, choice, axis=1).mean(axis=1)
                        omitted = np.setdiff1d(best[0], choice[0])
                        if "ens_mwh" in g and len(omitted):
                            missed = g.iloc[omitted].copy()
                            missed["gap_eur"] = (missed.cost - missed.reference_cost).clip(lower=0)
                            severity_rows.extend(dict(target=target, portfolio=portfolio, quota=quota,
                                method=method, group=r.group, delivered=bool(r.feasible),
                                y=r.y, ens_mwh=r.ens_mwh, gap_eur=r.gap_eur, exit_s=r.exit_s,
                                co2_t_per_served_mwh=r.co2_t_per_served_mwh)
                                for r in missed.itertuples())
                    metrics[method] = np.column_stack([recall, regret])
                    cohort_rows.append(dict(target=target, portfolio=portfolio, quota=quota,
                        method=method, n=n, k=k, hits=recall[0]*k, recall=recall[0], regret=regret[0]))
                by_quota[quota].append(metrics)
        if n_boot:
            for quota, strata in by_quota.items():
                # Equal portfolio sizes. The same within-portfolio samples serve every method.
                for method in [*methods, "uniform"]:
                    paired = np.mean([s[method] - s["balanced"] for s in strata], axis=0)
                    for i, metric in enumerate(("recall", "regret")):
                        lo, hi = np.quantile(paired[1:, i], [.025, .975])
                        intervals.append(dict(target=target, quota=quota, method=method,
                            comparator="balanced", metric=metric, difference=paired[0, i],
                            lower_95=lo, upper_95=hi))
    return pd.DataFrame(cohort_rows), pd.DataFrame(severity_rows), pd.DataFrame(intervals)


def sensitivity(zone_counts):
    directory = RUN / "checkpoints/sensitivity"
    settings = {}
    for target in PROFILES:
        for variant in sorted((directory / target).iterdir()):
            portions = []
            for fold in range(5):
                path = variant / f"{fold}_all_separate.json"
                records = json.loads(path.read_text())["predictions"]
                portions.append(pd.DataFrame(r for r in records if r["budget"] == 120).assign(portfolio=fold))
            settings[target, variant.name] = pd.concat(portions, ignore_index=True)
    assert len(settings) == 48
    tables = []
    for (target, variant), table in settings.items():
        # The same anchor/cost modification also applies to the balanced comparator.
        # Under changed dominant weights the equal-priority convention stays at 25% each.
        balanced = settings.get(("equal4", variant), settings["equal4", "main"])
        data = table.rename(columns={"p": "tailored"}).assign(target=target)
        data = data.merge(balanced[["group", "portfolio", "p"]].rename(columns={"p": "balanced"}),
                          on=["group", "portfolio"], validate="one_to_one")
        data["n_zones"] = data.group.map(zone_counts)
        per_group, _, _ = evaluate(data, ["tailored", "balanced", "n_zones"])
        tables.append(per_group.assign(variant=variant))
    return pd.concat(tables, ignore_index=True)


def exclusive_cases(old, updated):
    """The descriptive whole-sample top 60, distinct from the fold-wise forecasts."""
    profiles = {"service4_co2": "service", "cost_priority": "cost"}
    rankings = {}
    for label, data in [("original", old), ("strengthened", updated)]:
        table = data.pivot(index="group", columns="target", values="y")[list(profiles)]
        table = table.rename(columns=profiles)
        for profile in profiles.values():
            cutoff = table[profile].nlargest(60).min()
            assert (table[profile] >= cutoff).sum() == 60
            table[f"{profile}_rank"] = table[profile].rank(ascending=False, method="min").astype(int)
            table[f"{profile}_margin"] = table[profile] - cutoff
        rankings[label] = table
    table = rankings["strengthened"]
    rank_columns = ["service_rank", "cost_rank"]
    assert (table[rank_columns].le(60) == rankings["original"][rank_columns].le(60)).all().all()
    table = table.loc[table[rank_columns].le(60).sum(axis=1).eq(1)].copy()
    table["selected_by"] = np.where(table.service_rank.le(60), "service", "cost")
    physical = updated.query("target == 'service4_co2'").set_index("group").rename(
        columns={"cost": "dispatch_cost_eur"})
    table = table.rename(columns={"cost": "cost_score", "service": "service_score"})
    table = table.join(physical[["ens_mwh", "demand_mwh", "dispatch_cost_eur", "reference_cost",
                                 "exit_s", "thermal_mwh", "served_mwh"]], validate="one_to_one")
    table["unserved_fraction"] = table.ens_mwh / table.demand_mwh
    table["thermal_mwh_per_served_mwh"] = table.thermal_mwh / table.served_mwh
    assert (table.reference_cost.abs() > 1).all()
    table["relative_cost_gap"] = (table.dispatch_cost_eur - table.reference_cost) / table.reference_cost.abs()
    table.sort_values(["selected_by", "service_rank"]).to_csv(OUTPUT / "priority_only.csv")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    old, updated = historical()
    new = independent()
    exclusive_cases(old, updated)
    all_groups, all_cases, all_intervals = [], [], []
    for cohort, frame in [("development_original", old), ("development_strengthened", updated),
                          ("independent", new)]:
        expected = 400 if cohort == "independent" else 600
        assert len(frame) == expected * len(PROFILES)
        methods = ["tailored", "balanced", "n_zones"]
        if cohort == "independent":
            methods += ["trio", "dimensions"]
        groups, cases, bands = evaluate(frame, methods, N_BOOT)
        all_groups.append(groups.assign(cohort=cohort))
        all_cases.append(cases.assign(cohort=cohort))
        all_intervals.append(bands.assign(cohort=cohort))
    groups = pd.concat(all_groups, ignore_index=True)
    groups.to_csv(OUTPUT / "by_portfolio.csv", index=False)
    groups.groupby(["cohort", "target", "quota", "method"]).agg(
        k=("k", "sum"), hits=("hits", "sum"), recall=("recall", "mean"),
        regret=("regret", "mean")).to_csv(OUTPUT / "pooled.csv")
    pd.concat(all_intervals).to_csv(OUTPUT / "paired_intervals.csv", index=False)
    pd.concat(all_cases).to_csv(OUTPUT / "missed_cases.csv", index=False)
    zones = old.drop_duplicates("group").set_index("group").n_zones
    sensitivity(zones).to_csv(OUTPUT / "development_target_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
