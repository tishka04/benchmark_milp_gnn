"""Joint deadline/cost/ENS reporting; called explicitly by the analysis commands."""
from __future__ import annotations
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from .core import config, frozen, put, sealed
from .requirements import profiles, request_for, outcome, profile_signature, deadlines

GROUP = ["partition", "condition", "profile_id", "family", "method", "budget", "repetition"]
CELL = ["partition", "condition", "group", "key", "family", "profile_id", "budget"]

def expanded(df, cfg=None):
    cfg = cfg or config()
    rows = []
    grid = deadlines(cfg)
    for raw in df.to_dict("records"):
        if raw["budget"] not in grid:
            continue
        actual = raw.get("profile_id")
        actual = actual if isinstance(actual, str) and actual else None
        for p in profiles(cfg):
            # A routed execution belongs to its declared request only.
            if actual is not None and actual != p["id"]:
                continue
            req = request_for(p["id"], float(raw["budget"]), cfg)
            rows.append(dict(raw, profile_id=p["id"], priority=p["priority"], alpha=p["alpha"],
                             **outcome(raw, req, cfg)))
    return pd.DataFrame(rows)

def paired(df):
    from .statistics import interval
    results = []
    primary = df[df.repetition == 0]
    for keys, q in primary.groupby(["partition", "condition", "profile_id", "budget"], dropna=False):
        base = dict(zip(["partition", "condition", "profile_id", "budget"], keys))
        for a, b in itertools.combinations(sorted(q.method.unique()), 2):
            qa = q[q.method == a].set_index(["group", "key"])
            qb = q[q.method == b].set_index(["group", "key"])
            common = qa[["joint_success", "cost", "exit_s", "ens_mwh"]].join(
                qb[["joint_success", "cost", "exit_s", "ens_mwh"]], lsuffix="_a", rsuffix="_b", how="inner")
            known = common.dropna(subset=["joint_success_a", "joint_success_b"])
            for metric in ("joint_success", "cost", "exit_s", "ens_mwh"):
                eligible = known if metric == "joint_success" else known[
                    (known.joint_success_a == 1) & (known.joint_success_b == 1)]
                eligible = eligible.dropna(subset=[metric+"_a", metric+"_b"])
                diff = eligible[metric+"_a"] - eligible[metric+"_b"]
                value, lo, hi, n = interval(diff.to_numpy(), eligible.index.get_level_values("group"))
                results.append(dict(**base, first=a, second=b, metric=metric,
                                    difference=value, lower=lo, upper=hi, groups=n,
                                    common_known=len(known), paired_support=len(eligible),
                                    conditional_on_both_joint_success=metric != "joint_success"))
    return pd.DataFrame(results, columns=["partition", "condition", "profile_id", "budget",
        "first", "second", "metric", "difference", "lower", "upper", "groups",
        "common_known", "paired_support", "conditional_on_both_joint_success"])

def write_tables(df, tmp):
    from .statistics import interval, metrics
    df = df[df.budget.isin(deadlines())].copy()
    x = expanded(df)
    if x.empty:
        put(tmp/"tables"/"joint_unavailable.json", dict(reason="No observed allocation grid"))
        return
    coverage, quality, violations, reliability = [], [], [], []
    for keys, q in x.groupby(GROUP, dropna=False):
        base = dict(zip(GROUP, keys))
        y = pd.to_numeric(q.joint_success, errors="coerce")
        v, lo, hi, n = interval(y, q.group)
        coverage.append(dict(**base, expected=len(q), observed=int(q.observed.sum()),
                             known=int(y.notna().sum()), successes=int((y == 1).sum()),
                             failures=int((y == 0).sum()), unknown=int(y.isna().sum()),
                             estimate=v, lower=lo, upper=hi, groups=n,
                             risk_target=1-float(q.alpha.iloc[0]),
                             observed_failure_rate=None if v is None else 1-v))
        for flag in ("valid_exit_ok", "deadline_ok", "cost_ok", "ens_ok"):
            f = pd.to_numeric(q[flag], errors="coerce")
            violations.append(dict(**base, condition_flag=flag, known=int(f.notna().sum()),
                                   violations=int((f == 0).sum()), unknown=int(f.isna().sum())))
        for metric in ("cost", "exit_s", "ens_mwh"):
            z = pd.to_numeric(q.loc[y == 1, metric], errors="coerce").dropna()
            quality.append(dict(**base, metric=metric, support=len(z),
                                mean=float(z.mean()) if len(z) else None,
                                median=float(z.median()) if len(z) else None))
        if "routing_predicted_success" in q:
            z = q[q.joint_success.notna() & q.routing_predicted_success.notna()]
            if len(z):
                pred = z.routing_predicted_success.astype(float)
                reliability.append(dict(**base, **metrics(z.joint_success.astype(int), pred),
                                        unsupported=int((q.get("routing_risk_supported", pd.Series(None, index=q.index)) == False).sum()),
                                        fallback=int((q.get("routing_fallback", pd.Series(None, index=q.index)) == True).sum())))
    sensitivity = []
    cfg = config()
    fixed = df[~df.method.astype(str).str.startswith("router")].copy()
    grouping = ["partition", "condition", "family", "method", "budget", "repetition"]
    for keys, q in fixed.groupby(grouping, dropna=False):
        base = dict(zip(grouping, keys))
        for cap in config()["service_routing"]["ens_sensitivity_caps_mwh"]:
            for tau in (None, *config()["quality_thresholds"]):
                req = dict(request_for("balanced_service", float(base["budget"])),
                           ens_cap_mwh=float(cap), cost_tolerance=tau)
                values = [outcome(r, req, cfg)["joint_success"] for r in q.to_dict("records")]
                v, lo, hi, n = interval(values, q.group.to_numpy())
                sensitivity.append(dict(**base, ens_cap_mwh=cap, cost_tolerance=tau,
                    expected=len(q), known=sum(x is not None for x in values),
                    estimate=v, lower=lo, upper=hi, groups=n))
    pd.DataFrame(sensitivity).to_csv(tmp/"tables"/"joint_ens_sensitivity.csv", index=False)
    pd.DataFrame(coverage).to_csv(tmp/"tables"/"joint_coverage.csv", index=False)
    pd.DataFrame(quality).to_csv(tmp/"tables"/"joint_successful_quality.csv", index=False)
    pd.DataFrame(violations).to_csv(tmp/"tables"/"joint_violations.csv", index=False)
    pd.DataFrame(reliability, columns=GROUP+["n", "prevalence", "auc", "average_precision",
        "brier", "log_loss", "unsupported", "fallback"]).to_csv(tmp/"tables"/"joint_routing_reliability.csv", index=False)
    paired(x).to_csv(tmp/"tables"/"joint_paired.csv", index=False)
    # Retain group IDs for audit/reanalysis; does not make repeated profiles independent.
    x.to_csv(tmp/"tables"/"joint_cells.csv", index=False)
    conditions = []
    for keys, q in x[x.repetition == 0].groupby(["profile_id", "method", "budget"]):
        if q.duplicated(["group", "condition"]).any():
            continue  # combined partitions must not be conflated
        wide = q.pivot(index="group", columns="condition", values="joint_success")
        if "base" not in wide:
            continue
        for condition in wide.columns:
            if condition == "base":
                continue
            z = wide[["base", condition]].dropna()
            v, lo, hi, n = interval((z[condition]-z["base"]).to_numpy(), z.index)
            conditions.append(dict(profile_id=keys[0], method=keys[1], budget=keys[2],
                                   condition=condition, difference=v, lower=lo, upper=hi, groups=n))
    pd.DataFrame(conditions, columns=["profile_id", "method", "budget", "condition",
        "difference", "lower", "upper", "groups"]).to_csv(tmp/"tables"/"joint_paired_conditions.csv", index=False)
    put(tmp/"tables"/"joint_protocol.json", sealed(dict(
        profiles=profiles(), profile_signature=profile_signature(), deadlines_s=deadlines(),
        comparison_unit="one profile and one deadline; no pooling across deadlines",
        failure_rule="known failed condition dominates unknown; otherwise unknown remains unknown",
        allocation_rule="independent budget runs; no relabelling of a longer run",
        zero_rule="ENS <= declared cap in MWh; source snapshot zero counts are not assumed reproduced",
        inference="risk target is a model criterion, not a reliability certificate")))

def analyze_joint(run, input_path=None, name="main_joint"):
    from .statistics import frame, output_directory, publish
    run = Path(run)
    frozen(run)
    source = Path(input_path or run/"main_rows.json")
    final, tmp = output_directory(run, name, source)
    if tmp is None:
        return
    write_tables(frame(source), tmp)
    publish(final, tmp, source)
