"""Descriptive component/score sensitivity, never used to choose held-out features."""
from pathlib import Path
import numpy as np
import pandas as pd
from .dataio import COMPONENTS, DESCRIPTORS, data, frozen, output_directory, protocol, publish

def score_variants(base, cfg):
    a, b = base[COMPONENTS[:12]], base[COMPONENTS[12:]]
    s, h = a.mean(axis=1), b.mean(axis=1)
    variants = {f"mix_{v:g}": v*s+(1-v)*h for v in cfg["mixing_grid"]}
    for c in COMPONENTS:
        variants["remove_"+c] = (.5*a.drop(columns=c).mean(axis=1)+.5*h if c in a
                                 else .5*s+.5*b.drop(columns=c).mean(axis=1))
    return .5*s+.5*h, variants

def rank_change(reference, variant):
    x, y = np.asarray(reference), np.asarray(variant)
    i, j = np.triu_indices(len(x), 1)
    a, b = np.sign(x[i]-x[j]), np.sign(y[i]-y[j])
    eligible = (a != 0) & (b != 0)
    corr = pd.Series(x).corr(pd.Series(y), method="spearman") if np.std(x)>0 and np.std(y)>0 else np.nan
    return dict(scenarios=len(x), mean_abs_change=float(np.mean(np.abs(y-x))),
        max_abs_change=float(np.max(np.abs(y-x))), spearman=corr,
        comparable_pairs=int(eligible.sum()), reversed_pairs=int(((a*b < 0) & eligible).sum()),
        baseline_tied_pairs=int((a == 0).sum()), variant_tied_pairs=int((b == 0).sum()),
        bin_migrations=int((np.minimum(4, np.floor(x*5)) != np.minimum(4, np.floor(y*5))).sum()))

def audit(run):
    run = Path(run); frozen(run); cfg = protocol(); df = data(run)
    final, tmp = output_directory(run, "audit")
    if tmp is None:
        return
    base = df.drop_duplicates("group")
    stats, correlations, support = [], [], []
    scopes = [("all", base)] + list(base.groupby("family"))
    for family, q in scopes:
        for c in DESCRIPTORS:
            values = q[c].dropna().astype(float)
            raw = q["raw_"+c].dropna() if "raw_"+c in q else pd.Series(dtype=float)
            stats.append(dict(family=family, component=c, groups=len(q), known=len(values),
                missing=int(q[c].isna().sum()), unique=int(values.nunique()),
                constant=bool(len(values) and values.max()-values.min() <= cfg["constant_tolerance"]),
                mean=values.mean(), std=values.std(), minimum=values.min(), maximum=values.max(),
                raw_minimum=raw.min(), raw_maximum=raw.max(),
                lower_boundary=int((values <= 0).sum()) if c in COMPONENTS else None,
                upper_boundary=int((values >= 1).sum()) if c in COMPONENTS else None))
        for i, a in enumerate(COMPONENTS):
            for b in COMPONENTS[i+1:]:
                z = q[[a, b]].dropna()
                usable = len(z)>1 and (z.max()-z.min()).min()>cfg["constant_tolerance"]
                correlations.append(dict(family=family, first=a, second=b, groups=len(z),
                    pearson=z[a].corr(z[b]) if usable else None,
                    spearman=z[a].corr(z[b], method="spearman") if usable else None))
        for idx, (a, b) in enumerate(cfg["interactions"], 1):
            z = q[[a, b]].dropna()
            for ix in range(cfg["support_bins"]):
                for iy in range(cfg["support_bins"]):
                    bx = np.minimum(cfg["support_bins"]-1, np.floor(z[a]*cfg["support_bins"]))
                    by = np.minimum(cfg["support_bins"]-1, np.floor(z[b]*cfg["support_bins"]))
                    count = int(((bx == ix) & (by == iy)).sum())
                    support.append(dict(family=family, pair=idx, first=a, second=b,
                        x_bin=ix, y_bin=iy, groups=count,
                        supported=count >= cfg["minimum_surface_groups"]))
    pd.DataFrame(stats).to_csv(tmp/"tables/component_audit.csv", index=False)
    pd.DataFrame(correlations).to_csv(tmp/"tables/component_correlations.csv", index=False)
    pd.DataFrame(support).to_csv(tmp/"tables/interaction_support.csv", index=False)
    changes, scores = [], []
    valid = base.dropna(subset=COMPONENTS)
    for family, q in [("all", valid)] + list(valid.groupby("family")):
        if q.empty:
            continue
        original, variants = score_variants(q, cfg)
        for name, values in variants.items():
            changes.append(dict(family=family, variant=name, **rank_change(original, values)))
            if family == "all":
                scores.extend(dict(group=g, variant=name, value=float(v))
                              for g, v in zip(q.group, values))
    pd.DataFrame(changes).to_csv(tmp/"tables/score_sensitivity.csv", index=False)
    pd.DataFrame(scores).to_csv(tmp/"tables/score_variants.csv", index=False)
    contributions = valid[["key", "group", "family", "criticality"]].copy()
    for i, c in enumerate(COMPONENTS):
        contributions["contribution_"+c] = valid[c]*(1/24 if i < 12 else 1/20)
    cols = ["contribution_"+c for c in COMPONENTS]
    contributions["index_unrounded"] = contributions[cols].sum(axis=1)
    contributions["difference_from_stored_index"] = contributions.index_unrounded-contributions.criticality
    contributions.to_csv(tmp/"tables/score_contributions.csv", index=False)
    endpoint_support = df.groupby("budget")[cfg["endpoints"]].agg(["count", "sum"])
    endpoint_support.to_csv(tmp/"tables/endpoint_support.csv")
    df[["key", "group", "family", "outer_fold"]].drop_duplicates().to_csv(tmp/"tables/folds.csv", index=False)
    publish(run, final, tmp)
