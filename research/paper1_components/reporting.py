"""Per-deadline held-out scores, component contributions and supported pair plots."""
from contextlib import closing
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from .dataio import data, frozen, output_directory, protocol, publish, seed
from .fitting import predictions
from .parallel import results, worker_count

GROUP = ["endpoint", "budget", "family_control"]
MAIN = ["size", "index", "axes", "interpretable", "linear", "additive", "interactions"]

def losses(y, p, metric):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if metric == "brier":
        return (y-p)**2
    p = np.clip(p, 1e-7, 1-1e-7)
    return -(y*np.log(p)+(1-y)*np.log(1-p))

def interval(values, groups, cfg, salt):
    q = pd.DataFrame({"value": np.asarray(values), "group": np.asarray(groups)}).dropna()
    means = q.groupby("group").value.mean().to_numpy()
    if not len(means):
        return None, None, None, 0
    rng = np.random.default_rng(seed("paper1-bootstrap", *salt))
    samples = rng.choice(means, (cfg["bootstrap_samples"], len(means)), replace=True).mean(axis=1)
    return float(means.mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975)), len(means)

def score_tables(pred, cfg):
    scores, calibration = [], []
    for keys, q in pred.groupby(GROUP+["description"]):
        identity = dict(zip(GROUP+["description"], keys))
        known = q.dropna(subset=["y", "p"])
        y, p = known.y.to_numpy(dtype=float), known.p.to_numpy(dtype=float)
        r = dict(**identity, expected=len(q), known=len(known),
                 unknown_labels=int(q.y.isna().sum()), unpredicted=int(q.p.isna().sum()),
                 positives=int((y == 1).sum()), negatives=int((y == 0).sum()),
                 auc=float(roc_auc_score(y,p)) if len(set(y)) == 2 else None,
                 average_precision=float(average_precision_score(y,p)) if np.any(y == 1) else None)
        for metric in ("brier", "log_loss"):
            v, lo, hi, n = interval(losses(y, p, metric), known.group, cfg, list(keys[:3]))
            r.update({metric:v, metric+"_lower":lo, metric+"_upper":hi, "groups":n})
        scores.append(r)
        for b in range(cfg["calibration_bins"]):
            mask = np.minimum(cfg["calibration_bins"]-1, np.floor(p*cfg["calibration_bins"])) == b
            z = known.loc[mask]
            if len(z):
                v, lo, hi, n = interval(z.y, z.group, cfg, list(keys[:3])+[b])
                calibration.append(dict(**identity, bin=b, groups=n, predicted=float(z.p.mean()),
                                        observed=v, lower=lo, upper=hi))
    return pd.DataFrame(scores), pd.DataFrame(calibration, columns=GROUP+[
        "description", "bin", "groups", "predicted", "observed", "lower", "upper"])

def comparison_definitions(cfg):
    comps = [("description", "size", name, name) for name in MAIN if name != "size"]
    comps += [("curvature", "linear", "additive", "flexible_main_effects"),
              ("interaction_set", "additive", "interactions", "five_products")]
    comps += [("component", "drop_component_"+c.split("_")[0], "interactions", c)
              for c in cfg["targeted_components"]]
    comps += [("interaction", f"drop_interaction_{i+1}", "interactions", a+"*"+b)
              for i, (a,b) in enumerate(cfg["interactions"])]
    return comps

def contrast_tables(pred, cfg):
    overall, by_fold, by_family = [], [], []
    for keys, q in pred.groupby(GROUP):
        identity = dict(zip(GROUP, keys))
        q = q.dropna(subset=["y", "p"])
        wide = q.pivot(index=["group", "key", "family", "fold", "y"],
                       columns="description", values="p")
        for kind, baseline, candidate, term in comparison_definitions(cfg):
            if baseline not in wide or candidate not in wide:
                continue
            z = wide[[baseline, candidate]].dropna()
            y = z.index.get_level_values("y")
            for metric in ("brier", "log_loss"):
                gain = losses(y,z[baseline],metric)-losses(y,z[candidate],metric)
                v, lo, hi, n = interval(gain,z.index.get_level_values("group"),cfg,list(keys))
                row = dict(**identity, contrast_kind=kind, term=term, baseline=baseline,
                    candidate=candidate, metric=metric, gain=v, lower=lo, upper=hi, groups=n,
                    interpretation="positive = lower held-out loss with candidate")
                overall.append(row)
                for field, dest in (("fold", by_fold), ("family", by_family)):
                    for value in z.index.get_level_values(field).unique():
                        mask = z.index.get_level_values(field) == value
                        values = gain[mask]
                        dest.append({**row, field:value, "gain":float(np.mean(values)),
                            "lower":None, "upper":None,
                            "groups":int(z.index.get_level_values("group")[mask].nunique())})
    cols = GROUP+["contrast_kind", "term", "baseline", "candidate", "metric",
                  "gain", "lower", "upper", "groups", "interpretation"]
    return (pd.DataFrame(overall, columns=cols), pd.DataFrame(by_fold, columns=cols+["fold"]),
            pd.DataFrame(by_family, columns=cols+["family"]))

def surface_table(pred, base, cfg):
    """Observed joint cells only; no off-manifold synthetic feature combinations."""
    rows = []
    for keys, q in pred.groupby(GROUP):
        identity = dict(zip(GROUP, keys))
        q = q[q.description.isin(["additive", "interactions"])].dropna(subset=["y", "p"])
        wide = q.pivot(index=["group", "y"], columns="description", values="p").reset_index()
        if not {"additive", "interactions"}.issubset(wide):
            continue
        z = wide.merge(base, on="group", how="left", validate="one_to_one")
        for pair, (a,b) in enumerate(cfg["interactions"],1):
            valid = z.dropna(subset=[a,b,"additive","interactions"])
            bx = np.minimum(cfg["support_bins"]-1, np.floor(valid[a]*cfg["support_bins"]))
            by = np.minimum(cfg["support_bins"]-1, np.floor(valid[b]*cfg["support_bins"]))
            for ix in range(cfg["support_bins"]):
                for iy in range(cfg["support_bins"]):
                    part = valid[(bx == ix) & (by == iy)]
                    n = part.group.nunique()
                    supported = n >= cfg["minimum_surface_groups"]
                    rows.append(dict(**identity, pair=pair, first=a, second=b, x_bin=ix, y_bin=iy,
                        groups=int(n), supported=supported,
                        observed=float(part.y.mean()) if supported else None,
                        predicted_additive=float(part.additive.mean()) if supported else None,
                        predicted_interactions=float(part.interactions.mean()) if supported else None,
                        probability_difference=float((part.interactions-part.additive).mean()) if supported else None))
    return pd.DataFrame(rows, columns=GROUP+["pair", "first", "second", "x_bin", "y_bin",
        "groups", "supported", "observed", "predicted_additive", "predicted_interactions",
        "probability_difference"])

def figures(tmp, scores, contrasts, calibration, surfaces, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for endpoint in cfg["endpoints"]:
        for control in cfg["family_controls"]:
            suffix = f"{endpoint}_family{int(control)}"
            keep = (scores.endpoint == endpoint) & (scores.family_control == control)
            fig, ax = plt.subplots(figsize=(7,4))
            for name in MAIN:
                q = scores[keep & (scores.description == name)].sort_values("budget")
                ax.plot(q.budget, q.brier, marker="o", label=name)
            ax.set(xlabel="Independent deadline (s)", ylabel="Out-of-fold Brier score",
                   title=suffix, xticks=cfg["deadlines_s"])
            ax.legend(fontsize=7); fig.tight_layout()
            fig.savefig(tmp/"figures"/f"prediction_{suffix}.pdf"); plt.close(fig)
            fig, axes = plt.subplots(1,5,figsize=(16,3.5),sharex=True,sharey=True)
            for ax, budget in zip(axes,cfg["deadlines_s"]):
                ax.plot([0,1],[0,1],"k--",linewidth=.7)
                for name in ("size","additive","interactions"):
                    q = calibration[(calibration.endpoint == endpoint) &
                        (calibration.family_control == control) & (calibration.budget == budget) &
                        (calibration.description == name)]
                    if len(q):
                        ax.plot(q.predicted, q.observed, "o-", label=name)
                ax.set(title=f"{budget} s",xlabel="Predicted",xlim=(0,1),ylim=(0,1))
            axes[0].set_ylabel("Observed"); axes[-1].legend(fontsize=6)
            fig.tight_layout(); fig.savefig(tmp/"figures"/f"calibration_{suffix}.pdf"); plt.close(fig)
            q = contrasts[(contrasts.endpoint == endpoint) & (contrasts.family_control == control) &
                          (contrasts.metric == "brier") &
                          contrasts.contrast_kind.isin(["component","interaction","interaction_set"])]
            table = q.pivot(index="term",columns="budget",values="gain").reindex(columns=cfg["deadlines_s"])
            if len(table):
                fig, ax = plt.subplots(figsize=(8,7))
                vmax = max(1e-6, float(np.nanmax(np.abs(table.to_numpy()))))
                im = ax.imshow(table.to_numpy(),cmap="RdBu",vmin=-vmax,vmax=vmax,aspect="auto")
                ax.set_xticks(range(5),cfg["deadlines_s"]); ax.set_yticks(range(len(table)),table.index,fontsize=6)
                ax.set_xlabel("Independent deadline (s)")
                ax.set_title("Held-out Brier gain; positive favors complete model")
                fig.colorbar(im,ax=ax); fig.tight_layout()
                fig.savefig(tmp/"figures"/f"contributions_{suffix}.pdf"); plt.close(fig)
        # All deadlines remain in the CSV; this declared deadline is illustrated.
        for pair in range(1,len(cfg["interactions"])+1):
            q = surfaces[(surfaces.endpoint == endpoint) & (surfaces.family_control == True) &
                         (surfaces.budget == cfg["surface_deadline_s"]) & (surfaces.pair == pair)]
            if q.empty:
                continue
            table = q.pivot(index="y_bin",columns="x_bin",values="probability_difference")
            fig, ax = plt.subplots(figsize=(5,4))
            im = ax.imshow(table.to_numpy(dtype=float),origin="lower",cmap="RdBu",vmin=-.2,vmax=.2)
            a,b = cfg["interactions"][pair-1]
            labels = [f"{i/cfg['support_bins']:.1f}-{(i+1)/cfg['support_bins']:.1f}"
                      for i in range(cfg["support_bins"])]
            ax.set_xticks(range(len(labels)),labels,fontsize=7)
            ax.set_yticks(range(len(labels)),labels,fontsize=7)
            ax.set(xlabel=a,ylabel=b,title=f"{endpoint}, {cfg['surface_deadline_s']} s\nObserved cells; blank = insufficient support")
            fig.colorbar(im,ax=ax,label="Interaction minus additive prediction")
            fig.tight_layout(); fig.savefig(tmp/"figures"/f"pair{pair}_{endpoint}.pdf"); plt.close(fig)

def _report_task(job):
    """One independent endpoint/deadline/family-control cohort; no file writes."""
    pred, base, cfg = job
    scores, cal = score_tables(pred, cfg)
    contrasts, folds, families = contrast_tables(pred, cfg)
    surfaces = surface_table(pred, base, cfg)
    return scores, cal, contrasts, folds, families, surfaces


def numerical_tables(pred, base, cfg, workers=1):
    workers = worker_count(workers)
    # This is the original groupby order; sort returned jobs by index before
    # concatenation so process scheduling never changes CSV row order.
    jobs = [(q, base, cfg) for _, q in pred.groupby(GROUP)]
    if not jobs:
        raise ValueError("No prediction cohorts available for reporting")
    collected = {}
    with closing(results(_report_task, jobs, workers=min(workers, len(jobs)))) as completed:
        for index, tables in completed:
            collected[index] = tables
            print(f"Paper 1 report: {len(collected)}/{len(jobs)} cohorts computed", flush=True)
    ordered = [collected[i] for i in range(len(jobs))]
    return tuple(pd.concat([tables[j] for tables in ordered], ignore_index=True)
                 for j in range(6))


def report(run, workers=1):
    workers = worker_count(workers)
    run = Path(run); frozen(run); cfg = protocol()
    pred, models, coefficients = predictions(run)
    final, tmp = output_directory(run,"report")
    if tmp is None:
        return
    source = data(run).drop_duplicates("group")
    base = source[["group"]+cfg["targeted_components"]]
    scores, cal, contrasts, folds, families, surfaces = numerical_tables(pred, base, cfg, workers)
    tables = {"predictions":pred,"prediction_scores":scores,"calibration":cal,
        "predictive_contrasts":contrasts,"contrast_by_fold":folds,"contrast_by_family":families,
        "fit_diagnostics":models,"coefficients":coefficients,"interaction_surfaces":surfaces}
    for name, table in tables.items():
        table.to_csv(tmp/"tables"/(name+".csv"),index=False)
    figures(tmp,scores,contrasts,cal,surfaces,cfg)
    frozen(run)
    publish(run,final,tmp)
    print(f"Paper-1 tables and figures: {final}",flush=True)
