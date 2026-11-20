"""Reproducible operational coverage, physical costs and paired supports."""
from __future__ import annotations

from collections import defaultdict, Counter
import csv
from functools import lru_cache
import html
import itertools
from pathlib import Path
import statistics

from .common import BUDGETS, METHODS, FAMILIES, digest, read_json, load_record, atomic_json, finite
from .physics import validate_dispatch
from .budget_reserve import evaluation_schedule, RECEIPT_NAME as RESERVE_RECEIPT
from .workflows import report_path


def gap(cost, reference):
    if reference is None or not finite(reference):
        return None, None
    difference = cost - reference
    return (100 * difference / abs(reference) if reference != 0 else None), difference


def summarize(rows):
    summary, common, pairs = [], [], []
    primary = [r for r in rows if r["repetition"] == 0]
    by_case = defaultdict(dict)
    for r in primary:
        by_case[r["key"], r["budget"]][r["method"]] = r
    for (key, budget), methods in by_case.items():
        if all(m in methods and methods[m]["success"] and methods[m]["relative_gap_percent"] is not None for m in METHODS):
            common.extend(dict(methods[m], support="all_four") for m in METHODS)
        for a, b in itertools.combinations(METHODS, 2):
            if all(m in methods and methods[m]["success"] and methods[m]["relative_gap_percent"] is not None for m in (a, b)):
                pairs.append(dict(key=key, budget=budget, family=methods[a]["family"], method_a=a, method_b=b,
                                  gap_a=methods[a]["relative_gap_percent"], gap_b=methods[b]["relative_gap_percent"],
                                  gap_difference=methods[a]["relative_gap_percent"] - methods[b]["relative_gap_percent"]))
    for family in FAMILIES:
        for budget in BUDGETS:
            for method in METHODS:
                group = [r for r in primary if (r["family"], r["budget"], r["method"]) == (family, budget, method)]
                success = [r for r in group if r["success"]]
                supported = [r["relative_gap_percent"] for r in common if (r["family"], r["budget"], r["method"]) == (family, budget, method)]
                summary.append(dict(family=family, budget=budget, method=method, attempted=len(group), denominator=100,
                    successes=len(success), success_percent=len(success), without_ens=sum(r["ens_mwh"] <= 1e-3 for r in success),
                    ens_mwh=sum(r["ens_mwh"] for r in success), missing_reference=sum(not r["reference_valid"] for r in group),
                    zero_reference=sum(r["reference_cost"] == 0 for r in group), common_support=len(supported),
                    median_gap_common=statistics.median(supported) if supported else None,
                    **{f"coverage_gap_{limit}_percent": sum(r["success"] and r["relative_gap_percent"] is not None and r["relative_gap_percent"] <= limit for r in group)
                       for limit in (1, 5, 10)}))
    return summary, common, pairs


def write_csv(path, rows):
    path = Path(path)
    keys = list(dict.fromkeys(k for r in rows for k in r)) or ["no_observations"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def analyze(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    manifest = read_json(out / "manifest.json")
    mh = digest(manifest)
    cases = {c["key"]: c for c in manifest["cohort"]}
    dest = out / "analysis"
    dest.mkdir(parents=True, exist_ok=True)
    refs, rows, problems = {}, [], []
    @lru_cache(maxsize=8)
    def scenario(key):
        from src.milp.scenario_loader import load_scenario_data
        return load_scenario_data(Path(cases[key]["file"]))
    for key, case in cases.items():
        ref = load_record(report_path(out, case, "references"))
        valid, cost, bound, optimal = False, None, None, False
        if ref:
            if ref["identity"]["manifest_hash"] != mh:
                raise ValueError("Mixed reference versions")
            result = ref["result"]
            if result.get("valid"):
                check = validate_dispatch(scenario(key), result["dispatch"])
                valid = check["valid"]
                cost = check.get("cost") if valid else None
                bound, optimal = result.get("bound"), result.get("proven_optimal", False)
        refs[key] = dict(valid=valid, cost=cost, bound=bound, optimal=optimal)
    calibration = load_record(out / "calibration.json")
    amendment = load_record(out / RESERVE_RECEIPT)
    expected = (list(evaluation_schedule(manifest, calibration, amendment))
                + list(evaluation_schedule(manifest, calibration, amendment, extra=True)))
    for item in expected:
        identity = dict(item, manifest_hash=mh)
        path = out / "experiments" / (digest(identity) + ".json")
        record = load_record(path, identity)
        if not record or not record.get("complete"):
            continue
        result = record.get("result") or {}
        check = result.get("validation") or {}
        if record["success"]:
            actual = validate_dispatch(scenario(item["key"]), result.get("dispatch"))
            valid = (actual["valid"] and record["exit_s"] <= item["budget"]
                     and not record["forced"] and record["all_processes_exited"]
                     and finite(check.get("cost")) and abs(actual["cost"] - check["cost"]) <= 1e-3)
            if not valid:
                problems.append(dict(identity=identity, reason="late/invalid/nonreproducible success"))
                raise ValueError(f"Acceptance invariant failed: {path}")
        ref = refs[item["key"]]
        cost = check.get("cost") if record["success"] else None
        rg, delta = gap(cost, ref["cost"]) if cost is not None else (None, None)
        rows.append(dict(**item, family=cases[item["key"]]["family"], success=record["success"], status=record["status"],
            exit_s=record["exit_s"], preload_s=record["preload_s"], compute_s=record["compute_s"],
            finalization_s=record["finalization_s"], overshoot_s=record["deadline_overshoot_s"],
            physical_cost=cost, ens_mwh=check.get("ens_mwh") if record["success"] else None,
            overgen_mwh=check.get("overgen_mwh") if record["success"] else None,
            reference_valid=ref["valid"], reference_cost=ref["cost"], reference_proven_optimal=ref["optimal"],
            reference_bound=ref["bound"], relative_gap_percent=rg, absolute_cost_difference=delta))
    summary, common, pairs = summarize(rows)
    extras = [r for r in rows if r["key"] in manifest["variability_cases"]]
    failure_counts = Counter((r["method"], r["status"], r["repetition"]) for r in rows if not r["success"])
    failures = [dict(method=m, status=s, repetition=rep, count=n) for (m, s, rep), n in sorted(failure_counts.items())]
    for name, data in (("experiments", rows), ("coverage", summary), ("common_support", common),
                       ("paired_support", pairs), ("variability_all_repetitions", extras), ("failures", failures)):
        write_csv(dest / (name + ".csv"), data)
    acceptance = dict(accepted=len(rows) == len(expected) and not problems,
                      expected=18480, completed=len(rows), main_completed=sum(r["repetition"] == 0 for r in rows),
                      extra_completed=sum(r["repetition"] > 0 for r in rows),
                      valid_references=sum(r["valid"] for r in refs.values()),
                      missing_references=[k for k, r in refs.items() if not r["valid"]],
                      invalid_or_late_successes=0, physical_costs_recomputed=True, manifest_hash=mh,
                      timing_revision=amendment["sha256"] if amendment else None,
                      superseded_experiments_preserved=amendment["counts"].get("superseded", 0) if amendment else 0)
    atomic_json(dest / "acceptance.json", acceptance)
    plots = []
    def save(fig, name):
        fig.savefig(dest / (name + ".png"), dpi=160, bbox_inches="tight")
        fig.savefig(dest / (name + ".svg"), bbox_inches="tight")
        plt.close(fig)
        plots.append(name)
    for field, title in (("successes", "Solutions exploitables (nombre = % sur 100)"),
                         ("without_ens", "Solutions exploitables sans ENS (sur 100)"),
                         ("ens_mwh", "ENS totale des solutions exploitables (MWh)"),
                         ("coverage_gap_1_percent", "Délai respecté et gap ≤ 1 % (sur 100)"),
                         ("coverage_gap_5_percent", "Délai respecté et gap ≤ 5 % (sur 100)"),
                         ("coverage_gap_10_percent", "Délai respecté et gap ≤ 10 % (sur 100)")):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, family in zip(axes.flat, FAMILIES):
            for method in METHODS:
                group = sorted([r for r in summary if r["family"] == family and r["method"] == method], key=lambda r: r["budget"])
                ax.plot([r["budget"] for r in group], [r[field] for r in group], marker="o", label=method)
            ax.set(title=family.replace("_scenarios", ""), xlabel="Budget (s)", xscale="log")
            if field != "ens_mwh":
                ax.set_ylim(0, 100)
            ax.grid(alpha=.2)
        axes.flat[0].legend()
        fig.suptitle(title)
        fig.tight_layout()
        save(fig, field)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, family in zip(axes.flat, FAMILIES):
        for method in METHODS:
            group = [r for r in rows if r["repetition"] == 0 and r["family"] == family and r["method"] == method and r["success"] and r["reference_valid"]]
            ax.scatter([r["reference_cost"] for r in group], [r["physical_cost"] for r in group], s=5, alpha=.25, label=method)
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=.7)
        ax.set(title=family.replace("_scenarios", ""), xlabel="Référence 600 s", ylabel="Coût obtenu")
    axes.flat[0].legend()
    fig.tight_layout()
    save(fig, "individual_costs")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, family in zip(axes.flat, FAMILIES):
        for mi, method in enumerate(METHODS):
            for bi, budget in enumerate(BUDGETS):
                values = [r["relative_gap_percent"] for r in common if (r["family"], r["method"], r["budget"]) == (family, method, budget)]
                x = bi + (mi - 1.5) * .18
                if values:
                    ax.scatter([x] * len(values), values, s=3, alpha=.2, color=f"C{mi}")
                    ax.plot([x - .06, x + .06], [statistics.median(values)] * 2, color=f"C{mi}", linewidth=2)
            ax.plot([], [], color=f"C{mi}", label=method)
        ns = [sum(r["family"] == family and r["budget"] == b and r["method"] == METHODS[0] for r in common) for b in BUDGETS]
        ax.set_xticks(range(7), [f"{b}s\nn={n}" for b, n in zip(BUDGETS, ns)])
        ax.set(title=family.replace("_scenarios", ""), ylabel="Gap (%) ; support commun 4 méthodes")
        ax.axhline(0, color="black", linewidth=.5)
    axes.flat[0].legend()
    fig.tight_layout()
    save(fig, "gap_distributions_common")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for mi, method in enumerate(METHODS):
        for rep in (0, 1, 2):
            counts = [sum(r["method"] == method and r["budget"] == b and r["repetition"] == rep and r["success"] for r in extras) for b in BUDGETS]
            axes[0].plot(BUDGETS, counts, color=f"C{mi}", linestyle=("-", "--", ":")[rep], label=f"{method} rép. {rep}")
        group = [r for r in rows if r["method"] == method]
        axes[1].scatter([r["budget"] for r in group], [r["overshoot_s"] for r in group], s=4, alpha=.15, label=method)
    axes[0].set(xscale="log", xlabel="Budget (s)", ylabel="Succès sur 30", ylim=(0, 30))
    axes[0].legend(fontsize=6)
    axes[1].set(xscale="log", xlabel="Budget (s)", ylabel="Dépassement jusqu’à sortie effective (s)")
    axes[1].legend()
    fig.tight_layout()
    save(fig, "variability_and_overshoots")
    links = " ".join(f'<a href="{p.name}">{html.escape(p.name)}</a>' for p in sorted(dest.glob("*.csv")))
    body = f"""<!doctype html><html lang="fr"><meta charset="utf-8"><title>Operational exit v1</title>
<style>body{{font:16px system-ui;max-width:1300px;margin:40px auto;color:#172435}}img{{width:100%}}pre{{white-space:pre-wrap}}a{{margin-right:12px}}</style>
<h1>Sortie effective : {len(rows):,} / 18 480 expériences</h1>
{'<p><strong>TEST SYNTHÉTIQUE DE RENDU — aucun résultat scientifique.</strong></p>' if manifest.get('synthetic_test_only') else ''}
<p>État : {'complet, contrôles réussis' if acceptance['accepted'] else 'INCOMPLET — ne constitue pas une comparaison finale'}.</p>
<p>Quatre méthodes, sept processus indépendants par scénario. Lecture, HTE en ligne, sélection par coût physique, validation, écriture et sortie de tous les processus sont chronométrés. ENS permise et mesurée. La cascade RH-MO+LP peut appeler des MILP.</p>
<p>Chaque famille a un dénominateur fixe de 100. Les gaps utilisent |coût référence| ; les valeurs négatives sont conservées. Une référence nulle produit une différence absolue. Sans incumbent de référence valide, les succès restent comptés mais les gaps sont exclus. Les distributions communes indiquent leur effectif. Les répétitions ne sont jamais sélectionnées au meilleur résultat.</p>
<p>Références valides : {acceptance['valid_references']} / 600. Optimum uniquement si démontré ; sinon incumbent à 600 secondes solveur.</p>
<p>Révision de réserve : {acceptance['timing_revision'] or 'calibration initiale'}. {acceptance['superseded_experiments_preserved']} anciens essais conservés mais exclus de cette comparaison corrigée.</p>
<p>{links}</p><pre>{html.escape(str(acceptance))}</pre>
"""
    body += "".join(f'<h2>{html.escape(name.replace("_", " "))}</h2><img src="{name}.png"><a href="{name}.svg">SVG</a>' for name in plots)
    body += "<h2>Causes d’échec</h2><pre>" + html.escape(str(failures)) + "</pre></html>"
    (dest / "rapport.html").write_text(body, encoding="utf-8")
    return acceptance
