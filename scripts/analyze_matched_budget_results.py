"""Read-only audit and visual analysis of saved matched-budget experiments.

No solver/model is instantiated. Source results are never overwritten. Run from
the repository with its .venv; the companion PDF renderer uses reportlab.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator
import numpy as np

from scripts import benchmark_matched_budgets as benchmark

METHODS = benchmark.METHODS
FAMILIES = benchmark.FAMILIES
NAMES = {"milp": "HiGHS", "pipeline": "Pipeline K=16", "jepa": "JEPA"}
FAMILY_NAMES = dict(zip(FAMILIES, ["Low", "Medium", "High", "Persona VRE + batterie",
                                  "Persona réseau", "Persona mathématicien"]))
COLORS = dict(milp="#236b9e", pipeline="#d86929", jepa="#168579")
MARKERS = dict(milp="o", pipeline="s", jepa="^")
BUDGETS = list(benchmark.DEFAULT_BUDGETS)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def median(values):
    return statistics.median(values) if values else None


def checksum_tree(paths):
    return hashlib.sha256("\n".join(f"{p}:{benchmark.sha256(p)}" for p in sorted(paths)).encode()).hexdigest()


def audit(source):
    manifest = json.loads((source / "manifest.json").read_text("utf-8"))
    rows = json.loads((source / "per_scenario.json").read_text("utf-8"))
    assert manifest["protocol"] == benchmark.PROTOCOL
    assert manifest["budgets"] == BUDGETS and manifest["k"] == 16
    assert all(manifest[k] == 16 for k in ("milp_threads", "jepa_threads", "repair_workers"))
    assert manifest["repair_threads"] == 1
    assert len(manifest["cohort"]) == 600 and len(rows) == 10800
    assert Counter(r["family"] for r in manifest["cohort"]) == dict.fromkeys(FAMILIES, 100)
    lookup = {(r["family"], r["scenario_id"], r["budget_seconds"], r["method"]): r for r in rows}
    assert len(lookup) == len(rows) and all(r["completed"] for r in rows)
    assert len(list((source / "cases").rglob("*.json"))) == len(rows)
    manifest_hash = benchmark.digest(manifest)
    checkpoint_hashes, references, diagnostics = [], [], []
    wall_s = Counter()
    for i, item in enumerate(manifest["cohort"], 1):
        family, sid = item["family"], item["scenario_id"]
        reference = Path(manifest["root"]) / family / "reports" / f"{sid}.json"
        actual_hash = benchmark.sha256(reference)
        assert actual_hash == manifest["inputs_sha256"][str(reference.relative_to(Path(manifest["root"])))]
        # The MIP header precedes the large dispatch arrays; retain exact precision.
        with reference.open(encoding="utf-8") as stream:
            header = stream.read(8192)
        start = header.index('"mip"')
        mip = json.JSONDecoder().raw_decode(header[header.index(":", start) + 1:].lstrip())[0]
        references.append(dict(**item, objective=mip["objective"], status=mip["termination"], sha256=actual_hash))
        for bi, budget in enumerate(BUDGETS):
            for method in METHODS:
                path = benchmark.checkpoint_path(source, i, bi, method)
                outcome = benchmark.load_checkpoint(path, benchmark.identity(item, budget, method, manifest), manifest_hash)
                checkpoint_hashes.append(f"{path.relative_to(source)}:{benchmark.sha256(path)}")
                r = lookup[family, sid, budget, method]
                row = outcome.get("row") or {}
                feasible = benchmark.eligible(row)
                assert r["feasible"] == feasible
                assert r["reference_objective"] == mip["objective"] and r["reference_status"] == mip["termination"]
                assert r["decision_wall_s"] == outcome["decision_wall_s"]
                assert r["hard_timeout"] == outcome["hard_timeout"]
                slack = row.get("slack_mwh")
                assert r["zero_slack"] == bool(feasible and benchmark.finite(slack) and slack <= .001)
                if feasible:
                    assert r["objective"] == row["lp_objective"]
                    assert math.isclose(r["gap_reference_pct"], benchmark.gap(r["objective"], mip["objective"]), abs_tol=1e-10)
                else:
                    assert r["objective"] is None and r["gap_reference_pct"] is None
                wall_s[method] += outcome["experiment_wall_s"]
                if method == "pipeline":
                    gpu = outcome.get("generation") or {}
                    costs = row.get("all_objectives", [])
                    available = [v for v in costs if benchmark.finite(v)]
                    candidate_slacks = row.get("all_sample_slacks", [])
                    zero_available = any(benchmark.finite(c) and benchmark.finite(s) and s <= .001
                                         for c, s in zip(costs, candidate_slacks))
                    diagnostics.append(dict(**item, budget_seconds=budget,
                        generated=bool(gpu.get("ready")), generation_wall_s=outcome.get("generation_wall_s"),
                        gpu_batch_wall_s=gpu.get("gpu_batch_wall_s"),
                        selected_feasible=feasible, n_repairs_received=sum((m["row"] or {}).get("kind") == "repair" for m in outcome["members"])
                        if gpu.get("ready") else 0,
                        min_received_cost=min(available) if available else None,
                        selection_more_costly=bool(feasible and available and row["lp_objective"] > min(available) + .001),
                        zero_slack_available=zero_available,
                        selected_zero_slack=r["zero_slack"]))
        if i % 100 == 0:
            print(f"Audit: {i}/600 scénarios et leurs checkpoints vérifiés", flush=True)
    # Independently recompute published summaries and paired comparison supports.
    summary, pairs = benchmark.summarize(rows, list(FAMILIES), BUDGETS)
    assert summary == json.loads((source / "summary.json").read_text("utf-8"))
    assert pairs == json.loads((source / "paired_comparisons.json").read_text("utf-8"))
    info = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
                n_scenarios=600, n_experiments=10800, n_checkpoints_verified=len(checkpoint_hashes),
                n_references_verified=600, protocol=manifest["protocol"],
                checkpoint_inventory_sha256=hashlib.sha256("\n".join(checkpoint_hashes).encode()).hexdigest(),
                input_files_sha256={name: benchmark.sha256(source / name) for name in
                                   ("manifest.json", "per_scenario.json", "summary.json", "paired_comparisons.json")},
                reference_statuses=dict(Counter(r["status"] for r in references)),
                n_negative_references=sum(r["objective"] < 0 for r in references),
                experiment_wall_hours={m: wall_s[m] / 3600 for m in METHODS},
                note="Audit des identités, empreintes et métriques sauvegardées ; aucune nouvelle résolution ni revalidation physique complète.")
    return rows, lookup, summary, pairs, diagnostics, info


def derived_statistics(rows, lookup, diagnostics):
    coverage, fixed_costs, selector = [], [], []
    for family in ["ALL", *FAMILIES]:
        subset = [r for r in rows if family == "ALL" or r["family"] == family]
        cases = {(r["family"], r["scenario_id"]) for r in subset}
        common_fixed = [key for key in cases if all(lookup[*key, b, m]["feasible"] for b in (60, 120) for m in METHODS)]
        for budget in BUDGETS:
            for method in METHODS:
                group = [r for r in subset if r["budget_seconds"] == budget and r["method"] == method]
                coverage.append(dict(family=family, budget_seconds=budget, method=method,
                    n=len(group), feasible=sum(r["feasible"] for r in group),
                    zero_slack=sum(r["zero_slack"] for r in group),
                    within_1_pct=sum(r["feasible"] and r["gap_reference_pct"] <= 1 for r in group),
                    within_10_pct=sum(r["feasible"] and r["gap_reference_pct"] <= 10 for r in group)))
                if budget in (60, 120):
                    fixed_costs.append(dict(family=family, budget_seconds=budget, method=method,
                        n_common=len(common_fixed), median_gap=median([lookup[*key, budget, method]["gap_reference_pct"] for key in common_fixed])))
            d = [v for v in diagnostics if v["budget_seconds"] == budget and (family == "ALL" or v["family"] == family)]
            selector.append(dict(family=family, budget_seconds=budget, n=len(d),
                n_generated=sum(v["generated"] for v in d), n_any_repair_received=sum(v["n_repairs_received"] > 0 for v in d),
                n_selected_feasible=sum(v["selected_feasible"] for v in d),
                median_generation_s=median([v["generation_wall_s"] for v in d if v["generated"]]),
                median_gpu_batch_s=median([v["gpu_batch_wall_s"] for v in d if v["gpu_batch_wall_s"] is not None]),
                n_selection_more_costly=sum(v["selection_more_costly"] for v in d),
                n_zero_slack_available=sum(v["zero_slack_available"] for v in d),
                n_zero_slack_selected=sum(v["selected_zero_slack"] for v in d)))
    return dict(coverage=coverage, fixed_cohort_costs=fixed_costs, pipeline_diagnostics=selector)


def setup_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
        "axes.labelcolor": "#344050", "text.color": "#243344", "axes.edgecolor": "#bdc6ce",
        "grid.color": "#dce2e7", "grid.linewidth": .65, "svg.fonttype": "none",
        "savefig.facecolor": "white"})


def legend(fig, y=.91):
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], label=NAMES[m], lw=2.2) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, y), fontsize=11)


def save_figure(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=180)
    fig.savefig(out / f"{name}.svg")
    plt.close(fig)


def budget_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(.9, 140)
    ax.set_xticks(BUDGETS, labels=[str(b) for b in BUDGETS])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Budget de décision (secondes, échelle logarithmique)", fontsize=9)
    ax.grid(axis="y")


def percent_axis(ax):
    ax.set_ylim(-3, 105)
    ax.set_yticks([0, 25, 50, 75, 100])


def gap_axis(ax, maximum):
    ax.set_yscale("symlog", linthresh=1, linscale=.7)
    ticks = [0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
    ax.set_ylim(-.25, max(2, maximum * 1.5))
    ax.yaxis.set_major_locator(FixedLocator([v for v in ticks if v <= maximum * 1.5] or [0, 1]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}" if v < 1000 else f"{v:,.0f}".replace(",", " ")))
    ax.axhline(0, color="#9ca9b3", lw=.8)
    ax.grid(axis="y")


def draw_coverage(derived, out, field, title, subtitle, name):
    data = derived["coverage"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10.2))
    fig.subplots_adjust(top=.83, bottom=.095, left=.075, right=.975, hspace=.6, wspace=.18)
    fig.suptitle(title, fontsize=21, x=.075, ha="left", y=.98)
    fig.text(.075, .936, subtitle, fontsize=11)
    legend(fig, .917)
    for ax, family in zip(axes.flat, FAMILIES):
        for method in METHODS:
            values = [next(v for v in data if v["family"] == family and v["method"] == method and v["budget_seconds"] == b) for b in BUDGETS]
            ax.plot(BUDGETS, [100 * v[field] / v["n"] for v in values], color=COLORS[method], marker=MARKERS[method],
                    markersize=5, lw=2, alpha=.9, label=NAMES[method])
        ax.set_title(FAMILY_NAMES[family] + "  |  n = 100", loc="left", fontsize=12)
        budget_axis(ax); percent_axis(ax)
        ax.set_ylabel("Scénarios (%)")
    fig.text(.075, .025, "Même dénominateur : tous les scénarios. Chaque budget est une exécution indépendante ; les lignes guident la lecture.", fontsize=10)
    save_figure(fig, out, name)


def draw_costs(rows, lookup, out):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10.5))
    fig.subplots_adjust(top=.82, bottom=.1, left=.085, right=.97, hspace=.65, wspace=.22)
    fig.suptitle("Coûts : comparer les mêmes scénarios", fontsize=21, x=.085, ha="left", y=.98)
    fig.text(.085, .94, "Écart médian à la référence longue ; bandes = quartiles 25-75 %. Intersection faisable des trois méthodes à chaque budget.", fontsize=10)
    legend(fig, .916)
    for ax, family in zip(axes.flat, FAMILIES):
        ids = sorted({r["scenario_id"] for r in rows if r["family"] == family})
        groups = {b: [sid for sid in ids if all(lookup[family, sid, b, m]["feasible"] for m in METHODS)] for b in (60, 120)}
        maximum = 1
        for m in METHODS:
            q = np.array([np.percentile([lookup[family, sid, b, m]["gap_reference_pct"] for sid in groups[b]], [25, 50, 75]) for b in (60, 120)])
            maximum = max(maximum, q.max())
            ax.plot([0, 1], q[:, 1], color=COLORS[m], marker=MARKERS[m], lw=2)
            ax.fill_between([0, 1], q[:, 0], q[:, 2], color=COLORS[m], alpha=.12)
        ax.set_xticks([0, 1], [f"60 s\nn = {len(groups[60])}", f"120 s\nn = {len(groups[120])}"])
        ax.set_xlim(-.2, 1.2)
        gap_axis(ax, maximum)
        ax.set_title(FAMILY_NAMES[family], loc="left", fontsize=12)
        ax.set_ylabel("Écart à la référence (%)")
    fig.text(.085, .035, "L'effectif commun change entre 60 et 120 s : une hausse n'établit pas, à elle seule, une dégradation individuelle.", fontsize=10)
    fig.text(.085, .015, "Échelle Y symétrique logarithmique, linéaire entre -1 et +1 %. Aucun coût d'une solution infaisable n'entre dans ces statistiques.", fontsize=9)
    save_figure(fig, out, "04_couts_cohortes_communes")

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig.subplots_adjust(top=.84, bottom=.09, left=.09, right=.975, hspace=.52, wspace=.23)
    fig.suptitle("Chaque scénario face à sa référence MILP", fontsize=21, x=.09, ha="left", y=.98)
    fig.text(.09, .94, "Budget de 120 secondes. Tous les coûts faisables sont affichés, y compris les valeurs extrêmes ; aucune imputation des échecs.", fontsize=10)
    legend(fig, .92)
    for ax, family in zip(axes.flat, FAMILIES):
        data = [r for r in rows if r["family"] == family and r["budget_seconds"] == 120 and r["feasible"]]
        maximum = max(r["gap_reference_pct"] for r in data)
        counts = []
        for j, method in enumerate(METHODS):
            group = [r for r in data if r["method"] == method]
            counts.append(len(group))
            ax.scatter([int(r["scenario_id"].split("_")[-1]) + (j-1)*.12 for r in group],
                       [r["gap_reference_pct"] for r in group], color=COLORS[method], marker=MARKERS[method],
                       s=17, alpha=.75, linewidths=0)
        gap_axis(ax, maximum)
        ax.set_title(FAMILY_NAMES[family] + "  |  n H/P/J = " + "/".join(map(str, counts)), loc="left", fontsize=11)
        ax.set_xlim(0, 101); ax.set_xticks([1, 20, 40, 60, 80, 100])
        ax.set_xlabel("Numéro du scénario dans la famille")
        ax.set_ylabel("Écart à la référence (%)")
    fig.text(.09, .03, "0 % = coût de la référence ; plus bas = meilleur. Échelles Y adaptées par famille, logarithmiques au-delà de 1 %.", fontsize=10)
    fig.text(.09, .011, "Écart = 100 × (coût - référence) / |référence|. 475 références négatives : ce pourcentage n'est pas un ratio de coûts.", fontsize=9)
    save_figure(fig, out, "05_couts_par_scenario_120s")


def draw_pairs(pairs, out):
    fig, ax = plt.subplots(figsize=(14, 6.7))
    fig.subplots_adjust(top=.75, bottom=.15, left=.245, right=.97)
    fig.suptitle("Pipeline ou JEPA : qui a le coût le plus bas ?", fontsize=21, x=.055, ha="left", y=.97)
    fig.text(.055, .91, "À 120 secondes ; comparaison sur les scénarios où les deux méthodes sont faisables. Égalité : différence absolue ≤ 0,001.", fontsize=11)
    keys = ["left_wins", "ties", "right_wins", "unpaired"]
    colors = [COLORS["pipeline"], "#b9c3cc", COLORS["jepa"], "#edf0f3"]
    labels = ["Pipeline moins coûteux", "Égalité", "JEPA moins coûteux", "Paire non faisable"]
    data = [next(p for p in pairs if p["family"] == f and p["budget_seconds"] == 120 and p["left"] == "pipeline" and p["right"] == "jepa") for f in FAMILIES]
    left = np.zeros(6)
    for key, color, label in zip(keys, colors, labels):
        values = [100-p["n_paired"] if key == "unpaired" else p[key] for p in data]
        ax.barh(range(6), values, left=left, color=color, height=.6, label=label)
        for i, v in enumerate(values):
            if v >= 4: ax.text(left[i] + v/2, i, str(v), ha="center", va="center", color="white" if key in ("left_wins", "right_wins") else "#263445", fontsize=11)
        left += values
    ax.set_yticks(range(6), [FAMILY_NAMES[f] for f in FAMILIES]); ax.invert_yaxis()
    ax.set_xlim(0,100); ax.set_xlabel("Nombre de scénarios (100 par famille)")
    fig.legend(loc="upper center", bbox_to_anchor=(.51,.87), ncol=4, frameon=False, fontsize=10)
    fig.text(.055,.035,"Sur les 437 paires faisables : pipeline 150 victoires, JEPA 247, 40 égalités. Les 163 autres cas ne sont pas classés par coût.",fontsize=11)
    save_figure(fig,out,"06_duel_pipeline_jepa_120s")


def draw_pipeline_diagnostics(derived, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    fig.subplots_adjust(top=.77, bottom=.19, left=.07, right=.97, wspace=.23)
    fig.suptitle("Où le pipeline perd-il du temps et de la qualité ?", fontsize=20, x=.07, ha="left", y=.97)
    ds = [next(d for d in derived["pipeline_diagnostics"] if d["family"] == "ALL" and d["budget_seconds"] == b) for b in BUDGETS]
    ax=axes[0]
    for key,label,c in [("n_generated","Batch GPU disponible","#8d9ba8"),("n_any_repair_received","Au moins une réparation reçue","#627fa2"),("n_selected_feasible","Solution sélectionnée faisable",COLORS["pipeline"])]:
        ax.plot(BUDGETS,[100*d[key]/600 for d in ds],marker="o",label=label,color=c,lw=2)
    budget_axis(ax);percent_axis(ax);ax.set_ylabel("Scénarios (%)");ax.legend(fontsize=8,loc="upper left")
    ax.set_title("Disponibilité dans le budget",loc="left")
    ax=axes[1]
    d=[v for v in ds if v["budget_seconds"] in (60,120)]
    x=np.arange(2)
    ax.bar(x-.17,[v["n_zero_slack_selected"] for v in d],.32,label="Sélection actuelle",color=COLORS["pipeline"])
    ax.bar(x+.17,[v["n_zero_slack_available"] for v in d],.32,label="Au moins un candidat sans slack",color="#a7b9cd")
    for i,v in enumerate(d):
        ax.text(i-.17,v["n_zero_slack_selected"]+5,str(v["n_zero_slack_selected"]),ha="center")
        ax.text(i+.17,v["n_zero_slack_available"]+5,str(v["n_zero_slack_available"]),ha="center")
    ax.set_xticks(x,["60 s","120 s"]);ax.set_ylabel("Scénarios sur 600");ax.set_ylim(0,max(v["n_zero_slack_available"] for v in d)*1.3)
    ax.legend(fontsize=9);ax.set_title("Candidats déjà reçus, même budget",loc="left")
    fig.text(.07,.07,"La seconde vue est un diagnostic a posteriori des candidats disponibles, pas la performance d'un nouveau sélecteur évalué.",fontsize=10)
    fig.text(.07,.035,"Le sélecteur actuel minimise l'énergie EBM parmi les candidats faisables ; il ne minimise ni le coût réel ni le slack.",fontsize=10)
    save_figure(fig,out,"07_diagnostic_pipeline")


def write_dashboard(rows, out):
    template = Path(__file__).with_name("matched_budget_dashboard.html").read_text("utf-8")
    fields = ("family","scenario_id","budget_seconds","method","feasible","zero_slack","objective",
              "reference_objective","reference_status","gap_reference_pct","slack_mwh","status")
    content = dict(rows=[{k:r[k] for k in fields} for r in rows], families=FAMILY_NAMES, methods=NAMES, colors=COLORS)
    html = template.replace("__DATA_JSON__", json.dumps(content,ensure_ascii=False,separators=(",",":"),allow_nan=False).replace("</","<\\/"))
    (out/"explorer_resultats.html").write_text(html,"utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("outputs/load_shifting_v1/evaluation/budgets_v1"))
    parser.add_argument("--output",type=Path,default=Path("outputs/load_shifting_v1/evaluation/budgets_v1/analysis_20261027"))
    args=parser.parse_args();source=args.source.resolve();out=args.output.resolve()
    assert out!=source and out.name not in ("cases","w","resumes")
    rows,lookup,summary,pairs,diagnostics,info=audit(source)
    derived=derived_statistics(rows,lookup,diagnostics)
    out.mkdir(parents=True,exist_ok=True);figures=out/"figures";figures.mkdir(exist_ok=True)
    save_json(out/"audit.json",info)
    save_json(out/"analysis_data.json",dict(audit=info,summary=summary,pairs=pairs,derived=derived))
    setup_style()
    draw_coverage(derived,figures,"feasible","Faisabilité en fonction du temps disponible",
        "MILP HiGHS (16 threads), pipeline K=16 (GPU + 16 réparations CPU), JEPA (GPU + réparation).", "01_faisabilite_budget")
    draw_coverage(derived,figures,"zero_slack","Faisabilité sans énergie non servie ni surproduction",
        "Solution faisable et slack total ≤ 0,001 MWh. Les scénarios difficiles ne sont pas retirés du dénominateur.","02_sans_slack_budget")
    draw_coverage(derived,figures,"within_10_pct","Couverture à un coût proche de la référence",
        "Solution faisable et écart de coût ≤ +10 % par rapport à la référence MILP longue ; 100 scénarios par famille.","03_couverture_cout_10pct")
    draw_costs(rows,lookup,figures);draw_pairs(pairs,figures);draw_pipeline_diagnostics(derived,figures)
    write_dashboard(rows,out)
    print(json.dumps({"output":str(out),"audit":info,"global_diagnostics":[d for d in derived["pipeline_diagnostics"] if d["family"]=="ALL"],
                      "fixed_global":[d for d in derived["fixed_cohort_costs"] if d["family"]=="ALL"]},ensure_ascii=True,indent=2))


if __name__ == "__main__":
    main()
