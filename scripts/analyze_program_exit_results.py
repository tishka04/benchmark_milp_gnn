"""Plot saved execution-end observations only; never imports a solver or Torch.

RH-MO+LP root JSONs are shown as a distinct, non-comparable historical series.
Their per-scenario internal timer is not a cold-process exit measurement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("low_criticality_scenarios", "medium_criticality_scenarios", "high_criticality_scenarios",
            "persona_vre_battery_scenarios", "persona_network_operator_scenarios", "persona_mathematician_scenarios")
LABELS = dict(zip(FAMILIES, ("Low", "Medium", "High", "Persona VRE + batterie", "Persona réseau", "Persona mathématicien")))
METHODS = ("milp", "pipeline", "jepa")
NAMES = dict(milp="HiGHS", pipeline="Pipeline K=16", jepa="JEPA", rh="RH-MO+LP historique*")
COLORS = dict(milp="#236b9e", pipeline="#d86929", jepa="#168579", rh="#80758c")
MARKERS = dict(milp="o", pipeline="s", jepa="^", rh="D")
BUDGETS = (1, 2, 5, 10, 60, 120)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()).hexdigest()


def finite(x):
    return isinstance(x, (float, int)) and math.isfinite(x)


def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def write_json(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def json_header(path, key):
    with path.open(encoding="utf-8") as f:
        head = f.read(8192)
    start = head.index('"'+key+'"')
    return json.JSONDecoder().raw_decode(head[head.index(":", start)+1:].lstrip())[0]


def valid_solution(row, method):
    if not row or not row.get("success") or not finite(row.get("lp_objective")):
        return False
    if method == "milp":
        return finite(row.get("max_constraint_violation")) and row["max_constraint_violation"] <= .001
    return bool((row.get("dr_metrics") or {}).get("dr_feasible"))


def classify(outcome, method, budget):
    feasible = valid_solution(outcome.get("row"), method)
    normal = not outcome["hard_timeout"] and all(x["completed"] for x in outcome["members"])
    wall = outcome["experiment_wall_s"]
    if not finite(wall) or wall < 0:
        raise ValueError("Invalid end-of-execution time")
    return dict(snapshot_feasible=feasible, natural_end=normal,
        natural_solution=feasible and normal,
        normal_exit_within_budget=feasible and normal and wall <= budget+1e-9,
        returned_by_wall_deadline=feasible and wall <= budget+1e-9,
        hard_timeout=outcome["hard_timeout"], experiment_wall_s=wall,
        decision_wall_s=outcome["decision_wall_s"], cleanup_wall_s=outcome["cleanup_wall_s"],
        snapshot_available_after_s=outcome.get("snapshot_available_after_s"))


def extract(source):
    manifest = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    flat = json.loads((source/"per_scenario.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == "matched_budgets_gpu16_feasible_energy_v1"
    assert manifest["budgets"] == list(BUDGETS) and manifest["k"] == 16
    assert len(flat) == 10800 and Counter(x["method"] for x in flat) == dict.fromkeys(METHODS, 3600)
    assert Counter(x["family"] for x in manifest["cohort"]) == dict.fromkeys(FAMILIES, 100)
    lookup = {(x["family"], x["scenario_id"], x["budget_seconds"], x["method"]): x for x in flat}
    assert len(lookup) == 10800 and all(x["completed"] for x in flat)
    original = Path(manifest["root"])
    expected = {k.replace("\\", "/"): v for k, v in manifest["inputs_sha256"].items()}
    provenance = {str(source/name): sha(source/name) for name in ("manifest.json", "per_scenario.json")}
    references, rows, checkpoints, rh_records, rh_audit = {}, [], [], [], []
    for i, case in enumerate(manifest["cohort"], 1):
        family, sid = case["family"], case["scenario_id"]
        sp = original/family/(sid+".json")
        rp = original/family/"reports"/(sid+".json")
        for path in (sp, rp):
            actual = sha(path)
            assert actual == expected[path.relative_to(original).as_posix()]
            provenance[str(path)] = actual
        mip = json_header(rp, "mip")
        references[family, sid] = mip
        for bi, b in enumerate(BUDGETS):
            for method in METHODS:
                path = source/"cases"/f"{i:04}"/f"b{bi:02}"/(method+".json")
                raw = path.read_bytes()
                payload = json.loads(raw)
                claim = payload.pop("sha256")
                assert claim == digest(payload) and payload["manifest_sha256"] == digest(manifest)
                expected_id = dict(case, budget_seconds=float(b), method=method, k=16, protocol=manifest["protocol"])
                assert payload["identity"] == expected_id
                outcome = payload["outcome"]
                assert outcome["protocol"] == manifest["protocol"] and outcome["budget_seconds"] == b
                assert finite(outcome["decision_wall_s"]) and 0 <= outcome["decision_wall_s"] <= b
                state = classify(outcome, method, b)
                saved = lookup[family, sid, b, method]
                assert state["snapshot_feasible"] == saved["feasible"]
                assert saved["reference_objective"] == mip["objective"]
                assert saved["reference_status"] == mip["termination"]
                assert saved["decision_wall_s"] == outcome["decision_wall_s"]
                if state["snapshot_feasible"]:
                    row = outcome["row"]
                    assert row["scenario_id"] == sid and row["dr_formulation"] == "load_shifting_v1"
                    assert row["kind"] == method
                    assert row["family"] == family.removesuffix("_criticality_scenarios")
                    assert 0 <= outcome["snapshot_available_after_s"] < b
                    assert saved["objective"] == row["lp_objective"]
                    assert math.isclose(saved["gap_reference_pct"],
                        100*(row["lp_objective"]-mip["objective"])/max(abs(mip["objective"]), 1.), abs_tol=1e-8)
                rows.append(dict(family=family, scenario_id=sid, method=method, budget_seconds=b,
                    **state, objective=saved["objective"], reference_objective=mip["objective"],
                    reference_status=mip["termination"], gap_reference_pct=saved["gap_reference_pct"],
                    slack_mwh=saved["slack_mwh"]))
                checkpoints.append((str(path), hashlib.sha256(raw).hexdigest()))
        if i % 100 == 0:
            print(f"Checkpoints et références vérifiés : {i}/600", flush=True)

    for short, family in zip(("low", "medium", "high"), FAMILIES[:3]):
        path = ROOT/"outputs"/f"rhmo_{short}.json"
        provenance[str(path)] = sha(path)
        entries = json.loads(path.read_text(encoding="utf-8"))
        assert len(entries) == 100 and len({x["scenario_id"] for x in entries}) == 100
        ref_differences, identical_inputs = 0, 0
        for x in entries:
            sid = x["scenario_id"]
            ref = references[family, sid]
            legacy_input = ROOT/"outputs"/family/(sid+".json")
            legacy_report = ROOT/"outputs"/family/"reports"/(sid+".json")
            for p in (legacy_input, legacy_report): provenance[str(p)] = sha(p)
            identical_inputs += sha(legacy_input) == sha(original/family/(sid+".json"))
            old_ref = json_header(legacy_report, "mip")
            ref_differences += not math.isclose(ref["objective"], old_ref["objective"], rel_tol=1e-8, abs_tol=.001)
            good = bool(x["success"]) and finite(x["lp_objective"]) and x["lp_status"] not in ("infeasible", "error", "pending")
            gap = 100*(x["lp_objective"]-ref["objective"])/max(abs(ref["objective"]), 1.) if good else None
            old_gap = 100*(x["lp_objective"]-old_ref["objective"])/max(abs(old_ref["objective"]), 1.) if good else None
            rh_records.append(dict(family=family, scenario_id=sid, method="rh", declared_success=good,
                internal_time_s=x["time_total"], objective=x["lp_objective"] if good else None,
                reference_objective=ref["objective"], legacy_reference_objective=old_ref["objective"],
                numerical_gap_to_load_shifting_reference_pct=gap, numerical_gap_to_legacy_reference_pct=old_gap,
                dr_formulation=x.get("dr_formulation"), lp_status=x["lp_status"], slack_mwh=x["lp_slack"],
                comparable_protocol=False))
        rh_audit.append(dict(family=family, n=100, inputs_identical=identical_inputs,
            reference_costs_different=ref_differences,
            n_with_dr_formulation=sum("dr_formulation" in x for x in entries),
            n_with_dr_metrics=sum("dr_metrics" in x for x in entries),
            timer="internal per-scenario processing time, not cold-process end-to-end exit",
            protocol_comparable=False))
    return rows, rh_records, dict(input_sha256=provenance, checkpoint_sha256=checkpoints,
        reference_statuses=dict(Counter(x["termination"] for x in references.values())), rh_audit=rh_audit,
        n_scenarios=600, n_experiments=10800, thermal_note="Historical results before thermal_transitions_v1; no new physical validation.",
        exit_note="experiment_wall_s: parent measurement after workers stop/cleanup, before writing experiment checkpoint and campaign summaries; not full CLI runtime.")


def summarize(rows, rh):
    summary = []
    for scope, families in [("ALL600", FAMILIES), ("STRATIFIED300", FAMILIES[:3]), *[(f, (f,)) for f in FAMILIES]]:
        selection = [r for r in rows if r["family"] in families]
        for b in BUDGETS:
            by_method = {m: [r for r in selection if r["budget_seconds"] == b and r["method"] == m] for m in METHODS}
            support = {m: {(r["family"], r["scenario_id"]) for r in rr if r["normal_exit_within_budget"]} for m, rr in by_method.items()}
            common = set.intersection(*support.values())
            for m, rr in by_method.items():
                accepted = [r for r in rr if r["normal_exit_within_budget"]]
                gaps = [r["gap_reference_pct"] for r in accepted]
                paired = [r["gap_reference_pct"] for r in accepted if (r["family"], r["scenario_id"]) in common]
                summary.append(dict(scope=scope, method=m, budget=b, n=len(rr),
                    n_snapshot=sum(r["snapshot_feasible"] for r in rr),
                    n_natural_solution=sum(r["natural_solution"] for r in rr), n_exit=len(accepted),
                    n_wall_return=sum(r["returned_by_wall_deadline"] for r in rr),
                    median_gap=np.median(gaps) if gaps else None,
                    q25_gap=np.quantile(gaps, .25) if gaps else None, q75_gap=np.quantile(gaps, .75) if gaps else None,
                    n_common=len(common), median_common_gap=np.median(paired) if paired else None,
                    n_within_10pct=sum(x <= 10 for x in gaps), n_within_1pct=sum(x <= 1 for x in gaps),
                    n_zero_slack=sum(finite(r["slack_mwh"]) and r["slack_mwh"] <= .001 for r in accepted)))
            if scope != "ALL600" and all(f in FAMILIES[:3] for f in families):
                rr = [r for r in rh if r["family"] in families]
                accepted = [r for r in rr if r["declared_success"] and r["internal_time_s"] <= b]
                gaps = [r["numerical_gap_to_load_shifting_reference_pct"] for r in accepted]
                summary.append(dict(scope=scope, method="rh", budget=b, n=len(rr), n_exit=len(accepted),
                    median_gap=np.median(gaps) if gaps else None,
                    q25_gap=np.quantile(gaps, .25) if gaps else None, q75_gap=np.quantile(gaps, .75) if gaps else None,
                    n_zero_slack=sum(finite(r["slack_mwh"]) and r["slack_mwh"] <= .001 for r in accepted),
                    comparable_protocol=False))
    return clean(summary)


def axes_style(ax, count=False):
    ax.set_xscale("log"); ax.set_xticks(BUDGETS, labels=[str(x) for x in BUDGETS])
    ax.minorticks_off(); ax.set_xlim(.85, 145); ax.grid(axis="y", alpha=.25)
    ax.set_xlabel("Seuil / budget de temps (s)")
    if not count:
        ax.set_yscale("symlog", linthresh=1)
        ax.axhline(0, color="#777", lw=.8)


def legend(fig, methods, y=.91):
    fig.legend(handles=[Line2D([], [], color=COLORS[m], marker=MARKERS[m],
        ls="--" if m == "rh" else "-", label=NAMES[m]) for m in methods],
        loc="upper center", bbox_to_anchor=(.5, y), ncol=len(methods), frameon=False)


def save(fig, out, name):
    # Resolve text/axis positions before raster and vector export.
    fig.canvas.draw()
    for ext in ("png", "svg"):
        fig.savefig(out/(name+"."+ext), dpi=180)
    plt.close(fig)


def figures(summary, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "text.color": "#25394a", "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#3d5060", "svg.fonttype": "none"})
    lookup = {(r["scope"], r["method"], r["budget"]): r for r in summary}
    scopes = ("STRATIFIED300", *FAMILIES[:3])
    for costs in (False, True):
        fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.4))
        fig.subplots_adjust(left=.08, right=.97, top=.77, bottom=.20, hspace=.55, wspace=.26)
        title = "Coûts des solutions rendues" if costs else "Nombre de scénarios terminés avec une solution"
        fig.text(.055, .965, title, fontsize=20, weight="bold", va="top")
        fig.text(.055, .915, "Lecture descriptive à quatre méthodes — la série RH-MO+LP est historique et hors protocole commun.", fontsize=11, color="#865432", va="top")
        legend(fig, (*METHODS, "rh"), y=.875)
        for ax, scope in zip(axes.flat, scopes):
            n = 300 if scope == "STRATIFIED300" else 100
            ax.set_title(("Low + Medium + High" if scope == "STRATIFIED300" else LABELS[scope])+f"  |  n = {n}", loc="left", weight="bold")
            for method in (*METHODS, "rh"):
                rr = [lookup[scope, method, b] for b in BUDGETS]
                yy = [r["median_gap"] if costs else r["n_exit"] for r in rr]
                ax.plot(BUDGETS, [np.nan if x is None else x for x in yy], color=COLORS[method], marker=MARKERS[method],
                        ls="--" if method == "rh" else "-", lw=2, ms=4)
                if costs:
                    ax.fill_between(BUDGETS, [np.nan if r["q25_gap"] is None else r["q25_gap"] for r in rr],
                        [np.nan if r["q75_gap"] is None else r["q75_gap"] for r in rr], color=COLORS[method], alpha=.08)
            axes_style(ax, count=not costs)
            ns = " / ".join(str(lookup[scope, m, 120]["n_exit"]) for m in (*METHODS, "rh"))
            ax.text(.02, .98, "Effectifs à 120 s (ordre légende) : "+ns, transform=ax.transAxes, va="top", fontsize=8)
            if costs:
                ax.set_ylabel("Écart médian à la référence (%)")
                band = [lookup[scope, m, b][k] for m in (*METHODS, "rh") for b in BUDGETS for k in ("q25_gap", "q75_gap")]
                band = [0. if abs(v) < 1e-6 else v for v in band if v is not None]
                low, high = min([0., *band]), max([1., *band])
                ax.set_ylim(min(-.08, low*1.2), high*1.8)
            else:
                ax.set_ylim(-.03*n, 1.08*n); ax.set_ylabel("Scénarios avec solution")
        foot = ("Médiane et intervalle interquartile sur les seuls résultats retenus : effectifs différents selon méthode/budget.\n"
                "Écart = 100 × (coût − référence 600 s) / |référence| ; échelle symlog, linéaire autour de zéro.\n"
                "Pour RH : simple écart numérique, compatibilité physique non établie.\n"
                if costs else "HiGHS / pipeline / JEPA : fin normale des workers et temps total mesuré ≤ budget ; une exécution indépendante par budget.\n")
        foot += "* RH : succès déclaré et temps interne ≤ seuil ; un seul résultat par scénario, sans mesure du démarrage de processus ni contrôle DR sauvegardé.\nRésultats antérieurs à la correction thermique. Cette juxtaposition ne permet pas un classement équitable des quatre méthodes."
        fig.text(.055, .035, foot, fontsize=9, color="#526476", va="bottom")
        save(fig, out, "02_couts_quatre_methodes_historique" if costs else "01_sorties_quatre_methodes_historique")

    fig, axes = plt.subplots(3, 2, figsize=(13.3, 11.5))
    fig.subplots_adjust(left=.08, right=.97, top=.85, bottom=.10, hspace=.50, wspace=.20)
    fig.text(.055, .97, "Fin normale avec solution dans le budget", fontsize=20, weight="bold", va="top")
    fig.text(.055, .935, "Comparaison homogène des trois méthodes de budgets_v1 — 600 scénarios, 100 par famille.", fontsize=11)
    legend(fig, METHODS, y=.915)
    for ax, scope in zip(axes.flat, FAMILIES):
        for m in METHODS:
            ax.plot(BUDGETS, [lookup[scope, m, b]["n_exit"] for b in BUDGETS], color=COLORS[m], marker=MARKERS[m], lw=2, ms=4)
        ax.set_title(LABELS[scope], loc="left", weight="bold")
        axes_style(ax, count=True); ax.set_ylim(-3, 105); ax.set_ylabel("Scénarios / 100")
    fig.text(.055, .025, "Fin de l'exécution = arrêt/nettoyage des workers, hors écriture du checkpoint et des résumés. Les arrêts forcés sont exclus.\nUne solution peut utiliser de l'énergie non servie. Références et exécutions antérieures à la correction thermique.", fontsize=9)
    save(fig, out, "03_sorties_homogenes_six_familles")

    fig, axes = plt.subplots(1, 2, figsize=(13.3, 7.4))
    fig.subplots_adjust(left=.08, right=.97, top=.70, bottom=.23, wspace=.25)
    fig.text(.055, .965, "Qualité des solutions à la fin de l'exécution", fontsize=20, weight="bold", va="top")
    fig.text(.055, .88, "600 scénarios — comparaison homogène HiGHS / pipeline / JEPA.", fontsize=11, va="top")
    legend(fig, METHODS, y=.825)
    for m in METHODS:
        rr = [lookup["ALL600", m, b] for b in BUDGETS]
        axes[0].plot(BUDGETS, [np.nan if r["median_common_gap"] is None else r["median_common_gap"] for r in rr],
                     color=COLORS[m], marker=MARKERS[m], lw=2)
        axes[1].plot(BUDGETS, [r["n_within_10pct"] for r in rr], color=COLORS[m], marker=MARKERS[m], lw=2)
    axes[0].set_title("Coût : mêmes scénarios pour les trois méthodes", fontsize=11)
    axes[1].set_title("Couverture : solution à ≤ +10 % de la référence", fontsize=11)
    axes_style(axes[0]); axes[0].set_ylabel("Écart médian à la référence (%)")
    paired_gaps = [r["median_common_gap"] for r in summary if r["scope"] == "ALL600" and r["median_common_gap"] is not None]
    if paired_gaps and max(abs(v) for v in paired_gaps) < 10:
        axes[0].set_yscale("linear")
    axes_style(axes[1], count=True); axes[1].set_ylabel("Scénarios / 600"); axes[1].set_ylim(-15, 625)
    ns = "; ".join(f"{b} s : {lookup['ALL600','milp',b]['n_common']}" for b in BUDGETS)
    fig.text(.055, .12, "Support commun à gauche : "+ns, fontsize=9)
    fig.text(.055, .055, "Gauche : intersection des sorties normales avec solution à chaque budget ; cette intersection change entre budgets.\nDroite : dénominateur fixe de 600, échecs exclus du compteur. Coûts nets, références MILP de 600 s (pas toutes optimales).", fontsize=9)
    save(fig, out, "04_couts_homogenes_support_commun")


def make_report(out, summary, info):
    def table(rows, columns):
        head = "<tr>"+"".join("<th>"+html.escape(label)+"</th>" for _, label in columns)+"</tr>"
        body = []
        for row in rows:
            cells = []
            for key, _ in columns:
                x = row.get(key)
                text = "—" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)
                cells.append("<td>"+html.escape(text)+"</td>")
            body.append("<tr>"+"".join(cells)+"</tr>")
        return '<div class="scroll"><table>'+head+"".join(body)+"</table></div>"
    all_rows = [dict(r, method=NAMES[r["method"]]) for r in summary if r["scope"] == "ALL600"]
    protocol_table = table(all_rows, [("budget", "Budget (s)"), ("method", "Méthode"), ("n_snapshot", "Solution disponible avant budget"),
        ("n_natural_solution", "Fin normale + solution"), ("n_exit", "Fin normale + solution dans budget"),
        ("n_wall_return", "Fin mesurée dans budget, arrêt forcé inclus"), ("n_zero_slack", "Sortie sans slack"), ("median_gap", "Écart médian (%)"),
        ("n_common", "Support commun"), ("median_common_gap", "Écart médian apparié (%)")])
    different = sum(x["reference_costs_different"] for x in info["rh_audit"])
    last = {r["method"]: r for r in summary if r["scope"] == "ALL600" and r["budget"] == 120}
    findings = (f"À 120 s, sur 600 scénarios : {last['milp']['n_exit']} sorties normales avec solution pour HiGHS, "
        f"{last['pipeline']['n_exit']} pour le pipeline et {last['jepa']['n_exit']} pour JEPA. "
        f"Sur les {last['milp']['n_common']} scénarios communs, les écarts médians sont respectivement "
        f"{last['milp']['median_common_gap']:.2f} %, {last['pipeline']['median_common_gap']:.2f} % et {last['jepa']['median_common_gap']:.2f} %.")
    report = f'''<!doctype html><html lang="fr"><meta charset="utf-8"><title>Temps de sortie et coûts</title>
<style>body{{margin:0;background:#f4f7f9;color:#243a4d;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1450px;margin:auto;padding:32px}}section{{background:white;padding:24px;border-radius:12px;margin:20px 0}}.alert{{background:#fff2df;border-left:5px solid #ba7b36}}img{{width:100%}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:9px;border-bottom:1px solid #dbe2e8;text-align:right}}th{{background:#e6eef3}}.scroll{{overflow-x:auto}}a{{color:#007e79}}h1{{font-size:32px}}</style><main>
<h1>Sortie du programme avec une solution : ce que les fichiers permettent de mesurer</h1>
<p>Analyse des résultats sauvegardés, 30 octobre 2026. Aucune résolution, inférence ni modification des résultats sources.</p>
<section class="alert"><h2>La comparaison à quatre méthodes reste descriptive</h2>
<p>Les fichiers rhmo_low/medium/high.json couvrent les mêmes 300 identifiants et disposent d’un temps interne par scénario. Ils ne contiennent ni signature de formulation DR ni métriques DR. Le temps exclut le démarrage de processus ; les trois autres méthodes sont mesurées en processus frais, imports compris.</p>
<p>Les références historiques à la racine de outputs et celles de load_shifting_v1 ont des coûts différents dans <strong>{different}/300</strong> cas. Les courbes RH en pointillé sont donc des repères historiques, pas des résultats validés du même problème. Aucun résultat RH n’est inventé pour les 300 personas.</p></section>
<section><h2>Définition de la mesure</h2><p>Pour HiGHS, pipeline K=16 et JEPA, un cas est retenu si une solution est admissible selon les contrôles sauvegardés, tous les workers ont terminé normalement, aucun arrêt forcé n’est signalé, et experiment_wall_s ≤ budget. Cette durée inclut le nettoyage des workers, mais précède l’écriture du checkpoint et des résumés : le temps exact de fermeture de toute la commande n’est pas enregistré par scénario.</p>
<p>La faisabilité n’impose pas une ENS nulle. La colonne sans slack du tableau ajoute ce critère (slack ≤ 0,001 MWh). Les coûts de référence sont les MILP historiques de 600 s associés à la campagne, non les nouvelles références thermiques corrigées.</p>
<p>Chaque budget de 1, 2, 5, 10, 60 et 120 s correspond à une exécution indépendante. Les courbes RH reconstruisent des seuils sur une seule exécution par scénario. Aucune durée manquante n’est extrapolée.</p></section>
<section><h2>Quatre méthodes : repère historique visible</h2><img src="01_sorties_quatre_methodes_historique.png" alt="Nombre de scénarios avec solution à la fin du traitement"><img src="02_couts_quatre_methodes_historique.png" alt="Écarts numériques aux références MILP">
<p>Les écarts RH vers la référence load shifting sont des différences numériques, sans garantie que la solution respecte le même modèle. Ne pas les interpréter comme des gaps d’optimalité comparables. Les médianes conditionnelles portent sur des scénarios différents selon la méthode ; la figure suivante utilise un support commun pour les trois méthodes homogènes.</p></section>
<section><h2>Comparaison homogène : les 600 scénarios</h2><img src="03_sorties_homogenes_six_familles.png" alt="Sorties normales par famille"><img src="04_couts_homogenes_support_commun.png" alt="Coûts sur support commun et couverture qualité">{protocol_table}</section>
<section><h2>Interprétation</h2><p><strong>{findings}</strong></p><p>Passer de « solution disponible avant le délai » à « fin normale avec solution avant le délai » ajoute une condition. Pour chaque méthode, le second compteur est donc inférieur ou égal au premier. Le classement peut changer si certaines méthodes continuent longtemps après leur première solution, mais il faut appliquer la même règle à toutes. Attendre les 16 réparations du pipeline fait ici partie de sa terminaison normale.</p>
<p>Un arrêt forcé peut aussi restituer une solution déjà récupérée. Ce cas apparaît dans la dernière définition temporelle du tableau, et reste distinct d’une fin normale. La provenance des solutions est conservée.</p>
<p>Ce changement de mesure ne corrige ni la différence de protocole RH ni le défaut thermique des résultats historiques. Il ne peut pas valider à lui seul les conclusions initiales du papier. Les références non optimales restent des incumbents ; un écart négatif est possible. La comparaison avec une exécution MILP de 600 s complète serait une autre expérience, non un budget égal de 120 s.</p>
<p><a href="summary.json">Compteurs et coûts</a> · <a href="per_experiment.json">10 800 observations temporelles</a> · <a href="rh_historical.json">300 résultats RH et provenance limitée</a> · <a href="audit.json">Contrôles et empreintes</a></p></section></main></html>'''
    (out/"rapport.html").write_text(report, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT/"outputs/load_shifting_v1/evaluation/budgets_v1")
    p.add_argument("--output", type=Path, default=ROOT/"outputs/load_shifting_v1/evaluation/budgets_v1/exit_time_analysis_20261030")
    args = p.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if out == source or source in out.parents and out.relative_to(source).parts[0] in ("cases", "w", "resumes"):
        raise ValueError("Use a separate analysis directory")
    rows, rh, info = extract(source)
    summary = summarize(rows, rh)
    out.mkdir(parents=True, exist_ok=True)
    info["analysis_script_sha256"] = sha(Path(__file__))
    for name, data in (("per_experiment.json", rows), ("rh_historical.json", rh), ("summary.json", summary)):
        write_json(out/name, data)
    figures(summary, out)
    make_report(out, summary, info)
    # Confirm the analysis did not alter any input/result file.
    for path, expected in [*info["input_sha256"].items(), *info["checkpoint_sha256"]]:
        if sha(path) != expected:
            raise ValueError(f"Input changed during analysis: {path}")
    info["all_sources_unchanged"] = True
    write_json(out/"audit.json", info)
    for scope in ("ALL600", "STRATIFIED300"):
        print(scope)
        for r in summary:
            if r["scope"] == scope:
                print(r["budget"], r["method"], "snapshots", r.get("n_snapshot"), "exits", r["n_exit"],
                      "median_gap", r.get("median_gap"), "common", r.get("n_common"), "common_gap", r.get("median_common_gap"))
    print("RH audit:", json.dumps(info["rh_audit"], ensure_ascii=False))
    print("Report:", out/"rapport.html")


if __name__ == "__main__":
    main()
