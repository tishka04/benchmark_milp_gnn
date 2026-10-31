"""Audit corrected MILP references and their historical counterparts.

No solver calls, neural models, training labels or matched-budget inference rows.
All numerical conclusions in the HTML report are generated from saved results.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import analyze_criticality_existing as stats
from scripts.recompute_thermal_references import ROOT, THERMAL, DR, sha, digest, read_json, inventory, report_path
from src.analysis.criticality_index import compute_criticality, NORM_BOUNDS

ENDPOINTS = {k: stats.ENDPOINTS[k] for k in ("unserved_log", "overgen_log", "runtime_log", "ref_timeout")}
SHORT = dict(zip(ENDPOINTS, ("Énergie non servie", "Surproduction", "Temps MILP plafonné", "Limite de temps")))
MODELS = {"size": "Taille seule", "size_crit": "+ Criticité", "size_stress": "+ Stress",
          "size_hard": "+ Hardness", "size_both": "+ Stress et hardness",
          "size_components": "+ Composantes hors taille"}


def feature_row(scenario, report, personas, family, sid):
    c = compute_criticality(scenario)
    config = personas.get(family)
    stored = compute_criticality(scenario, alpha=config["alpha"], stress_weights=config["stress_weights"],
                                hardness_weights=config["hardness_weights"]) if config else c
    d = report["detail"]
    row = dict(family=family, family_label=stats.LABELS[family], scenario_id=sid, scenario_uuid=scenario["id"],
        stored_criticality=scenario.get("criticality_index"), stored_definition_recomputed=stored.criticality_index,
        criticality=c.criticality_index, stress=c.stress_index, hardness=c.hardness_index,
        n_zones=len(report["zones"]), n_lines=len(d["lines"]), n_regions=scenario["meta"]["regions"],
        n_assets=sum(scenario["meta"]["assets"].values()), n_timesteps=len(report["periods"]),
        binary_est=scenario["difficulty_indicators"]["n_binary_variables"],
        variables_est=scenario["estimates"]["vars_total"], constraints_est=scenario["estimates"]["cons_total"],
        thermal_ramp_pct=scenario["tech"]["thermal_ramp_pct"], n_thermal=scenario["meta"]["assets"]["thermal"],
        # Historical group membership, NEVER a claim about corrected feasibility.
        thermal_startup_blocked=scenario["tech"]["thermal_ramp_pct"] < .45 and scenario["meta"]["assets"]["thermal"] > 0)
    row.update(c.stress_normalized)
    row.update(c.hardness_normalized)
    row["hardness_without_size"] = np.mean([v for k, v in c.hardness_normalized.items()
                                           if k not in ("B1_n_zones", "B2_horizon", "B3_binary_est")])
    row.update({k+"_raw": v for k, v in (asdict(c.stress_metrics) | asdict(c.hardness_metrics)).items()})
    return row, list(c.stress_normalized)+list(c.hardness_normalized)


def outcome_row(report):
    mip, d = report["mip"], report["detail"]
    dt = d["dt_hours"]
    demand, ens, spill = (stats.total(d[k])*dt for k in ("demand", "unserved", "overgen_spill"))
    seconds, objective = float(mip["solve_seconds"]), float(mip["objective"])
    if demand <= 0 or min(ens, spill) < -.001 or seconds < 0 or not all(map(math.isfinite, (demand, ens, spill, seconds, objective))):
        raise ValueError("Reference incomplete ou non finie")
    ens, spill = (0. if x <= .001 else x for x in (ens, spill))
    if mip["termination"] not in {"optimal", "maxTimeLimit", "feasible", "maxIterations"}:
        raise ValueError("Reference sans solution exploitable")
    return dict(reference_objective=objective, reference_status=mip["termination"],
        ref_optimal=mip["termination"] == "optimal", ref_timeout=mip["termination"] == "maxTimeLimit",
        ref_cap=600, ref_seconds=seconds, runtime_log=np.log1p(min(seconds, 600)),
        demand_mwh=demand, unserved_mwh=ens, overgen_mwh=spill, unserved_pct=100*ens/demand,
        overgen_pct=100*spill/demand, unserved_log=np.log1p(100*ens/demand), overgen_log=np.log1p(100*spill/demand),
        ref_zero_slack=ens+spill <= .001, thermal_mwh=stats.total(d["thermal"])*dt,
        thermal_max_mw=max(max(v) for v in d["thermal"].values()))


def extract_pair(campaign):
    manifest = read_json(campaign/"manifest.json")
    source = Path(manifest["source_root"])
    if inventory(source) != {k: manifest[k] for k in ("source_root", "original_manifest_sha256", "cases")}:
        raise ValueError("Entrees historiques modifiees")
    before, after, hashes = [], [], {}
    personas = stats.persona_definitions()
    for case in manifest["cases"]:
        sp, old_path = source/case["scenario_path"], source/case["old_report_path"]
        new_path = report_path(campaign, case)
        old, new, scenario = read_json(old_path), read_json(new_path), read_json(sp)
        cp = new["batch_checkpoint"]
        content = {k: v for k, v in new.items() if k != "batch_checkpoint"}
        import hashlib
        checksum = hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
        if (cp["complete"] is not True or checksum != cp["report_sha256"]
                or new["reference_campaign_sha256"] != digest(manifest)
                or new["thermal_formulation"] != THERMAL or new["dr_formulation"] != DR
                or not new["thermal_validation"]["thermal_feasible"]
                or cp["provenance"]["scenario_sha256"] != case["scenario_sha256"]
                or cp["provenance"]["time_limit_seconds"] != 600
                or new["solver_resources"]["highs_threads"] != 16):
            raise ValueError(f"Reference corrigee incompatible : {new_path}")
        if old["zones"] != new["zones"] or old["periods"] != new["periods"] or old["detail"]["lines"] != new["detail"]["lines"]:
            raise ValueError("Structure du scenario modifiee")
        common, terms = feature_row(scenario, old, personas, case["family"], case["scenario_id"])
        b, a = {**common, **outcome_row(old)}, {**common, **outcome_row(new)}
        if not math.isclose(a["demand_mwh"], b["demand_mwh"], rel_tol=1e-10):
            raise ValueError("La demande a change")
        before.append(b)
        after.append(a)
        for path in (sp, old_path, new_path):
            hashes[str(path)] = sha(path)
        if len(before) % 100 == 0:
            print(f"Extraction avant/apres : {len(before)}/600", flush=True)
    b, a = pd.DataFrame(before), pd.DataFrame(after)
    if len(a) != 600 or a.scenario_uuid.nunique() != 600 or a.duplicated(["family", "scenario_id"]).any():
        raise ValueError("Cohorte incomplete ou dupliquee")
    return b, a, terms, hashes


def family_summary(frame):
    return frame.groupby("family", sort=False).agg(n=("scenario_id", "size"),
        optimal=("ref_optimal", "sum"), timeouts=("ref_timeout", "sum"),
        ens_positive=("unserved_mwh", lambda v: (v > .001).sum()), ens_total_mwh=("unserved_mwh", "sum"),
        spill_positive=("overgen_mwh", lambda v: (v > .001).sum()),
        runtime_median=("ref_seconds", "median"), mip_seconds_total=("ref_seconds", "sum"),
        cost_net_total=("reference_objective", "sum"), thermal_mwh=("thermal_mwh", "sum")).reset_index()


def compare_rows(before, after):
    keys = ["family", "scenario_id"]
    outcomes = ["reference_objective", "unserved_mwh", "overgen_mwh", "ref_seconds", "ref_optimal", "ref_timeout", "thermal_mwh"]
    joined = before[keys+outcomes].merge(after[keys+outcomes], on=keys, suffixes=("_before", "_after"), validate="one_to_one")
    joined["delta_cost"] = joined.reference_objective_after-joined.reference_objective_before
    joined["delta_cost_normalized_pct"] = 100*joined.delta_cost/np.maximum(abs(joined.reference_objective_before), 1.)
    joined["delta_ens_mwh"] = joined.unserved_mwh_after-joined.unserved_mwh_before
    joined["delta_seconds"] = joined.ref_seconds_after-joined.ref_seconds_before
    return joined


def csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def render(out, before, after, results, supports, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = out/"figures"
    figures.mkdir(exist_ok=True)
    labels = [stats.LABELS[f] for f in stats.FAMILIES]
    colors = {"before": "#71869e", "after": "#008b83"}
    legends = {"before": "Ancienne formulation", "after": "Thermique corrigé"}
    summaries = {k: family_summary(d).set_index("family").loc[stats.FAMILIES] for k, d in (("before", before), ("after", after))}
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.4))
    for k, offset in (("before", -.18), ("after", .18)):
        s = summaries[k]
        for ax, col, title in zip(axes, ("ens_positive", "runtime_median", "timeouts"),
                                  ("Scénarios avec énergie non servie", "Temps MILP médian (s)", "Arrêts à la limite de temps")):
            ax.barh(np.arange(6)+offset, s[col], height=.34, color=colors[k], label=legends[k])
            ax.set_yticks(range(6), labels)
            ax.invert_yaxis() if not ax.yaxis_inverted() else None
            ax.set_title(title, fontsize=11)
            ax.grid(axis="x", alpha=.18)
            ax.set_axisbelow(True)
    axes[0].set_xlim(0, 100); axes[2].set_xlim(0, 100)
    fig.suptitle("Références MILP : avant et après correction thermique", fontsize=17)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .07, 1, .93))
    for ext in ("png", "svg"):
        fig.savefig(figures/f"01_avant_apres.{ext}", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.3))
    models = [m for m in MODELS if m != "size"]
    for ax, endpoint in zip(axes.flat, ENDPOINTS):
        plotted = False
        for version_name, offset in (("before", -.12), ("after", .12)):
            table = results[version_name]["validation"]
            selected = table[(table.scheme == "random_family_balanced") & (table.endpoint == endpoint)].set_index("model")
            if selected.empty:
                ax.text(.02, .05 if version_name == "after" else .12, legends[version_name]+" : réponse non analysable",
                        transform=ax.transAxes, color=colors[version_name], fontsize=9)
            for i, model in enumerate(models):
                if model not in selected.index:
                    continue
                r = selected.loc[model]
                ax.plot([r.delta_low, r.delta_high], [i+offset]*2, color=colors[version_name], lw=2)
                ax.scatter(r.delta_r2, i+offset, color=colors[version_name], s=28)
                plotted = True
        ax.set_title(SHORT[endpoint])
        ax.set_yticks(range(len(models)), [MODELS[m] for m in models], fontsize=9)
        ax.set_ylim(len(models)+.5, -.6)
        ax.axvline(0, color="#aaa", lw=.8)
        ax.set_xlabel("Gain de R² par rapport à la taille seule")
        ax.grid(axis="x", alpha=.2)
        if not plotted:
            ax.set_xlim(-.05, .05)
    fig.suptitle("Apport de l’indice et des composantes au-delà de la taille", fontsize=16)
    fig.text(.04, .025, "Validation hors échantillon ; intervalles bootstrap appariés conditionnels aux prédictions.\nUne réponse constante ou trop rare est signalée, jamais remplacée par un gain nul.", fontsize=10)
    fig.tight_layout(rect=(0, .09, 1, .94))
    for ext in ("png", "svg"):
        fig.savefig(figures/f"02_apport_au_dela_taille.{ext}", dpi=160)
    plt.close(fig)

    def table_html(frame):
        return frame.to_html(index=False, border=0, float_format=lambda x: f"{x:,.4f}", escape=True)
    sections = []
    for label in ("before", "after"):
        v = results[label]["validation"]
        selected = v[v.scheme == "random_family_balanced"]
        score = selected.pivot(index="endpoint", columns="model", values="r2_cv").reindex(index=ENDPOINTS, columns=MODELS)
        score.index = [SHORT[x] for x in score.index]
        score.columns = [MODELS[x] for x in score.columns]
        sections.append(f"<h2>{legends[label]} : R² hors échantillon</h2>"+table_html(score.reset_index().rename(columns={"index": "Réponse"})))
    component_tables = []
    for label in ("before", "after"):
        a = results[label]["associations"]
        a = a[(a.scope == "all600") & (a.adjustment == "size_family") & a.feature.str.startswith(("A", "B"))].copy()
        a["abs_rho"] = a.rho.abs()
        a = a.sort_values("abs_rho", ascending=False).groupby("endpoint", sort=False).head(5)
        component_tables.append(f"<h3>{legends[label]}</h3>"+table_html(a[["endpoint", "feature", "n", "rho", "q_bh"]]))
    parts = []
    for k in ("before", "after"):
        s = summaries[k].reset_index().copy()
        s.insert(0, "version", legends[k])
        s["family"] = s.family.map(stats.LABELS)
        parts.append(s)
    total_before, total_after = before.unserved_mwh.sum(), after.unserved_mwh.sum()
    text = f'''<!doctype html><html lang="fr"><meta charset="utf-8"><title>Criticité — références thermiques corrigées</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;color:#243b4d;background:#f4f7f9}}main{{max-width:1500px;margin:auto;padding:28px}}section{{background:white;padding:24px;margin:20px 0;border-radius:12px}}img{{max-width:100%}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}th{{background:#e6edf3}}.scroll{{overflow-x:auto}}a{{color:#007e76}}</style><main>
<h1>Criticité, stress et hardness après correction thermique</h1>
<p>600 scénarios identiques, 100 par famille. Comparaison des références historiques et corrigées : HiGHS, 16 threads, 600 s par MILP. Aucun résultat neuronal historique n’est mélangé à la nouvelle formulation.</p>
<section><h2>Résultats observés</h2><p>ENS cumulée : <b>{total_before:,.1f} → {total_after:,.1f} MWh</b>.
Scénarios avec ENS : <b>{int((before.unserved_mwh > .001).sum())} → {int((after.unserved_mwh > .001).sum())}</b>.
Références déclarées optimales : <b>{int(before.ref_optimal.sum())} → {int(after.ref_optimal.sum())}</b>.
Arrêts au délai : <b>{int(before.ref_timeout.sum())} → {int(after.ref_timeout.sum())}</b>.</p>
<p>Ces valeurs décrivent le dispatch économique obtenu. Une ENS positive ne certifie pas une pénurie physiquement inévitable ; une ENS nulle ne démontre pas que le système résiste à d’autres perturbations.</p>
<img src="figures/01_avant_apres.png" alt="Comparaison des références par famille"><div class="scroll">{table_html(pd.concat(parts))}</div></section>
<section><h2>Information supplémentaire au-delà de la taille</h2>
<p>Comparer « taille seule » avec les mêmes modèles enrichis par la criticité, stress, hardness, les deux axes, puis les composantes hors taille.
Le modèle de composantes est une régression exploratoire distincte de l’indice initial. Un gain positif avec un intervalle recouvrant zéro reste incertain.
Les tableaux avant/après sont recalculés avec le même protocole ; la différence entre deux gains n’a pas ici d’intervalle de confiance propre.</p>
<img src="figures/02_apport_au_dela_taille.png" alt="Gains prédictifs au-delà de la taille">
<div class="scroll">{''.join(sections)}</div><p>NaN signifie non estimé : voir le tableau de support et les fichiers skipped_analyses.json, et non absence démontrée de signal.</p></section>
<section><h2>Support des réponses</h2><div class="scroll">{table_html(supports)}</div>
<p>Au minimum 10 observations ; réponse non constante ; pour ENS et surproduction, au moins 5 événements positifs ; pour l’arrêt au délai, au moins 5 événements et 5 non-événements.
Le critère est contrôlé à nouveau dans chaque sous-échantillon. Il constitue un filtre de diagnostic, pas une garantie de puissance statistique.</p></section>
<section><h2>Composantes : associations ajustées les plus fortes</h2><p>Au plus cinq composantes par réponse, triées par |ρ| après taille et famille. Tous les résultats, y compris non significatifs, sont conservés dans associations.csv. Les q-values BH portent sur l’ensemble des caractéristiques et réponses de chaque périmètre/ajustement.</p>
<div class="scroll">{''.join(component_tables)}</div></section>
<section><h2>Méthode et limites</h2><ul>
<li>Une observation par scénario. Indice commun inchangé : alpha = 0,5, poids uniformes. Scores personas d’origine vérifiés séparément avec leurs pondérations.</li>
<li>Taille : zones, lignes, régions, actifs, estimations des binaires/variables/contraintes, horizon ; log(1+x), splines et encodages exacts B1/B2/B3. Les estimations du générateur ne sont pas les tailles après presolve.</li>
<li>Standardisation et transformations apprises dans chaque pli. Ridge fixé à 1 ; cinq plis équilibrés par famille, répétitions déterministes. Sensibilités : effets de famille, famille entièrement exclue, références optimales, taille réduite, scores non linéaires, hardness sans termes de taille.</li>
<li>Le sous-ensemble nommé startup_allowed correspond aux scénarios dont le démarrage était possible dans l’ancienne formulation. Le champ thermal_startup_blocked conserve cette appartenance historique ; il ne décrit pas un blocage du modèle corrigé.</li>
<li>ENS et surproduction : log(1 + pourcentage de demande). Temps : log(1 + min(temps,600)) ; les cas censurés restent dans l’analyse, avec une réponse séparée pour les arrêts au délai. Ce n’est pas une estimation du temps exact d’optimalité.</li>
<li>La relaxation LP est calculée séparément pour conserver le format des références historiques ; son temps est exclu de la réponse « temps MILP ». Les temps muraux de campagne sont enregistrés séparément. Une seule exécution par scénario.</li>
<li>1 500 bootstraps appariés par scénario et stratifiés par famille, conditionnels aux prédictions hors échantillon. Associations de rang ajustées ; tests approximatifs et correction de multiplicité. Catalogue synthétique sélectionné : aucune interprétation causale ou validation universelle.</li>
<li>Coûts nets : recettes d’export incluses. Écarts avant/après normalisés par la valeur absolue de l’ancien coût, disponibles dans paired_scenarios.csv. Les références interrompues sont des incumbents, pas des optima prouvés.</li></ul>
<p><a href="audit.json">Provenance et protocole</a> · <a href="paired_scenarios.csv">600 comparaisons avant/après</a> · <a href="after/results.json">Audit corrigé complet</a> · <a href="before/results.json">Audit historique recalculé</a> · <a href="component_audit.csv">Composantes constantes / saturées</a></p></section></main></html>'''
    (out/"rapport.html").write_text(text, encoding="utf-8")


def run(campaign, repeats=10):
    campaign = campaign.resolve()
    out = campaign/"analysis"
    before, after, terms, hashes = extract_pair(campaign)
    out.mkdir(parents=True, exist_ok=True)
    results, support = {}, []
    for name, frame in (("before", before), ("after", after)):
        dest = out/name
        dest.mkdir(exist_ok=True)
        csv(frame, dest/"scenario_data.csv")
        skipped = []
        for endpoint in ENDPOINTS:
            y = frame[endpoint]
            support.append(dict(version=name, endpoint=endpoint, n=len(y), unique=y.nunique(),
                positive=int((y > 0).sum()), reason=stats.endpoint_support(frame, endpoint, 5) or "analysable"))
        assoc = stats.associations(frame, terms, dest, endpoints=ENDPOINTS, minimum_events=5, skipped=skipped)
        val = stats.validation(frame, terms, dest, repeats=repeats, endpoints=ENDPOINTS, minimum_events=5, skipped=skipped)
        summary = family_summary(frame)
        csv(summary, dest/"family_summary.csv")
        stats.save_json(dest/"skipped_analyses.json", skipped)
        results[name] = dict(associations=assoc, validation=val)
        stats.save_json(dest/"results.json", dict(version=name, n=len(frame), repeats=repeats,
            family_summary=summary.to_dict("records"), validation=val.to_dict("records"),
            associations=assoc.to_dict("records"), skipped=skipped))
    support = pd.DataFrame(support)
    comparison = compare_rows(before, after)
    csv(support, out/"outcome_support.csv")
    csv(comparison, out/"paired_scenarios.csv")
    component_audit = [dict(component=t, unique=after[t].nunique(), std=after[t].std(),
        n_zero=int((after[t] == 0).sum()), n_one=int((after[t] == 1).sum()), bounds=NORM_BOUNDS[t]) for t in terms]
    csv(pd.DataFrame(component_audit), out/"component_audit.csv")
    stats.save_json(out/"audit.json", dict(protocol="criticality_before_after_thermal_v1", n=600, repeats=repeats,
        endpoints=ENDPOINTS, minimum_events=5, model_additions=stats.MODEL_ADDS, size_features=stats.SIZES,
        sources=hashes, scenario_only_index_unchanged=True, historical_neural_results_used=False,
        original_definitions_matching=int((abs(after.stored_definition_recomputed-after.stored_criticality) < .00011).sum()),
        note="Indices identiques avant/apres; seule la reference physique/economique change. Sensibilites et predictions hors echantillon sauvegardees."))
    render(out, before, after, results, support, comparison)
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError(f"Source modifiee pendant l'analyse : {path}")
    print("Audit termine :", out/"rapport.html", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=10)
    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be positive")
    run(args.campaign, args.repeats)


if __name__ == "__main__":
    main()
