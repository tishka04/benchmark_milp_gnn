"""Read-only analysis of the separately saved thermal pilot (no solver imports)."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"low_criticality_scenarios":"Low", "medium_criticality_scenarios":"Medium",
          "high_criticality_scenarios":"High", "persona_vre_battery_scenarios":"Persona VRE",
          "persona_network_operator_scenarios":"Persona réseau", "persona_mathematician_scenarios":"Persona math."}


def number(x, digits=1):
    if x is None or not math.isfinite(x):
        return "n.d."
    if abs(x) < .5 * 10**(-digits):
        x = 0
    return f"{x:,.{digits}f}".replace(",", "\u202f").replace(".", ",")


def first_logged_incumbent(log):
    """First finite BestSol printed by HiGHS, on its internal rounded clock.

    This is a log observation, not a parent-certified wall-clock deadline.
    """
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        p=line.split()
        if p and p[0].isalpha():
            p=p[1:]
        if len(p)!=12 or not p[3].endswith("%") or not p[-1].endswith("s"):
            continue
        try:
            int(p[0]); int(p[1]); int(p[2])
            cost,seconds=float(p[5]),float(p[-1][:-1])
        except ValueError:
            continue
        if math.isfinite(cost):
            return dict(seconds=seconds, objective=cost, source="HiGHS text log; internal time rounded to 0.1 s")
    return None


def main(out):
    manifest = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    rows = []
    for c in manifest["cases"]:
        result = out/"cases"/c["family"]/(c["scenario_id"]+".json")
        new = json.loads(result.read_text(encoding="utf-8"))
        for kind in ("scenario", "reference"):
            assert hashlib.sha256((ROOT/c[kind+"_path"]).read_bytes()).hexdigest() == c[kind+"_sha256"]
        assert new["feasible"], f"Nonvalidated result: {result}"
        row = {**c, "label":LABELS[c["family"]],
               **{"new_"+k:new[k] for k in ("objective", "solve_seconds", "status", "unserved_mwh",
                                             "overgen_mwh", "thermal_mwh", "mip_gap", "dual_bound")}}
        row["delta_cost"] = new["objective"]-c["old_objective"]
        row["delta_cost_normalized_pct"] = 100*row["delta_cost"]/max(abs(c["old_objective"]),1)
        row["delta_ens_mwh"] = new["unserved_mwh"]-c["old_unserved_mwh"]
        row["old_ens_pct"] = 100*c["old_unserved_mwh"]/c["demand_mwh"]
        row["new_ens_pct"] = 100*new["unserved_mwh"]/c["demand_mwh"]
        row["time_ratio"] = new["solve_seconds"]/c["old_seconds"]
        row["first_logged_incumbent"] = first_logged_incumbent(result.with_suffix(".log"))
        timing = re.findall(r"^\s+Timing\s+([\d.]+)\s*$", result.with_suffix(".log").read_text(encoding="utf-8"), re.M)
        row["native_solver_seconds_from_log"] = float(timing[-1]) if timing else None
        row["old_censored"] = c["old_status"] != "optimal"
        row["new_censored"] = new["status"] != "optimal"
        row["max_constraint_violation"] = new["max_constraint_violation"]
        row["max_variable_violation"] = new["max_variable_violation"]
        row["thermal_feasible"] = new["thermal_metrics"]["thermal_feasible"]
        row["dr_feasible"] = new["dr_metrics"]["dr_feasible"]
        rows.append(row)
    assert len(rows) == 12
    summary = {
        "n":len(rows), "n_new_optimal":sum(r["new_status"]=="optimal" for r in rows),
        "n_old_optimal":sum(r["old_status"]=="optimal" for r in rows),
        "old_ens_total_mwh":sum(r["old_unserved_mwh"] for r in rows),
        "new_ens_total_mwh":sum(r["new_unserved_mwh"] for r in rows),
        "old_mip_seconds_sum":sum(r["old_seconds"] for r in rows),
        "new_mip_seconds_sum":sum(r["new_solve_seconds"] for r in rows),
        "max_constraint_violation":max(r["max_constraint_violation"] for r in rows),
        "max_variable_violation":max(r["max_variable_violation"] for r in rows),
        "n_old_ens_positive":sum(r["old_unserved_mwh"]>.001 for r in rows),
        "n_new_ens_positive":sum(r["new_unserved_mwh"]>.001 for r in rows),
    }
    (out/"comparison.json").write_text(json.dumps(dict(summary=summary, cases=rows,
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2, ensure_ascii=False), encoding="utf-8")

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    old_color, new_color = "#58799B", "#008A82"
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
                         "axes.spines.right":False, "axes.spines.left":False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.3), gridspec_kw={"width_ratios":[1.05,1,1.05]})
    fig.subplots_adjust(left=.21, right=.96, top=.84, bottom=.15, hspace=.48, wspace=.22)
    for g, group in enumerate(("blocked", "control")):
        selected = [r for r in rows if r["group"] == group]
        y = np.arange(len(selected))
        for ax in axes[g]:
            ax.set_ylim(5.65,-.65)
            ax.set_yticks(y)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x", alpha=.2)
            ax.set_axisbelow(True)
        axes[g,0].set_yticklabels([r["label"]+"  |  "+r["scenario_id"].replace("scenario_", "") for r in selected])
        axes[g,1].set_yticklabels([])
        axes[g,2].set_yticklabels([])
        delta = [r["delta_cost_normalized_pct"] for r in selected]
        axes[g,0].barh(y,delta,color=[new_color if x<=0 else "#c7633a" for x in delta],height=.5)
        axes[g,0].axvline(0,color="#666",lw=.8)
        lo, hi = min(0,min(delta)), max(0,max(delta))
        span = max(1, hi-lo)
        axes[g,0].set_xlim(lo-.18*span,hi+.30*span)
        for j,x in enumerate(delta):
            inside = group == "blocked" and x < -.1*span
            axes[g,0].text(x+.025*span,j,number(x,2 if group == "blocked" else 3)+" %",
                           va="center",ha="left",fontsize=9,color="white" if inside else "#333")
        for col, key_old, key_new in ((1,"old_ens_pct","new_ens_pct"),(2,"old_seconds","new_solve_seconds")):
            ax=axes[g,col]
            for j,r in enumerate(selected):
                a,b=r[key_old],r[key_new]
                ax.plot([a,b],[j,j],color="#bbc4cc",lw=2,zorder=1)
                ax.scatter(a,j,color=old_color,s=42,marker="^" if col==2 and r["old_censored"] else "o",zorder=3)
                ax.scatter(b,j,color=new_color,s=44,marker="^" if col==2 and r["new_censored"] else "o",zorder=4)
                if col==1:
                    ax.annotate(number(a,2)+" → "+number(b,2),(max(a,b),j),
                                xytext=(7,0),textcoords="offset points",fontsize=8,va="center")
            if col==1:
                ax.set_xlim(-.02, max(.5,max(r[key_old] for r in selected),max(r[key_new] for r in selected))*1.55)
            else:
                ax.set_xscale("log")
                ax.set_xlim(.25,1200)
                ax.set_xticks([1,10,60,600],labels=["1","10","60","600"])
                ax.axvline(600,color="#bbb",lw=.8,ls="--")
        axes[g,0].set_title("Variation du coût",loc="left",fontweight="bold")
        axes[g,1].set_title("Énergie non servie",loc="left",fontweight="bold")
        axes[g,2].set_title("Temps de résolution MILP",loc="left",fontweight="bold")
        axes[g,0].set_xlabel("100 × (nouveau − ancien) / |ancien| (%)")
        axes[g,1].set_xlabel("Part de la demande (%)")
        axes[g,2].set_xlabel("Secondes — échelle logarithmique")
    fig.text(.035,.955,"Correction thermique : comparaison sur 12 scénarios existants",fontsize=20,weight="bold",color="#233c50")
    fig.text(.035,.92,"Deux scénarios par famille, sélection fixée avant les nouveaux calculs. HiGHS : 16 threads, limite de 600 s par résolution.",fontsize=10,color="#506070")
    fig.legend(handles=[Line2D([0],[0],marker="o",color=old_color,label="Ancienne formulation",ls=""),
                        Line2D([0],[0],marker="o",color=new_color,label="Transitions corrigées",ls=""),
                        Line2D([0],[0],marker="^",color="#667",label="Limite de temps atteinte",ls="")],
               loc="upper center",bbox_to_anchor=(.64,.905),ncol=3,frameon=False,fontsize=10)
    fig.text(.035,.855,"Ancien démarrage\nimpossible",weight="bold",fontsize=11,color="#233c50",va="top")
    fig.text(.035,axes[1,0].get_position().y1+.015,"Témoins : ancien\ndémarrage possible",weight="bold",fontsize=11,color="#233c50",va="top")
    fig.text(.035,.035,"Coûts nets : les recettes d'export peuvent les rendre négatifs. Temps de fin de résolution, pas de première solution ; une seule mesure par cas.\nÉchantillon de diagnostic : aucune extrapolation aux 600 scénarios, aucune réévaluation des réseaux appris. Triangle : arrêt au délai, optimalité non prouvée.",fontsize=9,color="#506070")
    fig.savefig(out/"impact_thermique.png",dpi=170)
    fig.savefig(out/"impact_thermique.svg")
    plt.close(fig)

    def table(group):
        body=[]
        for r in rows:
            if r["group"]!=group:
                continue
            status=("Optimum déclaré" if r["new_status"]=="optimal" else "Limite de temps")
            if r["new_mip_gap"] is not None:
                status += " · gap "+number(100*r["new_mip_gap"],3)+" %"
            vals=[r["label"],r["scenario_id"].replace("scenario_", ""),str(r["n_zones"]),
                  number(r["old_objective"]/1e6,3),number(r["new_objective"]/1e6,3),
                  number(r["delta_cost_normalized_pct"],2),number(r["old_unserved_mwh"],1),number(r["new_unserved_mwh"],1),
                  number(r["old_seconds"],1)+( "*" if r["old_censored"] else ""),
                  number(r["new_solve_seconds"],1)+( "*" if r["new_censored"] else ""),status]
            body.append("<tr>"+"".join("<td>"+html.escape(v)+"</td>" for v in vals)+"</tr>")
        headers=["Famille","Scénario","Zones","Coût ancien<br>M€","Coût corrigé<br>M€","Δ coût<br>% normalisé",
                 "ENS ancienne<br>MWh","ENS corrigée<br>MWh","Temps ancien<br>s","Temps corrigé<br>s","Résultat corrigé"]
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'
    report = '''<!doctype html><html lang="fr"><meta charset="utf-8"><title>Pilote de correction thermique</title>
<style>body{font:16px/1.55 system-ui,sans-serif;color:#233c50;background:#f4f7f9;margin:0}main{max-width:1500px;padding:35px;margin:auto}h1{font-size:32px}h2{font-size:23px;margin-top:35px}.card{background:white;padding:24px;border-radius:12px;margin:20px 0}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:11px 9px;text-align:right;border-bottom:1px solid #dde4e9}th:first-child,td:first-child{text-align:left}th{background:#e7eef3}tr:nth-child(even){background:#f7f9fb}.muted{color:#607080;font-size:14px}img{width:100%}a{color:#007f79}.big{font-size:22px;font-weight:650}</style><main>
<h1>Démarrage et arrêt thermiques : pilote sur 12 scénarios</h1><p>Comparaison avant/après de la formulation MILP, sur deux scénarios de chaque famille. Les anciennes références sont lues sans modification.</p>'''
    report += '<div class="card"><p class="big">'+str(summary["n_new_optimal"])+ '/12 résolutions corrigées déclarées optimales ; 12/12 solutions vérifiées.</p>'
    report += '<p>Énergie non servie totale sur cet échantillon : <strong>'+number(summary["old_ens_total_mwh"])+ ' → '+number(summary["new_ens_total_mwh"])+ ' MWh</strong>. Cas avec déficit : '+str(summary["n_old_ens_positive"])+ ' → '+str(summary["n_new_ens_positive"])+'.</p>'
    report += '<p class="muted">Ce total décrit uniquement les 12 cas sélectionnés. Une baisse du coût net peut également provenir de recettes d’export supplémentaires.</p></div>'
    report += '<div class="card"><h2>Ce que montre le pilote</h2><p>Sur les cas medium, high, réseau et mathématicien auparavant bloqués, permettre le démarrage thermique élimine le déficit observé et abaisse fortement le coût net. Pour les cas low et VRE sélectionnés, le thermique reste inutilisé et le coût est pratiquement inchangé. Les six témoins conservent une ENS nulle et des coûts proches (écart inférieur à 0,005 % en valeur absolue).</p><p>Les temps changent fortement selon le cas. Le total des durées MILP passe de '+number(summary['old_mip_seconds_sum']/60)+' à '+number(summary['new_mip_seconds_sum']/60)+' minutes sur cet échantillon. Cela ne prédit pas le coût d’une campagne complète : l’échantillon est ciblé, deux nouvelles résolutions sont censurées à 600 s et les temps ne sont pas répétés.</p><p>Ce test confirme un effet physique et économique majeur du défaut de démarrage sur certains scénarios. Il ne mesure pas les performances relatives corrigées du MILP, du pipeline et de JEPA aux budgets courts.</p></div>'
    report += '<div class="card"><img src="impact_thermique.png" alt="Coûts, énergie non servie et temps avant/après"></div>'
    report += '<h2>Six cas auparavant bloqués</h2><p>Rampe inférieure à la puissance minimale : le démarrage était impossible depuis l’état initial arrêté.</p>'+table("blocked")
    report += '<h2>Six témoins non bloqués</h2><p>Le démarrage était déjà possible. Les écarts à budget limité et les temps peuvent varier avec la formulation et le parcours du solveur.</p>'+table("control")
    report += '<p class="muted">* Limite de temps atteinte : meilleur incumbent disponible, sans preuve d’optimalité. « Optimum déclaré » respecte les tolérances du solveur. M€ = millions d’euros ; ENS = énergie non servie. Δ coût = 100 × (corrigé − ancien) / |ancien|.</p>'
    medium=next(r for r in rows if r['family']=='medium_criticality_scenarios' and r['group']=='blocked')
    if medium['first_logged_incumbent']:
        report += '<p><strong>Temps de fin de résolution ≠ délai de première solution.</strong> Pour le cas medium bloqué, le premier incumbent est affiché à '+number(medium['first_logged_incumbent']['seconds'])+' s dans le journal interne HiGHS, contre '+number(medium['new_solve_seconds'])+' s pour terminer la résolution. Le journal ne fournit pas une certification de réception dans un budget mural strict ni le déficit de cette première solution.</p>'
    report += '<div class="card"><h2>Conditions et limites</h2><ul><li>12 scénarios fixes : un cas bloqué et un témoin par famille ; priorité aux anciennes références optimales, puis aux cas bloqués avec ENS quand disponibles ; médiane inférieure du nombre de zones. Sélection diagnostique, non aléatoire.</li><li>Mêmes fichiers d’entrée, même réglage HiGHS de 16 threads et 600 s. Nouvelle résolution à froid ; aucune solution ancienne utilisée comme point de départ.</li><li>Durée comparable à l’ancien champ solve_seconds : résolution MILP, hors construction du modèle. Pas de seconde résolution LP pour les duales. Une seule mesure par cas, sans conclusion statistique sur les temps.</li><li>Contraintes actives, bornes, intégralité, transitions thermiques et conservation de la demande déplacée vérifiées sur les solutions corrigées.</li><li>Le modèle corrigé suppose démarrage/arrêt en une période ; limites par défaut min(Pmax, max(Pmin, rampe)). Aucun réentraînement, aucune inférence pipeline/JEPA, aucune campagne complète.</li><li>Ancien témoin mathématicien déjà arrêté à 600 s : son coût est une référence d’incumbent, pas un optimum établi.</li></ul>'
    report += '<p>Violation maximale des contraintes : '+f'{summary["max_constraint_violation"]:.2e}'+'. Les empreintes des scénarios et références historiques sont inchangées.</p>'
    report += '<p class="muted">La limite de 600 s porte sur le solveur natif. Le temps de l’appel peut la dépasser légèrement, notamment lors du transfert du modèle et de la restitution de la solution. La preuve d’optimalité utilise une tolérance relative : sa traduction en euros dépend du coût obtenu. Le pilote ne sépare pas ces effets des changements de recherche combinatoire.</p>'
    report += '<p class="muted">Les coûts comparés sont les objectifs complets du modèle, démarrages thermiques inclus. Le détail historique cost_components omettait cette composante dans sa décomposition ; cost_audit.json la reconstitue et vérifie le total, sans modifier les objectifs ni les résultats de résolution.</p>'
    report += '<p><a href="comparison.json">Données détaillées</a> · <a href="manifest.json">Sélection et provenance</a> · <a href="impact_thermique.svg">Graphique vectoriel</a> · <a href="verification.json">Vérifications</a> · <a href="cost_audit.json">Décomposition complète des coûts</a></p></div></main></html>'
    (out/"rapport.html").write_text(report,encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    for r in rows:
        print(r["label"],r["group"],r["scenario_id"],"cost_delta_pct=",round(r["delta_cost_normalized_pct"],3),
              "ens=",round(r["old_unserved_mwh"],2),"->",round(r["new_unserved_mwh"],2),
              "time=",round(r["old_seconds"],1),"->",round(r["new_solve_seconds"],1),r["new_status"],"gap=",r["new_mip_gap"])


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/thermal_transition_pilot_20261027")
    main(parser.parse_args().output)
