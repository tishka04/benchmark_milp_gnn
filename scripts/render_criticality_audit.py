"""Render the saved statistical results as a PDF; no analysis/solver imports."""
import argparse
import csv
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--analysis",type=Path,default=Path("outputs/load_shifting_v1/evaluation/budgets_v1/criticality_audit_20261027"))
    ap.add_argument("--output",type=Path,default=Path("output/pdf/validation_criticite_20261027.pdf"))
    args=ap.parse_args();data=json.loads((args.analysis/"results.json").read_text("utf-8"))
    vals=data["validation"]
    def val(endpoint,model,scheme="random_family_balanced",field="r2_cv"):
        return next(r[field] for r in vals if r["scheme"]==scheme and r["model"]==model and r["endpoint"]==endpoint)
    def n(x,d=3):return f"{x:.{d}f}".replace(".",",")
    pdfmetrics.registerFont(TTFont("Audit","C:/Windows/Fonts/arial.ttf"))
    pdfmetrics.registerFont(TTFont("Audit-Bold","C:/Windows/Fonts/arialbd.ttf"))
    pdfmetrics.registerFontFamily("Audit",normal="Audit",bold="Audit-Bold")
    W,H=landscape(A3);M=48
    args.output.parent.mkdir(parents=True,exist_ok=True)
    c=canvas.Canvas(str(args.output),pagesize=(W,H))
    c.setTitle("Valeur de l'indice de criticité au-delà de la taille - audit des résultats existants")
    c.setAuthor("Analyse statistique benchmark_milp_gnn")
    body=ParagraphStyle("body",fontName="Audit",fontSize=15,leading=22,textColor=colors.HexColor("#26394d"))
    small=ParagraphStyle("small",parent=body,fontSize=12,leading=17)
    head=ParagraphStyle("head",parent=body,fontName="Audit-Bold",fontSize=19,leading=25)
    def para(text,x,y,width,style=body):
        p=Paragraph(text,style);_,h=p.wrap(width,H);assert y-h>=44,(c.getPageNumber(),y,h,text[:80])
        p.drawOn(c,x,y-h);return y-h-12
    def title(text,sub):
        c.setFillColor(colors.HexColor("#20374c"));c.setFont("Audit-Bold",29);c.drawString(M,H-57,text)
        return para(sub,M,H-78,W-2*M,small)-10
    def end():
        c.setFillColor(colors.HexColor("#65788a"));c.setFont("Audit",10)
        c.drawString(M,23,"600 scénarios | Résultats existants uniquement | 27 octobre 2026 | Aucun MILP ni entraînement HTE / JEPA / EBM relancé")
        c.drawRightString(W-M,23,str(c.getPageNumber()));c.showPage()
    def table(rows,y,widths):
        content=[[Paragraph(str(x),small) for x in r] for r in rows]
        t=Table(content,colWidths=widths,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e6eef4")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f4f7f9")]),
            ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LINEBELOW",(0,0),(-1,0),.7,colors.HexColor("#bbc8d3"))]))
        _,h=t.wrap(W-2*M,H);assert y-h>45;t.drawOn(c,M,y-h);return y-h-14

    y=title("Un signal dans les composantes, peu dans le score global",
        "Audit exploratoire des 600 scénarios et 10 800 expériences sauvegardées. Références MILP : 600 s, 16 threads ; comparaison : budgets de 1 à 120 s.")
    y=para("<b>Résultat principal :</b> l'indice du manuscrit, avec sa moyenne des composantes, ajoute très peu de pouvoir prédictif une fois la taille contrôlée. Les composantes prises séparément contiennent davantage d'information, mais une partie importante du signal est liée à une particularité de démarrage thermique de la formulation.",M,y,W-2*M)
    rows=[["<b>Résultat à expliquer</b>","<b>Taille seule</b>","<b>+ Criticité</b>","<b>+ Stress et hardness</b>","<b>+ 18 composantes</b>"]]
    for e,label in [("unserved_log","Énergie non servie / demande"),("runtime_log","Temps MILP plafonné à 600 s"),("ref_timeout","Référence arrêtée au délai"),("jepa_fail120","Échec JEPA à 120 s")]:
        rows.append([label]+[n(val(e,m)) for m in ["size","size_crit","size_both","size_components"]])
    y=table(rows,y,[350,150,150,200,W-2*M-850])
    y=para("R² hors échantillon, 5 plis équilibrés par famille × 10 répétitions. Énergie et temps transformés par log(1+x). Pour les résultats binaires : score sur l'erreur quadratique, équivalent à un score de Brier normalisé. Plus élevé = meilleure prédiction. Les 18 composantes sont ajustées conjointement par une régression exploratoire ; leur résultat ne valide pas le score initial.",M,y,W-2*M,small)
    y=para("<b>Ce qui est étayé :</b> des composantes techno-opérationnelles apportent une information supplémentaire dans ce catalogue synthétique. Les gains persistent avec des effets de famille et sur les seules références optimales.",M,y,W-2*M)
    y=para("<b>Ce qui ne l'est pas :</b> un indice global universel de criticité physique, une probabilité de défaillance, ou une capacité de transfert générale. Le score actuel n'est pas validé par cet audit comme mesure autonome de criticité au-delà de la taille.",M,y,W-2*M)
    para("<b>Point décisif :</b> 255 scénarios rendent le démarrage thermique impossible dans le modèle courant. Ils regroupent 146 des 148 références avec énergie non servie. Leur analyse doit rester distincte d'une validation générale de l'insuffisance de flexibilité.",M,y,W-2*M)
    end()

    captions=[
      ("01_indices_et_taille","Les scénarios low/medium/high ont été sélectionnés avec le score uniforme ; les personas utilisent des poids différents. L'audit recalcule une définition commune (alpha = 0,5 et poids uniformes) pour comparer les six familles. Les 600 scores enregistrés sont reproduits lorsque leurs pondérations d'origine sont utilisées : aucune corruption de ces scores n'est détectée."),
      ("02_correlations_ajustees","La corrélation brute criticité/temps MILP vaut 0,627 ; elle devient -0,005 après contrôle de la taille et -0,029 après ajout de la famille. Pour l'énergie non servie, les valeurs correspondantes sont 0,465 ; 0,082 ; 0,031. Ces associations ne démontrent pas de causalité et ne constituent pas des mesures de risque."),
      ("03_apport_hors_echantillon",f"Gains du modèle de composantes : +{n(val('unserved_log','size_components',field='delta_r2'))} de R² pour l'énergie non servie, +{n(val('runtime_log','size_components',field='delta_r2'))} pour le temps, +{n(val('ref_timeout','size_components',field='delta_r2'))} pour les limites de temps et +{n(val('jepa_fail120','size_components',field='delta_r2'))} pour les échecs JEPA. Les intervalles sont des bootstraps appariés de scénarios, conditionnels aux prédictions déjà calculées. Ils ne représentent pas des répétitions de la campagne ni l'incertitude d'un réseau réel."),
      ("04_composantes","A9 (flexibilité thermique) et B7 (rampes) sont liés à davantage d'énergie non servie, mais à moins de temps de calcul et de dépassements de délai, après contrôle de la taille et de la famille. Cette opposition contredit une lecture monotone simple de la difficulté et doit être reliée au blocage du démarrage thermique. A3 et A5 sont aussi liés aux échecs JEPA."),
      ("05_transfert",f"Pour le temps MILP, les deux axes séparés apportent un signal de transfert : R² = {n(val('runtime_log','size_both','leave_one_family_out'))}, contre {n(val('runtime_log','size','leave_one_family_out'))} pour la taille seule et {n(val('runtime_log','size_components','leave_one_family_out'))} pour les composantes. En revanche, le modèle de composantes donne des R² négatifs pour l'énergie non servie ({n(val('unserved_log','size_components','leave_one_family_out'))}), les limites de temps ({n(val('ref_timeout','size_components','leave_one_family_out'))}) et les échecs JEPA ({n(val('jepa_fail120','size_components','leave_one_family_out'))}). Le transfert dépend donc fortement de la réponse étudiée."),
      ("06_effet_formulation","Si r < 0,45, p(0)=0 et p(t) doit être soit nul soit au moins 0,45 Pmax, la contrainte p(t)-p(t-1) ≤ r Pmax interdit le premier démarrage, puis tous les suivants. Les 255 dispatchs concernés ont effectivement une production thermique nulle et un statut optimal. Ce constat vient du code et des résultats existants ; aucun contrefactuel corrigé n'a été résolu."),
    ]
    for name,caption in captions:
        im=ImageReader(str(args.analysis/"figures"/(name+".png")));iw,ih=im.getSize()
        avail=H-165;scale=min((W-50)/iw,avail/ih);ww,hh=iw*scale,ih*scale
        c.drawImage(im,(W-ww)/2,H-18-hh,ww,hh)
        para(caption,M,116,W-2*M,small)
        end()

    y=title("Vérifications de robustesse et limites de l'indice",
            "Les analyses ci-dessous utilisent les mêmes résultats sauvegardés. Elles ne constituent ni une nouvelle résolution ni un réglage des modèles neuronaux.")
    rows=[["<b>Vérification</b>","<b>Constat</b>","<b>Interprétation</b>"]]
    rows += [
      ["Références optimales uniquement (n = 495)",f"R² énergie : {n(val('unserved_log','size','optimal_references'))} → {n(val('unserved_log','size_components','optimal_references'))}<br/>R² temps : {n(val('runtime_log','size','optimal_references'))} → {n(val('runtime_log','size_components','optimal_references'))}","Les gains des composantes ne proviennent pas uniquement des références interrompues. Ce sous-ensemble inclut toutefois les 255 cas à démarrage thermique bloqué."],
      ["Contrôle de la taille + effets fixes de famille",f"R² énergie : {n(val('unserved_log','size','random_with_family'))} → {n(val('unserved_log','size_components','random_with_family'))}<br/>R² temps : {n(val('runtime_log','size','random_with_family'))} → {n(val('runtime_log','size_components','random_with_family'))}","Le signal des composantes persiste au sein des familles. Les agrégats stress/hardness restent d'un apport faible."],
      ["Retrait de B1/B2/B3 de hardness", "Résultats pratiquement identiques pour les agrégats ajustés", "Les composantes de taille de hardness n'expliquent pas à elles seules l'absence de gain du score agrégé."],
      ["Relations non linéaires avec les scores",f"Stress + hardness : gain R² énergie {n(val('unserved_log','size_both','nonlinear_scores',field='delta_r2'))} ; temps {n(val('runtime_log','size_both','nonlinear_scores',field='delta_r2'))}","Les intervalles des gains recouvrent zéro. La faible contribution des agrégats en validation mélangée n'est pas seulement due à une relation supposée linéaire."],
      ["Scénarios sans blocage thermique (n = 345)",f"R² temps : {n(val('runtime_log','size','startup_allowed'))} → {n(val('runtime_log','size_components','startup_allowed'))}<br/>Énergie non servie : seulement 2 cas", "Le gain sur le temps devient beaucoup plus faible. La faible variation de l'énergie non servie ne permet pas une validation physique dans ce sous-ensemble."],
      ["Composantes constantes / saturées", "B2 et B5 constants ; A6 = 1 dans 383 cas ; B8 = 1 dans 203 cas", "Deux dimensions ne sont pas testables ici. Les plafonnements réduisent le pouvoir de discrimination. Le nombre nominal de composantes ne garantit pas leur information indépendante."],
      ["Surproduction de référence", "Un seul scénario au-dessus de 0,001 MWh", "Aucune inférence sur cette réponse rare. Elle est décrite, sans régression ni test de corrélation."],
    ]
    y=table(rows,y,[285,270,W-2*M-555])
    para("<b>Portée scientifique :</b> le résultat soutient l'étude des mécanismes et d'une décomposition des difficultés. Il ne suffit pas à valider l'indice actuel comme mesure générale de criticité. Les effets de formulation, les approximations physiques et le transfert doivent être séparés avant une telle revendication.",M,y,W-2*M)
    end()

    y=title("Méthode, traçabilité et lecture des fichiers",
            "Une analyse statistique descriptive et prédictive de données existantes, sans modification du benchmark ou de ses résultats.")
    cw=(W-2*M-42)/2;xx=M+cw+42
    l=para("Construction et contrôles",M,y,cw,head)
    for text in [
      "<b>Unité :</b> 600 scénarios uniques, 100 par famille. Les méthodes et budgets ne sont pas traités comme 10 800 observations indépendantes. Les 1 200 fichiers de scénarios/références sont vérifiés contre leurs empreintes dans le manifeste.",
      "<b>Indice :</b> code actuel inchangé, alpha = 0,5, poids uniformes pour tous. Le score stocké de chaque persona est reproduit séparément avec ses poids et alpha propres (0,55 ; 0,55 ; 0,20).",
      "<b>Taille :</b> zones, lignes réellement décrites, régions, nombre d'actifs, estimations du générateur pour binaires/variables/contraintes. L'horizon est constant à 24 pas. Les estimations ne sont pas des décomptes de la matrice après presolve.",
      "<b>Régressions :</b> contrôles log(1+x), splines quadratiques à trois noeuds quantiles calculés dans chaque pli, encodages exacts B1/B2/B3 ; standardisation dans le pli et ridge fixé à 1, sans recherche d'hyperparamètre. Intercept non pénalisé. Sensibilité avec scores non linéaires.",
      "<b>Validation :</b> cinq plis équilibrés par famille, dix répétitions ; exclusion d'une famille entière ; contrôles avec famille, références optimales, taille réduite et absence de blocage thermique. Toutes les transformations prédictives sont calculées hors des scénarios évalués.",
    ]:l=para(text,M,l,cw,small)
    r=para("Réponses, incertitude et reproductibilité",xx,y,cw,head)
    for text in [
      "<b>Physique :</b> log(1 + 100 × énergie non servie / demande), calculé sur le dispatch de référence. Les énergies ≤ 0,001 MWh sont traitées comme nulles. Même un optimum économique avec délestage ne prouve pas une pénurie physiquement inévitable.",
      "<b>Calcul :</b> log(1 + min(temps, 600)), durée restreinte observée ; limite de temps séparée (105 cas). Aucun temps réel d'optimalité imputé. Prédictions bornées selon le domaine connu : énergie >= 0, temps entre 0 et log(601), indicateurs entre 0 et 1.",
      "<b>Coûts et méthodes :</b> échec à 120 s et absence de solution à moins de +10 % de la référence, sur tous les scénarios. Écarts conditionnels aux solutions faisables : asinh(écart % / 100), n = 545 pour le pipeline et 447 pour JEPA ; biais de sélection possible.",
      "<b>Associations :</b> rangs résidualisés sur les contrôles de taille non linéaires, leurs rangs et, en sensibilité, la famille. Tests t approximatifs ; correction BH sur toutes les composantes et réponses de chaque périmètre/ajustement. Analyses exploratoires, non causales.",
      "<b>Fichiers :</b> scenario_data.csv ; associations.csv ; validation.csv ; held_out_predictions.csv ; leave_family_out.csv ; audit.json ; results.json ; graphiques PNG/SVG. Le script d'analyse ne charge ni Pyomo/HiGHS ni Torch et ne modifie aucun poids.",
      "<b>Limites :</b> catalogue synthétique sélectionné, une seule configuration HiGHS, des familles non échangeables et une formulation particulière. Pas de validation sur réseau réel, autre horizon ou autre solveur ; pas de garantie d'une métrique universelle.",
    ]:r=para(text,xx,r,cw,small)
    end();c.save();print(args.output.resolve())


if __name__=="__main__":main()
