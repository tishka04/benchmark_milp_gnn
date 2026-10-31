"""Build the PDF report from audited analysis artifacts; no model/solver imports."""
import argparse
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

FAMILIES = ["low_criticality_scenarios", "medium_criticality_scenarios", "high_criticality_scenarios",
            "persona_vre_battery_scenarios", "persona_network_operator_scenarios", "persona_mathematician_scenarios"]
LABELS = ["Low", "Medium", "High", "Persona VRE + batterie", "Persona réseau", "Persona mathématicien"]
METHODS = ["milp", "pipeline", "jepa"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, default=Path("outputs/load_shifting_v1/evaluation/budgets_v1/analysis_20261027"))
    parser.add_argument("--output", type=Path, default=Path("output/pdf/analyse_matched_budgets_20261027.pdf"))
    args = parser.parse_args()
    data = json.loads((args.analysis / "analysis_data.json").read_text("utf-8"))
    pdfmetrics.registerFont(TTFont("Report", "C:/Windows/Fonts/arial.ttf"))
    pdfmetrics.registerFont(TTFont("Report-Bold", "C:/Windows/Fonts/arialbd.ttf"))
    pdfmetrics.registerFontFamily("Report", normal="Report", bold="Report-Bold")
    width, height = landscape(A3)
    margin = 48
    args.output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(args.output), pagesize=(width, height))
    c.setTitle("Comparaison MILP, pipeline K=16 et JEPA - budgets égaux")
    c.setAuthor("Analyse des résultats benchmark_milp_gnn")
    style = ParagraphStyle("body", fontName="Report", fontSize=15, leading=22, textColor=colors.HexColor("#26394d"))
    small = ParagraphStyle("small", parent=style, fontSize=12, leading=17)
    heading = ParagraphStyle("heading", parent=style, fontName="Report-Bold", fontSize=19, leading=25)

    def para(text, x, top, w, kind=style):
        p = Paragraph(text, kind)
        _, h = p.wrap(w, height)
        assert top-h >= 30, (text[:60], top, h)
        p.drawOn(c, x, top-h)
        return top-h-14

    def title(text, subtitle):
        c.setFillColor(colors.HexColor("#20364b"))
        c.setFont("Report-Bold", 29)
        c.drawString(margin, height-65, text)
        return para(subtitle, margin, height-84, width-2*margin, small)-15

    def end():
        c.setStrokeColor(colors.HexColor("#dce3e9"))
        c.line(margin, 34, width-margin, 34)
        c.setFont("Report", 10)
        c.setFillColor(colors.HexColor("#617285"))
        c.drawString(margin, 19, "Load shifting | 600 scénarios | Analyse des résultats sauvegardés | 27 octobre 2026")
        c.drawRightString(width-margin, 19, str(c.getPageNumber()))
        c.showPage()

    y = title("À budgets égaux, HiGHS reste devant", "Campagne complète : 10 800 expériences vérifiées, sans relancer de résolution. Résultats de cette implémentation et de cette machine.")
    y = para("<b>À 120 s :</b> HiGHS est faisable sur <b>600/600</b> scénarios, le pipeline sur <b>545/600 (90,8 %)</b> et JEPA sur <b>447/600 (74,5 %)</b>. Les solutions sans slack sont respectivement <b>420, 207 et 209</b>.", margin, y, width-2*margin)
    table = [["Famille (100 scénarios)", "Faisables H / P / J", "Sans slack H / P / J", "N commun", "Écart médian P / J"]]
    for family, label in zip(FAMILIES, LABELS):
        s = [next(r for r in data["summary"] if r["family"] == family and r["budget_seconds"] == 120 and r["method"] == m) for m in METHODS]
        table.append([label, " / ".join(str(r["n_feasible"]) for r in s), " / ".join(str(r["n_zero_slack"]) for r in s),
                      s[0]["n_common_all_three"], f"{s[1]['median_gap_reference_common']:.2f} % / {s[2]['median_gap_reference_common']:.2f} %".replace(".", ",")])
    t = Table(table, colWidths=[260, 215, 225, 105, width-2*margin-805], rowHeights=[37]*7)
    t.setStyle(TableStyle([("FONTNAME", (0,0), (-1,-1), "Report"), ("FONTNAME", (0,0), (-1,0), "Report-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 13), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8eff5")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f7f9")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("ALIGN", (1,0), (-1,-1), "CENTER"),
        ("LINEBELOW", (0,0), (-1,0), .7, colors.HexColor("#b7c4cf"))]))
    tw, th = t.wrap(width-2*margin, height)
    t.drawOn(c, margin, y-th); y -= th+15
    y = para("H = HiGHS, P = pipeline K=16, J = JEPA. Les médianes de coûts utilisent uniquement l'intersection faisable des trois méthodes, propre à chaque famille. Sans slack : énergie non servie + surproduction ≤ 0,001 MWh.", margin, y, width-2*margin, small)
    y = para("<b>Le gain du pipeline face à JEPA est surtout un gain de couverture.</b> En High, 81 scénarios sont faisables contre 24 pour JEPA, mais aucun dispatch sélectionné par le pipeline n'est sans slack. Sur leurs 20 cas communs, JEPA coûte moins cher dans 18 cas.", margin, y, width-2*margin)
    y = para("<b>La qualité économique reste en retrait.</b> Sur les 437 scénarios communs à 120 s, les écarts médians à la référence sont d'environ 0 % pour HiGHS, 10,86 % pour le pipeline et 5,29 % pour JEPA. JEPA remporte 247 duels de coût contre 150 pour le pipeline, avec 40 égalités.", margin, y, width-2*margin)
    para("<b>Conclusion limitée au protocole évalué :</b> cette campagne ne montre pas d'avantage du pipeline sur HiGHS en couverture faisable ou en coût à budget égal. Elle révèle des pistes concrètes sur les réparations CPU et la sélection des candidats.", margin, y, width-2*margin)
    end()

    captions = [
        ("01_faisabilite_budget", "À 10 s, HiGHS couvre 584/600 scénarios, JEPA 109/600, le pipeline 0/600. À 60 s, HiGHS atteint 600/600. La faisabilité inclut ici les variables de recours pénalisées : elle ne garantit pas l'absence d'énergie non servie."),
        ("02_sans_slack_budget", "La faisabilité sans slack distingue la qualité du service. Entre 60 et 120 s, le pipeline passe de 222 à 207 solutions sans slack, alors que sa couverture faisable augmente de 339 à 545. La sélection par énergie n'assure pas la conservation d'un candidat sans slack."),
        ("03_couverture_cout_10pct", "À 120 s, le nombre de scénarios faisables à moins de +10 % de la référence est de 577 pour HiGHS, 221 pour le pipeline et 247 pour JEPA. Ce critère combine couverture et coût, sans éliminer les échecs du dénominateur."),
        ("04_couts_cohortes_communes", "Les effectifs changent selon le budget : une moyenne ou médiane sur les seuls succès peut évoluer parce que des scénarios difficiles deviennent faisables. En gardant les mêmes 313 scénarios faisables pour les trois méthodes à 60 ET 120 s, l'écart médian du pipeline passe de 1,56 % à 1,80 % ; JEPA reste à 0,81 %."),
        ("05_couts_par_scenario_120s", "Les échelles sont adaptées à chaque famille et affichent tous les points. Cas extrême : persona mathématicien 82, référence -2 420,09 unités, coût pipeline 141,16 millions, coût JEPA 164,96 millions. Le faible dénominateur et les pénalités de slack gonflent fortement les écarts relatifs."),
        ("06_duel_pipeline_jepa_120s", "La comparaison de coûts ne classe que les paires faisables. Le pipeline est légèrement devant en nombre de victoires dans Medium (48 contre 45), et devant pour la persona mathématicien (42 contre 26). JEPA gagne plus souvent dans les quatre autres familles."),
        ("07_diagnostic_pipeline", "À 10 s, les 600 batches sont prêts, mais aucune réparation n'est reçue à temps. À 120 s, 260 scénarios disposent d'au moins un candidat faisable sans slack, contre 207 retenus. Dans 400/545 cas faisables, un candidat déjà reçu a un coût inférieur à celui sélectionné. Il s'agit de diagnostics a posteriori, pas d'un nouveau benchmark."),
    ]
    for name, caption in captions:
        path = args.analysis / "figures" / f"{name}.png"
        image = ImageReader(str(path)); iw, ih = image.getSize()
        available_height = height-150
        scale = min((width-2*margin)/iw, available_height/ih)
        draw_w, draw_h = iw*scale, ih*scale
        c.drawImage(image, (width-draw_w)/2, height-25-draw_h, draw_w, draw_h)
        para(caption, margin, 103, width-2*margin, small)
        end()

    y = title("Ce que l'on peut conclure, et ce qui reste à tester", "Les graphiques décrivent le protocole réellement exécuté. Les pistes ci-dessous ne sont pas des résultats déjà obtenus.")
    col_w = (width-2*margin-40)/2
    left_y = para("Lecture des résultats", margin, y, col_w, heading)
    for text in [
        "<b>Low et VRE :</b> les trois méthodes atteignent généralement des coûts proches de la référence sur leurs cas communs, mais HiGHS les atteint plus tôt. JEPA est souvent moins coûteux que le pipeline.",
        "<b>Medium :</b> à 120 s, les couvertures sont proches de 100 %. Les coûts restent très hétérogènes ; seulement 29 dispatchs du pipeline et 27 de JEPA sont sans slack, contre 83 pour HiGHS.",
        "<b>High et personas réseau/mathématicien :</b> l'augmentation du budget aide fortement la couverture du pipeline. Elle ne suffit pas à rétablir une qualité de service ou un coût comparable à HiGHS.",
        "<b>Temps enregistrés :</b> environ 12,74 h pour HiGHS, 22,66 h pour le pipeline et 11,88 h pour JEPA, soit 47,28 h d'expériences conservées. Cela exclut les tentatives interrompues, les pauses et les vérifications/écritures entre expériences.",
        "<b>Pistes à tester séparément :</b> mesurer finement le démarrage et les réparations CPU ; évaluer un service avec modèles préchargés sous une convention temporelle explicite ; comparer le sélecteur par énergie à une sélection par coût parmi les candidats faisables déjà reçus.",
    ]:
        left_y = para(text, margin, left_y, col_w)
    x = margin+col_w+40
    right_y = para("Conditions et limites", x, y, col_w, heading)
    for text in [
        "<b>Faisabilité :</b> critères enregistrés par le benchmark, avec vérification des contraintes MILP et des contrôles du pipeline. Cet audit vérifie les sauvegardes et les métriques ; il ne refait pas une validation physique complète des dispatchs.",
        "<b>Comparabilité :</b> mêmes scénarios, formulation load shifting, budgets et plafond CPU. Les méthodes apprises disposent en plus du GPU. Imports, chargements, transferts et réparations sont chronométrés ; entraînements et embeddings HTE précalculés sont exclus.",
        "<b>Référence longue :</b> 495 statuts optimaux et 105 limites de temps. Un écart négatif par rapport à un incumbent n'est pas anormal ; la référence n'est pas toujours un optimum certifié.",
        "<b>Coûts signés :</b> 475 références sont négatives. On utilise 100 × (coût - référence) / |référence|. Un écart de +100 % ne veut donc pas nécessairement dire 'deux fois plus cher'. Le tableau interactif conserve les coûts bruts.",
        "<b>Incertitude :</b> une exécution par scénario/budget/méthode, des scénarios synthétiques et des budgets indépendants. Aucune barre d'erreur sur les répétitions, aucun résultat pour Gurobi/CPLEX ou une autre machine ne peut être déduit de cette campagne.",
    ]:
        right_y = para(text, x, right_y, col_w)
    end()
    c.save()
    print(args.output.resolve())


if __name__ == "__main__":
    main()
