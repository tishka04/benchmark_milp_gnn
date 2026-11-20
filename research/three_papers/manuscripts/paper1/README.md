# Papier 1 — Criticité des scénarios et difficulté opérationnelle

Bibliographie consultée le **5 novembre 2026** : **47 références** dans [references.bib](references.bib). Le corpus est un réservoir de rédaction, pas une liste à citer intégralement. Les commentaires de chaque entrée indiquent sa pertinence, le statut de la version citée et les sources.

Le manuscrit associé utilise désormais \bibliography{paper1/references}, depuis le dossier manuscripts. Son texte et ses résultats n’ont pas été réécrits. Les citations déjà présentes sont conservées.

## Lectures prioritaires

- **leytonbrown2002** — [Learning the Empirical Hardness of Optimization Problems: The Case of Combinatorial Auctions](https://doi.org/10.1007/3-540-46135-3_37) (2002). Antécédent des modèles de difficulté empirique à partir de caractéristiques, avec une formulation résolue par CPLEX.
- **smithmiles2012** — [Measuring instance difficulty for combinatorial optimization problems](https://doi.org/10.1016/j.cor.2011.07.006) (2012). Fondement principal de la distinction entre caractéristiques d’instance, taille et difficulté empirique dépendante de l’algorithme.
- **smithmiles2014** — [Towards objective measures of algorithm performance across instance space](https://doi.org/10.1016/j.cor.2013.11.015) (2014). Évaluer les zones de réussite et d’échec dans l’espace des instances, au-delà des moyennes globales.
- **hutter2013** — [Algorithm runtime prediction: Methods & evaluation](https://doi.org/10.1016/j.artint.2013.10.003) (2014). Prédiction de performance, descripteurs et temps censurés; clé historique conservée malgré l’année de revue 2014.
- **oecd2008** — [Handbook on Constructing Composite Indicators: Methodology and User Guide](https://doi.org/10.1787/9789264043466-en) (2008). Normalisation, pondération, agrégation et sensibilité d’un indice composite; transfert méthodologique depuis les indicateurs de politique publique.
- **lannoye2012** — [Evaluation of Power System Flexibility](https://doi.org/10.1109/TPWRS.2011.2177280) (2012). Indicateurs de disponibilité des ressources de rampe et échelles temporelles; risque physique distinct de la difficulté informatique.
- **ulbig2015** — [Analyzing operational flexibility of electric power systems](https://doi.org/10.1016/j.ijepes.2015.02.028) (2015). Interprétation physique des capacités de flexibilité; ne mesure pas la difficulté MILP.
- **heggarty2019** — [Multi-temporal assessment of power system flexibility requirement](https://doi.org/10.1016/j.apenergy.2019.01.198) (2019). Besoins de flexibilité selon plusieurs échelles temporelles et leurs déterminants.
- **knueven2020** — [On Mixed-Integer Programming Formulations for the Unit Commitment Problem](https://doi.org/10.1287/ijoc.2019.0944) (2020). Comparer familles de formulations UC et effets sur le temps de calcul; fondement pour interpréter la difficulté.
- **beiranvand2017** — [Best practices for comparing optimization algorithms](https://doi.org/10.1007/s11081-017-9366-1) (2017). Comparaison équitable, ressources, jeux d’instances et présentation des résultats.
- **roberts2017** — [Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure](https://doi.org/10.1111/ecog.02881) (2017). Validation par groupes et transferts structurés; justification méthodologique, non étude de systèmes énergétiques.
- **gneiting2007** — [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437) (2007). Règles de score propres pour probabilités de succès; justification de log loss et Brier.
- **kapoor2023** — [Leakage and the reproducibility crisis in machine-learning-based science](https://doi.org/10.1016/j.patter.2023.100804) (2023). Prévenir la fuite entre scénarios apparentés, prétraitement, sélection de modèle et test final.
- **pezo2025** — [Automatic algorithm selection for Pseudo-Boolean optimization with given computational time limits](https://doi.org/10.1016/j.cor.2024.106836) (2025). Antécédent central pour une sélection dépendant explicitement du délai et de la qualité; la sortie complète et la validité physique restent spécifiques au projet.

## Où les mobiliser dans le manuscrit

| Passage | Clés proposées | Point à étayer |
| --- | --- | --- |
| Introduction : taille, difficulté et espace des instances | leytonbrown2002, smithmiles2012, smithmiles2014, hutter2013 | Difficulté empirique conditionnelle à une méthode et des ressources. |
| Section Criticality : construction et limites de l’indice | oecd2008, lannoye2012, ulbig2015, heggarty2019 | Agrégation/normalisation d’une part, stress physique et besoin de flexibilité d’autre part. |
| Modèle énergétique et structure combinatoire | carrion2006, moralesespana2013, knueven2020, huang2021, buchheim2023 | Le compte de variables ne résume ni le polyèdre ni les contraintes intertemporelles. |
| Comparaisons prédictives et calibration | brier1950, gneiting2007, niculescumizil2005, guo2017, saito2015 | Probabilités bien calibrées, métriques adaptées à la prévalence et baseline taille seule. |
| Validation groupée, rééchantillonnage et transfert | cawley2010, roberts2017, efron1979, kapoor2023, koh2021, li2025foundation | Unité scénario, choix sur développement et test indépendant. |
| Discussion : importance et usages du score | fisher2019, kerschke2019, pezo2025 | Valeur prédictive incrémentale; ni mécanisme causal démontré ni score universel. |

## Ensemble du corpus

Les autres références servent à préciser les méthodes, les limites ou les comparateurs. Les travaux AC-OPF, vision, TSP ou classification sont explicitement des références méthodologiques adjacentes. Les performances publiées ne sont pas des résultats de ce projet.

### Unit commitment: formulations and computational structure

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| carrion2006 | [A computationally efficient mixed-integer linear formulation for the thermal unit commitment problem](https://doi.org/10.1109/TPWRS.2006.876672), 2006 | Contraintes intertemporelles et compromis nombre de variables/contraintes et performance de résolution. |
| moralesespana2013 | [Tight and Compact MILP Formulation for the Thermal Unit Commitment Problem](https://doi.org/10.1109/TPWRS.2013.2251373), 2013 | Formulation thermique et effet du resserrement sur la résolution; référence de modèle, pas identité avec la formulation locale. |
| knueven2020 | [On Mixed-Integer Programming Formulations for the Unit Commitment Problem](https://doi.org/10.1287/ijoc.2019.0944), 2020 | Comparer familles de formulations UC et effets sur le temps de calcul; fondement pour interpréter la difficulté. |
| huang2021 | [Cutting planes for security-constrained unit commitment with regulation reserve](https://doi.org/10.1080/24725854.2020.1823533), 2021 | Lien entre réserves, structure polyédrale et difficulté de résolution; les contraintes du papier ne sont pas toutes celles du modèle local. |
| buchheim2023 | [The polytope of binary sequences with bounded variation](https://doi.org/10.1016/j.disopt.2023.100776), 2023 | Lecture polyédrale des transitions binaires; contexte pour des proxies de variabilité, sans assimiler ceux-ci à une mesure exacte de difficulté. |

### Physical stress and power-system flexibility

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| lannoye2012 | [Evaluation of Power System Flexibility](https://doi.org/10.1109/TPWRS.2011.2177280), 2012 | Indicateurs de disponibilité des ressources de rampe et échelles temporelles; risque physique distinct de la difficulté informatique. |
| ulbig2015 | [Analyzing operational flexibility of electric power systems](https://doi.org/10.1016/j.ijepes.2015.02.028), 2015 | Interprétation physique des capacités de flexibilité; ne mesure pas la difficulté MILP. |
| heggarty2019 | [Multi-temporal assessment of power system flexibility requirement](https://doi.org/10.1016/j.apenergy.2019.01.198), 2019 | Besoins de flexibilité selon plusieurs échelles temporelles et leurs déterminants. |

### Instance features and empirical computational difficulty

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| leytonbrown2002 | [Learning the Empirical Hardness of Optimization Problems: The Case of Combinatorial Auctions](https://doi.org/10.1007/3-540-46135-3_37), 2002 | Antécédent des modèles de difficulté empirique à partir de caractéristiques, avec une formulation résolue par CPLEX. |
| smithmiles2012 | [Measuring instance difficulty for combinatorial optimization problems](https://doi.org/10.1016/j.cor.2011.07.006), 2012 | Fondement principal de la distinction entre caractéristiques d’instance, taille et difficulté empirique dépendante de l’algorithme. |
| hutter2013 | [Algorithm runtime prediction: Methods & evaluation](https://doi.org/10.1016/j.artint.2013.10.003), 2014 | Prédiction de performance, descripteurs et temps censurés; clé historique conservée malgré l’année de revue 2014. |
| smithmiles2014 | [Towards objective measures of algorithm performance across instance space](https://doi.org/10.1016/j.cor.2013.11.015), 2014 | Évaluer les zones de réussite et d’échec dans l’espace des instances, au-delà des moyennes globales. |

### Composite indicators and criticality

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| oecd2008 | [Handbook on Constructing Composite Indicators: Methodology and User Guide](https://doi.org/10.1787/9789264043466-en), 2008 | Normalisation, pondération, agrégation et sensibilité d’un indice composite; transfert méthodologique depuis les indicateurs de politique publique. |

### Computational benchmarking and quality versus time

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| dolan2002 | [Benchmarking optimization software with performance profiles](https://doi.org/10.1007/s101070100263), 2002 | Comparer la performance sur un ensemble d’instances; ne remplace pas la mesure de sortie complète du processus. |
| berthold2013 | [Measuring the impact of primal heuristics](https://doi.org/10.1016/j.orl.2013.08.007), 2013 | Situer les courbes qualité-temps et l’intégrale primale; distinguer incumbent disponible et programme terminé. |
| beiranvand2017 | [Best practices for comparing optimization algorithms](https://doi.org/10.1007/s11081-017-9366-1), 2017 | Comparaison équitable, ressources, jeux d’instances et présentation des résultats. |
| gleixner2021 | [MIPLIB 2017: data-driven compilation of the 6th mixed-integer programming library](https://doi.org/10.1007/s12532-020-00194-3), 2021 | Diversité et sélection d’instances MILP; bibliothèque externe, pas description du générateur énergétique local. |

### Learning for unit commitment: direct related work

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| xavier2021 | [Learning to Solve Large-Scale Security-Constrained Unit Commitment Problems](https://doi.org/10.1287/ijoc.2020.0976), 2021 | Antécédent direct pour contraintes, solutions initiales et réduction d’espace apprises en SCUC; année du fascicule 2021, mise en ligne 2020. |
| park2024 | [Confidence-Aware Graph Neural Networks for Learning Reliability Assessment Commitments](https://doi.org/10.1109/TPWRS.2023.3298735), 2024 | Confiance, fixation partielle et réparation de faisabilité sur un problème opérationnel de grande taille. |
| pourahmadi2025 | [Unit Commitment Predictor With a Performance Guarantee: A Support Vector Machine Classifier](https://doi.org/10.1109/TPWRS.2024.3400405), 2025 | Prédiction UC et garanties sous hypothèses; comparateur non GNN pour discuter performance hors échantillon. |

### Generalization, graph size and distribution shift

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| koh2021 | [WILDS: A Benchmark of in-the-Wild Distribution Shifts](https://proceedings.mlr.press/v139/koh21a.html), 2021 | Évaluation séparée des changements de distribution; benchmark externe, sans preuve de transfert pour les cas énergétiques locaux. |
| yehudai2021 | [From Local Structures to Size Generalization in Graph Neural Networks](https://proceedings.mlr.press/v139/yehudai21a.html), 2021 | La capacité à traiter différentes tailles n’implique pas une généralisation à leurs structures locales. |
| li2025foundation | [Towards Foundation Models for Mixed Integer Linear Programming](https://openreview.net/forum?id=6yENDA7J4G), 2025 | Diversité des classes MILP et transfert hors des classes d’entraînement; contexte récent pour les limites des cohortes synthétiques. |

### Power-grid graph learning: adjacent applications

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| zhang2024risk | [Graph neural networks for power grid operational risk assessment under evolving grid topology](https://arxiv.org/abs/2405.07343), 2024 — prépublication | Prédiction du risque physique et topologies changeantes; distinguer prédiction de risque et prédiction du succès informatique. |
| pjetri2025 | [Graph Neural Network Surrogates for Energy System Modeling](https://doi.org/10.1109/ISGTEurope64741.2025.11305443), 2025 | Prédiction rapide de sorties d’un modèle énergétique et identification de cas critiques; surrogate de planification, pas résolution UC complète sous délai. |

### Algorithm selection, portfolios and selection costs

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| rice1976 | [The Algorithm Selection Problem](https://doi.org/10.1016/S0065-2458(08)60520-3), 1976 | Cadre historique reliant instance, caractéristiques, algorithme et critère de performance. |
| xu2008 | [SATzilla: Portfolio-based Algorithm Selection for SAT](https://doi.org/10.1613/jair.2490), 2008 | Sélection par prédiction de performance, caractéristiques et coût de préparation. |
| bischl2016 | [ASlib: A benchmark library for algorithm selection](https://doi.org/10.1016/j.artint.2016.04.003), 2016 | Scénarios de sélection, coûts des caractéristiques et références meilleur algorithme fixe/oracle. |
| kerschke2019 | [Automated Algorithm Selection: Survey and Perspectives](https://doi.org/10.1162/evco_a_00242), 2019 | Revue de cadrage de la sélection automatique et de ses critères d’évaluation. |
| tornede2020 | [Run2Survive: A Decision-theoretic Approach to Algorithm Selection based on Survival Analysis](https://proceedings.mlr.press/v129/tornede20a.html), 2020 | Gestion des temps censurés et décision selon le risque; distinguer temps d’atteinte d’une cible et sortie de processus mesurée ici. |
| alissa2023 | [Automated Algorithm Selection: from Feature-Based to Feature-Free Approaches](https://doi.org/10.1007/s10732-022-09505-4), 2023 | Alternative récente aux caractéristiques conçues manuellement; étude méthodologique sur domaines séquentiels, pas preuve en énergie. |
| pezo2025 | [Automatic algorithm selection for Pseudo-Boolean optimization with given computational time limits](https://doi.org/10.1016/j.cor.2024.106836), 2025 | Antécédent central pour une sélection dépendant explicitement du délai et de la qualité; la sortie complète et la validité physique restent spécifiques au projet. |

### Statistical validation, calibration and uncertainty

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| brier1950 | [Verification of forecasts expressed in terms of probability](https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml), 1950 | Définition historique du score de Brier pour événements binaires. |
| efron1979 | [Bootstrap Methods: Another Look at the Jackknife](https://doi.org/10.1214/aos/1176344552), 1979 | Fondement du rééchantillonnage; dans ce projet l’unité doit rester le groupe scénario, pas chaque cellule budget. |
| niculescumizil2005 | [Predicting Good Probabilities With Supervised Learning](https://doi.org/10.1145/1102351.1102430), 2005 | Calibration de modèles variés, dont arbres et modèles probabilistes; pertinence directe pour les sélecteurs simples. |
| demsar2006 | [Statistical Comparisons of Classifiers over Multiple Data Sets](https://jmlr.org/papers/v7/demsar06a.html), 2006 | Comparaisons multiples et unités indépendantes; ne pas traiter les budgets répétés d’un même scénario comme des jeux indépendants. |
| gneiting2007 | [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437), 2007 | Règles de score propres pour probabilités de succès; justification de log loss et Brier. |
| cawley2010 | [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html), 2010 | Séparer choix des modèles et évaluation finale; prévenir l’optimisme dû aux comparaisons répétées. |
| saito2015 | [The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets](https://doi.org/10.1371/journal.pone.0118432), 2015 | Justifier l’analyse précision-rappel lorsque le succès ou l’échec opérationnel est rare. |
| guo2017 | [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html), 2017 | Distinguer discrimination et calibration des probabilités; la calibration thermique n’est pas supposée déjà implémentée. |
| roberts2017 | [Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure](https://doi.org/10.1111/ecog.02881), 2017 | Validation par groupes et transferts structurés; justification méthodologique, non étude de systèmes énergétiques. |
| fisher2019 | [All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously](https://www.jmlr.org/papers/v20/18-760.html), 2019 | Dépendance de l’importance des variables à la classe de modèles; éviter une lecture causale des scores d’importance. |
| kapoor2023 | [Leakage and the reproducibility crisis in machine-learning-based science](https://doi.org/10.1016/j.patter.2023.100804), 2023 | Prévenir la fuite entre scénarios apparentés, prétraitement, sélection de modèle et test final. |

### Software and external benchmark resources

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| pedregosa2011 | [Scikit-learn: Machine Learning in Python](https://www.jmlr.org/papers/v12/pedregosa11a.html), 2011 | Bibliothèque prévue pour les modèles statistiques et le routage; préciser la version effectivement utilisée lors des futurs calculs. |
| huangfu2018 | [Parallelizing the dual revised simplex method](https://doi.org/10.1007/s12532-017-0130-5), 2018 | Référence recommandée par HiGHS; article sur le simplexe LP, pas validation de performances UC ou de tous les composants MIP. |
| xavier2024ucjl | [UnitCommitment.jl: A Julia/JuMP Optimization Package for Security-Constrained Unit Commitment](https://doi.org/10.5281/zenodo.4269874), 2024 — logiciel/documentation | Benchmark externe et formulations publiques; ce logiciel n’est pas utilisé comme provenance des résultats locaux. |
| gurobi2026 | [Gurobi Optimizer Reference Manual](https://www.gurobi.com), 2026 — logiciel/documentation | Référence fournisseur pour l’adaptateur et les futurs essais; ne présume pas que Gurobi a été exécuté ni qu’il est une action du sélecteur. |

## Traçabilité

Voir le [catalogue commun](../../bibliography/catalogue.json), le [journal de recherche](../../bibliography/RECHERCHE.md) et le [tri des 40 références initiales](../../bibliography/TRI_BIBLIOGRAPHIE_ORIGINE.md). Une référence commune est reproduite à l’identique dans les bibliographies qui en ont besoin.

Aucune compilation BibTeX/LaTeX ni exécution scientifique n’a été effectuée. Les champs optionnels non confirmés ont été omis; les particularités de métadonnées sont documentées dans le catalogue et dans [BIBLIOGRAPHIE.md](../../BIBLIOGRAPHIE.md).
