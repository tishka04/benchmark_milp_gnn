# Papier 3 — Complémentarité et routage avant résolution

Bibliographie consultée le **5 novembre 2026** : **42 références** dans [references.bib](references.bib). Le corpus est un réservoir de rédaction, pas une liste à citer intégralement. Les commentaires de chaque entrée indiquent sa pertinence, le statut de la version citée et les sources.

Le manuscrit associé utilise désormais \bibliography{paper3/references}, depuis le dossier manuscripts. Son texte et ses résultats n’ont pas été réécrits. Les citations déjà présentes sont conservées.

## Lectures prioritaires

- **rice1976** — [The Algorithm Selection Problem](https://doi.org/10.1016/S0065-2458(08)60520-3) (1976). Cadre historique reliant instance, caractéristiques, algorithme et critère de performance.
- **gomes2001** — [Algorithm portfolios](https://doi.org/10.1016/S0004-3702(00)00081-3) (2001). Complémentarité et variabilité des algorithmes; exécution parallèle ou entrelacée à distinguer du routage vers une seule méthode.
- **xu2008** — [SATzilla: Portfolio-based Algorithm Selection for SAT](https://doi.org/10.1613/jair.2490) (2008). Sélection par prédiction de performance, caractéristiques et coût de préparation.
- **bischl2016** — [ASlib: A benchmark library for algorithm selection](https://doi.org/10.1016/j.artint.2016.04.003) (2016). Scénarios de sélection, coûts des caractéristiques et références meilleur algorithme fixe/oracle.
- **hutter2013** — [Algorithm runtime prediction: Methods & evaluation](https://doi.org/10.1016/j.artint.2013.10.003) (2014). Prédiction de performance, descripteurs et temps censurés; clé historique conservée malgré l’année de revue 2014.
- **kerschke2019** — [Automated Algorithm Selection: Survey and Perspectives](https://doi.org/10.1162/evco_a_00242) (2019). Revue de cadrage de la sélection automatique et de ses critères d’évaluation.
- **kotthoff2014** — [Algorithm Selection for Combinatorial Search Problems: A Survey](https://doi.org/10.1609/aimag.v35i3.2460) (2014). Revue structurante sur les caractéristiques, cibles de sélection et domaines combinatoires.
- **lindauer2015** — [AutoFolio: An Automatically Configured Algorithm Selector](https://doi.org/10.1613/jair.4726) (2015). Sélection automatique et configuration du sélecteur; référence d’état de l’art, pas obligation de complexifier le routage local.
- **kadioglu2011** — [Algorithm Selection and Scheduling](https://doi.org/10.1007/978-3-642-23786-7_35) (2011). Sélection et ordonnancement avec budget; mécanisme plus large que le routage mono-action proposé.
- **tornede2020** — [Run2Survive: A Decision-theoretic Approach to Algorithm Selection based on Survival Analysis](https://proceedings.mlr.press/v129/tornede20a.html) (2020). Gestion des temps censurés et décision selon le risque; distinguer temps d’atteinte d’une cible et sortie de processus mesurée ici.
- **pezo2025** — [Automatic algorithm selection for Pseudo-Boolean optimization with given computational time limits](https://doi.org/10.1016/j.cor.2024.106836) (2025). Antécédent central pour une sélection dépendant explicitement du délai et de la qualité; la sortie complète et la validité physique restent spécifiques au projet.
- **alissa2023** — [Automated Algorithm Selection: from Feature-Based to Feature-Free Approaches](https://doi.org/10.1007/s10732-022-09505-4) (2023). Alternative récente aux caractéristiques conçues manuellement; étude méthodologique sur domaines séquentiels, pas preuve en énergie.
- **elkan2001** — [The Foundations of Cost-Sensitive Learning](https://ijcai.org/proceedings/2001/133) (2001). Distinguer précision de classification, coûts d’erreurs et utilité de la décision; ici la cible primaire est le succès conjoint.
- **gneiting2007** — [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437) (2007). Règles de score propres pour probabilités de succès; justification de log loss et Brier.
- **cawley2010** — [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html) (2010). Séparer choix des modèles et évaluation finale; prévenir l’optimisme dû aux comparaisons répétées.
- **roberts2017** — [Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure](https://doi.org/10.1111/ecog.02881) (2017). Validation par groupes et transferts structurés; justification méthodologique, non étude de systèmes énergétiques.

## Où les mobiliser dans le manuscrit

| Passage | Clés proposées | Point à étayer |
| --- | --- | --- |
| Introduction : sélection, portefeuille et prédiction | rice1976, gomes2001, xu2008, hutter2013, kotthoff2014, kerschke2019 | Cadre antérieur et périmètre précis de la contribution énergétique. |
| Complémentarité, oracle et méthode fixe | bischl2016, lindauer2015, lindauer2017 | Oracle rétrospectif comme diagnostic; meilleure méthode fixe choisie sur développement. |
| Sélecteur dépendant du délai et coût de décision | pezo2025, tornede2020, elkan2001 | Antécédents de sélection sous budget, censure et pénalités de décision. |
| Choix des caractéristiques et modèles | smithmiles2012, smithmiles2014, alissa2023, kadioglu2010 | Caractéristiques physiques simples versus représentation apprise; configuration et sélection restent distinctes. |
| Frontière avec les calendriers de solveurs | kadioglu2011, amadini2014, gomes2001 | Le manuscrit choisit une action; il ne partage pas le budget entre plusieurs membres. |
| Calibration, développement groupé et test final | gneiting2007, niculescumizil2005, cawley2010, roberts2017, kapoor2023, efron1979 | Évaluer les probabilités et le gain net après sélection sans réutiliser les résultats du test. |

## Ensemble du corpus

Les autres références servent à préciser les méthodes, les limites ou les comparateurs. Les travaux AC-OPF, vision, TSP ou classification sont explicitement des références méthodologiques adjacentes. Les performances publiées ne sont pas des résultats de ce projet.

### Unit commitment: formulations and computational structure

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| moralesespana2013 | [Tight and Compact MILP Formulation for the Thermal Unit Commitment Problem](https://doi.org/10.1109/TPWRS.2013.2251373), 2013 | Formulation thermique et effet du resserrement sur la résolution; référence de modèle, pas identité avec la formulation locale. |
| knueven2020 | [On Mixed-Integer Programming Formulations for the Unit Commitment Problem](https://doi.org/10.1287/ijoc.2019.0944), 2020 | Comparer familles de formulations UC et effets sur le temps de calcul; fondement pour interpréter la difficulté. |

### Instance features and empirical computational difficulty

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| leytonbrown2002 | [Learning the Empirical Hardness of Optimization Problems: The Case of Combinatorial Auctions](https://doi.org/10.1007/3-540-46135-3_37), 2002 | Antécédent des modèles de difficulté empirique à partir de caractéristiques, avec une formulation résolue par CPLEX. |
| smithmiles2012 | [Measuring instance difficulty for combinatorial optimization problems](https://doi.org/10.1016/j.cor.2011.07.006), 2012 | Fondement principal de la distinction entre caractéristiques d’instance, taille et difficulté empirique dépendante de l’algorithme. |
| hutter2013 | [Algorithm runtime prediction: Methods & evaluation](https://doi.org/10.1016/j.artint.2013.10.003), 2014 | Prédiction de performance, descripteurs et temps censurés; clé historique conservée malgré l’année de revue 2014. |
| smithmiles2014 | [Towards objective measures of algorithm performance across instance space](https://doi.org/10.1016/j.cor.2013.11.015), 2014 | Évaluer les zones de réussite et d’échec dans l’espace des instances, au-delà des moyennes globales. |

### Computational benchmarking and quality versus time

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| dolan2002 | [Benchmarking optimization software with performance profiles](https://doi.org/10.1007/s101070100263), 2002 | Comparer la performance sur un ensemble d’instances; ne remplace pas la mesure de sortie complète du processus. |
| berthold2013 | [Measuring the impact of primal heuristics](https://doi.org/10.1016/j.orl.2013.08.007), 2013 | Situer les courbes qualité-temps et l’intégrale primale; distinguer incumbent disponible et programme terminé. |
| beiranvand2017 | [Best practices for comparing optimization algorithms](https://doi.org/10.1007/s11081-017-9366-1), 2017 | Comparaison équitable, ressources, jeux d’instances et présentation des résultats. |
| gleixner2021 | [MIPLIB 2017: data-driven compilation of the 6th mixed-integer programming library](https://doi.org/10.1007/s12532-020-00194-3), 2021 | Diversité et sélection d’instances MILP; bibliothèque externe, pas description du générateur énergétique local. |

### Learning to guide mixed-integer optimization

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| bengio2021 | [Machine learning for combinatorial optimization: A methodological tour d'horizon](https://doi.org/10.1016/j.ejor.2020.07.063), 2021 | Taxonomie des interactions entre apprentissage et optimisation et choix des distributions d’entraînement. |

### Learning for unit commitment: direct related work

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| xavier2021 | [Learning to Solve Large-Scale Security-Constrained Unit Commitment Problems](https://doi.org/10.1287/ijoc.2020.0976), 2021 | Antécédent direct pour contraintes, solutions initiales et réduction d’espace apprises en SCUC; année du fascicule 2021, mise en ligne 2020. |

### Generalization, graph size and distribution shift

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| koh2021 | [WILDS: A Benchmark of in-the-Wild Distribution Shifts](https://proceedings.mlr.press/v139/koh21a.html), 2021 | Évaluation séparée des changements de distribution; benchmark externe, sans preuve de transfert pour les cas énergétiques locaux. |

### Decision-focused learning: related but distinct problem setting

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| elmachtoub2022 | [Smart ``Predict, then Optimize''](https://doi.org/10.1287/mnsc.2020.3922), 2022 | Évaluer l’effet aval d’une prédiction; ne pas présenter le routage ou JEPA comme une implémentation de SPO. |

### Algorithm selection, portfolios and selection costs

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| rice1976 | [The Algorithm Selection Problem](https://doi.org/10.1016/S0065-2458(08)60520-3), 1976 | Cadre historique reliant instance, caractéristiques, algorithme et critère de performance. |
| elkan2001 | [The Foundations of Cost-Sensitive Learning](https://ijcai.org/proceedings/2001/133), 2001 | Distinguer précision de classification, coûts d’erreurs et utilité de la décision; ici la cible primaire est le succès conjoint. |
| gomes2001 | [Algorithm portfolios](https://doi.org/10.1016/S0004-3702(00)00081-3), 2001 | Complémentarité et variabilité des algorithmes; exécution parallèle ou entrelacée à distinguer du routage vers une seule méthode. |
| xu2008 | [SATzilla: Portfolio-based Algorithm Selection for SAT](https://doi.org/10.1613/jair.2490), 2008 | Sélection par prédiction de performance, caractéristiques et coût de préparation. |
| kadioglu2010 | [ISAC -- Instance-Specific Algorithm Configuration](https://ai.dmi.unibas.ch/research/reading_group/kadioglu-et-al-ecai2010.pdf), 2010 | Configuration conditionnée par groupes d’instances; distinction à conserver avec le choix entre quatre méthodes figées. |
| kadioglu2011 | [Algorithm Selection and Scheduling](https://doi.org/10.1007/978-3-642-23786-7_35), 2011 | Sélection et ordonnancement avec budget; mécanisme plus large que le routage mono-action proposé. |
| amadini2014 | [SUNNY: a Lazy Portfolio Approach for Constraint Solving](https://doi.org/10.1017/S1471068414000179), 2014 | Approche par voisinage et ordonnancement; comparateur conceptuel, non assimilé au sélecteur mono-action du papier. |
| kotthoff2014 | [Algorithm Selection for Combinatorial Search Problems: A Survey](https://doi.org/10.1609/aimag.v35i3.2460), 2014 | Revue structurante sur les caractéristiques, cibles de sélection et domaines combinatoires. |
| lindauer2015 | [AutoFolio: An Automatically Configured Algorithm Selector](https://doi.org/10.1613/jair.4726), 2015 | Sélection automatique et configuration du sélecteur; référence d’état de l’art, pas obligation de complexifier le routage local. |
| bischl2016 | [ASlib: A benchmark library for algorithm selection](https://doi.org/10.1016/j.artint.2016.04.003), 2016 | Scénarios de sélection, coûts des caractéristiques et références meilleur algorithme fixe/oracle. |
| lindauer2017 | [Open Algorithm Selection Challenge 2017: Setup and Scenarios](https://proceedings.mlr.press/v79/lindauer17a.html), 2017 | Protocoles de comparaison de sélecteurs et scénarios de référence. |
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
| kapoor2023 | [Leakage and the reproducibility crisis in machine-learning-based science](https://doi.org/10.1016/j.patter.2023.100804), 2023 | Prévenir la fuite entre scénarios apparentés, prétraitement, sélection de modèle et test final. |

### Software and external benchmark resources

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| pedregosa2011 | [Scikit-learn: Machine Learning in Python](https://www.jmlr.org/papers/v12/pedregosa11a.html), 2011 | Bibliothèque prévue pour les modèles statistiques et le routage; préciser la version effectivement utilisée lors des futurs calculs. |
| huangfu2018 | [Parallelizing the dual revised simplex method](https://doi.org/10.1007/s12532-017-0130-5), 2018 | Référence recommandée par HiGHS; article sur le simplexe LP, pas validation de performances UC ou de tous les composants MIP. |
| gurobi2026 | [Gurobi Optimizer Reference Manual](https://www.gurobi.com), 2026 — logiciel/documentation | Référence fournisseur pour l’adaptateur et les futurs essais; ne présume pas que Gurobi a été exécuté ni qu’il est une action du sélecteur. |

## Traçabilité

Voir le [catalogue commun](../../bibliography/catalogue.json), le [journal de recherche](../../bibliography/RECHERCHE.md) et le [tri des 40 références initiales](../../bibliography/TRI_BIBLIOGRAPHIE_ORIGINE.md). Une référence commune est reproduite à l’identique dans les bibliographies qui en ont besoin.

Aucune compilation BibTeX/LaTeX ni exécution scientifique n’a été effectuée. Les champs optionnels non confirmés ont été omis; les particularités de métadonnées sont documentées dans le catalogue et dans [BIBLIOGRAPHIE.md](../../BIBLIOGRAPHIE.md).
