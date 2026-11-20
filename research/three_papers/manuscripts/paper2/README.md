# Papier 2 — Méthodes apprises, utilité sous délai et généralisation

Bibliographie consultée le **5 novembre 2026** : **75 références** dans [references.bib](references.bib). Le corpus est un réservoir de rédaction, pas une liste à citer intégralement. Les commentaires de chaque entrée indiquent sa pertinence, le statut de la version citée et les sources.

Le manuscrit associé utilise désormais \bibliography{paper2/references}, depuis le dossier manuscripts. Son texte et ses résultats n’ont pas été réécrits. Les citations déjà présentes sont conservées.

## Lectures prioritaires

- **bengio2021** — [Machine learning for combinatorial optimization: A methodological tour d'horizon](https://doi.org/10.1016/j.ejor.2020.07.063) (2021). Taxonomie des interactions entre apprentissage et optimisation et choix des distributions d’entraînement.
- **cappart2023** — [Combinatorial Optimization and Reasoning with Graph Neural Networks](https://www.jmlr.org/papers/v24/21-0449.html) (2023). Revue étendue des GNN pour optimisation; remplace la version courte IJCAI 2021.
- **xavier2021** — [Learning to Solve Large-Scale Security-Constrained Unit Commitment Problems](https://doi.org/10.1287/ijoc.2020.0976) (2021). Antécédent direct pour contraintes, solutions initiales et réduction d’espace apprises en SCUC; année du fascicule 2021, mise en ligne 2020.
- **gasse2019** — [Exact Combinatorial Optimization with Graph Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html) (2019). Branchement appris sur graphe biparti; distinguer guidage de recherche exacte et génération de candidats.
- **nair2021** — [Solving Mixed Integer Programs Using Neural Networks](https://arxiv.org/abs/2012.13349v3) (2021). Neural Diving et Neural Branching; point de comparaison direct pour les affectations apprises complétées par un solveur.
- **sonnerat2022** — [Learning a Large Neighborhood Search Algorithm for Mixed Integer Programs](https://arxiv.org/abs/2107.10201v3) (2022). Choix appris des voisinages MILP; comparer le rôle du solveur et le coût complet de l’amélioration.
- **qin2025** — [Solve Large-Scale Unit Commitment Problems by Physics-Informed Graph Learning](https://doi.org/10.1109/TPWRS.2025.3562886) (2025). Graphe physique hiérarchique, neural diving et branching pour UC; voisin scientifique central.
- **park2024** — [Confidence-Aware Graph Neural Networks for Learning Reliability Assessment Commitments](https://doi.org/10.1109/TPWRS.2023.3298735) (2024). Confiance, fixation partielle et réparation de faisabilité sur un problème opérationnel de grande taille.
- **wei2025** — [Data-augmentation acceleration framework by graph neural network for near-optimal unit commitment](https://doi.org/10.1016/j.apenergy.2024.124332) (2025). Solutions proches de l’optimum pour augmenter l’apprentissage puis réparation; voisin direct du pipeline à documenter dans l’état de l’art.
- **yang2025lns** — [Solving Unit Commitment Problems with Graph Neural Network based Initial Commitment Prediction and Large Neighborhood Search](https://arxiv.org/abs/2505.14408) (2025). Prédiction initiale, restauration de faisabilité puis voisinages appris; comparaison très proche de la chaîne candidats-réparation.
- **venkatesh2025** — [Feasibility-guaranteed machine learning unit commitment: Fuzzy Optimization approaches](https://doi.org/10.1016/j.apenergy.2024.124923) (2025). Intégrer des prédictions comme contraintes non obligatoires; autre stratégie de traitement des erreurs de prédiction.
- **pourahmadi2025** — [Unit Commitment Predictor With a Performance Guarantee: A Support Vector Machine Classifier](https://doi.org/10.1109/TPWRS.2024.3400405) (2025). Prédiction UC et garanties sous hypothèses; comparateur non GNN pour discuter performance hors échantillon.
- **fritz2026** — [Learning to Fix: Optimisation-Aware Machine Learning for Accelerated Unit Commitment](https://arxiv.org/abs/2609.39396) (2026). Seuils de confiance tenant compte des conséquences économiques des erreurs de fixation; antécédent récent à discuter avec prudence.
- **assran2023** — [Self-Supervised Learning From Images With a Joint-Embedding Predictive Architecture](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html) (2023). Origine du principe de prédiction dans l’espace latent; I-JEPA image n’est pas la méthode énergétique locale ni une validation de celle-ci.
- **lecun2006** — [A Tutorial on Energy-Based Learning](https://yann.lecun.org/exdb/publis/pdf/lecun-06.pdf) (2006). Définition de l’énergie comme fonction de compatibilité et distinction entre inférence, apprentissage et normalisation probabiliste.
- **du2019** — [Implicit Generation and Modeling with Energy Based Models](https://papers.nips.cc/paper_files/paper/2019/hash/378a063b8fdb1db941e34f4bde584c7d-Abstract.html) (2019). Entraînement et génération avec EBM continus; titre final différent de la version arXiv sur la généralisation.
- **goshvadi2023** — [DISCS: A Benchmark for Discrete Sampling](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f9ad87c1ebbae8a3555adb31dbcacf44-Abstract-Datasets_and_Benchmarks.html) (2023). Comparaison contrôlée des échantillonneurs; le pipeline local échantillonne un latent continu, distinction à conserver.
- **chen2023** — [On Representing Mixed-Integer Linear Programs by Graph Neural Networks](https://openreview.net/forum?id=4gc3MGZra1d) (2023). Limites d’expressivité des GNN sur les MILP; empêcher d’inférer une garantie de résolution à partir de l’architecture.
- **yehudai2021** — [From Local Structures to Size Generalization in Graph Neural Networks](https://proceedings.mlr.press/v139/yehudai21a.html) (2021). La capacité à traiter différentes tailles n’implique pas une généralisation à leurs structures locales.
- **joshi2022** — [Learning the travelling salesperson problem requires rethinking generalization](https://doi.org/10.1007/s10601-022-09327-y) (2022). Protocole de généralisation en optimisation neuronale; TSP est un domaine méthodologique adjacent, pas UC.
- **li2025foundation** — [Towards Foundation Models for Mixed Integer Linear Programming](https://openreview.net/forum?id=6yENDA7J4G) (2025). Diversité des classes MILP et transfert hors des classes d’entraînement; contexte récent pour les limites des cohortes synthétiques.
- **beiranvand2017** — [Best practices for comparing optimization algorithms](https://doi.org/10.1007/s11081-017-9366-1) (2017). Comparaison équitable, ressources, jeux d’instances et présentation des résultats.
- **berthold2013** — [Measuring the impact of primal heuristics](https://doi.org/10.1016/j.orl.2013.08.007) (2013). Situer les courbes qualité-temps et l’intégrale primale; distinguer incumbent disponible et programme terminé.

## Où les mobiliser dans le manuscrit

| Passage | Clés proposées | Point à étayer |
| --- | --- | --- |
| Introduction et état de l’art ML–MILP | bengio2021, cappart2023, gasse2019, nair2021, sonnerat2022 | Distinguer branchement, affectations partielles et recherche de voisinages. |
| Travaux directement concurrents en UC | xavier2021, park2024, qin2025, wei2025, yang2025lns, yang2025fixation, venkatesh2025, pourahmadi2025 | Comparer formulation, information disponible, traitement des erreurs et coût total avant de rapprocher les gains publiés. |
| Travaux récents de 2026 | fritz2026, xu2026uc, wang2026cuts, dong2026, pramita2026 | Actualiser la discussion, en signalant les deux prépublications Fritz et Pramita. |
| Encodeur temporel et prédicteur latent | cho2014, vaswani2017, lee2019, brody2022, hu2020, assran2023, bardes2022 | Citer le bloc réellement utilisé; I-JEPA vision et HGT sont des antécédents conceptuels. |
| Énergie, candidats et réparation | lecun2006, du2019, welling2011, sun2022, goshvadi2023, fischetti2003, danna2005, fischetti2005, zeng2024 | Distinguer génération, sélection, intégralité et validation; ne pas attribuer la qualité au générateur si elle vient de la réparation. |
| Taille, permutations, topologie et horizon | chen2023, xu2019, zaheer2017, yehudai2021, joshi2022, li2025foundation, liu2023 | Compatibilité d’une dimension et transfert réussi sont deux propriétés différentes. |
| Protocole comparatif et chronométrage | dolan2002, berthold2013, beiranvand2017, gleixner2021, roberts2017, kapoor2023 | Équité des ressources et des cohortes; aucune référence citée ne suffit à certifier la sortie complète du programme local. |
| Discussion : proximité avec decision-focused learning | donti2017, wilder2019, elmachtoub2022, mandi2024, mandi2025, wu2026 | Prédire des paramètres incertains n’est pas générer des décisions pour une instance entièrement connue. |

## Ensemble du corpus

Les autres références servent à préciser les méthodes, les limites ou les comparateurs. Les travaux AC-OPF, vision, TSP ou classification sont explicitement des références méthodologiques adjacentes. Les performances publiées ne sont pas des résultats de ce projet.

### Unit commitment: formulations and computational structure

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| carrion2006 | [A computationally efficient mixed-integer linear formulation for the thermal unit commitment problem](https://doi.org/10.1109/TPWRS.2006.876672), 2006 | Contraintes intertemporelles et compromis nombre de variables/contraintes et performance de résolution. |
| moralesespana2013 | [Tight and Compact MILP Formulation for the Thermal Unit Commitment Problem](https://doi.org/10.1109/TPWRS.2013.2251373), 2013 | Formulation thermique et effet du resserrement sur la résolution; référence de modèle, pas identité avec la formulation locale. |
| knueven2020 | [On Mixed-Integer Programming Formulations for the Unit Commitment Problem](https://doi.org/10.1287/ijoc.2019.0944), 2020 | Comparer familles de formulations UC et effets sur le temps de calcul; fondement pour interpréter la difficulté. |
| huang2021 | [Cutting planes for security-constrained unit commitment with regulation reserve](https://doi.org/10.1080/24725854.2020.1823533), 2021 | Lien entre réserves, structure polyédrale et difficulté de résolution; les contraintes du papier ne sont pas toutes celles du modèle local. |

### Physical stress and power-system flexibility

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| lannoye2012 | [Evaluation of Power System Flexibility](https://doi.org/10.1109/TPWRS.2011.2177280), 2012 | Indicateurs de disponibilité des ressources de rampe et échelles temporelles; risque physique distinct de la difficulté informatique. |
| ulbig2015 | [Analyzing operational flexibility of electric power systems](https://doi.org/10.1016/j.ijepes.2015.02.028), 2015 | Interprétation physique des capacités de flexibilité; ne mesure pas la difficulté MILP. |
| heggarty2019 | [Multi-temporal assessment of power system flexibility requirement](https://doi.org/10.1016/j.apenergy.2019.01.198), 2019 | Besoins de flexibilité selon plusieurs échelles temporelles et leurs déterminants. |

### Instance features and empirical computational difficulty

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
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
| gasse2019 | [Exact Combinatorial Optimization with Graph Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html), 2019 | Branchement appris sur graphe biparti; distinguer guidage de recherche exacte et génération de candidats. |
| bengio2021 | [Machine learning for combinatorial optimization: A methodological tour d'horizon](https://doi.org/10.1016/j.ejor.2020.07.063), 2021 | Taxonomie des interactions entre apprentissage et optimisation et choix des distributions d’entraînement. |
| nair2021 | [Solving Mixed Integer Programs Using Neural Networks](https://arxiv.org/abs/2012.13349v3), 2021 — prépublication | Neural Diving et Neural Branching; point de comparaison direct pour les affectations apprises complétées par un solveur. |
| sonnerat2022 | [Learning a Large Neighborhood Search Algorithm for Mixed Integer Programs](https://arxiv.org/abs/2107.10201v3), 2022 — prépublication | Choix appris des voisinages MILP; comparer le rôle du solveur et le coût complet de l’amélioration. |
| cappart2023 | [Combinatorial Optimization and Reasoning with Graph Neural Networks](https://www.jmlr.org/papers/v24/21-0449.html), 2023 | Revue étendue des GNN pour optimisation; remplace la version courte IJCAI 2021. |

### Learning for unit commitment: direct related work

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| demars2021 | [Applying reinforcement learning and tree search to the unit commitment problem](https://doi.org/10.1016/j.apenergy.2021.117519), 2021 | Apprentissage guidant une recherche UC sous incertitude; les informations et objectifs diffèrent du benchmark déterministe local. |
| xavier2021 | [Learning to Solve Large-Scale Security-Constrained Unit Commitment Problems](https://doi.org/10.1287/ijoc.2020.0976), 2021 | Antécédent direct pour contraintes, solutions initiales et réduction d’espace apprises en SCUC; année du fascicule 2021, mise en ligne 2020. |
| he2024 | [Fast Unit Commitment Constraint Screening with Learning-Based Cost Model](https://doi.org/10.1109/SmartGridComm60555.2024.10738113), 2024 | Réduction des contraintes par modèle de coût; méthode d’accélération distincte des candidats appris. |
| park2024 | [Confidence-Aware Graph Neural Networks for Learning Reliability Assessment Commitments](https://doi.org/10.1109/TPWRS.2023.3298735), 2024 | Confiance, fixation partielle et réparation de faisabilité sur un problème opérationnel de grande taille. |
| pourahmadi2025 | [Unit Commitment Predictor With a Performance Guarantee: A Support Vector Machine Classifier](https://doi.org/10.1109/TPWRS.2024.3400405), 2025 | Prédiction UC et garanties sous hypothèses; comparateur non GNN pour discuter performance hors échantillon. |
| qin2025 | [Solve Large-Scale Unit Commitment Problems by Physics-Informed Graph Learning](https://doi.org/10.1109/TPWRS.2025.3562886), 2025 | Graphe physique hiérarchique, neural diving et branching pour UC; voisin scientifique central. |
| venkatesh2025 | [Feasibility-guaranteed machine learning unit commitment: Fuzzy Optimization approaches](https://doi.org/10.1016/j.apenergy.2024.124923), 2025 | Intégrer des prédictions comme contraintes non obligatoires; autre stratégie de traitement des erreurs de prédiction. |
| wei2025 | [Data-augmentation acceleration framework by graph neural network for near-optimal unit commitment](https://doi.org/10.1016/j.apenergy.2024.124332), 2025 | Solutions proches de l’optimum pour augmenter l’apprentissage puis réparation; voisin direct du pipeline à documenter dans l’état de l’art. |
| yang2025fixation | [Stable Variable Fixation for Accelerated Unit Commitment via Graph Neural Network and Linear Programming Hybrid Learning](https://doi.org/10.3390/app15084498), 2025 | Fixation stable et hybride GNN-LP; comparer explicitement la récupération de faisabilité. |
| yang2025lns | [Solving Unit Commitment Problems with Graph Neural Network based Initial Commitment Prediction and Large Neighborhood Search](https://arxiv.org/abs/2505.14408), 2025 — prépublication | Prédiction initiale, restauration de faisabilité puis voisinages appris; comparaison très proche de la chaîne candidats-réparation. |
| dong2026 | [A deep learning unit commitment method based on confidence intervals](https://doi.org/10.1016/j.epsr.2026.112908), 2026 | Initialisations différenciées selon la confiance des prédictions; numéro de volume non renseigné faute de confirmation primaire. |
| fritz2026 | [Learning to Fix: Optimisation-Aware Machine Learning for Accelerated Unit Commitment](https://arxiv.org/abs/2609.39396), 2026 — prépublication | Seuils de confiance tenant compte des conséquences économiques des erreurs de fixation; antécédent récent à discuter avec prudence. |
| pramita2026 | [Reinforcement learning on the EPRI AI-ccelerating Unit Commitment benchmark: diagnosing over-commitment and guiding MILP with learned policies](https://arxiv.org/abs/2610.25544), 2026 — prépublication | Résultats négatifs et diagnostic du sur-engagement; comparaison entre politique autonome et politique guidant MILP. |
| wang2026cuts | [Adaptive cut generation via hypergraph neural networks for unit commitment problems](https://doi.org/10.1016/j.eswa.2026.132519), 2026 | Accélération interne de MILP par hypergraphes et coupes, distincte de la génération externe de stratégies. |
| xu2026uc | [Graph neural network with physics-informed optimization layer for unit commitment and economic dispatch](https://doi.org/10.1016/j.ijepes.2026.111821), 2026 | Séparation décisions discrètes/dispatch continu et correction physique; comparer hypothèses et contraintes avant tout rapprochement chiffré. |

### Feasibility repair and search neighborhoods

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| fischetti2003 | [Local branching](https://doi.org/10.1007/s10107-003-0395-5), 2003 | Voisinages MILP autour d’un candidat; antécédent conceptuel de réparation/amélioration, sans prétendre que la cascade locale implémente local branching. |
| danna2005 | [Exploring relaxation induced neighborhoods to improve MIP solutions](https://doi.org/10.1007/s10107-004-0518-7), 2005 | RINS et plongeons guidés; situer les améliorations par solveur face à une proposition apprise. |
| fischetti2005 | [The feasibility pump](https://doi.org/10.1007/s10107-004-0570-3), 2005 | La faisabilité des entiers est un problème distinct de l’arrondi; référence de contexte, pas identité algorithmique. |
| zeng2024 | [GLinSAT: The General Linear Satisfiability Neural Network Layer By Accelerated Gradient Descent](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dd73f39426a03131c38c8d943153d44b-Abstract-Conference.html), 2024 | Projection différentiable sous contraintes linéaires; distinguer satisfiabilité linéaire bornée et intégralité de la solution UC. |

### Representation and neural-architecture foundations

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| cho2014 | [Learning Phrase Representations using RNN Encoder--Decoder for Statistical Machine Translation](https://doi.org/10.3115/v1/D14-1179), 2014 | Source du GRU employé dans les encodeurs temporels; l’application initiale est la traduction. |
| vaswani2017 | [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html), 2017 | Attention et encodage positionnel; ne constitue pas une garantie de transfert vers de nouveaux horizons. |
| lee2019 | [Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks](https://proceedings.mlr.press/v97/lee19d.html), 2019 | Agrégation de cardinalité variable par attention et tokens; distinguer cette propriété de l’équivariance du décodeur entier. |
| hu2020 | [Heterogeneous Graph Transformer](https://doi.org/10.1145/3366423.3380027), 2020 | Attention avec types de nœuds et de relations; référence conceptuelle, le sigle HTE local ne signifie pas que HGT est implémenté à l’identique. |
| bardes2022 | [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://openreview.net/forum?id=iWpcWZ8phD), 2022 | Référence complémentaire sur la non-dégénérescence des représentations; ne pas prétendre utiliser VICReg sans vérifier les pertes locales. |
| brody2022 | [How Attentive Are Graph Attention Networks?](https://openreview.net/forum?id=F72ximsx7C1), 2022 | Référence GATv2 pour l’attention de graphe; utiliser seulement pour les blocs effectivement présents. |
| assran2023 | [Self-Supervised Learning From Images With a Joint-Embedding Predictive Architecture](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html), 2023 | Origine du principe de prédiction dans l’espace latent; I-JEPA image n’est pas la méthode énergétique locale ni une validation de celle-ci. |

### Energy-based models and candidate sampling

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| lecun2006 | [A Tutorial on Energy-Based Learning](https://yann.lecun.org/exdb/publis/pdf/lecun-06.pdf), 2006 — manuscrit de tutoriel | Définition de l’énergie comme fonction de compatibilité et distinction entre inférence, apprentissage et normalisation probabiliste. |
| welling2011 | [Bayesian Learning via Stochastic Gradient Langevin Dynamics](https://ora.ox.ac.uk/objects/uuid:13ec7031-519b-4223-81cb-dd8de9213836), 2011 | Fondement des mises à jour gradient-bruit; ne justifie pas une convergence stationnaire pour seize candidats ou un nombre fini de pas. |
| du2019 | [Implicit Generation and Modeling with Energy Based Models](https://papers.nips.cc/paper_files/paper/2019/hash/378a063b8fdb1db941e34f4bde584c7d-Abstract.html), 2019 | Entraînement et génération avec EBM continus; titre final différent de la version arXiv sur la généralisation. |
| sun2022 | [Annealed Training for Combinatorial Optimization on Graphs](https://arxiv.org/abs/2207.11542), 2022 — prépublication | Lien EBM et optimisation combinatoire avec recuit; ne pas confondre énergie d’objectif et énergie latente apprise. |
| goshvadi2023 | [DISCS: A Benchmark for Discrete Sampling](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f9ad87c1ebbae8a3555adb31dbcacf44-Abstract-Datasets_and_Benchmarks.html), 2023 | Comparaison contrôlée des échantillonneurs; le pipeline local échantillonne un latent continu, distinction à conserver. |
| alcazar2024 | [Enhancing combinatorial optimization with classical and quantum generative models](https://doi.org/10.1038/s41467-024-46959-5), 2024 | Génération de candidats au service de la recherche; retenir le cadre générique, sans revendiquer d’avantage quantique pour ce projet. |

### Generalization, graph size and distribution shift

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| zaheer2017 | [Deep Sets](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html), 2017 | Invariance aux permutations des ensembles; utile pour l’audit de renumérotation et la conception de la future architecture. |
| xu2019 | [How Powerful are Graph Neural Networks?](https://openreview.net/forum?id=ryGs6iA5Km), 2019 | Pouvoir expressif des agrégateurs de graphes et limites de distinction entre structures. |
| koh2021 | [WILDS: A Benchmark of in-the-Wild Distribution Shifts](https://proceedings.mlr.press/v139/koh21a.html), 2021 | Évaluation séparée des changements de distribution; benchmark externe, sans preuve de transfert pour les cas énergétiques locaux. |
| yehudai2021 | [From Local Structures to Size Generalization in Graph Neural Networks](https://proceedings.mlr.press/v139/yehudai21a.html), 2021 | La capacité à traiter différentes tailles n’implique pas une généralisation à leurs structures locales. |
| joshi2022 | [Learning the travelling salesperson problem requires rethinking generalization](https://doi.org/10.1007/s10601-022-09327-y), 2022 | Protocole de généralisation en optimisation neuronale; TSP est un domaine méthodologique adjacent, pas UC. |
| chen2023 | [On Representing Mixed-Integer Linear Programs by Graph Neural Networks](https://openreview.net/forum?id=4gc3MGZra1d), 2023 | Limites d’expressivité des GNN sur les MILP; empêcher d’inférer une garantie de résolution à partir de l’architecture. |
| li2025foundation | [Towards Foundation Models for Mixed Integer Linear Programming](https://openreview.net/forum?id=6yENDA7J4G), 2025 | Diversité des classes MILP et transfert hors des classes d’entraînement; contexte récent pour les limites des cohortes synthétiques. |

### Power-grid graph learning: adjacent applications

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| donon2019 | [Graph Neural Solver for Power Systems](https://doi.org/10.1109/IJCNN.2019.8851855), 2019 | Transfert en taille pour calcul de flux; domaine adjacent, distinct de l’UC mixte entier et de la certification de faisabilité. |
| liu2023 | [Topology-Aware Graph Neural Networks for Learning Feasible and Adaptive AC-OPF Solutions](https://doi.org/10.1109/TPWRS.2022.3230555), 2023 | Adaptation topologique et régularisation physique en AC-OPF; ne pas assimiler adaptation avec réentraînement et transfert figé. |
| zhang2024risk | [Graph neural networks for power grid operational risk assessment under evolving grid topology](https://arxiv.org/abs/2405.07343), 2024 — prépublication | Prédiction du risque physique et topologies changeantes; distinguer prédiction de risque et prédiction du succès informatique. |
| pjetri2025 | [Graph Neural Network Surrogates for Energy System Modeling](https://doi.org/10.1109/ISGTEurope64741.2025.11305443), 2025 | Prédiction rapide de sorties d’un modèle énergétique et identification de cas critiques; surrogate de planification, pas résolution UC complète sous délai. |

### Decision-focused learning: related but distinct problem setting

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| donti2017 | [Task-based End-to-end Model Learning in Stochastic Optimization](https://proceedings.neurips.cc/paper/2017/file/3fc2c60b5782f641f76bcefc39fb2392-Paper.pdf), 2017 | Aligner prédiction et coût de décision, avec applications énergétiques; distinct de l’imitation de stratégies connues. |
| wilder2019 | [Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization](https://doi.org/10.1609/aaai.v33i01.33011658), 2019 | Apprentissage à travers une relaxation de problème combinatoire; cadre voisin à distinguer du générateur latent local. |
| elmachtoub2022 | [Smart ``Predict, then Optimize''](https://doi.org/10.1287/mnsc.2020.3922), 2022 | Évaluer l’effet aval d’une prédiction; ne pas présenter le routage ou JEPA comme une implémentation de SPO. |
| mandi2024 | [Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities](https://doi.org/10.1613/jair.1.15320), 2024 | Revue et benchmark pour distinguer apprentissage de paramètres, de décisions et de politiques de résolution. |
| mandi2025 | [Feasibility-Aware Decision-Focused Learning for Predicting Parameters in the Constraints](https://arxiv.org/abs/2510.04951), 2025 — prépublication | Tension entre qualité et faisabilité lorsque les contraintes contiennent des paramètres prédits; contexte différent de la réparation sous paramètres connus. |
| wu2026 | [Smart Predict-Then-Optimize-Based Unit Commitment for Integrated Energy Systems](https://doi.org/10.1109/TII.2026.3671325), 2026 | Application énergétique récente de SPO à des prévisions; différencier prévision des entrées et génération d’une stratégie. |

### Algorithm selection, portfolios and selection costs

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| gomes2001 | [Algorithm portfolios](https://doi.org/10.1016/S0004-3702(00)00081-3), 2001 | Complémentarité et variabilité des algorithmes; exécution parallèle ou entrelacée à distinguer du routage vers une seule méthode. |

### Statistical validation, calibration and uncertainty

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| efron1979 | [Bootstrap Methods: Another Look at the Jackknife](https://doi.org/10.1214/aos/1176344552), 1979 | Fondement du rééchantillonnage; dans ce projet l’unité doit rester le groupe scénario, pas chaque cellule budget. |
| demsar2006 | [Statistical Comparisons of Classifiers over Multiple Data Sets](https://jmlr.org/papers/v7/demsar06a.html), 2006 | Comparaisons multiples et unités indépendantes; ne pas traiter les budgets répétés d’un même scénario comme des jeux indépendants. |
| cawley2010 | [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html), 2010 | Séparer choix des modèles et évaluation finale; prévenir l’optimisme dû aux comparaisons répétées. |
| roberts2017 | [Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure](https://doi.org/10.1111/ecog.02881), 2017 | Validation par groupes et transferts structurés; justification méthodologique, non étude de systèmes énergétiques. |
| kapoor2023 | [Leakage and the reproducibility crisis in machine-learning-based science](https://doi.org/10.1016/j.patter.2023.100804), 2023 | Prévenir la fuite entre scénarios apparentés, prétraitement, sélection de modèle et test final. |

### Software and external benchmark resources

| Clé | Référence | Utilité / limite |
| --- | --- | --- |
| huangfu2018 | [Parallelizing the dual revised simplex method](https://doi.org/10.1007/s12532-017-0130-5), 2018 | Référence recommandée par HiGHS; article sur le simplexe LP, pas validation de performances UC ou de tous les composants MIP. |
| xavier2024ucjl | [UnitCommitment.jl: A Julia/JuMP Optimization Package for Security-Constrained Unit Commitment](https://doi.org/10.5281/zenodo.4269874), 2024 — logiciel/documentation | Benchmark externe et formulations publiques; ce logiciel n’est pas utilisé comme provenance des résultats locaux. |
| gurobi2026 | [Gurobi Optimizer Reference Manual](https://www.gurobi.com), 2026 — logiciel/documentation | Référence fournisseur pour l’adaptateur et les futurs essais; ne présume pas que Gurobi a été exécuté ni qu’il est une action du sélecteur. |

## Traçabilité

Voir le [catalogue commun](../../bibliography/catalogue.json), le [journal de recherche](../../bibliography/RECHERCHE.md) et le [tri des 40 références initiales](../../bibliography/TRI_BIBLIOGRAPHIE_ORIGINE.md). Une référence commune est reproduite à l’identique dans les bibliographies qui en ont besoin.

Aucune compilation BibTeX/LaTeX ni exécution scientifique n’a été effectuée. Les champs optionnels non confirmés ont été omis; les particularités de métadonnées sont documentées dans le catalogue et dans [BIBLIOGRAPHIE.md](../../BIBLIOGRAPHIE.md).
