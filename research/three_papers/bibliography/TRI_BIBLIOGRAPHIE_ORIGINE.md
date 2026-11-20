# Sélection depuis MILP-GNN.bib

Examen des **40 entrées** du [fichier fourni](../../../pdfs/MILP-GNN.bib), au 5 novembre 2026 : **25 reprises et 15 laissées hors du périmètre actuel**. Le fichier source est conservé intact. Une exclusion signifie une moindre pertinence pour ces trois projets, et non un jugement sur la qualité du travail.

Les références reprises peuvent changer de clé et de version pour citer la publication définitive. Les clés déjà utilisées par les trois manuscrits sont, elles, conservées. Les fichiers par papier ajoutent également 76 travaux absents de ce fichier de départ.

## Entrées reprises

| Clé initiale | Clé retenue | Papiers | Référence et usage |
| --- | --- | --- | --- |
| `yang_solving_nodate` | `yang2025lns` | 2 | [Solving Unit Commitment Problems with Graph Neural Network based Initial Commitment Prediction and Large Neighborhood Search](https://arxiv.org/abs/2505.14408) — Prédiction initiale, restauration de faisabilité puis voisinages appris; comparaison très proche de la chaîne candidats-réparation. |
| `zhang_graph_2024` | `zhang2024risk` | 1, 2 | [Graph neural networks for power grid operational risk assessment under evolving grid topology](https://arxiv.org/abs/2405.07343) — Prédiction du risque physique et topologies changeantes; distinguer prédiction de risque et prédiction du succès informatique. |
| `donon_graph_2019` | `donon2019` | 2 | [Graph Neural Solver for Power Systems](https://doi.org/10.1109/IJCNN.2019.8851855) — Transfert en taille pour calcul de flux; domaine adjacent, distinct de l’UC mixte entier et de la certification de faisabilité. |
| `chen_representing_2023` | `chen2023` | 2 | [On Representing Mixed-Integer Linear Programs by Graph Neural Networks](https://openreview.net/forum?id=4gc3MGZra1d) — Limites d’expressivité des GNN sur les MILP; empêcher d’inférer une garantie de résolution à partir de l’architecture. |
| `yang_stable_2025` | `yang2025fixation` | 2 | [Stable Variable Fixation for Accelerated Unit Commitment via Graph Neural Network and Linear Programming Hybrid Learning](https://doi.org/10.3390/app15084498) — Fixation stable et hybride GNN-LP; comparer explicitement la récupération de faisabilité. |
| `gasse_exact_2019` | `gasse2019` | 2 | [Exact Combinatorial Optimization with Graph Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html) — Branchement appris sur graphe biparti; distinguer guidage de recherche exacte et génération de candidats. |
| `qin_solve_2023` | `qin2025` | 2 | [Solve Large-Scale Unit Commitment Problems by Physics-Informed Graph Learning](https://doi.org/10.1109/TPWRS.2025.3562886) — Graphe physique hiérarchique, neural diving et branching pour UC; voisin scientifique central. |
| `park_confidence-aware_2023` | `park2024` | 1, 2 | [Confidence-Aware Graph Neural Networks for Learning Reliability Assessment Commitments](https://doi.org/10.1109/TPWRS.2023.3298735) — Confiance, fixation partielle et réparation de faisabilité sur un problème opérationnel de grande taille. |
| `liu_topology-aware_2023` | `liu2023` | 2 | [Topology-Aware Graph Neural Networks for Learning Feasible and Adaptive AC-OPF Solutions](https://doi.org/10.1109/TPWRS.2022.3230555) — Adaptation topologique et régularisation physique en AC-OPF; ne pas assimiler adaptation avec réentraînement et transfert figé. |
| `zeng_glinsat_2024` | `zeng2024` | 2 | [GLinSAT: The General Linear Satisfiability Neural Network Layer By Accelerated Gradient Descent](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dd73f39426a03131c38c8d943153d44b-Abstract-Conference.html) — Projection différentiable sous contraintes linéaires; distinguer satisfiabilité linéaire bornée et intégralité de la solution UC. |
| `sun_annealed_2022` | `sun2022` | 2 | [Annealed Training for Combinatorial Optimization on Graphs](https://arxiv.org/abs/2207.11542) — Lien EBM et optimisation combinatoire avec recuit; ne pas confondre énergie d’objectif et énergie latente apprise. |
| `goshvadi_discs_2023` | `goshvadi2023` | 2 | [DISCS: A Benchmark for Discrete Sampling](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f9ad87c1ebbae8a3555adb31dbcacf44-Abstract-Datasets_and_Benchmarks.html) — Comparaison contrôlée des échantillonneurs; le pipeline local échantillonne un latent continu, distinction à conserver. |
| `alcazar_enhancing_2024` | `alcazar2024` | 2 | [Enhancing combinatorial optimization with classical and quantum generative models](https://doi.org/10.1038/s41467-024-46959-5) — Génération de candidats au service de la recherche; retenir le cadre générique, sans revendiquer d’avantage quantique pour ce projet. |
| `pjetri_graph_nodate` | `pjetri2025` | 1, 2 | [Graph Neural Network Surrogates for Energy System Modeling](https://doi.org/10.1109/ISGTEurope64741.2025.11305443) — Prédiction rapide de sorties d’un modèle énergétique et identification de cas critiques; surrogate de planification, pas résolution UC complète sous délai. |
| `buchheim_polytope_2023` | `buchheim2023` | 1 | [The polytope of binary sequences with bounded variation](https://doi.org/10.1016/j.disopt.2023.100776) — Lecture polyédrale des transitions binaires; contexte pour des proxies de variabilité, sans assimiler ceux-ci à une mesure exacte de difficulté. |
| `huang_cutting_2021` | `huang2021` | 1, 2 | [Cutting planes for security-constrained unit commitment with regulation reserve](https://doi.org/10.1080/24725854.2020.1823533) — Lien entre réserves, structure polyédrale et difficulté de résolution; les contraintes du papier ne sont pas toutes celles du modèle local. |
| `morales_espana_tight_2013` | `moralesespana2013` | 1, 2, 3 | [Tight and Compact MILP Formulation for the Thermal Unit Commitment Problem](https://doi.org/10.1109/TPWRS.2013.2251373) — Formulation thermique et effet du resserrement sur la résolution; référence de modèle, pas identité avec la formulation locale. |
| `brody_how_2022` | `brody2022` | 2 | [How Attentive Are Graph Attention Networks?](https://openreview.net/forum?id=F72ximsx7C1) — Référence GATv2 pour l’attention de graphe; utiliser seulement pour les blocs effectivement présents. |
| `vaswani2017attention` | `vaswani2017` | 2 | [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) — Attention et encodage positionnel; ne constitue pas une garantie de transfert vers de nouveaux horizons. |
| `wu_smart_2026` | `wu2026` | 2 | [Smart Predict-Then-Optimize-Based Unit Commitment for Integrated Energy Systems](https://doi.org/10.1109/TII.2026.3671325) — Application énergétique récente de SPO à des prévisions; différencier prévision des entrées et génération d’une stratégie. |
| `donti_task_2017` | `donti2017` | 2 | [Task-based End-to-end Model Learning in Stochastic Optimization](https://proceedings.neurips.cc/paper/2017/file/3fc2c60b5782f641f76bcefc39fb2392-Paper.pdf) — Aligner prédiction et coût de décision, avec applications énergétiques; distinct de l’imitation de stratégies connues. |
| `wilder_melding_2019` | `wilder2019` | 2 | [Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization](https://doi.org/10.1609/aaai.v33i01.33011658) — Apprentissage à travers une relaxation de problème combinatoire; cadre voisin à distinguer du générateur latent local. |
| `elmachtoub_smart_2022` | `elmachtoub2022` | 2, 3 | [Smart ``Predict, then Optimize''](https://doi.org/10.1287/mnsc.2020.3922) — Évaluer l’effet aval d’une prédiction; ne pas présenter le routage ou JEPA comme une implémentation de SPO. |
| `mandi_decision_2024` | `mandi2024` | 2 | [Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities](https://doi.org/10.1613/jair.1.15320) — Revue et benchmark pour distinguer apprentissage de paramètres, de décisions et de politiques de résolution. |
| `mandi_feasibility_2025` | `mandi2025` | 2 | [Feasibility-Aware Decision-Focused Learning for Predicting Parameters in the Constraints](https://arxiv.org/abs/2510.04951) — Tension entre qualité et faisabilité lorsque les contraintes contiennent des paramètres prédits; contexte différent de la réparation sous paramètres connus. |

## Entrées non reprises

| Clé initiale | Motif au regard des manuscrits actuels |
| --- | --- |
| `xia_scalable_2025` | Compensation séquentielle multi-niveau de marchés couplés, avec gaz et carbone : mécanisme de marché distinct du benchmark UC et de ses quatre actions. |
| `hobbs_strategic_2000` | Comportements stratégiques et pouvoir de marché formulés comme MPEC ; hors du problème de résolution UC sous délai. |
| `conejo_investment_2016` | Ouvrage sur l'investissement et l'expansion de production et de transport ; périmètre de planification distinct de l'exploitation étudiée. |
| `micheli_two-stage_2023` | Planification stochastique d'expansion à deux étapes ; ne fournit pas le comparateur opérationnel du protocole. |
| `owerko_optimal_2019` | Application à l'AC-OPF ; domaine adjacent déjà représenté par Donon et Liu, sans en faire un concurrent UC discret. |
| `deihim_initial_2024` | Prédiction de points initiaux pour l'AC-OPF continu ; transfert éventuel, mais hors du noyau UC/MILP des trois papiers. |
| `ahmed_advancing_2024` | Restauration de charge après perturbation : autre tâche de décision énergétique, non évaluée dans ces manuscrits. |
| `marquant_holarchic_2017` | Conception énergétique multi-échelle et agrégation de MILP ; cette hiérarchie ne correspond pas à l'encodeur neuronal ni au benchmark UC étudié. |
| `yokoyama_hierarchical_2021` | Conception robuste et optimisation hiérarchique ; le problème et la décomposition ne sont pas ceux du protocole. |
| `pozo_three-level_2013` | Planification d'expansion à trois niveaux ; hors du périmètre opérationnel sous délai. |
| `talebi_graph_2025` | Surrogate graphique de calcul de flux AC ; domaine continu adjacent, sans nécessité supplémentaire pour le positionnement actuel. |
| `mancilla_non-convex_2026` | Application de modèles d'énergie au suivi d'indice financier ; les fondements EBM et les travaux d'optimisation combinatoire plus proches sont privilégiés. |
| `li_structured_2021` | Apprentissage par renforcement coopératif avec actions variables ; composante non établie dans les méthodes étudiées. |
| `oord2018representation` | Contrastive Predictive Coding : ne pas le présenter comme fondement de l'implémentation sans établir que sa perte contrastive est effectivement utilisée. |
| `loshchilov2019decoupled` | Référence AdamW utile si la section d'entraînement en décrit précisément l'usage ; non indispensable au positionnement comparatif actuel. Cette exclusion ne signifie pas qu'AdamW est absent du code. |

## Précautions de réemploi

Les références AC-OPF, réseaux électriques et apprentissage orienté décision conservées sont annotées comme voisines, avec leurs différences de problème. La présence de JEPA, VICReg, Set Transformer, GRU ou GNN dans le corpus sert à établir la provenance des idées pertinentes ; elle ne remplace pas la vérification exacte des mécanismes et des pertes utilisés par le modèle local.

Les principales mises à jour concernent les versions publiées de Qin, Park et Pjetri, les titres définitifs et les statuts de prépublication. Les exceptions de métadonnées figurent dans [la synthèse bibliographique](../BIBLIOGRAPHIE.md) et dans les commentaires des notices.
