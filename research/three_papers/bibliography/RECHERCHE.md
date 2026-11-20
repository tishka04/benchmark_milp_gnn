# Recherche bibliographique — 5 novembre 2026

## Périmètre et résultat

La recherche porte sur les trois projets de manuscrit, leur protocole commun et leurs questions scientifiques : (1) criticité, stress physique et difficulté de résolution ; (2) méthodes apprises pour le MILP énergétique, faisabilité, délai et généralisation ; (3) complémentarité et choix d'une méthode avant résolution.

Le corpus comporte **101 références distinctes**, réparties en **47, 75 et 42 entrées** dans les trois bibliographies. Les recouvrements sont volontaires : une même référence peut soutenir plusieurs papiers. Les 40 entrées de [MILP-GNN.bib](../../../pdfs/MILP-GNN.bib) ont été examinées ; 25 sont reprises, avec une notice parfois remplacée par la publication définitive, et 15 sont laissées hors du périmètre actuel. Les 76 autres références complètent ce fichier de départ. Le [relevé de sélection](TRI_BIBLIOGRAPHIE_ORIGINE.md) rend ces choix explicites.

Il s'agit d'une **recherche étendue et documentée**, arrêtée au 5 novembre 2026. Ce travail ne constitue pas une revue systématique certifiant l'exhaustivité : aucune base bibliométrique complète n'a été exportée, aucun protocole PRISMA n'a été appliqué et tous les textes intégraux n'ont pas été lus.

## Démarche

1. Lire les trois manuscrits, les six références initialement partagées et les 40 notices du fichier fourni ; identifier les notions et les comparateurs réellement mobilisés.
2. Rechercher les fondements, les synthèses et les travaux directement concurrents, puis élargir aux méthodes de validation nécessaires au protocole.
3. Compléter les recherches par titres, auteurs et DOI, suivre les références pertinentes des synthèses et rechercher spécifiquement les publications 2025–2026.
4. Confronter les notices d'éditeurs et de conférences aux dépôts des auteurs ; préférer la publication définitive lorsqu'elle est identifiée et accessible.
5. Dédupliquer les versions, attribuer les références aux papiers et annoter leur rôle ainsi que les limites du rapprochement.

Les titres, résumés et notices primaires ont servi au filtrage ; plusieurs textes intégraux ont été consultés pour préciser une méthode ou une information bibliographique. L'annotation de pertinence ne prétend pas remplacer une lecture intégrale critique avant rédaction.

## Familles de requêtes

Le tableau consigne des requêtes représentatives effectivement utilisées ; les reprises par titre exact, auteur et DOI ne sont pas toutes reproduites.

| Axe | Exemples de requêtes |
| --- | --- |
| Formulations et apprentissage UC | `"unit commitment" "machine learning" "review" 2025 2026` ; `"Learning to Solve Large-Scale Security-Constrained Unit Commitment Problems" Xavier Qiu Ahmed` ; `"On Mixed-Integer Programming Formulations for the Unit Commitment Problem" Knueven` |
| Stress physique et flexibilité | `"Evaluation of Power System Flexibility" Lannoye Flynn O Malley 2012 doi` ; `"Analyzing operational flexibility of electric power systems" Ulbig Andersson 2015` ; `"Multi-temporal assessment of power system flexibility requirement" Heggarty 2019` |
| Difficulté empirique et indice | `"Measuring instance difficulty" "Smith-Miles" Lopes 2012` ; `"Learning the empirical hardness of optimization problems" Leyton-Brown Nudelman Shoham` ; `"Handbook on Constructing Composite Indicators" 2008 OECD` |
| Évaluation qualité–temps | `"Benchmarking optimization software with performance profiles" Dolan More 2002` ; `"Measuring the impact of primal heuristics" Berthold 2013` ; `"Best practices for comparing optimization algorithms" Beiranvand Hare Lucet 2017` ; `"MIPLIB 2017" Gleixner 2021` |
| Guidage neuronal des MILP | Titres exacts de Gasse, Nair, Sonnerat et Chen ; `"Machine learning for combinatorial optimization: A methodological tour"` ; `"Combinatorial Optimization and Reasoning with Graph Neural Networks"` |
| UC récent et concurrents directs | `"unit commitment" "graph neural" "2026"` ; `"Learning to Fix: Optimisation-Aware Machine Learning for Accelerated Unit Commitment"` ; `"Unit Commitment Predictor With a Performance Guarantee"` ; `"Fast Unit Commitment Constraint Screening with Learning-Based Cost Model"` ; `"Data-augmentation acceleration framework by graph neural network for near-optimal unit commitment"` |
| UC et renforcement | `"Reinforcement learning on the EPRI AI-ccelerating Unit Commitment benchmark"` ; `"Applying reinforcement learning and tree search to the unit commitment problem"` |
| Architecture et généralisation | Titres exacts de I-JEPA, VICReg, Set Transformer, GRU, Transformer, GATv2 et HGT ; `"From Local Structures to Size Generalization in Graph Neural Networks"` ; `"Learning TSP Requires Rethinking Generalization"` ; `"MILP-Evolve"` |
| Énergie et échantillonnage | Titres exacts du tutoriel de LeCun, de Du et Mordatch, de DISCS, d'Annealed Training et de l'optimisation géométrique d'Alcazar ; recherche du fondement Langevin de Welling et Teh |
| Apprentissage orienté décision | Titres exacts de Donti, Wilder, Elmachtoub et Grigas, de la synthèse de Mandi et de Feasibility-Aware Decision-Focused Learning ; recherche du papier Smart Predict-Then-Optimize pour l'UC |
| Sélection et coût du choix | `"The Algorithm Selection Problem" Rice 1976` ; `"Algorithm portfolios" Gomes Selman 2001` ; titres exacts de SATzilla, ASlib, AutoFolio, SUNNY et ISAC ; `"algorithm selection" "survival analysis" Tornede` |
| Sélection récente sous budget | `"Automated Algorithm Selection: from Feature-Based to Feature-Free Approaches"` ; `"Automatic algorithm selection for Pseudo-Boolean optimization with given computational time limits"` ; `"algorithm selection" "2025" "2026" "benchmark"` |
| Validation et calibration | Titres exacts de Cawley–Talbot, Roberts et al., Kapoor–Narayanan, Gneiting–Raftery, Guo et al., Saito–Rehmsmeier et WILDS |
| Réparation et logiciels | Titres exacts de Local Branching, RINS et Feasibility Pump ; `"HiGHS" "Citing" "Huangfu"` ; citation officielle de Gurobi et de UnitCommitment.jl |

## Sources et provenance

Les liens exacts sont conservés pour chaque entrée dans les commentaires des fichiers BibTeX et dans [catalogue.json](catalogue.json). Les familles de sources utilisées sont :

- Actes officiels : [PMLR](https://proceedings.mlr.press/), [NeurIPS](https://proceedings.neurips.cc/), [OpenReview](https://openreview.net/), [CVF](https://openaccess.thecvf.com/), [ACL Anthology](https://aclanthology.org/) et [IJCAI](https://www.ijcai.org/proceedings/).
- Revues et éditeurs : IEEE, INFORMS, Elsevier, Springer, JAIR et les pages DOI correspondantes.
- Dépôts d'auteurs et établissements : notamment arXiv, TUM, DTU, HKUST, Georgia Tech et Cardiff, lorsque la page éditeur ne donne pas les métadonnées complètes.
- Documentation officielle des logiciels pour les références HiGHS, Gurobi et UnitCommitment.jl.

L'accès direct à certaines pages d'éditeurs a échoué. Des notices primaires indexées par le moteur et des dépôts institutionnels ont alors permis les recoupements. Les champs non confirmés sont omis ou signalés, sans attribuer à une notice inaccessible une vérification qui n'a pas eu lieu.

## Critères de sélection

Une référence est retenue si elle étaye directement une question du manuscrit, un comparateur, une composante réellement pertinente de l'architecture, une méthode de validation ou une limite de portée. Les guides distinguent les lectures prioritaires des compléments.

Les domaines voisins sont conservés seulement lorsque le transfert est explicite : AC-OPF n'est pas UC, TSP n'est pas MILP énergétique, apprentissage des paramètres d'une décision sous incertitude n'est pas génération de décisions pour une instance connue, et un portefeuille ordonnancé n'est pas le choix préalable d'une seule action.

La recherche ne traite pas toute la planification énergétique, tous les marchés électriques, tous les modèles graphiques ni toutes les méthodes d'optimisation. Les références d'investissement, d'équilibre de marché et d'applications éloignées ne sont pas ajoutées uniquement parce qu'elles emploient un MILP ou un GNN.

## Versions, dates et doublons

- Une seule notice est conservée par travail et par papier. La publication en revue ou en conférence remplace le préprint correspondant quand elle est identifiée ; les titres alternatifs ne créent pas de références supplémentaires.
- L'année désigne la version citée. L'année du fascicule peut différer de celle du DOI ou de la première mise en ligne.
- Les huit notices de type prépublication sont explicitement signalées. Ce statut décrit la version retenue ; il ne prouve pas l'absence de toute publication ultérieure.
- Les auteurs sont nommés dans les champs BibTeX ; « et al. » n'est pas utilisé comme substitut à une liste disponible.
- Les notes de métadonnées documentent les exceptions, notamment la notice IEEE de Liu et al. héritée du fichier fourni, quelques champs facultatifs omis et les conflits de titre ou d'année.

Le corpus inclut notamment les travaux 2025–2026 de Pezo et al. sur la sélection sous limite de temps, Wei et al. sur l'augmentation de données UC, les études UC de Xu, Wang, Dong et Wu, ainsi que les prépublications récentes Learning to Fix et EPRI. Leur présence ne vaut ni validation indépendante de leurs performances ni preuve de nouveauté des trois manuscrits.

## Limites d'utilisation

Les références soutiennent le positionnement et la rédaction ; elles ne valident aucun résultat local. Le stress physique ne doit pas être présenté comme une cause universelle de difficulté algorithmique. Les performances d'une méthode apprise doivent conserver leurs coûts de génération, réparation et sortie complète. La sélection d'algorithmes doit démontrer une complémentarité exploitable au-delà du meilleur choix fixe.

La recherche couvre principalement les publications scientifiques en anglais repérables par les sources et requêtes ci-dessus. Des travaux non indexés, des publications très récentes, des variantes terminologiques ou des textes non accessibles peuvent manquer. Une affirmation précise de nouveauté exigera une lecture détaillée des concurrents les plus proches au moment de la soumission.
