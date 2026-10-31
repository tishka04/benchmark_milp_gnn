# Comparaison à sortie effective — operational_exit_v1

## État de livraison

Cette livraison prépare le protocole. L'inventaire et les configurations sont figés à l'étape 1. Les tests utilisent uniquement des données artificielles et de petits MILP/LP synthétiques. Aucun des 5 000 labels, des 600 nouveaux calculs de référence, des entraînements ou des pilotes réels n'est lancé pendant cette préparation.

L'ancienne campagne `outputs/thermal_transitions_v1/reference_criticality/state.json` déclare 600 références terminées. Leur présence ne vaut pas certification pour ce protocole : chaque réutilisation sera contrôlée. Les résultats et fichiers historiques restent à leur emplacement.

Tout le nouveau travail va dans `outputs/thermal_transitions_v1/operational_exit_v1`. Une modification du code, des configurations sources, de l'environnement ou des scénarios après préparation entraîne un refus de reprise. Utiliser un nouveau `-OutputDir` pour un protocole modifié ; ne pas supprimer les échecs pour les rejouer.

## Éléments conservés et adaptations

| Élément existant | Utilisation dans le nouveau protocole |
| --- | --- |
| `src/milp/scenario_loader.py`, `src/milp/model.py`, `src/thermal.py` | Chargement et modèle thermique corrigé, sans modification de ces fichiers pour cette livraison. |
| `src/milp/lp_worker_two_stage.py` | Même cascade, limitée aux trois premiers étages, avec échéance absolue commune et extraction du dispatch complet. Certaines décisions restent entières : ce calcul peut être un MILP. |
| `src/heuristics/rolling_horizon.py` | Candidat RH existant, suivi de la même cascade de réparation. |
| Encodeur HTE contrastif et sa normalisation | Réutilisation conditionnelle : mêmes tableaux d'entrée, arêtes, types et hiérarchie, même empreinte du checkpoint. Aucune étiquette n'entre dans le HTE. |
| Architectures et configurations AE / JEPA / Latent EBM | Architectures et pertes actuelles conservées. Nouvelles initialisations et nouveaux labels ; plafonds de 60 / 50 / 20 époques. |
| Trois entraîneurs `src/uc_jepa/training/train_*.py` | Ajout d'un mode optionnel : partitions figées, RTX 4050 obligatoire, accumulation et reprise complète. Les configurations historiques gardent leur comportement. |
| `src/operational_exit/` | Inventaire, coût/validation indépendants, supervision, génération K16, exécution, checkpoints et analyse. |
| `scripts/run_operational_exit.ps1` | Point d'entrée dédié, progression et temps réellement écoulés. |

Les modèles UC conservent leurs huit canaux. L'interface historique de réparation consomme les sept canaux zonaux ; le mode d'import/export demeure une décision du modèle de réparation. Cette règle est identique pour RH, JEPA et le pipeline complet.

## Audit physique et coût

`physics.validate_dispatch` recalcule les équations depuis les paramètres et les valeurs du dispatch, sans utiliser les résidus ou l'objectif retournés par le solveur. Tolérance absolue physique : `1e-3` ; intégralité : `1e-5`. L'absence d'une variable, une valeur non finie, un axe incohérent ou une fraction binaire hors tolérance invalide le résultat.

Les contrôles couvrent : bilan de chaque zone et chaque pas ; capacités et minimums ; transitions thermiques exactes, état initial et rampes de fonctionnement/démarrage/arrêt ; nucléaire ; production et écrêtement renouvelables ; DR par blocs, activation, limite de périodes, rampe, récupération causale, dette énergétique et remboursement terminal ; batteries et STEP, rendements, rétention, états initiaux/terminaux et exclusivité ; réservoirs ; flux et capacités des lignes ; exclusivité import/export ; capacités nulles.

Le périmètre physique existant est conservé : réseau de transport à capacités, sans ajout de lois de Kirchhoff, N−1, durées minimales de marche/arrêt ou autres contraintes. Les conventions existantes de réservoir hydroélectrique et de conditions terminales sont reproduites dans le validateur ; aucune nouvelle contrainte terminale hydro n'est introduite.

Deux adaptations nécessaires ont été isolées dans le nouveau protocole :

1. Le coût historique additionne des puissances. `apply_cost_units` multiplie les termes énergétiques par `dt_hours`, conserve les coûts de démarrage par événement et conserve séparément les pénalités de réparation. Il traite aussi l'objectif actif des étages de réparation souple. À un pas horaire, le coût physique est identique. Le coût indépendant inclut combustible, démarrages, DR, ENS, écrêtements, cyclage et échanges signés ; la sélection finale exclut les pénalités de proximité au décodeur.
2. Les nœuds de zone du graphe HTE sont ordonnés lexicalement, tandis que les dispatchs utilisent l'ordre des zones du scénario. Les embeddings sont explicitement réordonnés avant apprentissage et inférence. Cela ne change ni les entrées ni les poids du HTE.

L'ENS est autorisée et mesurée en MWh. Elle n'est jamais assimilée à une violation des équations. Le rapport distingue couverture avec ENS permise, couverture sans ENS (`ENS <= 1e-3 MWh`) et quantité d'ENS.

Avant les labels, l'étape 2 doit valider douze scénarios réels, deux par famille couvrant les tailles extrêmes. Pour chacun : référence corrigée compatible ou nouveau MILP, validation complète, fixation de toutes les décisions binaires, résolution LP et comparaison des coûts. Tout écart inexpliqué arrête la chaîne. Cet audit réel reste à exécuter.

## Données, labels et apprentissage

L'inventaire reprend les identifiants de la cohorte historique, sans reprendre ses anciens coûts : 600 cas, exactement 100 dans chacune des six familles. Il exige les 5 000 cas `scenarios_v3`, l'unicité des UUID et aucune intersection entraînement/évaluation par UUID ou empreinte de fichier. L'horizon reste de 24 pas.

La partition historique est reconstruite depuis l'ordre trié des anciens fichiers UC, avant toute nouvelle classification gold/silver, avec `torch.randperm(seed=42)` et 10 % de validation. La partition préparée contient 4 500 identifiants d'entraînement et 500 de validation. Elle reste attachée aux identifiants, même si un nouveau label change de classe ou est exclu. Les 600 cas d'évaluation sont exclus des trois apprentissages et de la calibration.

Les labels sont recalculés avec quatre processus de quatre threads HiGHS, 600 secondes **solveur** par scénario. Les autres bibliothèques sont limitées à un thread. La relaxation LP additionnelle est désactivée par défaut ; `-WithLabelRelaxation` permet de l'activer dès la préparation et doit ensuite rester identique à chaque reprise. Les anciens producteurs/consommateurs de rapports ne sont pas modifiés.

Chaque rapport contient statut, incumbent si disponible, borne/gap lorsqu'ils existent, validation, ENS et temps. Un calcul terminé sans incumbent est conservé et exclu des labels : il n'est pas rejoué pour améliorer la sélection. Une interruption d'infrastructure reste incomplète et peut être reprise. Un incumbent qui échoue à la validation physique interrompt la production pour diagnostic.

Les références utilisent 600 secondes solveur et 16 threads. La réutilisation est limitée aux références corrigées dont scénario, sources physiques, versions HiGHS/Pyomo, ressources, durée et contrôles d'intégrité coïncident. Le dispatch complet est revalidé et son coût recomputé ; l'équivalence du coût historique exige `dt_hours=1`. Sinon, une nouvelle référence est calculée. « Optimal » conserve le sens du statut du solveur, à ses tolérances ; un incumbent limité en temps n'est pas présenté comme un optimum.

Les nouveaux labels valides deviennent gold si HiGHS termine optimal, sinon silver. Leur partition reste celle de l'identifiant. Les embeddings du nouvel ensemble sont reconstruits avec le HTE conditionnellement réutilisé et l'alignement des zones corrigé. L'index final et chaque fichier sont empreintés ; les dépendances AE/JEPA sont ensuite figées avant l'étape suivante.

AE, JEPA et EBM exigent CUDA sur la RTX 4050 ; il n'existe aucun repli CPU. Avant la première époque, le plus grand exemple de l'ensemble interne train/validation est répété pour tester les micro-lots 8, 4, 2 puis 1, sans mise à jour des poids. L'état aléatoire est restauré après cette sonde. Les moments Adam sont réservés pendant ce test. Le micro-lot retenu et le pic mémoire sont enregistrés ; une accumulation conserve huit exemples par mise à jour, sauf dernier lot incomplet. Les termes de perte dépendant des statistiques du lot restent calculés par micro-lot : cette accumulation n'affirme pas une identité numérique avec un lot physique de huit exemples.

Après chaque époque, `resume.pt` et son engagement `resume.json` conservent modèle, optimiseur, scheduler, époque, historique, meilleur score, compteur d'arrêt anticipé, seuils, micro-lot et RNG Python/NumPy/Torch/CUDA. L'EBM conserve aussi son prior latent. Une copie de la dernière époque engagée permet la récupération après une écriture interrompue. Un premier checkpoint sans engagement est conservé comme non engagé et l'apprentissage reprend depuis l'initialisation déterministe. Les seuils JEPA sont ajustés sur validation interne ; l'EBM hérite des nouveaux seuils, sans reprendre les anciens seuils manuels.

## Horloge et ressources

Chaque couple scénario–méthode–budget est une exécution indépendante. Les budgets sont `1, 2, 5, 10, 30, 60, 120` secondes. Il n'existe ni poursuite d'un budget précédent, ni récupération d'un snapshot tardif.

| Méthode | Génération | Réparation / solveur |
| --- | --- | --- |
| HiGHS | Modèle construit depuis le scénario | 16 threads, modèle initial vierge |
| Pipeline | HTE + JEPA + Langevin K=16 sur GPU, flux aléatoires indépendants | 16 processus, 1 thread HiGHS chacun |
| JEPA direct | HTE + JEPA + décodage sur GPU, un candidat | 1 processus, 16 threads |
| RH | Heuristique rolling horizon, un candidat | 1 processus, 16 threads |

Les bibliothèques, poids, contexte CUDA et processus de réparation sont préchargés sans identifiant, fichier, données, graphe, label ou embedding du scénario. Le signal READY termine le préchargement. L'horloge haute résolution démarre immédiatement avant l'envoi du scénario au processus principal. Elle inclut lecture et empreinte du scénario, graphe et HTE en ligne, génération, transfert, construction/réparation, sélection au plus faible coût physique valide, validation finale, écriture et fermeture de **tous** les processus.

Un objet Job Windows contient l'interpréteur et tous ses descendants. Le lanceur indirect de l'environnement virtuel est évité pour que son propre mécanisme de fermeture ne tue pas silencieusement un descendant. Le superviseur exige simultanément un Job vide et la sortie du processus principal. Un fichier déjà écrit ne termine pas l'horloge. Un résultat écrit alors qu'un descendant reste actif échoue.

La limite coopérative de calcul est l'échéance globale moins la réserve de finalisation. Le budget n'est pas divisé par K ; tous les candidats partagent cette échéance. Les messages tardifs peuvent être vidés pour fermer les canaux, mais ne peuvent plus remplacer le candidat sélectionné. À l'échéance, le superviseur tue l'ensemble encore actif, confirme sa disparition et enregistre un échec, même si un fichier résultat existe. Les temps de préchargement, calcul, finalisation, sortie et dépassement observé sont séparés.

L'observation de sortie se fait par sondage conservateur à environ 1 ms : une sortie observée après l'échéance échoue. Les interruptions forcées et les ralentissements de fermeture ne sont pas soustraits du délai.

## Calibration, ordre et répétitions

Trente identifiants de validation interne sont sélectionnés par quantiles de taille, y compris les extrêmes. Le pilote exécute chaque méthode avec 120 secondes et une réserve provisoire de 30 secondes. Il doit confirmer la génération K16, les seize processus préchargés, une observation mémoire GPU et une finalisation naturelle, sans erreur d'implémentation/validation. Une erreur de pilote reste enregistrée ; elle ne doit pas être effacée pour recalibrer favorablement.

La réserve commune est ensuite figée : `max(0.050 s, 2 × maximum des finalisations observées)`. Elle peut rendre certains très petits budgets inutilisables ; ce sera un résultat du protocole. Les empreintes des modèles et des seuils sont enregistrées dans `calibration.json`. Elles sont vérifiées avant comparaison et à chaque préchargement. Le code/configuration est également vérifié à chaque nouvelle expérience. Une recalibration après démarrage de l'évaluation est interdite.

La comparaison principale comporte `600 × 7 × 4 = 16 800` expériences. Les familles sont entrelacées ; des rotations équilibrent les positions des méthodes et des budgets. Une seule expérience est active à la fois, sans concurrence GPU/solveur entre méthodes. La graine dépend de l'identifiant et de la répétition ; les graines de candidat ajoutent son index et sont indépendantes du budget.

Trente cas d'évaluation, cinq par famille couvrant les tailles, reçoivent deux répétitions supplémentaires : `30 × 7 × 4 × 2 = 1 680`. Leurs trois répétitions restent distinctes ; aucune sélection au meilleur résultat. Les sommes des budgets sont respectivement **152 h** et **15,2 h**, et ne constituent pas une estimation du temps total : préchargements, labels, références, apprentissages et pilote s'y ajoutent ; les exécutions peuvent aussi sortir avant leur budget.

## Reprise et fichiers

Les checkpoints portent une identité complète et une empreinte SHA-256. Une copie du même résultat est publiée avant le fichier principal ; en cas d'interruption entre les deux écritures, la copie terminale prime sur un ancien marqueur incomplet. Les copies corrompues sont mises à part. Une erreur d'accès ne déclenche pas une nouvelle expérience.

Un résultat terminal est réutilisé qu'il soit un succès, une absence d'incumbent, une invalidité, une erreur de worker ou un arrêt forcé. Seule une tentative sans résultat terminal du superviseur est rejouable comme interruption externe/d'infrastructure. Son dossier est conservé, et son éventuel fichier brut ne devient jamais un succès après coup.

| Chemin sous la sortie | Contenu |
| --- | --- |
| `manifest.json`, `configs/` | Identités, sources, versions, partitions, sous-ensembles et configurations préparées |
| `audit.json` | Porte d'entrée physique des douze cas réels |
| `labels/`, `references/` | Rapports indépendants et sauvegardes |
| `dataset/`, `dataset.json` | Nouveaux labels/embeddings certifiés et exclusions |
| `training/` | Modèles, historiques, sonde mémoire et états complets de reprise |
| `calibration_runs/`, `calibration.json` | Observations de validation et réserve figée |
| `experiments/`, `experiments_attempts/` | Résultat terminal par expérience ; journaux et fichiers bruts des tentatives |
| `state.json` | État des étapes et durées réelles de chaque invocation |
| `analysis/` | Rapport HTML, neuf figures PNG/SVG, CSV et contrôles d'acceptation |

Le verrou partagé empêche une exécution concurrente avec la campagne historique compatible ; un verrou propre à la nouvelle sortie empêche deux lanceurs sur le même dossier. L'option `-AnalyzeOnly` ne lance aucun worker.

## Analyse

Les courbes par famille gardent un dénominateur de 100, même dans un rapport partiel. Les figures séparent couverture exploitable, couverture sans ENS, ENS totale, couverture avec gap au plus 1/5/10 %, coûts individuels face à la référence, distributions de gaps sur support commun aux quatre méthodes, répétitions et dépassements de délai.

Le gap est `100 × (coût − référence) / abs(référence)`. Les gaps négatifs restent présents. Une référence nulle donne seulement une différence absolue ; une référence sans incumbent valide exclut le gap mais ne supprime pas le succès de la couverture. Les comparaisons communes et les six paires de méthodes sont exportées avec leurs supports. Les répétitions et causes d'échec restent identifiables dans les CSV. L'analyse revalide les dispatchs réussis et recalcule leurs coûts ; elle refuse un succès tardif ou non reproductible. Un ensemble de moins de 18 480 résultats terminaux porte explicitement la mention « INCOMPLET ».

## Étapes et commandes PowerShell

| Étape | Travail |
| --- | --- |
| 1 | Préparer inventaire, partitions et configurations |
| 2 | Audit physique réel sur douze cas |
| 3 | Produire les 5 000 labels |
| 4 | Certifier/réutiliser ou calculer les 600 références |
| 5 | Vérifier HTE et produire le nouvel ensemble |
| 6–8 | Entraîner AE, JEPA puis Latent EBM |
| 9 | Pilote et calibration internes |
| 10 | Comparaison principale |
| 11 | Répétitions supplémentaires |
| 12 | Analyse et rapport |

Depuis PowerShell :

```powershell
Set-Location 'C:\Users\coudr\OneDrive\projects\benchmark\benchmark_milp_gnn'

# Aperçu en lecture seule, sans résolution ni écriture.
.\scripts\run_operational_exit.ps1 -Preview

# Préparation seule, déjà effectuée dans cette livraison.
.\scripts\run_operational_exit.ps1 -Resume -StartStep 1 -EndStep 1

# Lancement futur de toute la chaîne après préparation.
.\scripts\run_operational_exit.ps1 -Resume -StartStep 2 -EndStep 12

# Reprise globale après interruption : chaque résultat terminal est conservé.
.\scripts\run_operational_exit.ps1 -Resume

# Exemples d'exécution par étapes.
.\scripts\run_operational_exit.ps1 -Resume -StartStep 2 -EndStep 2
.\scripts\run_operational_exit.ps1 -Resume -StartStep 3 -EndStep 5
.\scripts\run_operational_exit.ps1 -Resume -StartStep 6 -EndStep 9
.\scripts\run_operational_exit.ps1 -Resume -StartStep 10 -EndStep 11

# Régénérer seulement l'analyse, y compris d'une campagne partielle.
.\scripts\run_operational_exit.ps1 -AnalyzeOnly
```

Une nouvelle variante doit avoir un autre dossier. Exemple de préparation avec relaxation LP additionnelle :

```powershell
.\scripts\run_operational_exit.ps1 -EndStep 1 -WithLabelRelaxation `
  -OutputDir 'outputs/thermal_transitions_v1/operational_exit_v1_with_lp'
```

Les tests synthétiques reproductibles sont :

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_operational_exit.py' -v
.\tests\test_operational_exit_launcher.ps1
```

Les observations de préparation ne démontrent ni la capacité K16 sur les plus grands graphes réels, ni une couverture opérationnelle, ni une accélération. Les étapes physiques, mémoire et pilote sont précisément les contrôles à exécuter avant la comparaison complète.
