# Comparaison à sortie effective — operational_exit_v1

## État de livraison

Cette livraison prépare le protocole. L'inventaire et les configurations sont figés à l'étape 1. Les tests utilisent uniquement des données artificielles et de petits MILP/LP synthétiques. Aucun des 5 000 labels, des 600 nouveaux calculs de référence, des entraînements ou des pilotes réels n'est lancé pendant cette préparation.

L'ancienne campagne `outputs/thermal_transitions_v1/reference_criticality/state.json` déclare 600 références terminées. Leur présence ne vaut pas certification pour ce protocole : chaque réutilisation sera contrôlée. Les résultats et fichiers historiques restent à leur emplacement.

Tout le nouveau travail va dans `outputs/thermal_transitions_v1/operational_exit_v1`. Une modification du code, des configurations sources, de l'environnement ou des scénarios après préparation entraîne un refus de reprise. Utiliser un nouveau `-OutputDir` pour un protocole modifié ; ne pas supprimer les échecs pour les rejouer.

Exception documentée : la réparation HTE du 2 novembre 2026 ci-dessous autorise uniquement les adaptations nécessaires pour reproduire l'encodage météo historique. Elle conserve le manifeste initial et les identités des résultats physiques, avec un reçu séparé des sources corrigées.

La réparation d'écriture Windows du 3 novembre 2026, décrite en fin de document, conserve également les étapes 1 à 8 et le premier pilote échoué. Elle ouvre une révision distincte du pilote avant toute calibration achevée ou évaluation, sans modifier les modèles ni les limites de temps.

La correction d'intégralité du 4 novembre 2026 rétablit les domaines binaires d'origine dans la cascade de réparation. Elle conserve les modèles, le validateur et les pilotes antérieurs ; la calibration repart avec une révision explicite du code corrigé.

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

Les embeddings produits à l'étape 5 servent à l'apprentissage. Pour chaque essai d'inférence JEPA ou pipeline, `Generator.candidates` rappelle `OnlineHTE.encode` après le démarrage de l'horloge : graphe, normalisation, passage HTE et synchronisation CUDA sont recalculés dans le budget. Un seul embedding du scénario est partagé entre les 16 candidats du pipeline au sein de cet essai. Le préchargement statique reste dans `preload_s`, séparé du budget `exit_s` ; ce dernier mesure donc une réponse avec modèles déjà chargés, pas un démarrage à froid.

Un objet Job Windows contient l'interpréteur et tous ses descendants. Le lanceur indirect de l'environnement virtuel est évité pour que son propre mécanisme de fermeture ne tue pas silencieusement un descendant. Le superviseur exige simultanément un Job vide et la sortie du processus principal. Un fichier déjà écrit ne termine pas l'horloge. Un résultat écrit alors qu'un descendant reste actif échoue.

La limite coopérative de calcul est l'échéance globale moins la réserve de finalisation. Le budget n'est pas divisé par K ; tous les candidats partagent cette échéance. Les messages tardifs peuvent être vidés pour fermer les canaux, mais ne peuvent plus remplacer le candidat sélectionné. À l'échéance, le superviseur tue l'ensemble encore actif, confirme sa disparition et enregistre un échec, même si un fichier résultat existe. Les temps de préchargement, calcul, finalisation, sortie et dépassement observé sont séparés.

L'observation de sortie se fait par sondage conservateur à environ 1 ms : une sortie observée après l'échéance échoue. Les interruptions forcées et les ralentissements de fermeture ne sont pas soustraits du délai.

## Calibration, ordre et répétitions

Trente identifiants de validation interne sont sélectionnés par quantiles de taille, y compris les extrêmes. Le pilote exécute chaque méthode avec 120 secondes et une réserve provisoire de 30 secondes. Un dépassement (`forced_timeout`, `late_exit`) ou un arrêt coopératif sans résultat (`cooperative_deadline`) est compté comme un échec, puis le pilote poursuit les essais suivants. La disparition de tous les processus et le préchargement K16 restent contrôlés. Sur les essais disposant d'un résultat normal, la génération requise et la mesure mémoire GPU sont vérifiées. Les erreurs d'implémentation/validation restent bloquantes. Aucun échec terminé ne doit être effacé ou rejoué pour obtenir un succès.

La réserve commune est ensuite figée : `max(0.050 s, 2 × maximum des finalisations observées)`. Les sorties forcées ou tardives sans mesure exploitable restent inconnues et ne sont pas remplacées par zéro. Si aucune finalisation n'est observable, la réserve provisoire de 30 secondes est conservée, sans prétendre qu'elle a été mesurée. Les nombres d'essais par statut et de finalisations observées/inconnues figurent dans `calibration.json`. La réserve peut rendre certains très petits budgets inutilisables ; ce sera un résultat du protocole. Les empreintes des modèles et des seuils sont également enregistrées, puis vérifiées avant comparaison et à chaque préchargement. Le code/configuration est vérifié à chaque nouvelle expérience. Une recalibration après démarrage de l'évaluation est interdite.

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
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_operational_embedding_timing.py' -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_operational_pilot_io.py' -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_operational_repair_integrality.py' -v
.\tests\test_operational_exit_launcher.ps1
```

Le test de chronométrage des embeddings utilise le superviseur, le processus principal et le générateur réels, avec modèles CPU et réparations synthétiques. Pour chacune des deux méthodes apprises, un délai injecté dans `HTE.encode` doit apparaître dans les temps de calcul et de sortie ; un délai dépassant le budget doit provoquer un arrêt forcé et un échec. Ce test ne charge aucun checkpoint et ne lance ni calcul GPU ni résolution.

Les observations de préparation ne démontrent ni la capacité K16 sur les plus grands graphes réels, ni une couverture opérationnelle, ni une accélération. Les étapes physiques, mémoire et pilote sont précisément les contrôles à exécuter avant la comparaison complète.

## Reprise après le refus de compatibilité HTE du 2 novembre 2026

Le constructeur historique utilise `hash(weather_profile) % 100` pour la deuxième caractéristique des nœuds météo. Le hachage Python des chaînes dépend du processus : reconstruire le même scénario dans un autre processus change donc cette colonne. Fixer une nouvelle graine de hachage ne reproduirait pas nécessairement celle qui a servi à entraîner le HTE.

`hte_compatibility.py` retrouve la correspondance profil → code dans les seuls graphes d'entraînement historiques. Il exige une valeur unique par profil dans l'ensemble du corpus et à tous les instants. Il fige cette table dans `hte_feature_contract.json`, associée aux empreintes du HTE, du manifeste, des scénarios et des graphes examinés. Un profil inconnu ou des codes historiques contradictoires provoquent un refus explicite.

Les graphes reconstruits doivent retrouver exactement les caractéristiques effectivement consommées par le HTE, les arêtes, leurs types, les identifiants de nœud et les correspondances hiérarchiques. La distance des lignes est une métadonnée auxiliaire historique : ni `TemporalGraphDatasetV3`, ni l'adaptateur en mémoire, ni le HTE ne la lisent. Seules les différences sur cette métadonnée sont admises ; les capacités de ligne et les autres attributs restent contrôlés.

La table fixe est réutilisée pour reconstruire les graphes en ligne depuis chaque scénario. Elle ne contient ni label, ni embedding, ni graphe d'évaluation. Les poids et la normalisation du HTE sont conservés. Le contrat d'entrée fait aussi partie des empreintes figées avant le pilote et la comparaison.

Pour la campagne existante, `runtime_amendment.json` consigne exactement les sources corrigées et l'empreinte du contrat. Le manifeste d'origine, l'audit physique, les 5 000 rapports de labels et les 600 références restent en place avec leurs identités initiales. L'enregistrement est interdit après production d'embeddings, entraînement ou évaluation ; toute modification extérieure aux fichiers de cette réparation est refusée. Pour une campagne préparée avec le code corrigé, le contrat est produit au début de l'étape 5.

La commande de réparation, exécutée une seule fois sur la campagne existante, ne lance ni MILP, ni apprentissage, ni évaluation :

```powershell
.\.venv\Scripts\python.exe -B -u -m src.operational_exit.hte_compatibility
```

Une fois le reçu enregistré, la reprise reste :

```powershell
.\scripts\run_operational_exit.ps1 -Resume -StartStep 5 -EndStep 12
```

## Reprise après l'échec d'écriture du pilote du 3 novembre 2026

Le premier essai HiGHS du pilote (`scenario_04790`) s'est terminé avec une erreur d'écriture, après environ 1,23 seconde et sans arrêt forcé. Le journal indiquait `FileNotFoundError` dans `atomic_json` : le résultat avait un chemin de 247 caractères, mais l'ajout du suffixe temporaire le portait à 284 caractères, au-delà de la limite classique Windows. Les chemins longs étaient désactivés sur la machine. Le message de calibration masquait cette cause par l'absence de mesure de finalisation.

Les accès JSON atomiques utilisent désormais la syntaxe Windows de chemin étendu, y compris pour le fichier temporaire, le remplacement atomique et la récupération des copies. Le réglage système n'est pas modifié. L'écriture, sa synchronisation disque et la fermeture restent dans le même budget. Les erreurs du pilote indiquent désormais leur statut, le scénario, la méthode et le journal à consulter.

`pilot_io_amendment.json` lie le correctif aux sources précédentes, au manifeste, aux modèles entraînés, aux fichiers conservés et au journal de l'échec. L'enregistrement est limité à cette panne d'écriture, après les étapes 1 à 8, avant toute calibration achevée ou évaluation. Il refuse un succès, un dépassement de temps, une erreur d'une autre origine ou une modification de la physique, de l'apprentissage ou de l'inférence.

Le résultat échoué et son journal restent en place. Les nouveaux essais portent une identité `runtime_revision` différente : ils ne remplacent pas l'échec et aucune meilleure observation n'est choisie parmi des répétitions. La calibration enregistrée mentionne cette révision. Les étapes 1 à 8 ne sont pas rejouées.

Vérification locale du 3 novembre : l'essai HiGHS qui bloquait a ensuite enregistré un dispatch valide et la sortie de tous les processus en 1,794 seconde, après 10,298 secondes de préchargement. Cette seule observation du pilote corrigé est enregistrée ; elle ne constitue pas une calibration complète. Les 16 221 fichiers existants de labels, références, configurations, dataset et entraînements ont conservé leurs tailles et dates ; les empreintes du manifeste, des modèles et des fichiers d'identité protégés ont aussi été vérifiées inchangées.

Enregistrement du correctif, une seule fois :

```powershell
.\.venv\Scripts\python.exe -B -u -m src.operational_exit.pilot_io_repair
```

Puis reprise normale :

```powershell
.\scripts\run_operational_exit.ps1 -Resume -StartStep 9 -EndStep 12
```

## Réserve proportionnelle au budget et reprise de l'étape 10

La calibration du 5 novembre 2026 a mesuré une finalisation maximale de 19,2298 s, soit une réserve de 38,4596 s au budget pilote de 120 s. La soustraire telle quelle aux budgets de 1 à 30 s empêchait toute résolution. La réserve effective devient `min(B / 2, max(0.05, R120 * min(1, B / 120)))`, où `B` est le budget total et `R120` la réserve issue du pilote. Cette règle utilise les observations du pilote, sans ajustement sur les coûts d'évaluation, et laisse au moins la moitié de chaque budget au calcul. Elle ne garantit pas la finalisation : les dépassements restent des échecs.

Pour cette campagne, les réserves sont respectivement 0,3205 ; 0,6410 ; 1,6025 ; 3,2050 ; 9,6149 ; 19,2298 ; 38,4596 secondes aux budgets 1, 2, 5, 10, 30, 60, 120 secondes. Le chronométrage et le superviseur restent inchangés, avec embeddings en ligne et sortie complète de tous les processus inclus.

La migration s'effectue après arrêt de l'invocation active, sous les deux verrous de campagne. `budget_reserve_amendment.json` conserve les empreintes du pilote, des modèles et de tous les anciens résultats terminés. Les étapes 1 à 9 restent acquises. Les résultats à 120 s conservent exactement leurs identités et sont réutilisés, succès comme échecs. Tous les essais aux budgets modifiés reçoivent une nouvelle identité incluant la révision et la réserve effective : les anciens restent présents, mais sont exclus du rapport corrigé. Un essai interrompu est archivé avant reprise ; aucun résultat brut d'un processus arrêté n'est récupéré comme succès.

Enregistrement unique puis reprise habituelle :

```powershell
.\.venv\Scripts\python.exe -B -u -m src.operational_exit.budget_reserve
.\scripts\run_operational_exit.ps1 -Resume -StartStep 10 -EndStep 12
```

Les nouvelles campagnes enregistrent directement cette règle dans la calibration. L'analyse des étapes 10 et 11 suit les identités corrigées ; elle ne choisit jamais le meilleur résultat entre deux révisions.

## Correction de l'intégralité de la réparation du 4 novembre 2026

Le pilote pipeline de `scenario_04790` avait généré ses 16 candidats, puis terminé et écrit son résultat en 3,616 secondes. Les 16 réparations annonçaient un optimum du premier étage, mais le validateur indépendant refusait leur dispatch. La cascade historique relaxait tous les domaines entiers au premier étage, dont le mode import/export resté libre. La reproduction du premier candidat donne un résidu physique maximal de `7,5e-12`, mais un résidu d'intégralité de `0,4912` sur `import_mode`, avec des imports et exports simultanés.

L'adaptateur opérationnel restaure les domaines entiers d'origine avant toute résolution de réparation. Les décisions fixées par le décodeur restent fixées ; le mode import/export libre est résolu comme une décision binaire. Cette règle s'applique à la même cascade pour pipeline, JEPA et RH. Les contraintes physiques, seuils de validation, coûts, budgets, limites de threads et frontières du chronométrage restent inchangés. Les résidus et motifs du validateur sont désormais conservés pour chaque candidat dans `candidate_diagnostics`.

Sur le candidat reproduit, le dispatch corrigé passe le même validateur, avec un résidu d'intégralité nul. Les labels et modèles appris ne sont pas régénérés : leur formulation MILP imposait déjà les domaines binaires corrects. La correction concerne l'exécution de la réparation à l'inférence.

Vérification locale : les 225 tests passent. Sur le seul premier scénario du pilote, les quatre méthodes terminent avec un résultat valide et tous les processus fermés : HiGHS en 1,779 s, pipeline en 4,539 s, JEPA en 2,191 s et RH en 2,285 s, préchargement exclu selon le protocole. Les 16 réparations du pipeline sont valides. Ces quatre observations sont enregistrées et seront réutilisées à la reprise ; elles ne constituent pas une calibration complète. Les 16 232 fichiers préexistants contrôlés, dont les anciens pilotes, sont conservés inchangés, ainsi que les 34 empreintes vérifiées du manifeste, des modèles et des autres artefacts protégés.

`repair_integrality_amendment.json` lie le correctif au précédent reçu d'écriture, aux sources, aux modèles, aux diagnostics avant/après et à tous les pilotes antérieurs conservés. L'enregistrement est interdit après calibration achevée ou début d'évaluation. La nouvelle identité `runtime_revision` s'applique à toute la calibration des quatre méthodes ; aucune meilleure observation n'est sélectionnée entre les révisions.

La commande d'enregistrement vérifie les deux diagnostics et les artefacts conservés, sans relancer l'apprentissage :

```powershell
.\.venv\Scripts\python.exe -B -u -m src.operational_exit.repair_integrality
```

La reprise utilise ensuite la même commande des étapes 9 à 12.

## Conservation des décisions physiques fixes dans la réparation

Le pilote JEPA de `scenario_02050` échouait au deuxième étage de réparation avec une seule violation : `zero_capacity:b_charge_mode = 1`. Le modèle physique fixe ce mode à zéro lorsqu'une zone ne possède pas de batterie. La cascade historique libérait cette variable avec les autres décisions du sous-ensemble à réparer ; la pénalité d'écart au décodeur pouvait alors la ramener à un. L'intégralité était respectée, mais la décision restait interdite par le modèle physique et le validateur.

L'adaptateur mémorise désormais les décisions entières fixées lors de la construction du modèle, avant l'application du candidat. Il les rétablit après la libération d'un sous-ensemble et avant chaque résolution. Les cibles correspondantes sont ajustées à leur valeur physique avant la construction des pénalités et du budget de changements : une demande impossible sur un équipement absent ne consomme donc aucun changement. Les décisions proposées pour les équipements présents restent réparables. La règle couvre aussi le thermique absent, le pompage absent, la flexibilité indisponible à une période et les échanges extérieurs indisponibles.

Les modèles entraînés, les labels, les références, les contraintes physiques et les seuils du validateur restent identiques. Les embeddings calculés à l'inférence et la réparation restent à l'intérieur du chronométrage de décision existant.

Vérification du 4 novembre 2026 : les 231 tests passent. Le scénario bloquant `scenario_02050` passe avec les quatre méthodes, sortie effective de tous les processus comprise : HiGHS 54,010 s, pipeline 44,555 s, JEPA 15,969 s et RH 6,522 s (préchargement exclu selon le protocole). Les 16 candidats du pipeline sont valides. JEPA conserve une énergie non servie de 1 015,662 MWh, autorisée et mesurée par le protocole ; la correction du dispatch ne constitue donc pas une validation de sa qualité comparative. Ces quatre essais sont enregistrés pour la reprise, mais ne constituent pas la calibration complète. Les 16 276 fichiers préexistants contrôlés restent inchangés, ainsi que les 80 empreintes vérifiées du manifeste, des modèles et des résultats conservés.

Le reçu `repair_structural_fixes_amendment.json` lie cette correction aux reçus précédents et conserve tous les anciens pilotes, réussis ou échoués. Il exige un échec diagnostiqué de capacité nulle dans la réparation, les entraînements terminés et aucune calibration achevée ni évaluation commencée. La calibration des quatre méthodes utilise une nouvelle identité commune ; les observations des révisions antérieures ne sont pas mélangées avec les nouvelles.

Enregistrement du correctif pour cette campagne :

```powershell
.\.venv\Scripts\python.exe -B -u -m src.operational_exit.repair_structural_fixes
```

Après cet enregistrement, reprise sans réentraînement :

```powershell
.\scripts\run_operational_exit.ps1 -Resume -StartStep 9 -EndStep 12
```

## Correction du transfert de solution HiGHS sur les grands scénarios

Le pilote HiGHS de `scenario_02177` a atteint l'arrêt forcé à 120 secondes, après 76 essais réussis de la révision précédente. Son fichier brut a été inspecté uniquement pour le diagnostic : l'optimisation avait été interrompue après 85,93 secondes de solveur, mais la récupération des valeurs consommait encore une grande partie de la réserve. Le fichier présent ne rend pas cet essai réussi : le critère de sortie effective et son statut `forced_timeout` restent inchangés.

`highspy` construit une nouvelle liste Python à chaque accès à `solution.col_value`. Le transfert parcourait toutes les variables en relisant cette propriété pour chacune, ce qui copiait le vecteur complet autant de fois qu'il contenait de variables. Le correctif lit ce vecteur une fois avant la boucle, puis indexe la liste obtenue. Sur les 66 168 colonnes du modèle bloquant, un diagnostic utilisant le même vecteur synthétique mesure 27,553 secondes avant et 0,025 seconde après ; les valeurs transférées sont identiques. Ce diagnostic ne résout pas de MILP et ne constitue pas un résultat de comparaison.

`solution_transfer_amendment.json` conserve le manifeste, les modèles, les diagnostics et tous les pilotes antérieurs, y compris le dépassement. Les budgets, le superviseur, les équations physiques, les tolérances et les frontières du chronométrage ne changent pas. La nouvelle calibration des quatre méthodes utilise une révision commune du code corrigé ; aucune sortie brute d'un essai forcé n'est récupérée comme succès et aucune meilleure répétition n'est sélectionnée. Les étapes 1 à 8 restent réutilisées.

Enregistrement de cette correction avant reprise :

```powershell
.\.venv\Scripts\python.exe -B -u -m src.operational_exit.solution_transfer_repair
```

Les essais réels du code corrigé confirment une sortie HiGHS valide en 92,643 secondes sur `scenario_02177` et en 92,728 secondes sur `scenario_01777`, le plus grand scénario du pilote. Les quatre méthodes passent sur `scenario_02177`. Le pipeline sur `scenario_01777` atteint en revanche `forced_timeout` à 120,001 secondes : cet échec reste enregistré et son éventuel fichier brut ne compte pas comme résultat. Le contrôle ciblé s'est arrêté à cette observation ; JEPA et RH n'ont pas été lancés sur ce dernier cas lors de ce contrôle.

## Dépassements comptés comme échecs, sans bloquer la calibration

Le 5 novembre 2026, la règle a été explicitée : un programme qui n'a pas fourni un résultat valide et terminé à temps échoue ; le protocole doit poursuivre les autres essais. La calibration exigeait à tort une finalisation normale pour chaque observation. Elle conserve maintenant les dépassements avec leur statut d'échec et les inclut dans son décompte. `-Resume` réutilise tout enregistrement terminé, réussi ou échoué, sans répétition à la recherche d'un succès. Les étapes de comparaison appliquaient déjà cette règle de poursuite.

`calibration_outcomes_amendment.json` trace ce changement de comptabilisation. Les workers, solveurs, budgets et règles d'acceptation restent identiques à la révision d'exécution précédente. Cette modification ne crée donc pas de nouvelles identités d'essais : les cinq succès et l'échec du contrôle ciblé ci-dessus restent réutilisables. Les modèles et tous les anciens résultats sont conservés.

Validation : les 240 tests passent, notamment la poursuite après échecs, leur réutilisation à la reprise, et le cas où aucune finalisation n'a pu être observée. Une vérification sur l'enregistrement réel du pipeline de `scenario_01777` confirme sa réutilisation inchangée, sans lancement de worker ni récupération du fichier brut. Les 16 584 fichiers préexistants contrôlés et les 390 empreintes protégées restent inchangés. La calibration complète reste à terminer à la reprise.

```powershell
.\.venv\Scripts\python.exe -B -u -m src.operational_exit.calibration_outcomes
.\scripts\run_operational_exit.ps1 -Resume -StartStep 9 -EndStep 12
```
