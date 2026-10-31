# Relancer le pipeline load shifting

Le point d'entrée est `scripts/run_load_shifting_pipeline.ps1` (PowerShell 5.1
ou 7). Il appelle l'environnement `.venv` du dépôt et utilise les scénarios
existants. Le script et les modules ont été préparés sans exécuter les
résolutions, entraînements ou inférences.

## Commandes

Depuis la racine du dépôt :

```powershell
# Afficher les étapes, sans créer de fichiers ni lancer Python
.\scripts\run_load_shifting_pipeline.ps1 -ListSteps

# Campagne complète : MILP train 4 x 4, comparaison HiGHS 16 threads
# K=1,2,5,8,12,16 ; budgets séparés standard / JEPA / chaque comparaison
.\scripts\run_load_shifting_pipeline.ps1 `
    -TrainMilpWorkers 4 `
    -TrainMilpThreads 4 `
    -HighsThreads 16 `
    -InferenceK 5 `
    -GlobalTimeBudgetSeconds 120 `
    -JepaTimeBudgetSeconds 120 `
    -ParallelBudgetSeconds 120

# Reprendre à la première étape inachevée, avec les paramètres sauvegardés
# Si l'étape 2 est interrompue : conserver les MILP valides déjà terminés
# Si l'étape 10 est interrompue : conserver les inférences déjà terminées
.\scripts\run_load_shifting_pipeline.ps1 -Resume

# Recommencer à la génération des embeddings, en conservant le HTE entraîné
.\scripts\run_load_shifting_pipeline.ps1 -StartStep 5

# Exécuter uniquement la comparaison parallèle et ses métriques
# (les modèles, labels et références doivent déjà avoir été produits)
.\scripts\run_load_shifting_pipeline.ps1 -StartStep 12

# Recalculer uniquement ses tableaux
.\scripts\run_load_shifting_pipeline.ps1 -StartStep 13

# S'arrêter après l'entraînement du Latent EBM
.\scripts\run_load_shifting_pipeline.ps1 -EndStep 9
```

`-Resume` et `-StartStep` sont exclusifs. `-EndStep` est inclusif. Une nouvelle
invocation sans ces options refuse de redémarrer une campagne déjà enregistrée.
L'état est écrit avant chaque étape et après son succès, dans
`outputs/load_shifting_v1/_pipeline/state.json`. Chaque invocation écrit un
journal `run-<date>.log` dans ce même dossier. Un verrou empêche deux instances
du script d'écrire simultanément les sorties.

## Étapes

| Numéro | Action |
| --- | --- |
| 1 | Copier les scénarios train/test ; vérifier les horizons et chemins des configurations |
| 2 | Recalculer les MILP et classer les rapports d'entraînement gold/silver |
| 3 | Reconstruire les graphes temporels |
| 4 | Entraîner le HTE |
| 5 | Générer les embeddings train/test avec le meilleur checkpoint HTE |
| 6 | Extraire les labels UC et vérifier la couverture des familles de test |
| 7 | Entraîner l'autoencodeur UC |
| 8 | Entraîner JEPA |
| 9 | Entraîner le Latent EBM |
| 10 | Inférence standard, sélection par énergie, et baseline JEPA + LP |
| 11 | Métriques standard |
| 12 | Best-of-K : batch CUDA puis réparations CPU concurrentes ; baseline MILP HiGHS |
| 13 | Métriques de la comparaison parallèle |

Les familles sont `scenarios_v3` pour l'entraînement, les trois familles de
criticité et les trois familles persona pour le test. Les configurations UC
doivent pointer sur `outputs/load_shifting_v1` et utiliser 24 pas de temps.
Une conversion incomplète des graphes ou des labels de test arrête la campagne.
Le pipeline ne supprime pas silencieusement des scénarios difficiles du test.

À l'étape 2, les 5 000 scénarios d'entraînement utilisent par défaut
**4 processus × 4 threads HiGHS** (`-TrainMilpWorkers 4 -TrainMilpThreads 4`).
Les familles de test restent séquentielles, avec **16 threads par solveur**
pour la commande ci-dessus. Ces ressources sont indépendantes de celles
de l'entraînement ; `-HighsThreads 16` reste aussi le réglage de la baseline
MILP de comparaison finale. Les bibliothèques BLAS/OpenMP sont limitées à un
thread pendant chaque batch de références, puis leur environnement et celui
de HiGHS sont restaurés, y compris en cas d'erreur.

Le batch ne maintient qu'au plus `workers` scénarios soumis à la fois.
Avec `--fail-fast`, la première erreur reçue empêche toute nouvelle soumission,
annule les tâches en attente et termine les workers encore actifs, sans attendre
leurs limites de résolution. Le résumé partiel indique les échecs et le nombre
de scénarios sans résultat collecté ; le processus sort avec un code non nul,
ce qui arrête le lanceur. Chaque nouveau rapport est écrit dans un fichier
temporaire, puis remplacé atomiquement après réussite du MIP, de la relaxation
LP et de l'export. Un fichier temporaire laissé par un worker arrêté est ignoré.
Le contrôle d'arrêt utilise l'API publique de Python 3.14 lorsqu'elle existe,
et les processus de l'exécuteur sur Python 3.11, qui ne fournit pas cette API.

Le temps réel est suivi séparément des sommes de temps des solveurs :

- Chaque `batch_summary.json` contient `elapsed_wall_seconds`, mesuré dans
  le processus parent : préparation du batch, démarrage des workers, MIP,
  relaxation LP, exports, validation des rapports repris et arrêt des workers.
  Ce compteur porte uniquement sur l'invocation courante. Le démarrage initial de Python
  et ses imports sont hors de ce compteur, mais inclus dans celui de l'étape.
- `solve_seconds_total` et `lp_solve_seconds_total` additionnent les temps
  des solutions disponibles, y compris celles réutilisées. Les compteurs
  `executed_solve_seconds_total`, `executed_lp_solve_seconds_total` et
  `scenario_wall_seconds_total` ne portent que sur les résultats collectés
  pendant l'invocation courante, sans les workers interrompus. Ce sont des
  sommes, pas des durées du batch. Chaque ancien résumé de batch est archivé.
- Le journal affiche les scénarios complétés et le temps écoulé, avec un point
  d'attente toutes les 30 secondes si aucun worker ne retourne de résultat.
- `state.json` conserve `stage_timings`, une liste des tentatives de chaque étape
  avec début, fin, statut et `elapsed_wall_seconds`, y compris après une erreur
  interceptée. La reprise conserve cet historique ; l'attente entre deux
  invocations n'est pas du temps de calcul. Un arrêt brutal de la machine ne
  permet pas d'enregistrer la fin de la tentative en cours.

Les deux paramètres d'entraînement sont sauvegardés et repris avec `-Resume`.
Pour un état ancien sans ces paramètres, une reprise avant ou pendant l'étape 2
adopte 4 × 4. Si l'étape 2 était déjà terminée, son ancienne allocation est
conservée comme provenance ; pour recalculer ces références en 4 × 4, utiliser
`-StartStep 2 -TrainMilpWorkers 4 -TrainMilpThreads 4`.

Avec **`-Resume` à l'étape 2**, chaque famille train/test est inspectée : seuls
les scénarios manquants, échoués, incomplets ou incompatibles sont recalculés.
Les rapports valides restent en place, sans réécriture. Une solution réalisable
obtenue à la limite de temps (`maxTimeLimit`, silver) est conservée ; une limite
de temps sans incumbent exploitable est un échec à retenter. Une famille déjà
complète ne lance aucun solveur ni worker, même si son résumé de batch manque.

Les nouveaux rapports comportent un marqueur de fin et des empreintes du
scénario, du contenu du rapport et des sources de formulation/chargement/résolution.
La formulation load shifting, le solveur, son nombre de threads et sa limite
de temps doivent correspondre. Le MIP et le LP doivent être exploitables, et
les trajectoires DR doivent respecter les contrôles existants sur l'horizon complet.
Les anciens rapports sans ces métadonnées sont recalculés, car leur compatibilité
ne peut pas être vérifiée. Le nombre de processus n'entre pas dans l'empreinte
individuelle ; ses paramètres restent contrôlés par l'état de campagne.

Le journal affiche les rapports réutilisés puis la progression globale.
Le résumé distingue `reused_count`, `executed_count` (résultats de tentatives
collectés cette fois), `new_success_count`, `failure_count` et
`scenarios_unfinished`. La reprise repose sur les rapports individuels, ce qui
préserve aussi un rapport terminé juste avant l'arrêt du processus parent.
Les dossiers de classement gold/silver/bronze et leur index sont archivés puis
reconstruits après validation de toutes les familles, pour éviter les copies obsolètes.
Pour forcer un recalcul complet des références et des étapes suivantes, utiliser
**`-StartStep 2`** : cette commande archive toujours les anciens rapports.
Le module Python expose aussi `--resume --reports-dir ...` pour les batches de
rapports JSON seuls ; la reprise des exports CSV/HDF et des figures n'est pas prise en charge.

Une étape d'entraînement interrompue recommence à l'époque 1. Les étapes 3 à 6 et 10
recommencent aussi leurs calculs lorsqu'elles sont réexécutées. Leurs anciens
dossiers de sorties et ceux des entraînements sont déplacés dans
`outputs/load_shifting_v1/_archive/<date>/` avant remplacement. Les scénarios
d'origine et leurs anciens résultats hors de `load_shifting_v1` sont conservés.
Repartir d'une étape amont invalide la progression des étapes suivantes.

Pour une reprise après une erreur **au sein de l'étape 12**, le module conserve
chaque expérience terminée `(famille, scénario, K)` ainsi que la baseline MILP.
Seul un groupe interrompu est recommencé, génération GPU comprise. Les empreintes des données,
références, checkpoints et sources, les versions et les réglages doivent être
identiques pour réutiliser ces résultats. Après modification d'un modèle ou du
protocole, utiliser de nouvelles sorties ou repartir des étapes correspondantes.
`-StartStep 12` archive l'ancienne comparaison et la recommence ; `-Resume`
conserve ses groupes terminés. Le protocole `gpu_batch_cpu_repairs_v1` empêche
de mélanger les anciennes expériences entièrement CPU avec celles sur GPU.

Les checkpoints de l'étape 12 sont désormais vidés sur disque avant leur
publication atomique, munis d'une empreinte de contrôle, et accompagnés d'une
copie `.json.bak` de la sauvegarde précédente. Les verrous temporaires Windows
sont retentés pendant au plus 2 s pour ces écritures/lectures du parent, hors
fenêtre de décision. Un fichier JSON vide, rempli d'octets nuls, tronqué ou dont
l'empreinte ne correspond plus est conservé sous un nom `.corrupt-...`. La
reprise restaure la copie précédente si elle est valide ; sinon, seul le
scénario concerné est recalculé. Les scénarios intacts et leurs échecs/timeout
déjà mesurés sont conservés. Les sorties brutes des workers ne sont jamais
récupérées après coup, faute de preuve de réception avant l'échéance.

La révision précisément identifiée du lanceur antérieure à cette correction
des sauvegardes est autorisée à la reprise. Tous les autres contrôles de
compatibilité restent stricts. Le manifeste d'origine est conservé, les
reprises sont tracées dans `resumes/`, et le calcul des candidats, leur sélection
et leurs budgets restent identiques. Aucun résultat n'est recalculé par la
seule installation de cette correction ; il faut lancer `-Resume`.

## Comparaison parallèle demandée dans le commentaire de Thibault

Source : commentaire page 28, transcrit dans
`docs/review_audio_2026-09-14/Commentaires_Thibault_extraits.txt`. Il propose
plusieurs exécutions réellement concurrentes, sans diviser le budget de
120 secondes par K.

Le script fixe explicitement `P = -HighsThreads`. Sans valeur explicite, P est
le nombre de processeurs logiques retourné par PowerShell à la création de la
campagne ; cette valeur est ensuite sauvegardée. P est un **nombre de threads
autorisé**, pas une mesure de l'utilisation effective de chaque cœur par HiGHS.
Le script ne prétend pas retrouver le parallélisme d'une ancienne expérience.

- La baseline MILP utilise HiGHS avec `threads=P`, `parallel=on` si P > 1.
  Les références de test de l'étape 2 utilisent également ce réglage ; celles
  d'entraînement utilisent les paramètres `TrainMilpWorkers`/`TrainMilpThreads`.
  La limite
  existante des références reste 600 s par appel au solveur, pour le MIP puis
  sa relaxation LP ; cette étape n'est pas la baseline à budget court.
- À l'étape 12, **chaque K est exécuté séparément**, avec la grille par défaut
  **K=1,2,5,8,12,16**. Seules les valeurs ne dépassant pas P sont retenues.
  Un processus charge une seule copie des modèles sur `cuda:0`, prédit l'ancre
  JEPA, exécute les K chaînes Latent EBM dans **un batch**, puis décode les K
  candidats dans un batch. CUDA est obligatoire, sans repli silencieux sur CPU.
- Après transfert des candidats en mémoire CPU et fermeture du processus GPU,
  K processus de réparation/dispatch sont lancés avant d'attendre leurs résultats.
  Chacun dispose d'un thread PyTorch/BLAS et d'un thread HiGHS. Les modèles neuronaux
  ne sont pas rechargés pour les réparations ; le GPU leur est masqué.
  La cascade de réparation reste celle de l'évaluation standard, avec au plus
  trois étapes et les mêmes contrôles de load shifting.
- Le meilleur objectif parmi les dispatchs faisables retournés à temps est
  retenu. Cette sélection diffère de la sélection par énergie de l'étape 10.
  La graine GPU est enregistrée par scénario ; les candidats du batch reçoivent
  des tirages de bruit distincts. Les expériences de tailles différentes ne
  sont pas des préfixes identiques d'un même pool. Chaque réparation possède
  également une graine HiGHS enregistrée.
- Le budget mural commun est 120 s par scénario et méthode, configurable par
  `-ParallelBudgetSeconds`. Il commence avant le lancement du premier
  processus GPU et n'est **jamais divisé par K**. Les réparations disposent
  chacune du temps restant jusqu'à la même échéance absolue, sans remise à zéro.
  Par exemple, si génération et démarrage consomment 8 s sur 120 s, les K
  réparations disposent chacune d'au plus 112 s en parallèle.
  La baseline utilise la même
  fenêtre, dans une exécution séparée pour éviter la concurrence avec le pipeline.
- Le chronométrage inclut démarrage Python, imports, chargement des modèles,
  transferts GPU/CPU, construction du problème, calcul et réception du résultat. Les embeddings
  HTE précalculés et les entraînements restent hors de cette fenêtre.
  Il s'agit donc d'une comparaison avec **démarrage à froid et embeddings
  précalculés**, pas d'une mesure d'un service entièrement préchargé ni du
  temps complet depuis la construction du graphe.
- Un processus encore actif à l'échéance est arrêté et compté sans solution
  retournée à temps. Le nettoyage est chronométré séparément. Un incumbent
  interne non retourné avant l'échéance n'est pas comptabilisé.
- Les erreurs de programme/import/fichier arrêtent l'expérience ; elles ne
  deviennent pas artificiellement des échecs de faisabilité.

Pour P=16, la grille par défaut représente 6 batchs GPU et jusqu'à 44 réparations
par scénario, en groupes concurrents de tailles 1, 2, 5, 8, 12 et 16, plus la
baseline MILP. Pour 600 scénarios et 120 s par expérience, cela représente
jusqu'à **120 h de budget pipeline + 20 h de baseline HiGHS = 140 h**, hors
écritures et nettoyage. Une grille explicite reste possible :

```powershell
.\scripts\run_load_shifting_pipeline.ps1 -HighsThreads 16 -BestOfK 1,2,5,8,12,16
```

Une reprise conserve la grille sauvegardée. Pour remplacer celle d'une ancienne
campagne à 16 valeurs, utiliser `-StartStep 12 -BestOfK 1,2,5,8,12,16` ; les
anciens résultats de comparaison seront archivés avant le recalcul.

Les poids sont chargés une fois par batch ; la mémoire GPU des activations
augmente avec K. La RAM CPU augmente aussi avec les K problèmes de réparation.
Un manque de mémoire arrête l'expérience et permet une reprise après correction ;
le script ne réduit pas K automatiquement. Cette comparaison mesure le pipeline
sur **un GPU et jusqu'à P threads CPU**, face à HiGHS sur **P threads CPU** :
le budget mural est commun, mais les ressources matérielles ne sont pas identiques.
Les threads sont des plafonds configurés ; les temps mesurés sont conservés.
Le temps CPU inclut les processus terminés de génération et de réparation.
Celui des processus tués est inconnu et signalé comme incomplet.

Les sorties de l'étape 13 sont dans
`outputs/load_shifting_v1/evaluation/parallel_best_of_k/` :

- `manifest.json` : protocole, matériel, versions, empreintes et cohorte ;
- `cases/<famille>/<scenario>.json` : baseline et résultats de chaque K ;
- `jobs/` : journaux de génération/réparation et batchs de candidats, utiles en cas d'erreur ;
- `best_of_k_parallel_per_scenario.csv` : coût, faisabilité, temps total et par phase,
  temps du batch synchronisé (calcul et transfert CPU), nom du GPU, mémoire GPU,
  nombre de candidats produits/réparations terminées et écarts ;
- `best_of_k_parallel_summary.csv` et `.json` : synthèses par famille et K.

La faisabilité est calculée sur **tous** les scénarios. Les écarts de coûts sont
fournis à la fois sur les paires faisables pipeline/MILP pour chaque K et sur
l'intersection faisable commune à tous les K et au MILP ; les effectifs sont
explicites. Les écarts visent l'incumbent de la baseline sous budget, pas un
optimum supposé. Les coûts des références longues restent disponibles avec
leur statut. Un coût de référence nul n'engendre pas de pourcentage artificiel :
la différence de coût absolue est aussi exportée.

## Inférence standard et environnement

Les étapes 10/11 conservent l'évaluation standard à K=5, configurable via
`-InferenceK`. Ses candidats sont séquentiels, avec sélection par énergie.
L'étape 10 appelle `scripts/run_strict_standard_inference.py` :

- `-GlobalTimeBudgetSeconds 120` fixe désormais une limite **externe stricte**
  par scénario pour le pipeline standard. La valeur par défaut est 120 ;
  zéro est refusé.
- `-JepaTimeBudgetSeconds 120` donne à JEPA + réparation/dispatch une fenêtre
  **indépendante** de 120 s. Elle commence dans un nouveau processus après
  l'exécution standard, même si celle-ci a épuisé son budget. La baseline
  ne fait pas d'échantillonnage EBM.
- Chaque fenêtre inclut démarrage Python, imports, chargement des modèles,
  génération/décodage, construction du problème et réparations jusqu'à la
  réception du résultat. Les embeddings HTE précalculés sont hors budget.
  Ce sont donc des mesures avec démarrage à froid.
- Le processus est tué à l'échéance. Les derniers résultats de candidats
  réparés **déjà reçus** restent disponibles ; la sélection par énergie se
  fait sur ce sous-ensemble. Un candidat encore en cours ou reçu trop tard
  ne compte pas. Sans résultat, le scénario est conservé comme échec dans
  les métriques, pour chacune des deux méthodes.
- Les sorties enregistrent la limite, `hard_timeout`, le nombre de candidats
  complétés et les temps muraux. L'arrêt du processus et le retard éventuel
  de détection dû à l'ordonnanceur sont mesurés séparément ; aucun résultat
  reçu après l'échéance n'est accepté. Le temps réel de la campagne comprend
  ce nettoyage et les écritures de résultats.
- Un accès temporairement refusé par Windows à un snapshot ou au fichier
  de fin est retenté dans la boucle de surveillance, pendant au plus 2 s
  et toujours dans la fenêtre initiale de 120 s. Le dernier snapshot déjà
  reçu reste disponible ; un snapshot plus ancien encore non lu peut aussi
  être retenu pendant que le plus récent est verrouillé. La fin du worker
  ne valide pas prématurément ce résultat ancien si le dernier reste à lire.
  À l'échéance, aucun fichier supplémentaire n'est lu. Un refus persistant
  au-delà des 2 s arrête explicitement la campagne ; les JSON corrompus et
  les identités incohérentes restent des erreurs. `file_access_retries`
  conserve le nombre de tentatives de lecture refusées dans les résultats.

Le protocole standard est `strict_standard_separate_v1`. Pour une ancienne
campagne arrêtée avant ou pendant l'étape 10, `-Resume` adopte ces nouveaux
réglages (l'ancien budget nul devient 120 s). Si les anciennes inférences
ont déjà été réalisées, repartir avec `-StartStep 10` ; aucun entraînement
n'est nécessaire pour ce seul changement de protocole temporel.

À l'étape 10, `-Resume` conserve maintenant les paires pipeline/JEPA présentes
dans `cases/<famille>/<scénario>.json`, après contrôle du manifeste, des
identités et des budgets. Un résultat infaisable ou un timeout déjà enregistré
reste un résultat terminé : il n'est pas rééchantillonné à la reprise. Les
scénarios manquants, dont celui interrompu par une erreur du worker, sont repris.
Les données, modèles, paramètres et sources doivent être identiques ; seule
la révision précisément identifiée du lanceur antérieure à cette correction
des chemins/reprises est également acceptée. Le manifeste d'origine reste
intact et chaque reprise enregistre sa provenance dans `resumes/`.

Les nouveaux workers utilisent des chemins courts `jobs/0401/pipeline/a001/`
(index du scénario dans le manifeste), afin d'éviter la limite Windows de
260 caractères lors des exports de familles. Le nom complet et le scénario
restent dans `job.json`. Chaque méthode terminée possède son propre checkpoint,
ce qui permet aussi de garder le pipeline si JEPA plante ensuite. Les anciennes
tentatives restent intactes ; les retries utilisent `a002/`, etc. Les snapshots
d'une tentative interrompue sans checkpoint du parent ne sont jamais récupérés
après coup, car leur réception avant l'échéance n'est plus vérifiable.
`-StartStep 10` reste un redémarrage explicite complet avec archivage.

Les budgets standard et JEPA ne modifient pas celui des étapes 12/13 :
`-ParallelBudgetSeconds 120` reste une fenêtre commune au batch GPU et aux
réparations CPU de chaque K, sans division par K.

Le lanceur charge pandas avant PyTorch, conformément au contournement local du
conflit de DLL PyArrow/PyTorch. Il n'installe ni ne modifie les dépendances.
`-PythonExe` permet de désigner un autre environnement compatible.

Les contrôles réalisés lors de la préparation portent sur la syntaxe PowerShell,
les blocs Python, l'assemblage du batch avec des modèles simulés et des tests
à processus simulés (arrêt des workers, soumission bornée, chronométrage et
restauration des threads compris). Ils ne constituent pas une
validation numérique des futurs entraînements ou benchmarks.
