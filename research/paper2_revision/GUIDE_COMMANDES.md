# Commandes PowerShell — papier 2

Lancer depuis le dépôt. Aucun bloc coûteux ci-dessous n'a été exécuté pour la
préparation du papier. Les méthodes historiques ne sont pas relancées.

```powershell
$Project = 'C:\Users\coudr\OneDrive\projects\benchmark\benchmark_milp_gnn'
$P2 = Join-Path $Project 'research\paper2_revision\run_paper2.ps1'
$ErrorActionPreference = 'Stop'

& $P2 -Stage Check
& $P2 -Stage AnalyzeExisting
& $P2 -Stage Estimate
```

`Check` utilise des petits cas CPU et ne lance aucun entraînement. Le test
Gurobi est ignoré tant que son module ou sa licence ne sont pas disponibles.
Les analyses historiques partent de l'extraction déjà validée, avec vérification
de son sceau et de son lien au gel des 18 480 observations.

## Préparer les entrées et le développement

```powershell
& $P2 -Stage Prepare
& $P2 -Stage ReferencesDevelopment
& $P2 -Stage Development
```

`Prepare` génère et audite les cohortes mais ne résout aucun scénario.
L'audit des milliers d'entrées d'apprentissage peut prendre plusieurs minutes.
`ReferencesDevelopment` : 330 références (120 cas indépendants et 210 contrôles).
`Development` : 7 200 essais, comprenant quatre méthodes, ablation appariée et
contrôles structurels. Exécuter une seule campagne à la fois sur cette machine.

## Modèles multi-horizons

```powershell
& $P2 -Stage TrainingLabels
& $P2 -Stage TrainMultiHorizon
& $P2 -Stage ValidateMultiHorizon
```

`TrainingLabels` : 540 références, puis embeddings d'apprentissage sur GPU.
`TrainMultiHorizon` : AE/JEPA/EBM pour chacune des trois graines, avec reprise
de l'état d'optimisation ; plafonds 60/50/20 époques. `ValidateMultiHorizon` :
3 150 essais sur les contrôles de développement, pour la graine principale.
La RTX 4050 et les versions de l'environnement historique sont requises ; aucun
repli CPU silencieux n'est permis pour les modèles appris.

## Verrouiller et évaluer une seule fois

```powershell
& $P2 -Stage Lock
& $P2 -Stage ReferencesFinal
& $P2 -Stage Final
```

`Lock` vérifie les étapes terminées et fige les trois chaînes apprises.
`ReferencesFinal` : 330 références. `Final` : 12 150 essais au total :
7 200 pour la comparaison/ablation/structure, 3 150 multi-horizons,
1 080 pour les deux graines supplémentaires et 720 répétitions supplémentaires.
Un échec observé n'est pas relancé pour obtenir un meilleur résultat.

## Gurobi

Après installation de `gurobipy` compatible avec Python 3.11 et activation d'une
licence autorisant les tailles étudiées, sans mise à jour des autres dépendances :

```powershell
& $P2 -Stage Check
& $P2 -Stage Gurobi
```

1 200 essais : 120 cas historiques sélectionnés à l'avance et 120 cas finaux,
aux cinq délais. L'étape vérifie la licence avant de lancer les essais et
conserve les références communes HiGHS. Un problème Gurobi laisse les autres
blocs utilisables. Le contrôle synthétique de licence ne garantit pas une
licence couvrant toutes les grandes instances.

## Analyser et mettre à jour le papier

```powershell
& $P2 -Stage Analyze
& $P2 -Stage Manuscript
```

L'analyse conserve les inconnus et les échecs, les observations appariées et les
graines séparées. Les tables d'effets, les 25 tests corrigés et les graphiques
sont intégrés automatiquement. Une relecture scientifique du texte accompagne
l'intégration définitive de nouvelles expériences ; les conclusions actuelles
portent sur la campagne historique.

Pour seulement reconstruire le draft actuel après `AnalyzeExisting` :

```powershell
& $P2 -Stage Manuscript
```

La compilation cherche Tectonic, puis latexmk. Un Tectonic déjà présent dans les
outils temporaires du dépôt est aussi reconnu. Sans compilateur, les sources,
figures et l'archive de reproduction restent disponibles, avec statut explicite.
Le PDF produit doit être inspecté visuellement avant diffusion.

## Reprise et autre version

Pour reprendre toute la campagne après une interruption, quelle que soit l'étape,
utiliser cette commande PowerShell autonome :

```powershell
& 'C:\Users\coudr\OneDrive\projects\benchmark\benchmark_milp_gnn\research\paper2_revision\run_paper2.ps1' -Stage Resume
```

La reprise contrôle l'environnement, puis repasse sur les étapes dont les calculs
manquants doivent être complétés. Les résultats terminaux déjà enregistrés sont
réutilisés et les entraînements interrompus reprennent leur état sauvegardé.
Après le verrouillage du test final, la reprise commence aux références finales.
Toute erreur arrête l'enchaînement. Pour une campagne utilisant un autre dossier,
ajouter le même argument `-Run` qu'au lancement initial.

La reprise réutilise désormais les scénarios préparés après vérification de
leurs empreintes, du manifeste et de l'audit des entrées. Elle ne les régénère
pas et ne refait pas un audit déjà validé. Cela corrige le faux conflit
`Immutable output changed` lié aux tuples du générateur devenant des listes
dans les fichiers JSON. Si la préparation a été interrompue avant publication
du manifeste, le lanceur compare les valeurs dans leur représentation JSON ;
une modification réelle reste refusée et aucun fichier existant n'est écrasé.
Les sources scientifiques gelées, les résultats et leurs identités sont conservés.

Pour vérifier seulement la préparation, sans reprendre les expériences :

```powershell
& "$Project\.venv\Scripts\python.exe" -B -m research.paper2_revision.recovery --run "$Project\research\paper2_revision\runs\methods_v1" --verify-only
```

Le lanceur PowerShell utilise cette reprise pour `Prepare` et `Resume`.
L'ancien appel Python `-m research.paper2_revision --stage Prepare` reste
inchangé dans le protocole gelé : utiliser le lanceur PowerShell.

### Refus d'accès Windows pendant un entraînement

`TrainMultiHorizon` et `Resume` utilisent désormais une écriture des checkpoints
qui réessaie les refus d'accès ou de partage Windows (jusqu'à 26,8 secondes
d'attente par opération). Le modèle est sérialisé une seule fois ; seule
l'opération d'entrée-sortie bloquée est réessayée. La copie de secours `.bak`
et le checkpoint principal sont remplacés atomiquement. Si le blocage persiste,
l'étape s'arrête en conservant le checkpoint précédent et les fichiers
temporaires nécessaires au diagnostic.

La reprise recharge le checkpoint principal avec son modèle, son optimiseur,
sa position dans l'époque et les états aléatoires. Les fichiers `.tmp` ou `.bak`
ne sont jamais promus automatiquement. Un pas d'optimisation non publié pourra
être recalculé. La logique scientifique, les graines et les plafonds d'époques
restent ceux du protocole gelé. L'identité du correctif d'écriture est conservée
dans `runs/.../recovery/checkpoint_writer_<empreinte>.json` au lancement de
l'entraînement.

La commande habituelle `-Stage Resume` reprend toute la suite. Pour ne reprendre
que les entraînements, utiliser `-Stage TrainMultiHorizon`. Pour contrôler
les checkpoints sur CPU, sans entraîner ni modifier les fichiers :

```powershell
& "$Project\.venv\Scripts\python.exe" -B -m research.paper2_revision.recovery --run "$Project\research\paper2_revision\runs\methods_v1" --stage TrainMultiHorizon --verify-only
```

Utiliser le lanceur PowerShell pour bénéficier du correctif : l'ancien appel
Python `-m research.paper2_revision --stage TrainMultiHorizon` reste inchangé
dans les sources gelées. L'avertissement sur l'absence de Flash Attention est
distinct du refus d'accès lors de la sauvegarde.

Tests de reprise et d'écriture sur des cas synthétiques CPU :

```powershell
& "$Project\.venv\Scripts\python.exe" -B -m unittest discover -s "$Project\research\paper2_revision\recovery\tests" -v
```

Relancer exactement la même étape reprend les calculs manquants. Pour un nouveau
protocole après modification des sources, fournir un autre dossier à chaque appel :

```powershell
$Run = Join-Path $Project 'research\paper2_revision\runs\methods_v2'
& $P2 -Stage Prepare -Run $Run
```

Ne pas copier les résultats du test final vers le développement et ne pas
retoucher un gel existant.

## Suivre l'avancement et le temps restant

La même commande affiche désormais un tableau par étape, utilisable aussi dans
un second terminal pendant la campagne :

```powershell
& $P2 -Stage Estimate
# Facultatif : convertir en jours à raison de 8 heures de calcul par jour.
& $P2 -Stage Estimate -HoursPerDay 8
# Sortie structurée pour un outil de suivi.
& $P2 -Stage Estimate -Json
```

Le tableau distingue les résultats terminaux des calculs encore manquants,
les durées mesurées et le temps restant à planifier. Les échecs terminaux
comptent comme réalisés ; une tentative interrompue sans résultat ne compte pas.
Les observations de la comparaison principale et de l'ablation restent séparées.

Les estimations additionnent préchargement et temps de sortie des processus.
Elles utilisent d'abord la moyenne des nouvelles observations comparables
(même famille, transformation, horizon, méthode et délai, au moins trois
mesures), puis les durées historiques pour les configurations de base à 24 h.
Ces extrapolations historiques restent indicatives, les tirages étant différents.
À défaut, la colonne « Dont provision » réserve le délai alloué plus un temps de
chargement observé. Pour les nouvelles méthodes, le chargement de leur méthode
historique correspondante sert de proxy ; Gurobi utilise celui de HiGHS.
Si aucun chargement n'a été mesuré, son plafond est provisionné (300 s pour
les références, 600 s pour les essais). Les provisions sont remplacées au fil
de l'arrivée de mesures comparables ; elles ne constituent pas une prévision
ni une borne supérieure du temps total.

Les entraînements sont comptés sur leurs neuf reçus de fin. Dès qu'une partie
AE, JEPA ou EBM est terminée, sa durée peut servir à estimer cette même partie
sur une autre graine. Un entraînement partiel reste non chiffré : le suivi ne
charge pas les gros checkpoints. Les embeddings, contrôles entre essais,
analyses, compilation et pauses sont signalés séparément. Le total est donc
un temps de calcul à planifier **plus les postes non chiffrés**, sans date de
fin promise. `HoursPerDay` ne modifie pas l'exécution ni son parallélisme.

Le suivi est en lecture seule, sans verrou d'écriture ni accès au solveur/GPU.
Il est isolé dans `monitoring/` pour préserver les empreintes scientifiques des
campagnes déjà gelées. Le lanceur PowerShell utilise ce nouveau suivi ; l'ancien
appel Python `-m research.paper2_revision --stage Estimate` conserve sa sortie
d'allocations afin de ne pas modifier les sources gelées. Pour utiliser le
nouveau rapport directement en Python :

```powershell
& "$Project\.venv\Scripts\python.exe" -B -m research.paper2_revision.monitoring --run "$Project\research\paper2_revision\runs\methods_v1"
```

Tests du suivi, sans solveur ni entraînement :

```powershell
& "$Project\.venv\Scripts\python.exe" -B -m unittest discover -s "$Project\research\paper2_revision\monitoring\tests" -v
```
