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

Relancer exactement la même étape reprend les calculs manquants. Pour un nouveau
protocole après modification des sources, fournir un autre dossier à chaque appel :

```powershell
$Run = Join-Path $Project 'research\paper2_revision\runs\methods_v2'
& $P2 -Stage Prepare -Run $Run
```

Ne pas copier les résultats du test final vers le développement et ne pas
retoucher un gel existant. `Estimate` affiche les nombres planifiés, réalisés
et restants ; les durées sont des sommes d'allocations, hors temps auxiliaires.
