# Papier 1 — composantes et interactions

Implémentation isolée préparée le 11 novembre 2026. Aucun import Python, test, ajustement, analyse, solveur ou compilation n'a été exécuté par l'assistant. Les fichiers ont été relus textuellement ; la validation d'exécution reste à faire avec les commandes ci-dessous.

Ce module ne modifie ni n'importe `research/three_papers` ou `src`. Il lit les observations scellées du routeur dans `research/three_papers/runs/v2_service_by_deadline`, contrôle leurs empreintes et celles des sources figées, et écrit exclusivement dans `research/paper1_components/runs`. Il ne charge aucun modèle du routeur ou de JEPA, ne génère pas de scénario et ne lance pas de résolution.

## Protocole livré

- Les 600 scénarios historiques, six familles de 100 cas ; HiGHS, répétition primaire, cinq allocations indépendantes : 5, 10, 30, 60 et 120 s.
- Deux cibles séparées : sortie valide dans le délai et coût à 5 % de la référence ; même exigence avec ENS ≤ 0,001 MWh. Près d'une référence de valeur absolue ≤ 1 euro, le dépassement autorisé reste 1 euro. Les gaps négatifs sont conservés, et les labels inconnus sont distingués des échecs.
- Sept descriptions principales : taille, taille + indice, taille + deux axes, six composantes interprétables, 22 composantes linéaires, 22 composantes avec courbures additives, et modèle additif enrichi des cinq interactions.
- Modèles distincts pour chaque délai et cible, avec et sans indicateur de famille.
- Cinq plis externes par scénario, équilibrés par famille sans consulter les résultats ; trois plis internes pour choisir C parmi 0,1 / 1 / 10 selon le score de Brier. Les mêmes groupes sont utilisés dans toutes les comparaisons. Imputation, centrage des produits, codage des familles et standardisation sont ajustés seulement sur l'apprentissage.
- Base additive fixée : z, max(0,z−0,25), max(0,z−0,75). Elle permet une courbure individuelle avant d'attribuer un gain à un produit.
- Chaque produit conserve ses deux effets individuels. Les composantes constantes et les interactions non identifiables sont signalées. Une cible constante utilise un prédicteur constant explicite.
- Une non-convergence interrompt l'ajustement : elle n'est pas présentée comme un résultat valide.

| Paire | Composantes |
| --- | --- |
| 1 | A2 volatilité résiduelle × A8 inadéquation du stockage énergétique |
| 2 | A2 volatilité résiduelle × B7 contrainte de rampe |
| 3 | A6 pointe/puissance ferme × A10 manque de réserve d'effacement |
| 4 | A8 inadéquation du stockage énergétique × B8 contrainte d'état de charge |
| 5 | B3 nombre estimé de variables binaires × B9 densité d'interconnexion |

Huit retraits de composante suppriment tous ses termes et ses produits. Cinq retraits d'interaction suppriment seulement le produit en conservant ses parents. Chaque modèle réduit est réajusté et sa régularisation est choisie à l'intérieur des mêmes plis externes.

## Étapes

Depuis la racine du dépôt, dans le même environnement Python que le dernier lancement :

```powershell
$Python = '.\.venv\Scripts\python.exe'
$Run = '.\research\paper1_components\runs\components_v1'

& $Python -B -m unittest discover -s .\research\paper1_components\tests -p 'test_*.py' -v
& $Python -B -m research.paper1_components --run $Run prepare
& $Python -B -m research.paper1_components --run $Run audit
& $Python -B -m research.paper1_components --run $Run fit --workers 8
& $Python -B -m research.paper1_components --run $Run report --workers 8
```

Le script `run_paper1_components.ps1` applique ces étapes avec arrêt automatique sur erreur. Il ne relance ni les tests de l'ancien module, ni la campagne historique, ni l'entraînement du routeur.

`fit` comprend 2 000 ajustements externes (20 modèles × 2 contrôles de famille × 2 cibles × 5 délais × 5 plis), plus la sélection interne. Ce sont de petites régressions CPU, pas des MILP ; leur durée n'a pas été mesurée. Un point de reprise scellé est écrit après chaque ajustement externe. Relancer la même étape réutilise les points terminés, après contrôle de provenance. Le reçu final d'ajustement est requis avant le rapport.

Ne pas modifier les sources/configuration après `prepare`. Une correction demande un nouveau dossier pour le papier 1 uniquement ; le dossier du routeur reste inchangé. Les répertoires de tentatives incomplètes sont conservés.


## Exécution sur plusieurs cœurs

Le lanceur accepte maintenant le nombre de processus :

```powershell
& 'C:\Users\coudr\OneDrive\projects\benchmark\benchmark_milp_gnn\research\paper1_components\run_paper1_components.ps1' -Workers 8
```

Huit processus sont proposés pour le Core Ultra 9 285H utilisé ici. `-Workers 4` réduit la charge ; `-Workers 1` rend les phases d'analyse séquentielles. Les tests synthétiques comprennent leurs propres essais à deux processus.

Les 2 000 ajustements externes sont répartis entre les processus, et chacun conserve sa sélection interne séquentielle. Les calculs du rapport (scores, rééchantillonnages, calibration et contrastes) sont répartis entre les vingt combinaisons cible × délai × contrôle de famille. La préparation, l'audit, l'écriture des tableaux et le dessin des figures restent séquentiels.

Chaque processus utilise un seul fil BLAS/OpenMP, y compris lors d'un lancement direct par Python. Le démarrage utilise explicitement `spawn` pour Windows. Au plus deux tâches par processus restent soumises simultanément ; les petites données préparées sont transmises aux processus sans relire les sorties historiques.

Seul le processus principal publie les points de reprise et les exports, sous le verrou du répertoire. Les plis, graines, modèles, budgets, critères et règles statistiques sont inchangés. L'ordre des lignes des tableaux est rétabli avant l'export, indépendamment de l'ordre de fin des tâches. Le nombre de processus est un réglage d'exécution : il peut changer lors d'une reprise, sans invalider les points terminés de cette version du code.

En cas d'erreur ou d'interruption, les tâches encore annulables sont annulées, les tâches déjà démarrées sont attendues, et aucun reçu final incomplet n'est publié. Relancer la même commande reprend les ajustements sauvegardés. Les versions antérieures du code restent incompatibles avec un manifeste déjà préparé : utiliser alors un nouveau dossier, sans effacer les résultats existants.

Aucune accélération n'a été mesurée. Les tests de démarrage Windows, d'équivalence séquentiel/parallèle, de reprise et d'interruption ont été ajoutés, mais n'ont pas été exécutés.

## Résultats produits lors du lancement

Sous `audit/tables` :

- `component_audit.csv` : variance, constantes, bornes atteintes, valeurs brutes et support par famille.
- `component_correlations.csv` et `interaction_support.csv` : dépendances entre composantes et combinaisons observées.
- `score_contributions.csv` : contributions arithmétiques et reconstruction non arrondie de l'indice.
- `score_sensitivity.csv` / `score_variants.csv` : mélange stress/difficulté 0 / 0,25 / 0,5 / 0,75 / 1, et retrait de chacune des 22 composantes avec renormalisation de son bloc ; changements de score, classement, inversions de paires, ex æquo et migrations entre cinq intervalles fixes.
- `folds.csv` et `endpoint_support.csv` : groupes de validation et supports des cibles.

Sous `report/tables` :

- `prediction_scores.csv`, `predictions.csv`, `calibration.csv` : prédictions hors pli, Brier, log loss, AUC, précision moyenne et calibration.
- `predictive_contrasts.csv` : gains de perte prédictive, avec intervalles groupés, séparés par cible/délai/contrôle de famille.
- `contrast_by_fold.csv`, `contrast_by_family.csv` : stabilité descriptive.
- `fit_diagnostics.csv`, `coefficients.csv` : régularisation retenue, support et termes écartés.
- `interaction_surfaces.csv` : prédictions additive et avec interactions dans des cellules réellement observées ; aucune combinaison synthétique hors support. Une cellule de moins de dix scénarios reste masquée.

Un gain positif signifie une perte prédictive plus faible pour le candidat. Pour une ablation, le candidat est le modèle complet : un gain positif signifie que le retrait détériore la prédiction. Les cartes de probabilités comparent deux modèles réajustés ; elles n'isolent pas un effet causal d'un produit. Les courbes/matrices PDF couvrent les cinq délais ; les cartes bidimensionnelles illustrent le délai déclaré de 30 s, et leurs tableaux couvrent les cinq.

## Portée scientifique

Les 600 cas ont déjà servi à développer les questions. Ces résultats resteront exploratoires. Les intervalles rééchantillonnent les pertes hors pli par scénario, sans réajuster les modèles dans chaque réplication. Ils sont ponctuels, sans garantie simultanée sur toutes les comparaisons ; les différences entre familles, coefficients et retraits de variables corrélées ne sont pas des effets causaux.

Le test principal des interactions compare le modèle complet au modèle additif autorisant déjà des courbures individuelles. Les retraits sont descriptifs. Une faible perte au retrait peut refléter des informations redondantes avec d'autres composantes ou avec la taille.

La pondération supervisée sous contraintes, les normalisations alternatives, les transferts de famille/structure/solveur et l'application à la cohorte finale ne sont pas exécutés par ce module. Les variantes de score servent à la description, pas à choisir une pondération sur les résultats de test. Les choix devront être verrouillés avant d'utiliser les résultats des 120 nouveaux scénarios comme confirmation.

## Validation préparée

Les dix-huit tests synthétiques initiaux couvrent la syntaxe, l'absence d'import du code figé, les frontières d'écriture, les labels, la grille, le regroupement, la hiérarchie des interactions, le prétraitement appris hors test, les constantes, un exemple d'interaction connue, la sélection interne, le signe et la séparation des contrastes, les cellules rares, les ex æquo et la reprise. Dix tests supplémentaires portent sur le parallélisme : paramètres, limitation des fils, démarrage Windows, file bornée, erreur de processus, équivalence des ajustements et des tableaux, reprise et refus d'un point de reprise altéré. Au total, 28 tests sont préparés. Aucun n'a été lancé pendant la préparation.
