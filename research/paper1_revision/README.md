# Révision du papier 1

Les analyses historiques sont exécutées et intégrées au manuscrit. Les longues
résolutions sont **préparées, non lancées**, conformément au choix de l'utilisateur.

La comparaison supplémentaire avec les descripteurs du MILP construit est
implémentée dans [realized/README.md](realized/README.md). Elle compare huit
représentations avec les mêmes cibles et plis. Sa préparation ne résout aucun
MILP ; ses résultats sont conservés séparément du manuscrit actuel.

## État du plan

| Chantier | Résultat livré |
| --- | --- |
| 1 — Définitions | Audit des 22 composantes, corrections A2/A3/B3, dictionnaire et distinction entre paramètres générés et contraintes actives. |
| 2 — Échelle et direction | Comparaison commune / direction commune avec échelles / directions séparées ; 50 contrastes et intervalles simultanés. |
| 3 — Références simples | Taille, indice original, trois/six/22 composantes et boosting, avec réglage imbriqué. |
| 4 — Origine des échecs | Décomposition livraison/économie/service, modèles directs et conditionnels, projection et calibration. |
| 5 — Redondance | Diagnostics, réduction apprise dans les plis, groupes physiques et normalisation non saturante. |
| 6 — Conventions du score | 48 spécifications réajustées ; performances, rangs et poids regroupés conservés. |
| 7 — Références économiques | Sous-groupes audités ; 217 renforcements préparés. Changements d'étiquettes et de cibles non encore observés. |
| 8 — Transfert entre familles | Six validations externes par famille, avec réglage interne par famille. |
| 9 — Test indépendant | 400 scénarios et 33 prédicteurs gelés ; aucune performance indépendante encore mesurée. |
| 10 — Temps et répétitions | Répétitions historiques analysées ; 450 nouvelles exécutions instrumentées préparées. |
| 11 — Allocation du temps | Portefeuilles externes séparés, optimisation exacte et bootstrap avec réallocation. |
| 12 — Manuscrit | Texte réorganisé, figures et tableaux, définition du scénario, formulation MILP et PDF de 35 pages vérifié. |

Les différences essentielles du modèle sont explicites dans le papier : une
instance est déterministe ; le délai et la répétition définissent une exécution
de cette instance. Les paramètres générés, les données physiques réalisées et
les 22 descripteurs ne sont pas interchangeables. La définition figure page 3,
le tableau des paramètres page 4 et le MILP en annexe A, pages 20–22.

## Reproduire les analyses

Depuis la racine du dépôt, utiliser l'environnement `.venv` (Python 3.11.9,
NumPy 2.1.3, SciPy 1.17.1, pandas 3.0.2, scikit-learn 1.8.0, pyarrow).

```powershell
.\.venv\Scripts\python.exe -m research.paper1_revision prepare
.\.venv\Scripts\python.exe -m research.paper1_revision fit
.\.venv\Scripts\python.exe -m research.paper1_revision lofo
.\.venv\Scripts\python.exe -m research.paper1_revision probability
.\.venv\Scripts\python.exe -m research.paper1_revision robustness
.\.venv\Scripts\python.exe -m research.paper1_revision allocation
.\.venv\Scripts\python.exe -m research.paper1_revision report
.\.venv\Scripts\python.exe -m research.paper1_revision.supplement
.\.venv\Scripts\python.exe -m research.paper1_revision.coefficient_sensitivity
```

Les ajustements reprennent les points de contrôle seulement si les empreintes
correspondent. Les sorties historiques restent inchangées. `prepare` vérifie les
données scellées et reconstruit les quatre tables communes ; il exige la présence
des sources historiques dans ce dépôt. Les tables Parquet suffisent ensuite aux
nouveaux ajustements. Les rapports de reproduction utilisent aussi les anciennes
prédictions de `paper1_weights` et `paper1_components`.

## Résultats et portée

- MSE : modèles commun, direction commune avec échelles par délai, séparé,
  taille seule, trois/six/22 composantes, réduction, groupes, normalisation et boosting.
- Transfert : six validations externes par famille, réglage interne par famille.
- Probabilités : modèles directs et produits conditionnels cohérents ; pertes
  continues en deux parties ; calibration et Brier Skill Score avec référence d'entraînement.
- Robustesse : 48 spécifications de cible, certification, seuils économiques,
  saturation, redondance et répétitions archivées.
- Décision : allocation exacte dans chaque portefeuille externe, ES principal,
  E secondaire, 200 bootstrap de portefeuilles réoptimisés par condition.

Le dossier `runs/revision_v1` contient les tables, prédictions, plis, diagnostics
et empreintes. Les figures et fragments LaTeX générés sont dans
`research/three_papers/manuscripts/paper1/results`. Le manuscrit principal est
`research/three_papers/manuscripts/paper1_criticality.tex` ; le PDF vérifié est
`output/pdf/paper1_criticality.pdf`.

L'audit corrige A3 : `round(1.8*d,2)`, et non le ratio réel pointe-creux.
Il précise aussi que le paramètre historique `battery_e_to_p_hours` ne modifie
pas les batteries installées, dont le chargeur conserve une durée de 4 h.
La durée DR nominale n'impose pas une durée maximale de report ; le maximum
d'événements limite des périodes actives. Les anciennes valeurs ne sont pas
remplacées. Les 22 indicateurs restent des approximations issues du générateur.

## Campagne physique complète préparée

`runs/revision_v1/physical_campaign/manifest.json` contient : 217 renforcements
de référence, 400 nouveaux scénarios avec leur référence et 2 000 exécutions
aux cinq délais, plus 450 répétitions chronométrées. Le plafond cumulé des
réserves est d'environ **390,5 heures**, hors préchargements. Ce plafond n'est
pas une prédiction de durée réelle. L'ordre est randomisé et fixé à l'avance.
Douze configurations représentatives ont été chargées et leurs MILP construits
sans appeler le solveur. Les prédicteurs sont ajustés uniquement sur l'historique
et gelés avant toute ouverture du test indépendant.

Commandes complètes, incluant la nouvelle comparaison et son gel avant test :

```powershell
# Extraction et ajustements statistiques, sans résolution physique.
.\.venv\Scripts\python.exe -m research.paper1_revision.complete_campaign prepare
# Cette commande lance les résolutions physiques, puis analyse les résultats.
.\.venv\Scripts\python.exe -m research.paper1_revision.complete_campaign run --max-hours 450
# Réexécuter la commande run pour reprendre les travaux non terminés.
# Pour analyser seulement les résultats déjà disponibles :
.\.venv\Scripts\python.exe -m research.paper1_revision.complete_campaign analyze
```

Le lanceur arrête de démarrer des tâches quand leur réserve ne tient plus dans
l'enveloppe de l'invocation. Les préchargements et terminaisons peuvent ajouter
du temps à cette enveloppe ; ce n'est pas une garantie murale stricte. Chaque
exécution utilise le superviseur Windows mesurant la sortie de l'ensemble du
processus. Les anciennes références sont conservées et les nouvelles étiquettes
évaluées séparément. Aucun résultat manquant n'est transformé en échec observé.
Les analyses indépendantes automatiques portent sur les cinq cibles continues
et les modèles directs E/ES. Le traitement des nouvelles références recalcule
les cibles continues et binaires, la matrice des changements d'étiquette et
les pertes des prédictions historiques conservées. De grands changements
nécessiteraient ensuite un réajustement versionné. Les traces chronométrées
sont agrégées avec leurs effectifs ; une étape interrompue reste inconnue.
PGLib-UC et un second
solveur sont des extensions facultatives, non réalisées ici.

## Contrôles et limites

Les tests vérifient notamment l'identification du modèle intermédiaire sur des
données synthétiques, la séparation des directions, les probabilités ordonnées,
les cas limites des indicateurs, l'allocation exacte et les bornes économiques
avec coûts négatifs ou proches de zéro. La suite détecte les fuites de groupes,
reproduit les anciennes pertes et compare les anciennes prédictions aux
réajustements. Les intervalles bootstrap restent conditionnels aux modèles
ajustés ; les variations des poids entre plis ne sont pas des intervalles de
population. Les nouveaux résultats demeurent des analyses de développement.

Le fichier `array_copy_fix.json` documente une correction de compatibilité
pandas 3 : rendre modifiable une copie de matrice avant la transformation
non saturante. Les empreintes de points de contrôle antérieurs à cette
correction conservent leur provenance initiale ; leurs calculs sont inchangés.
