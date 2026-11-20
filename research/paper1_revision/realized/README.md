# Comparaison des descripteurs historiques et du MILP construit

Cette extension ne modifie ni les scénarios, ni la formulation, ni les résultats
historiques. Elle définit une version distincte des descripteurs à partir des
profils et paramètres chargés, et du modèle Pyomo construit avec ses fixations
structurelles. L'extraction n'appelle aucun solveur. Les statistiques de
présolve, les solutions, les coûts obtenus et les temps de résolution n'entrent
jamais dans les variables explicatives.

## Exécution complète

Depuis la racine du dépôt :

```powershell
# Extraction, validations historiques, gel des prédicteurs et rapports.
# Cette commande ne résout aucun MILP.
.\.venv\Scripts\python.exe -m research.paper1_revision.complete_campaign prepare

# Lance ensuite les résolutions physiques et produit les analyses disponibles.
.\.venv\Scripts\python.exe -m research.paper1_revision.complete_campaign run --max-hours 450

# Refaire uniquement les analyses, sans nouvelles résolutions :
.\.venv\Scripts\python.exe -m research.paper1_revision.complete_campaign analyze
```

La préparation conserve des points de contrôle par scénario et ajustement.
Une interruption se reprend avec la même commande. Le lancement physique
vérifie le gel supplémentaire avant d'appeler le lanceur historique ; il ne
duplique pas ses travaux déjà terminés. Les anciens résultats et prédicteurs
restent conservés. Ne pas lancer deux campagnes physiques simultanément.

Les étapes séparées restent disponibles :

```powershell
.\.venv\Scripts\python.exe -m research.paper1_revision.realized prepare
.\.venv\Scripts\python.exe -m research.paper1_revision.realized fit
.\.venv\Scripts\python.exe -m research.paper1_revision.realized freeze
.\.venv\Scripts\python.exe -m research.paper1_revision.realized report
```

## Comparaison déclarée

| Représentation | Contenu |
| --- | --- |
| `legacy_normalized` | Les 22 composantes normalisées historiques. |
| `legacy_raw` | Les 22 valeurs avant normalisation historique ; les écrêtages intrinsèques des anciennes formules demeurent. |
| `legacy_without_disputed` | Valeurs brutes sans A3, B3, B8 et B9. |
| `corrected_only` | Valeurs brutes avec uniquement A3, B3, B8 et B9 remplacées par leurs nouvelles définitions. |
| `realized` | 22 descripteurs calculés sur l'instance et le modèle construits. |
| `realized_compact` | Nombre de zones, binaires libres et proportion de zones thermiques. |
| `size` | Nombre de zones seul. |
| `model_size` | Zones, pas de temps, lignes, variables déclarées, binaires libres et contraintes actives. |

Les nouvelles coordonnées reprennent les identifiants A1–B10 pour faciliter la
correspondance, mais plusieurs changent de signification. Elles ne remplacent
pas silencieusement l'ancien indice. Les formules exactes sont conservées dans
`features.py` et dans `feature_manifest.json` après extraction.

En particulier : A3 mesure le vrai ratio pointe/creux agrégé, avec plancher
de 1 MW au dénominateur ; B3 compte les variables binaires qui ne sont fixées
ni explicitement ni par leurs bornes, avant présolve ; B8 utilise puissance,
énergie et fenêtres terminales réellement imposées aux batteries et au stockage
pompé ; B9 utilise les lignes réellement construites. A2 et A4 viennent des
séries de demande résiduelle. Les données de demande sont antérieures à
l'optimisation. Aucun flux optimisé ne sert à calculer une « congestion ».

Les autres rapports restent des descriptions, pas des certificats physiques :
la capacité d'échange n'est pas une dépendance aux importations observée ;
additionner des capacités ne prouve pas leur disponibilité simultanée ;
les rampes sont celles du modèle, exprimées par transition. Les absences de
stockage/réseau et les dénominateurs nuls sont traités explicitement.

## Évaluation et prévention des fuites

- Même population historique, mêmes cinq délais, mêmes cibles et mêmes plis
  que la révision précédente. Les cinq préférences continues et E/ES sont inclus.
- Comparaison principale : modèles linéaires à coefficients positifs par délai
  pour les cibles continues ; régression logistique pour E/ES. Une régression
  signée complète la comparaison `legacy_raw` / `realized`.
- Toutes les représentations sont centrées/réduites sur l'entraînement. Ce
  traitement est réappris dans chaque pli interne. Une constante d'entraînement
  reste sans effet appris, même si elle varie dans le test indépendant.
- Cinq plis externes groupés par scénario, puis six validations laissant une
  famille entière de côté, avec réglage interne entre les familles restantes.
- 66 spécifications ajustées dans chaque pli ; 726 points de contrôle externes.
  Chaque spécification contient cinq modèles propres aux délais.
- Contrastes principaux `realized` contre `legacy_raw` : 35 intervalles
  simultanés, par schéma de validation, à partir de 2 000 bootstrap appariés
  par scénario. Ils sont conditionnels aux prédictions ajustées.
- Allocation ES dans chaque portefeuille externe : budget réservé, temps
  observé et 200 rééchantillonnages avec nouvelle optimisation du portefeuille.
  Cette expérience ne valide pas directement les poids des cibles continues.
- Les 66 spécifications finales et les 132 000 prédictions indépendantes sont
  gelées avant toute tentative de référence ou d'essai sur les 400 nouveaux cas.
  Aucun résultat indépendant ne sert au choix des formules ou modèles.

Le contraste avec `legacy_raw` et la correction ciblée permettent de préciser
ce qui vient des définitions. Ils ne garantissent pas une attribution causale
de chaque différence de performance. Les résultats sur l'ancien corpus restent
des analyses de développement.

## Coût et résultats

Les données de 600 cas historiques et de 400 cas indépendants sont extraites
une fois, avec temps de chargement, construction et calcul enregistrés.
La construction complète est nécessaire au comptage des variables. Ce coût
est **hors des délais des expériences historiques**. Les gains d'allocation
décrivent donc une situation où les descripteurs sont déjà disponibles ; ils
ne constituent pas une preuve de gain net pour une application en temps réel.

Les sorties sont dans `runs/revision_v1/realized_milp_v1` :

- `feature_audit.csv`, `feature_manifest.json` : valeurs, dimensions, formules et provenance ;
- `nested_scores.csv`, `lofo_scores.csv`, `pooled_scores.csv` et prédictions Parquet ;
- `*_primary_contrasts.csv`, `*_calibration.csv` : comparaisons et calibration ;
- `allocation_totals.csv`, `allocation_intervals.csv` : utilité décisionnelle ;
- `definition_comparison.csv`, `feature_computation_times.csv` : audit des écarts et surcoût ;
- `freeze.json`, `frozen_models.json`, `external_predictions.parquet` : gel avant test ;
- `external_scores.csv`, `external_contrasts_*.csv`, `external_status.json` après résolution.
- `reference_prediction_sensitivity.csv` : réévaluation des prédictions conservées
  lorsque les références renforcées modifient les cibles ; aucun réajustement automatique.

Les références ou essais manquants restent non observés. Les contrastes
indépendants ne sont produits que pour les lots complets. Les résultats restent
séparés du manuscrit tant qu'ils n'ont pas été examinés ; aucune amélioration
des descripteurs réalisés n'est présumée.
