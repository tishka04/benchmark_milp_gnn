# Contrats, réemploi et choix scientifiques

## Isolation

La fonction `src/operational_exit/protocol.py::source_signature` a été lue avant de choisir les emplacements. Elle couvre `src/**/*.py`, `configs/uc_jepa/*.yaml` et trois scripts existants. Tous les ajouts de cette livraison se trouvent dans `research/three_papers/`, hors de ce périmètre. Aucune modification du runtime historique, des modèles, des réserves ou de `pdfs/main.tex`.

Les fonctions historiques `load_record` peuvent réparer les fichiers. Elles ne sont pas utilisées pour inventorier, figer ou extraire la campagne. `core.record` lit les JSON, vérifie les sceaux et identités, et refuse une divergence entre copie principale et sauvegarde. Il ne répare rien. L'amendement de réserve fournit la carte actuelle des sources ; celle-ci doit correspondre au runtime lu avant le gel.

## Correspondance avec l'existant

| Source | Réutilisation | Adaptation ou séparation |
| --- | --- | --- |
| `pdfs/main.tex`, sections formulation, graphes, AE/JEPA/EBM et reconstruction | Description du problème, représentation en huit canaux, tokens, BiGRU, conditionnement, cascade | Réécriture des affirmations numériques ; correction DR, temps total, statut des références et nature MILP de la réparation. Paragraphes méthodologiques adaptés dans le papier 2 et le protocole commun. |
| `pdfs/criticality_index.tex`, `src/analysis/criticality_index.py` | 12 + 10 composantes, bornes fixes et agrégation | Définition commune aux six familles ; export brut/normalisé/saturation ; prédiction groupée au lieu de validation par séparation des familles. |
| `pdfs/scenario_generator_explained.tex`, `src/generator/generator_v1.py` | Paramètres, samplers et métadonnées | UUID déterministes et nouveaux jeux disjoints ; nouveau tirage déclaré, sans reproduire le classement historique. |
| `scripts/generate_eval_families.py`, `scripts/generate_persona_scenarios.py` | Constructeurs d'espaces de paramètres | Import des fonctions seulement ; pas d'appel aux commandes de génération qui écrivent les familles historiques. |
| `src/operational_exit/budget_reserve.py::evaluation_schedule` | Sélection pure des identités attendues, budgets courts amendés | Aucun enregistrement ou changement de réserve. |
| `src/operational_exit/physics.py` | Validation indépendante et coût physique | Identique pour nouvelles références, quatre méthodes, Gurobi et routeur. |
| `src/operational_exit/solving.py` | HiGHS direct et cascade de réparation corrigée | Même backend de réparation, même intégralité et mêmes fixes structurels. |
| `src/operational_exit/supervisor.py` | Job Windows, arbre de processus, sortie effective, arrêt forcé classé en échec | Nouveau module worker et répertoires de tentatives autonomes. |
| `src/operational_exit/worker.py` | Cycle de vie, collecte avant échéance, sélection au coût, attente naturelle des enfants | Worker local avec adaptateur de scénarios, décomposition du temps, diagnostics de réparation, Gurobi, contrôle K16, multi-horizons, routage. |
| `src/operational_exit/generation.py` | Générateur AE/JEPA/EBM courant | Copie adaptée `fixed_generator.py` : import absolu, instrumentation et contrôle supprimant uniquement le gradient EBM. Le fichier source n'est pas modifié. |
| Modèles sous `operational_exit_v1/training` et HTE historique | Étape A gelée, hashes vérifiés | Aucun fine-tuning. Le nouvel AE/JEPA/EBM multi-horizons ne charge pas leurs poids. |
| `scripts/analyze_corrected_criticality.py`, `scripts/run_criticality_driver_analysis.py` | Lecture des conventions et des analyses précédentes | Nouvelles cibles de succès économique, groupes communs aux budgets, aucun sentinel de coût pour les échecs. |

Les conventions numériques et temporelles historiques incompatibles ne sont pas transférées. Les fichiers HTE sont réutilisés avec leur contrat de caractéristiques et leur normalisation, sans recalibrage sur le test.

## Entrées et sorties

| Fichier | Contrat |
| --- | --- |
| `config.json` | Paramètres scientifiques explicites, ressources, seuils, seeds, tailles des cohortes et plafonds d'entraînement. Inclus dans la signature de la nouvelle étude. |
| `runs/v1/freeze.json` | Sceau, sources, versions, chemins et hashes des modèles, identités courantes, hashes des références et scénario/manifest. Créé seulement après les étapes 10/11 achevées. |
| `runs/v1/cases.json` | `entries`: clé, groupe de base, famille, partition, condition, chemin, SHA-256 et horizon attendu. Sélection Gurobi sans accès à ses résultats. |
| `runs/v1/references/<digest(case)>.json` | Référence HiGHS commune : identité des ressources, résultat valide ou absence de référence, borne, certification, dispatch et supervision. |
| `runs/v1/experiments/<digest(identity)>.json` | Résultat terminal, succès physique/temporal, code de sortie, arrêt forcé, temps de sortie et de préchargement, dispatch complet si disponible, détails de réparation. |
| `runs/v1/attempts`, `ref_attempts` | Journaux et sorties brutes de chaque tentative, y compris interruption. Jamais exploités comme succès sans reçu terminal du superviseur. |
| `main_rows.json`, `rows_<partition>_<suite>.json` | Grille d'observations et caractéristiques. `observed=false`, `feasible=null`, `economic_5=null` pour un essai non observé ; échec observé connu = 0 lorsque la référence existe. |
| `dimension_audit.json` | Formes réellement présentes dans les labels et partition ; nombre réel de variables, contraintes et entiers libres des 600 cas. |
| `artifacts/<analyse>/tables/*.csv` | Supports, couvertures, contrastes appariés entre méthodes et conditions, qualité conditionnelle, temps, correction des propositions, prédiction/calibration. |
| `routing/diagnostic.json` | Gain oracle, support complet, incertitude et décision `proceed`. Peut légitimement arrêter la formation. |
| `routing/selector.joblib`, `model_receipt.json` | Modèle retenu et traçabilité, seulement si le diagnostic autorise sa formation. Le fichier sérialisé est un artefact local de confiance, pas une entrée à télécharger arbitrairement. |
| `multi_horizon/dataset.json` | Hashes des couples `u,h`, groupes, horizons, split par base, exclusions et statut de certification des labels. |
| `multi_horizon/*_last.pt`, `*_best.pt`, `*_receipt.json` | Reprise mutable du nouvel entraînement et artefacts finaux distincts. Les anciens checkpoints ne sont jamais remplacés. |
| `design_lock.json` | Décisions et modèles verrouillés avant toute référence ou exécution sur `final`/`controlled_test`. |

## Unités et estimands

Coûts en euros, puissances en MW, énergies en MWh, durées en secondes. Les horizons utilisent une résolution horaire. Le seuil principal est 5 %, avec 1 % et 10 % secondaires. Le gap vaut `(coût - référence) / abs(référence)` et peut être négatif. Pour `abs(référence) <= 1 euro`, le gap relatif est absent ; le succès utilise une tolérance absolue de 1 euro. Cette convention doit apparaître dans les publications.

Les observations d'un scénario restent groupées à travers budgets, méthodes et répétitions. Les variantes partagent leur base. La couverture primaire utilise la répétition 0 ; les deux répétitions complémentaires sont séparées. Les IC sont des bootstraps de moyennes par groupe. Les intervalles présentés sont descriptifs et ne corrigent pas automatiquement toutes les comparaisons multiples : les cinq descriptions, le seuil primaire et le test final sont annoncés à l'avance.

Le contrôle du papier 1 compare taille, taille + indice, taille + deux axes, taille + 22 composantes et taille + six composantes présélectionnées. La régression logistique a une régularisation fixée ; aucun réglage sur le test. Les coefficients standardisés sont des diagnostics associatifs, pas une attribution causale. Les transferts sont des applications sans recalibrage du prédicteur appris sur HiGHS.

Le diagnostic du papier 3 utilise les cellules complètes des quatre méthodes, séparément pour chaque couple profil–délai sur la grille 5/10/30/60/120 s. La méthode fixe est choisie hors du groupe évalué pour le même couple. La poursuite exige soit un gain de couverture ≥ 0,02 avec borne inférieure à 95 % positive, soit un gain conditionnel de coût/temps avec les contraintes de couverture et risque définies dans SERVICE_ROUTING.md. Le support connu doit atteindre 90 %, avec au moins 20 groupes. Aucune moyenne entre délais ni stabilité sur un autre délai ne conditionne cette décision. Ces seuils sont des choix de protocole ; les intervalles ne sont pas corrigés pour les comparaisons multiples. Les scores rétrospectifs d'un sélecteur n'incluent pas son coût en ligne ; seules les nouvelles exécutions permettent la conclusion opérationnelle.

## Transformations et limites de généralisation

- Taille : deux copies physiques, indices et régions distincts, import total doublé et ligne de liaison. Cela change la taille **et** la structure ; ce n'est pas une intervention pure sur le nombre de variables.
- Lignes : chaîne connectée sur un ordre aléatoire, complétée avec des arêtes jusqu'au nombre disponible, capacités et distances reprises de la base. Le compte peut augmenter si la base est trop peu connectée ; le compte réellement utilisé est exporté.
- Parc : capacité, minimum, rampes et état initial thermiques × 0,8 ; puissance, énergie et états batterie × 1,2. Les contraintes restent cohérentes. Cela ne crée pas artificiellement davantage d'engagements individuels.
- Permutation : bijection des noms de zones appliquée aux données, lignes et ancrage des imports, avec liste canonique des positions conservée. Le problème est physiquement isomorphe ; le test synthétique fourni le vérifie après exécution par l'utilisateur.
- Horizons : troncature à 12 h ou extension du jour initial à 48/72 h, facteurs journaliers explicites, capacités respectées et conditions terminales réappliquées. Ces profils synthétiques structurés ne représentent pas tous les décalages météorologiques possibles.
- Indice : ses proxys de générateur peuvent ignorer certains changements réalisés de réseau ou de parc. Cela est enregistré et discuté, pas masqué par une redéfinition après examen du test. Les modifications de taille/horizon mettent à jour leurs descripteurs et les composantes de saturation restent visibles.
- Nouvelle architecture : HTE gelé conservé, buffer sinusoïdal étendu à 72 ; AE/JEPA/EBM entièrement nouveaux, requêtes conditionnées par le contenu sans embedding appris d'indice de zone, longueur temporelle variable. La normalisation du HTE reste celle d'entraînement. La généralisation réussie n'est pas présumée.

## Reprise

Les résultats expérimentaux terminaux sont immuables, y compris les échecs. Les tentatives sans reçu terminal restent conservées et ne fournissent pas d'observation valide ; la commande peut reprendre ces calculs non conclus. Les publications d'analyse passent par un répertoire de tentative puis un renommage ; une analyse achevée est vérifiée par ses hashes et réutilisée. Produire les exports uniquement après achèvement de leur suite.

La formation conserve modèle, optimiseur, progression et états aléatoires après chaque accumulation. Un arrêt pendant une accumulation ne valide pas le pas incomplet. Une copie précédente est gardée ; aucun mécanisme silencieux ne substitue un checkpoint incompatible. En cas de fichier corrompu, examiner les deux copies et leur identité avant toute récupération manuelle dans le nouveau dossier.

## Amendement service v2 — 11 novembre 2026

Le contrat de [SERVICE_ROUTING.md](SERVICE_ROUTING.md) et les profils de `config.json` remplacent le seul seuil de 5 % pour les analyses conjointes et le routeur. Les anciens indicateurs économiques restent disponibles pour les comparaisons historiques et le papier 1. La nouvelle graine est 20261111, le dossier initial `runs/v2_service`, remplacé pour les décisions séparées par `runs/v2_service_by_deadline`. Les demandes sont incluses dans les identités ; les pivots et fusions distinguent désormais `profile_id`.

Le gel et le verrou final enregistrent les profils et les modèles avec leurs empreintes. Les champs `routing_*` exposent probabilité, coût/temps prévus, support, repli et temps restant. Les méthodes inchangées sont évaluées sous chaque profil ; une sortie routée conserve uniquement sa demande effective.

L'amendement par délai a été demandé après lecture du premier diagnostic agrégé. Il conserve les seuils numériques, change la grille et l'unité de décision, et exige un nouveau gel. `enabled_requests` autorise des couples précis ; `enabled_profiles` n'autorise pas tous les délais d'un profil. Les sources historiques et le diagnostic antérieur restent intacts.
