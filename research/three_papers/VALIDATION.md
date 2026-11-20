# État réel de validation

## Effectué pendant la préparation

- Lecture directe des fichiers du protocole, de sa fonction de signature, des interfaces physiques, des générateurs, des modèles et des sources des manuscrits.
- Lecture de l'état enregistré de la campagne : seules les étapes 1 à 9 y étaient déclarées terminées. Le verrou était occupé lors d'une tentative de lecture ; aucun processus n'a été inspecté, arrêté ou relancé. L'état courant ultérieur est à relire avec la commande d'inventaire.
- Création et relecture de fichiers uniquement sous `research/three_papers/`.
- Vérification en ligne des six notices bibliographiques primaires, consignée dans `BIBLIOGRAPHIE.md`.
- Relecture manuelle des conventions : sources surveillées, sélection amendée, temps de sortie, canaux d'activité stockage, bornes/références, groupes, chemins et commandes.

## Non effectué

**Aucune commande shell, aucun import/exécution des programmes Python du projet, aucun test, aucune analyse, aucune génération de scénarios, aucune installation, aucune résolution, aucun entraînement ni aucune compilation LaTeX.** Les lectures/écritures directes de fichiers n'ont pas servi à effectuer ces calculs par un autre outil.

En conséquence, la syntaxe et les interfaces Python n'ont pas été vérifiées par un interpréteur ; les tests fournis n'ont pas tourné ; aucun taux de succès, gain prédictif ou gain d'oracle n'a été calculé. Les figures n'existent pas encore. Les trois `.tex` n'ont pas été compilés et les PDF n'ont pas été inspectés. La licence Gurobi et sa capacité pour les instances réelles restent à vérifier.

## Vérifications préparées pour l'utilisateur

| Test | Question vérifiée lors de son exécution |
| --- | --- |
| `test_contracts.py` | Syntaxe des nouveaux modules, conservation des gaps négatifs, convention zéro, distinction manquant/échec, non-mutation des sources, identités amendées, règle de sortie, groupement, canaux stockage |
| `test_multihorizon.py` | Formes 12/24/48/72, invariance des tokens et équivariance du décodeur aux permutations de zones |
| `test_physical_adapters.py` | Préservation du premier jour, cohérence des axes, duplication physique, bornes du parc et égalité de coût sous renumérotation sur petits MILP |
| `test_integration.py` | Sur option explicite : cycle complet du nouveau worker, terminaison et sortie physique ; adaptateur Gurobi contre HiGHS sur cas synthétique |
| `audit` | Dimensions réellement présentes dans l'entraînement et entiers effectivement libres, sans solveur |
| `extract` / `extract-extension` | Cohérence des hashes, identités, coûts et validation physique indépendante |
| `latexmk` puis inspection des PDF | Compilation et qualité du rendu final, références et placeholders |

L'ordre exact et les garde-fous sont dans `GUIDE_COMMANDES.md`. Arrêter avant les calculs coûteux si un test échoue. Toute correction des sources après un gel change la version scientifique ; ne pas contourner le contrôle de provenance.

## Limites qui restent même après les tests techniques

Les nouveaux échantillons ne reproduisent pas exactement la sélection historique ; les cas à horizons prolongés restent synthétiques ; plusieurs composantes de criticité sont des proxys bornés ou constants ; le doublement de taille modifie aussi la structure. Les références peuvent être des incumbents. Les succès de réparation ne démontrent pas à eux seuls le transfert d'une proposition apprise. Les intervalles ne rendent pas les associations causales. Le papier 3 peut légitimement s'arrêter au diagnostic négatif de complémentarité.

## Modification du 11 novembre 2026 : critères conjoints et routeur

Ajout des profils, de la conjonction à trois états (succès/échec/inconnu), des analyses par profil et de sensibilité ENS, du diagnostic par demande, des modèles de probabilité calibrés par groupes et des régressions conditionnelles coût/temps. Les identités des essais routés portent la demande, le worker déduit le temps de lecture/caractéristiques/sélection et les exports conservent les supports et replis.

La convention ENS est celle des drafts relus : 0,001 MWh. Les seuils de risque et de poursuite sont des paramètres explicites, non des résultats validés. La version `v2_service` ne réécrit ni les résultats historiques ni v1.

23 tests synthétiques sont préparés dans `tests/test_service_routing.py`. Aucun test Python, import du programme, calcul statistique, solveur, entraînement, lancement d’expérience ou compilation n’a été exécuté par l’assistant pour cette modification. La vérification effectuée est une relecture textuelle des interfaces et des fichiers. L’exécution des tests puis un essai d’intégration restent nécessaires avant les calculs réels.

## Amendement par délai du 11 novembre 2026

Le journal fourni par l'utilisateur pour la version antérieure v2_service indique 35 tests réussis, 2 ignorés, les 18 480 sorties inventoriées et la production des analyses conjointes et du diagnostic. Ce diagnostic agrégé a arrêté l'entraînement. Ces résultats concernent le code antérieur, pas l'amendement présent.

La grille demandée est maintenant 5/10/30/60/120 s. Le diagnostic, la sélection et les exports de scores/calibration distinguent chaque couple profil–délai. Neuf tests supplémentaires sont préparés dans `tests/test_service_deadlines.py` : filtrage de la grille, refus d'agrégation, absence de dilution du gain, risque/support par délai, choix de modèles distincts, autorisation sur le délai nominal malgré un temps restant inférieur, calibration séparée et planification des seuls couples autorisés.

Vérification de cet amendement : relecture textuelle des sources, tests et commandes ; vérification des fichiers sauvegardés. Aucun test, import Python, calcul statistique, entraînement, solveur ou compilation exécuté par l'assistant. Les nouveaux résultats restent à produire dans `runs/v2_service_by_deadline`. Le diagnostic précédent et ses sorties sont conservés.
