# Papier 1 · Trente journées sur un réseau fixe

Le pilote mesure si une équipe peut identifier les six journées les plus préoccupantes d’un même système avant de lancer l’optimisation. Il compare les journées sous cinq mandats illustratifs (service, coût, temps, émissions directes et équilibre des quatre). Les cinq évaluations utilisent les mêmes dispatchs. La mesure principale est le **regret moyen de sélection** à quota fixe. Elle indique de combien la perte moyenne des six journées retenues est inférieure à celle des six journées les plus préoccupantes observées après résolution.

## Lancement dans PowerShell

Depuis le projet, avec son environnement Python Windows et HiGHS déjà installés :

~~~powershell
Set-Location 'C:\Users\coudr\OneDrive\projects\benchmark\benchmark_milp_gnn'
.\.venv\Scripts\python.exe -m research.paper1_fixed_days prepare
.\.venv\Scripts\python.exe -m research.paper1_fixed_days run
.\.venv\Scripts\python.exe -m research.paper1_fixed_days analyze
~~~

La commande de préparation crée le manifeste des journées et des prévisions dans **research/paper1_fixed_days/runs/pilot_v1**. Elle construit les modèles pour mesurer leurs dimensions, sans appeler le solveur. Elle est réexécutable avec les mêmes paramètres. Sur la machine du projet, le manifeste par défaut a déjà été préparé.

La commande **run** effectue les essais dans l’ordre fixé avant leur exécution, une résolution à la fois, avec 16 threads HiGHS par défaut. Chaque journée a deux essais indépendants à 30 et 120 secondes, chronométrés du début du traitement du cas jusqu’à la sortie du processus, et une résolution de référence avec une limite solveur de 600 secondes. Les 90 résolutions totalisent 6 h 15 de budgets nominaux. Le superviseur accorde jusqu’à 60 secondes supplémentaires par référence pour la préparation et la finalisation, soit 6 h 45 de plafonds cumulés hors préchargements et écritures de contrôle. La durée effective dépend du solveur.

Pour exécuter seulement trois nouveaux travaux, consulter le décompte ou reprendre après interruption :

~~~powershell
.\.venv\Scripts\python.exe -m research.paper1_fixed_days run --max-jobs 3
.\.venv\Scripts\python.exe -m research.paper1_fixed_days status
.\.venv\Scripts\python.exe -m research.paper1_fixed_days run
~~~

Chaque observation terminée est conservée avec son empreinte, y compris les non-livraisons à l’échéance. Si le processus est interrompu avant l’enregistrement du verdict d’un essai, cet essai peut être repris. Deux lanceurs ne peuvent pas écrire dans le même pilote en parallèle. En cas d’erreur d’infrastructure, le lanceur indique le journal du worker concerné. L’analyse demande que tous les travaux aient un verdict et indique séparément les références valides et les références certifiées optimales. Une référence invalide laisse inconnue la qualité économique d’un dispatch livré.

## Système et chroniques

Le réseau par défaut est le [premier scénario numéroté de la famille de difficulté moyenne](../../outputs/medium_criticality_scenarios/scenario_00001.json), choisi selon son identifiant. Il comprend 56 zones, 55 lignes et 24 pas horaires. La topologie, les longueurs et capacités des lignes, le parc, les apports hydroélectriques, les coûts, les états initiaux et les règles physiques sont fixés. Les journées sont **synthétiques et indépendantes**, chacune partant du même état de stockage.

Chaque journée reçoit une graine et des paramètres archivés dans le manifeste. Un profil journalier de demande est combiné à celui de chaque zone, avec un multiplicateur de pointe entre 0,75 et 1,25. L’éolien et le solaire combinent des profils régionaux de météo avec les chroniques du système de base. Les limites de modulation de demande, exprimées en proportion de la consommation, sont recalculées en MW. La préparation vérifie pour chaque journée l’empreinte des champs physiques fixes, l’unicité des chroniques et l’invariance des descripteurs structurels. Le worker régénère la journée et contrôle ses empreintes avant la résolution.

Pour étudier un autre réseau du benchmark, créer un pilote distinct avec **prepare --anchor chemin_du_scenario.json --run-dir chemin_du_pilote** et repasser le même **--run-dir** à **run**, **status** et **analyze**. L’entrée attendue est un scénario de 24 pas horaires avec un identifiant UUID. L’option **--threads** se fixe à la préparation. Les comparaisons de temps doivent conserver les mêmes réglages et le même environnement matériel.

## Comparaisons

- Les **trois descripteurs gelés** sont le nombre de zones, le nombre de binaires libres et la fraction de zones comportant une capacité thermique. Les **six dimensions** sont les nombres de zones, de pas, de lignes, de variables, de binaires libres et de contraintes du modèle. Ces deux représentations donnent des prévisions identiques entre journées, à mandat et échéance fixés.
- Le comparateur à **22 descripteurs du MILP construit** utilise aussi des entrées sensibles aux conditions horaires. Les trois prédicteurs utilisent les ajustements gelés du papier. Le classement utilise les prévisions bornées entre zéro et un. Le manifeste conserve également les prévisions brutes.
- Les règles physiques classent les journées par **pointe de charge résiduelle**, **plus forte hausse horaire**, **plus faible marge à la pointe** et **plus faible stock initial**. La charge résiduelle de ces règles est la demande moins le solaire et l’éolien disponibles. La marge est la capacité thermique et nucléaire plus le fil de l’eau au moment de la pointe résiduelle, moins cette pointe. Le stock initial, fixe dans ce protocole, est un témoin supplémentaire. Le calcul des descripteurs gelés reprend les conventions historiques, dont le retrait du fil de l’eau dans certaines mesures de charge résiduelle.
- Le tirage uniforme donne le niveau de référence. Les ex æquo sont traités par leur probabilité d’inclusion, à priorité égale. Pour six places sur trente, le hasard retrouve en moyenne 1,2 journée du top six. Les intervalles descriptifs sont rééchantillonnés par journée pour tenir compte de l’appariement des stratégies.

## Lire les sorties

Ouvrir **research/paper1_fixed_days/runs/pilot_v1/analysis/summary.md** une fois l’analyse terminée.

| Fichier dans le dossier d’analyse | Question à laquelle il répond |
|---|---|
| observations.csv | Chaque journée a-t-elle livré un dispatch ? Quels étaient sa demande, ses indicateurs journaliers, son coût, ses MWh non servis, ses émissions directes et son temps ? |
| profile_scores.csv et score_dispersion.csv | Quel score chaque mandat attribue-t-il à une journée et de combien les scores varient-ils entre journées ? |
| shortlist.csv | Pour chaque mandat, échéance et règle, combien de journées graves sont retrouvées, quel est le regret, quelles conséquences mesurables sont omises ? |
| shortlist_days.csv | Quelle journée entre dans chaque liste et avec quelle probabilité en cas d’égalité au seuil ? |
| prediction_errors.csv | Quelle est l’erreur moyenne de chaque modèle sur les cinq scores et sur les probabilités de succès économique (E) ou économique avec service (ES) ? |

Les cinq scores sont des **pertes composites**. Le service utilise l’énergie non servie, le coût l’écart défavorable à la référence propre à la journée, le temps la fraction du délai consommée et les émissions le thermique interne par MWh servi. Les grandeurs physiques restent accessibles dans les observations. Une non-livraison observée vaut la perte maximale pour chaque mandat ; aucun délestage ni aucune émission physique ne lui sont imputés.

Les tests de régression du pilote sont exécutables sans résolution :

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s research/paper1_fixed_days/tests -p test_pilot.py -v
~~~
