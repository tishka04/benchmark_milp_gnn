# Révision du papier 1 — protocole du 12 novembre 2026

Ce protocole met en œuvre le plan joint par l'utilisateur. Les 600 scénarios ont
déjà été explorés : les analyses historiques sont du développement, même avec
validation imbriquée. Aucune modification des données historiques ou du routeur.
Les cinq préférences du manuscrit sont conservées, dont `service4_co2`.

## Comparaisons principales

1. Indice original avec pente commune puis propre au délai ; coefficients
   non négatifs communs ; direction commune et amplitudes propres aux délais
   (rang un) ; coefficients non négatifs séparés. Pénalité sur les coefficients
   effectifs, moyenne sur les délais. Trois initialisations du modèle bilinéaire,
   diagnostic de convergence et dispersion des objectifs. Marge exploratoire
   d'équivalence : 0,001 de MSE absolue, déclarée avant ces nouveaux ajustements.
2. Même flexibilité temporelle pour constante, B1, B1+B3+B4, six et 22 composantes.
   Un gradient boosting fournit une référence flexible ; aucun choix de modèle
   après examen des plis externes. Les constantes sont retirées sur l'entraînement.
3. Transfert : six familles externes, validation interne laissant aussi une
   famille de côté. La famille sert au découpage, jamais de prédicteur.

Les plis historiques de l'étude des poids sont conservés. Chaque transformation
apprise et chaque choix de pénalité sont recalculés dans chaque pli interne.
Tous les délais d'un même scénario restent ensemble. Les résultats binaires
antérieurs ont leur propre découpage, conservé pour leur reproduction.

## Analyses secondaires déclarées

- Décomposition F, Q|F et service|F,Q, puis produits cohérents des probabilités.
  Repli vers une fréquence de Laplace quand le conditionnement laisse moins de
  dix observations ; constante exacte si la classe est unique. Comparaison
  directe / projection [0,1] / modèle continu en deux parties, calibration.
- Réduction des constantes et corrélations absolues >0,95 apprise sur train ;
  groupes physiques fixes (A2,A4), (A3,A5), (A7,A8) ; transformation arctangente
  monotone non saturante de A6 et B8 seulement, appliquée aux valeurs brutes.
- Sensibilités une à la fois : ancres ENS 0,005/0,01/0,02, coût 0,025/0,05/0,10,
  carbone 0,36/0,72/1,44 ; priorités 0,4/0,6/0,8 ; seuils E 1/5/10 %.
  Le facteur d'émission physique reste 0,72. Coût sans pénalités ENS/carbone
  évalué uniquement si les deux décompositions, candidat et référence, existent.
- Sous-groupes de certification, disponibilité des bornes, changements
  d'étiquettes entre références versionnées ; coûts négatifs et proches de zéro
  traités explicitement. Aucun incumbent n'est rebaptisé optimum.
- Répétitions historiques : fréquences par scénario, délai et taille ; durées
  observées seulement. Pas de reconstruction des étapes non journalisées.
- Allocation indépendante dans chaque pli externe, ES principal, E secondaire,
  budgets moyens 10/20/30/60 s. Comparateurs uniforme/mélange prédéfini, taille,
  indice, 22 composantes, oracle rétrospectif. Budget réservé et temps observé
  séparés. Bootstrap des portefeuilles avec nouvelle optimisation à chaque tirage.

Les contrastes principaux partagent les scénarios ; intervalles bootstrap
appariés, et intervalles simultanés par bootstrap joint du maximum standardisé sur les 50 contrastes
préférence × délai × (commun→rang un, rang un→séparé). Ils sont conditionnels aux
prédictions hors pli et ne quantifient pas toute la variabilité de réapprentissage.
Amendement numérique : le maximum standardisé remplace les quantiles extrêmes
Bonferroni, insuffisamment résolus avec 2 000 tirages ; les 50 contrastes restent
inclus sans sélection. Ces intervalles sont approximatifs.
La variabilité des poids entre plis inclut le choix de pénalité ; elle ne doit pas
être présentée comme un intervalle bootstrap de population.

## Calculs physiques distincts

Le protocole complet prévoit 217 références à 1 800 s solveur (plafond 2 100 s
par processus), quatre lots indépendants de 100 scénarios et trois répétitions
chronométrées sur 30 scénarios stratifiés. Avec 100 observations indépendantes,
la demi-largeur normale maximale d'une proportion à 95 % est environ 0,098 ;
cette précision descriptive ne garantit pas la puissance d'un contraste.
Les nouveaux lots : tirages non filtrés, horizons/résolutions, topologies,
paramètres découplés. Le modèle physique de transport reste identique. Toute
adaptation après ouverture des résultats exige un nouveau test indépendant.
Le coût maximal de calcul, la commande de lancement, les diagnostics et les
statuts sont enregistrés séparément ; préparation et exécution sont distinguées.

## Livrables et ordre

`prepare` : tables scénarios/exécutions/références, plis, empreintes, audit et
reproduction des pertes historiques. `fit` : analyses principales imbriquées.
`robustness` : représentations, cibles et sous-groupes. `allocation` : décisions
par portefeuille. `campaign-prepare` : manifestes de nouvelles résolutions.
`report` : tableaux, figures et texte LaTeX générés depuis les résultats.
Le manuscrit ne peut annoncer que les expériences effectivement terminées.
