# Papier 2 — comparaison opérationnelle et transfert

Ce module est autonome vis-à-vis du routeur du papier 3. Il réutilise la physique,
le superviseur Windows, les modèles historiques et les conventions de la campagne
complète, sans les modifier. Les adaptations de `three_papers` sont locales :
scénarios, générateurs, réparation, extraction et apprentissage multi-horizons.

## Démarrage

Lire [GUIDE_COMMANDES.md](GUIDE_COMMANDES.md). Chaque étape doit être appelée
explicitement. `Check`, `AnalyzeExisting`, `Estimate` et `Manuscript` ne lancent
ni campagne ni entraînement. `Check` résout seulement quelques MILP synthétiques
minuscules, avec des plafonds solveur de deux secondes pour les tests de physique
et des délais globaux de cinq secondes pour les deux tests de processus CPU.
Il ignore Gurobi s'il manque.

Les autres étapes peuvent durer longtemps. L'appel explicite `-Stage Resume`
enchaîne la reprise de la campagne jusqu'au manuscrit, en tenant compte du
verrouillage du test final. Le plan contient 23 700 essais et 1 200 références : 311,25 h
d'allocations d'essais et 300 h de plafonds de références, hors préchargements,
entraînements et analyses. Ce ne sont pas des prévisions de durée réelle.

`-Stage Estimate` affiche maintenant un suivi par étape et le temps restant,
préchargements inclus, à partir des durées observées. Il distingue explicitement
les estimations, les provisions pour les configurations encore non mesurées et
les postes non chiffrés. Il peut être consulté pendant la campagne, en lecture
seule. Voir le guide pour `-HoursPerDay`, `-Json` et la méthode d'estimation.

## Conventions scientifiques

- Délais indépendants : 5, 10, 30, 60, 120 s. Une exécution de 120 s n'est jamais
  requalifiée en observation à un autre délai.
- Critère principal : sortie valide et terminée à temps, écart de coût <= 5 %,
  ENS <= 0,001 MWh. Échecs conservés ; observations inconnues séparées.
- Référence commune : HiGHS, 600 s solveur, 16 threads, plafond global 900 s.
  Incumbent, borne et optimum certifié restent distingués. Les gaps négatifs
  sont conservés ; près de zéro, la tolérance absolue est 1 EUR.
- Bootstrap apparié par base, stratifié par famille, 2 000 rééchantillonnages.
  Les 25 tests confirmatoires utilisent les discordances appariées et Holm.
- Les suites `four` et `ablation` ont des identités distinctes. Leur observation
  pipeline respective n'est pas fusionnée ; l'ablation est analysée en paires.
- Le contrôle `perturb16` conserve graines, initialisation, bruit, dérive,
  seuillage et ressources, en supprimant seulement le gradient d'énergie.
- La fraction modifiée par réparation exclut les décisions sans équipement
  applicable ; le taux non masqué et les dénominateurs sont aussi conservés.
- Les cellules historiques au-delà de 24 périodes sont inapplicables et absentes
  de la grille d'essais ; elles ne sont pas des échecs.
- Une seule graine multi-horizons est principale (20261112). Les deux autres
  (20261113/20261114) servent uniquement à la sensibilité prévue.

## Provenance et reprise

`Prepare` gèle sources, environnement principal, modèles et campagne historique,
puis génère des groupes nouveaux et audite les entrées. `Lock` exige les blocs
de développement et les neuf checkpoints AE/JEPA/EBM. Il ne dépend d'aucun
diagnostic ou modèle de routage. Les références et essais finaux sont interdits
avant ce verrou. Un nouvel entraînement est interdit après verrouillage.

Chaque résultat terminal est signé et immuable, y compris un échec. Relancer
une étape reprend les observations manquantes. Une tentative interrompue sans
résultat terminal reste inconnue ; ses journaux sont conservés. Changer le
protocole ou le code après gel exige un nouveau répertoire d'exécution.

Gurobi est optionnel pour l'environnement principal : son installation tardive
ne modifie pas les méthodes ouvertes. Sa version, son adaptateur et sa licence
sont contrôlés séparément avant ses essais. Aucune substitution de solveur de
réparation n'est effectuée. L'environnement doit rester identique entre les
essais comparés ; ne pas mettre à jour les dépendances historiques.

L'audit distingue données supervisées, entrée de l'encodeur et normalisation.
L'absence de liste complète des cas de préentraînement HTE est une limite
explicite, jamais une preuve d'absence de fuite. Les transformations portent sur
les données physiques réalisées, utilisées directement par l'heuristique,
le graphe et les modèles de résolution.

## Sorties

- `runs/methods_v1/analyses/` : instantanés adressés par contenu, tables et figures.
- `research/three_papers/manuscripts/paper2/` : texte, annexe et figures publiables.
- `output/pdf/paper2/` : PDF et journal de compilation, si LaTeX est disponible.
- `paper2_analysis_reproduction.zip` : archive transportable des observations,
  analyses et sources LaTeX. Elle ne prétend pas redistribuer les logiciels
  sous licence ou les données/modèles nécessaires à une nouvelle campagne.

Une publication scientifique externe n'est pas créée automatiquement. Les
expériences nouvelles ne sont jamais déclarées réalisées lorsqu'elles sont
seulement préparées. L'intégration automatique produit les tables, effets et
figures ; l'interprétation de nouveaux résultats exige ensuite une relecture
de l'abstract, de la discussion et de la conclusion.
