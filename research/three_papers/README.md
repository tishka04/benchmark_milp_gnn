# Trois papiers — préparation isolée

Livraison du 5 novembre 2026. Le code, les tests, trois manuscrits et les commandes sont préparés. **Aucune commande, analyse, génération de scénarios, résolution, installation, formation ou compilation n'a été exécutée pour cette livraison.** Les limites et vérifications à effectuer sont dans [VALIDATION.md](VALIDATION.md).

Commencer par [GUIDE_COMMANDES.md](GUIDE_COMMANDES.md). Toutes les commandes se lancent depuis la racine du dépôt. Les calculs nouveaux et le gel définitif attendent la fin des étapes 10 et 11 de la campagne actuelle. Le fichier d'état lu pendant la préparation ne déclarait que les étapes 1 à 9 terminées ; cela ne constitue pas un état actualisé à l'heure où vous lirez ce document.

## Livrables

| Objet | Fichier |
| --- | --- |
| Papier 2, priorité de rédaction | [manuscripts/paper2_methods_generalization.tex](manuscripts/paper2_methods_generalization.tex) |
| Papier 1 | [manuscripts/paper1_criticality.tex](manuscripts/paper1_criticality.tex) |
| Papier 3, conditionné au diagnostic | [manuscripts/paper3_routing.tex](manuscripts/paper3_routing.tex) |
| Méthode physique et protocole communs | [manuscripts/common_protocol.tex](manuscripts/common_protocol.tex) |
| Bibliographie du papier 1 — 47 références | [references.bib](manuscripts/paper1/references.bib), [guide](manuscripts/paper1/README.md) |
| Bibliographie du papier 2 — 75 références | [references.bib](manuscripts/paper2/references.bib), [guide](manuscripts/paper2/README.md) |
| Bibliographie du papier 3 — 42 références | [references.bib](manuscripts/paper3/references.bib), [guide](manuscripts/paper3/README.md) |
| Recherche, provenance et vérification bibliographiques | [BIBLIOGRAPHIE.md](BIBLIOGRAPHIE.md) |
| Paramètres fixés | [config.json](config.json) |
| Interfaces, réemploi et choix scientifiques | [CONTRATS_ET_REEMPLOI.md](CONTRATS_ET_REEMPLOI.md) |
| Tests préparés | [tests](tests) |

## Organisation du code

- `core.py` : lectures sans réparation, identités amendées, gel des sources et ressources, écritures isolées.
- `cases.py` : nouvelles cohortes déterministes, groupes de variantes et transformations physiques.
- `extract.py` : revalidation, grille d'observations, indice commun, audit des dimensions.
- `statistics.py` : couverture, incertitudes, contrastes appariés, prédiction groupée, calibration et transfert.
- `experiments.py`, `worker.py`, `gurobi.py` : références et exécutions supervisées, arrêt effectif, Gurobi direct, instrumentation.
- `fixed_generator.py` : adaptation isolée du générateur courant et contrôle `perturb16`.
- `multihorizon.py` : corpus, architecture variable et entraînement AE → JEPA → EBM avec reprise.
- `routing.py` : diagnostic d'oracle, arrêt conditionnel, modèles simples et sélection avant résolution.
- `__main__.py` : commandes `python -B -m research.three_papers ...`.

Le dossier initial était `research/three_papers/runs/v1/`. La version courante utilise `research/three_papers/runs/v2_service_by_deadline/` ; les anciens résultats restent conservés. Les brouillons compilent avec des cadres explicites si les figures n'existent pas encore. Aucun fichier existant sous `src`, `configs`, `scripts`, `training`, `outputs` ou `pdfs` n'a été modifié par cette préparation.

## Extension délai–coût–ENS — 11 novembre 2026

La version préparée utilise `runs/v2_service`. Les profils, analyses conjointes, calibration groupée et sélection coût/temps avec signalement du repli sont décrits dans [SERVICE_ROUTING.md](SERVICE_ROUTING.md). Les commandes actualisées sont dans [GUIDE_COMMANDES.md](GUIDE_COMMANDES.md). Les nouveaux tests sont préparés mais n’ont pas été exécutés ; aucune expérience ni analyse de campagne n’a été lancée pendant cette modification. Les sorties v1 sont conservées.

## Cinq délais distincts — amendement du 11 novembre 2026

Le diagnostic, le choix de modèle et les scores du routeur sont désormais séparés pour **5, 10, 30, 60 et 120 s**, sous chacun des quatre profils. Un seul délai peut autoriser un profil ; les autres restent évalués indépendamment. Les prédicteurs peuvent partager un apprentissage avec le délai en entrée. La version `runs/v2_service_by_deadline` conserve le diagnostic agrégé antérieur dans `runs/v2_service`. Le script [run_service_by_deadline.ps1](run_service_by_deadline.ps1) prépare les tests, analyses et entraînement éventuel, sans relancer la campagne historique. Aucun de ces calculs n'a été exécuté par l'assistant lors de cet amendement.
