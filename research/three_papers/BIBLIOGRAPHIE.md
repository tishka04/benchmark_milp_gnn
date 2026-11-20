# Bibliographies des trois papiers

Recherche arrêtée au **5 novembre 2026**. Les trois manuscrits disposent chacun de leur propre fichier `references.bib`, accompagné d'un guide de lecture et de suggestions d'emplacement des citations.

| Papier | Fichier BibTeX | Guide de lecture | Nombre |
| --- | --- | --- | ---: |
| 1 — Criticité et difficulté opérationnelle | [paper1/references.bib](manuscripts/paper1/references.bib) | [Lectures et utilisation](manuscripts/paper1/README.md) | 47 |
| 2 — Méthodes apprises et généralisation | [paper2/references.bib](manuscripts/paper2/references.bib) | [Lectures et utilisation](manuscripts/paper2/README.md) | 75 |
| 3 — Complémentarité et routage | [paper3/references.bib](manuscripts/paper3/references.bib) | [Lectures et utilisation](manuscripts/paper3/README.md) | 42 |

Il y a **101 références distinctes**, soit 164 occurrences réparties entre les trois fichiers. Les références communes sont volontairement reprises pour que chaque papier dispose d'une bibliographie autonome. Le corpus contient 90 publications, 8 prépublications, 2 références de logiciel ou de documentation et 1 manuscrit technique. « Publication » ne signifie pas que toutes les métadonnées ont pu être revérifiées directement chez l'éditeur : l'exception Liu et al. est signalée ci-dessous.

## Traçabilité

- [Méthode, périmètre, requêtes et limites de la recherche](bibliography/RECHERCHE.md).
- [Examen des 40 références du fichier initial](bibliography/TRI_BIBLIOGRAPHIE_ORIGINE.md) : 25 reprises, 15 écartées du périmètre actuel, avec motifs.
- [Catalogue structuré](bibliography/catalogue.json) : métadonnées, affectation aux papiers, rôle, statut, sources primaires et date de consultation.
- [Contrôle textuel des livrables](bibliography/VERIFICATION.md).

La recherche est approfondie et inclut des travaux récents de 2025–2026 ; elle ne garantit pas une exhaustivité absolue. Les notices, résumés et plusieurs textes intégraux ont été consultés. Il ne s'agit pas d'une revue systématique avec export intégral de bases bibliométriques ni d'une lecture complète de chacun des 101 travaux.

## Utilité scientifique

**Papier 1.** Le noyau relie la difficulté empirique des instances et les espaces d'instances à la construction d'indices composites, aux formulations UC et aux besoins physiques de flexibilité. Les références de calibration et de validation groupée soutiennent l'évaluation prédictive. La bibliographie distingue un indicateur de stress physique, une corrélation avec les échecs et une propriété universelle de difficulté, qui n'est pas démontrée.

**Papier 2.** Le noyau couvre le guidage appris des MILP, les méthodes UC proches, les réparations et les recherches de voisinage, les représentations JEPA/GNN et les modèles d'énergie, puis la généralisation en taille et hors distribution. Les travaux AC-OPF et d'apprentissage orienté décision sont marqués comme voisins. Les comparaisons doivent mesurer le coût complet jusqu'à la sortie valide ; la réussite d'une réparation ne suffit pas à isoler l'apport du générateur.

**Papier 3.** Le noyau couvre la sélection d'algorithmes, les portefeuilles, les coûts de caractéristiques, la sélection sous délai et la validation des probabilités. Les guides distinguent le choix d'une seule méthode, la configuration et l'ordonnancement d'un portefeuille. Une absence de complémentarité exploitable reste un résultat possible.

## Corrections de notices et choix de versions

| Clé | Choix retenu |
| --- | --- |
| `hutter2013` | Clé historique conservée pour les citations existantes ; version publiée dans *Artificial Intelligence* en **2014**, DOI [10.1016/j.artint.2013.10.003](https://doi.org/10.1016/j.artint.2013.10.003). |
| `bischl2016` | Version de revue ASlib, *Artificial Intelligence* 237, 41–58, au lieu de la seule notice arXiv. |
| `xavier2021` | Année du fascicule **2021** ; première publication en ligne en 2020. |
| `qin2025`, `park2024` | Publications IEEE définitives retenues à la place des préprints de 2023. |
| `chen2023` | Version ICLR 2023 et liste complète des cinq auteurs, dont Jianfeng Lu. |
| `cappart2023` | Version complète JMLR, sans doublon avec le texte IJCAI antérieur. |
| `pjetri2025` | Publication ISGT Europe 2025 et cinq auteurs, à la place de la notice Authorea sans date. |
| `du2019` | Titre final NeurIPS *Implicit Generation and Modeling with Energy Based Models* ; pas de doublon sous le titre arXiv différent. |
| `mandi2024` | Volume **80** selon JAIR, qui prime sur des notices institutionnelles indiquant 81. |
| `nair2021`, `sonnerat2022` | Versions arXiv explicitement datées ; aucune publication de revue non confirmée n'est attribuée. |
| `sun2022` | Prépublication retenue ; une mention de soumission n'est pas transformée en acceptation ICLR. |
| `lecun2006` | Manuscrit technique de 2006, sans reconstruire une notice de chapitre dont l'année est ambiguë. |
| `liu2023` | Titre, auteurs et méthode contrôlés dans le dépôt auteur. Les coordonnées IEEE, dont DOI et pages, proviennent du fichier fourni ; la page éditeur n'a pas pu être récupérée durant cette recherche. |
| `dong2026`, `kadioglu2010`, `hu2020` | Quelques champs facultatifs non confirmés sont laissés vides : respectivement volume ; pagination et DOI ; pagination finale. Les sources accessibles restent indiquées. |

Les références de logiciels suivent les recommandations de leurs projets. Elles n'attestent ni une version installée, ni un calcul local. L'article de Huangfu et Hall recommandé pour HiGHS porte sur le simplexe parallèle ; il ne démontre pas à lui seul les performances MILP revendiquées dans ce projet.

## Utilisation dans les manuscrits

Les directives bibliographiques pointent maintenant vers `paper1/references`, `paper2/references` et `paper3/references`, depuis le dossier `manuscripts`. Les clés déjà citées ont été conservées. Le texte scientifique et les résultats n'ont pas été réécrits.

Ces bibliographies sont des **réservoirs de rédaction** : seules les entrées citées avec `\cite{...}` apparaîtront dans les articles. Aucun `\nocite{*}` n'a été ajouté. Les guides proposent des citations supplémentaires, à intégrer lors de la rédaction.

L'ancien [manuscripts/references.bib](manuscripts/references.bib), qui contient les six notices initiales, est conservé comme historique ; il n'est plus utilisé par les trois manuscrits. Le fichier source [pdfs/MILP-GNN.bib](../../pdfs/MILP-GNN.bib) est inchangé.

La vérification porte sur les fichiers textuels livrés, leurs notices, leurs clés et leurs renvois. **Aucune commande, compilation LaTeX/BibTeX, installation, analyse statistique ou expérience n'a été lancée.** La compilation reste à effectuer selon le [guide existant](GUIDE_COMMANDES.md).
