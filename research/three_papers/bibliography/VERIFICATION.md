# Contrôle textuel des bibliographies

Contrôle effectué le **5 novembre 2026**, par relecture des fichiers livrés et vérification de leur structure textuelle. Aucun programme externe, commande, test scientifique, BibTeX ou compilateur LaTeX n'a été lancé.

| Contrôle | Papier 1 | Papier 2 | Papier 3 |
| --- | ---: | ---: | ---: |
| Entrées BibTeX | 47 | 75 | 42 |
| Entrées détaillées dans le guide | 47 | 75 | 42 |
| Clés déjà citées dans le manuscrit | 3 | 3 | 3 |
| Clés citées absentes de la bibliographie | 0 | 0 | 0 |
| Clés dupliquées dans le fichier | 0 | 0 | 0 |
| DOI dupliqués dans le fichier | 0 | 0 | 0 |
| Déséquilibre d'accolades | 0 | 0 | 0 |
| Caractères non ASCII dans les champs BibTeX, hors commentaires | 0 | 0 | 0 |

Le catalogue réunit **101 clés distinctes**, sans DOI dupliqué. Les listes par papier totalisent **164 entrées**. Chaque notice dispose d'un titre, d'auteurs, d'une année et d'au moins une URL source. Les notices d'article possèdent un champ de revue et celles de communication un champ d'actes. Les champs facultatifs volontairement omis sont signalés dans [la synthèse](../BIBLIOGRAPHIE.md).

Les fichiers BibTeX relus correspondent aux données du catalogue ; les guides contiennent toutes leurs clés, sans ajout inconnu. Les statuts se répartissent en 90 publications, 8 prépublications, 2 références de logiciel ou de documentation et 1 manuscrit technique.

Les trois manuscrits ne diffèrent de leur état avant cette intervention que par la directive bibliographique, qui pointe vers le fichier propre au papier. Toutes les citations existantes se résolvent, y compris après prise en compte du protocole commun. Aucun `\nocite{*}` n'a été ajouté.

Le fichier `pdfs/MILP-GNN.bib` relu après les écritures est identique au texte lu au départ. Le relevé de sélection couvre ses 40 clés : 25 reprises et 15 non reprises avec motif. L'ancien fichier partagé à six notices reste conservé et n'est plus la cible des trois manuscrits.

Les renvois locaux de l'index bibliographique, des trois guides, du relevé de sélection et du document de recherche pointent vers des fichiers existants. Les liens de recherche sont conservés pour la provenance ; ce contrôle textuel ne constitue pas une vérification automatique de disponibilité de toutes les pages web.

## Limites du contrôle

L'équilibrage des accolades et la résolution des clés ne remplacent pas une compilation. Le rendu final, l'ordre des noms, les coupures d'URL et la compatibilité avec le style bibliographique n'ont donc pas été vérifiés dans un PDF.

Les vérifications bibliographiques restent distinctes de la validation scientifique : elles ne démontrent ni la reproductibilité des résultats cités ni la validité des futures expériences du projet. L'exception de métadonnées concernant Liu et al., les champs facultatifs manquants et les statuts de prépublication sont conservés explicitement dans les livrables.
