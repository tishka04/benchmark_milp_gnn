# Routage délai–coût–ENS : version 2, décisions par délai

Implémentation préparée le 11 novembre 2026. Aucun calcul de campagne, entraînement, test Python ou lancement d'expérience n'a été effectué pendant cette modification.

## Critères et conventions

Une demande comporte délai, tolérance économique relative (ou plafond absolu), plafond d'ENS en MWh, probabilité d'échec tolérée et priorité coût/temps. Les demandes du benchmark sont des profils figés dans config.json ; leurs identifiants et valeurs font partie des identités d'exécution. Une demande modifiée sans nouveau gel est refusée.

La convention ENS nulle est **ENS ≤ 0,001 MWh**, conforme aux drafts actuellement présents. La valeur n'est pas confondue avec la tolérance des bilans en MW. La sensibilité 0,001/1/10/100 MWh ne relance aucune résolution : elle change les critères appliqués aux sorties déjà enregistrées. Un coût est l'objectif complet, pénalités ENS incluses.

Une condition inactive est codée par null, et non par l'infini dans les variables des modèles. Le gap conserve les valeurs négatives ; près d'une référence de valeur absolue ≤ 1 euro, la convention reste un dépassement absolu ≤ 1 euro. Aucun coût/référence/certificat/dispatch réel n'entre dans les caractéristiques d'inférence.

Un essai non observé reste inconnu. Pour un essai observé, une violation connue suffit à établir l'échec conjoint, même si un autre critère est inconnu. Sinon, une condition active inconnue laisse le succès inconnu. Cette conjonction explicite diffère de certains anciens labels économiques qui exigeaient une référence même pour identifier un échec. Aucun coût ni temps de terminaison naturel fictif n'est attribué aux échecs.

Les cinq délais demandés sont **5, 10, 30, 60 et 120 secondes**. Chacun est croisé avec les quatre profils, soit 20 demandes distinctes. Les observations historiques à 1 et 2 s restent dans les sources, mais sont exclues des analyses conjointes et de l'apprentissage du routeur dans cette version. Un résultat obtenu avec une allocation de 120 s ne devient jamais une observation à 30 s. Chaque budget reste une exécution indépendante.

## Modèles et choix

Le routeur apprend, par méthode, une probabilité de succès conjoint et deux régressions conditionnelles aux sorties acceptables : coût et temps de sortie complet. Les caractéristiques sont la taille, les 22 composantes de criticité, le budget disponible et les exigences actives. Risque et priorité agissent dans la règle de choix.

Les prétraitements et la calibration sont ajustés sans le groupe évalué. La calibration de type Platt utilise des prédictions hors pli internes regroupées par scénario, si son support le permet. Les cas constants ou insuffisants sont signalés. Aucun budget, profil ou répétition d'une même base ne passe d'un côté à l'autre de ces séparations.

Un minimum conservateur entre les probabilités des profils plus larges impose qu'un événement plus strict ne reçoive pas une probabilité supérieure à celle d'un événement qui le contient. La calibration hors pli du résultat de cette correction est rapportée ; cette correction ne crée pas une garantie de fiabilité. Aucune monotonie artificielle n'est imposée entre allocations indépendantes.

Parmi les méthodes prédites admissibles à p ≥ 1 − alpha, le routeur minimise le coût ou le temps attendu demandé. En cas d'égalité : probabilité de succès puis ordre déterministe. En l'absence de choix soutenu, il choisit la plus grande probabilité prévue et marque explicitement le repli et la non-satisfaction du risque demandé.

Les supports minimums sont 8 groupes connus et 8 groupes avec cible conditionnelle disponible. Les demandes hors de la plage de budgets apprise ou les caractéristiques hors des bornes marginales observées ne sont pas déclarées soutenues. Les bornes marginales ne constituent pas un test complet de distribution ; les composants saturés peuvent encore masquer des différences physiques. Le modèle peut être interrogé à un budget restant intermédiaire, signalé comme interpolation. Les règles empiriques utilisent le budget observé inférieur, sans inventer de label intermédiaire.

Les comparateurs sont HiGHS seul, un choix restreint à HiGHS/RH, une règle fixe, une règle par famille, une règle par quintile fixe d'indice, la logistique et l'arbre. Le boosting est optionnel avant le premier entraînement. Les pondérations apprises de criticité, les balayages de normalisation et l'étude complète des sous-ensembles de caractéristiques restent des extensions distinctes.

## Diagnostic et sélection de modèle

Le diagnostic conserve seulement les portefeuilles complets pour son plafond, et indique le support exclu. La méthode fixe est choisie hors des groupes évalués pour chaque profil et budget.

Les paramètres par défaut sont des choix de protocole à revoir avant freeze, pas des résultats :

- couverture, pour un profil et un délai fixés : gain moyen sur les scénarios ≥ 2 points et borne inférieure de l'intervalle groupé à 95 % positive ; aucun autre délai ne conditionne cette décision ;
- qualité conditionnelle : gain de coût relatif au choix fixe ≥ 1 %, ou temps gagné ≥ 0,5 s ; borne inférieure positive sur support commun acceptable ;
- pour cette seconde voie : couverture de l'oracle ≥ 1 − alpha et borne inférieure du gain de couverture ≥ −1 point ;
- dans les deux voies : ≥ 90 % de portefeuilles connus et au moins 20 groupes ; au moins 20 groupes de paires acceptables pour la seconde voie.

Chaque seuil ci-dessus s'applique séparément à un couple profil–délai, y compris le taux de satisfaction 1 − alpha. Le diagnostic publie les 20 décisions dans requests et continuation_by_request.csv ; enabled_requests contient seulement les couples autorisés. enabled_profiles est un résumé, jamais une autorisation pour les autres délais. Aucune moyenne entre délais ne déclenche la poursuite. Les intervalles sont ponctuels : ils ne constituent pas une garantie simultanée sur les 20 demandes.

L'oracle utilise les sorties futures et ignore le coût de sélection. Sa voie de non-infériorité est un filtre d'opportunité, pas une validation du sélecteur. Si aucun couple ne passe, aucun modèle n'est publié.

Les prédicteurs peuvent apprendre conjointement sur les cinq délais, avec le délai en entrée et les scénarios regroupés. Le choix du modèle par couple profil–délai utilise les scores hors pli : priorité aux candidats dont le taux de satisfaction atteint 1 − alpha et dont le contraste de couverture respecte la marge, puis au gain conditionnel déclaré. Si aucun candidat ne satisfait ces critères, la meilleure couverture hors pli détermine le choix, sans déclarer le risque garanti. Toutes les comparaisons, replis, supports et tables de calibration sont exportés séparément par délai. Les modèles sont autorisés uniquement pour les couples déclarés, même si un autre délai du même profil passe. Le test indépendant reste nécessaire.

## Temps en ligne et traces

Le chronomètre du superviseur commence avant lecture du scénario. Après calcul des caractéristiques, le routeur est interrogé avec le temps restant, diminué d'une réserve de sélection déclarée de 0,01 s. Le temps réel de sélection est ensuite déduit et la même règle de réserve de finalisation est appliquée au budget restant.

Si la sélection dépasse cette réserve déclarée, routing_selection_overrun le signale et le drapeau de risque soutenu est retiré. Cela ne change pas le succès physique mesuré : seule la sortie complète sous le délai permet d'établir le succès observé. Les travailleurs inutilisés doivent également se fermer avant l'échéance.

Le modèle prévoit le temps d'une méthode à partir des exécutions historiques complètes. Le portefeuille préchargé et la réutilisation des données peuvent changer ce temps : seule l'exécution chronométrée du routeur valide le bénéfice. Le préchargement reste séparé et doit être rapporté.

Les sorties contiennent la demande, la méthode choisie, les prédictions des quatre méthodes, les méthodes admissibles, les supports et alertes, le repli, le budget interrogé et le budget effectivement restant. Le choix de modèle et les résultats d'exécution distinguent chaque couple profil–délai. L'autorisation est vérifiée sur le délai total demandé, tandis que la prédiction utilise le temps réellement restant.

## Fichiers et version

- requirements.py : critères et profils sans dépendance aux solveurs.
- joint_analysis.py : couverture, violations, supports, contrastes, sensibilité ENS.
- routing.py : diagnostic, calibration groupée, régressions, choix et publication des modèles.
- experiments.py / worker.py / extract.py : identités par demande, temps restant et traçabilité.
- tests/test_service_routing.py : tests synthétiques préparés, non exécutés.
- GUIDE_COMMANDES.md et run_service_by_deadline.ps1 : lancement manuel dans runs/v2_service_by_deadline.

La graine des nouveaux cas est 20261111. Les anciens résultats v1 restent intacts ; leurs gels ne sont pas compatibles avec le code modifié. Les CSV nouvellement préparés ne remplacent pas les analyses déjà citées dans les drafts tant qu'ils n'ont pas été produits et examinés.

## Amendement après le premier diagnostic

Le premier lancement v2_service effectué par l'utilisateur a réussi ses tests (35 réussis, 2 ignorés), produit les tables conjointes et arrêté l'entraînement sur le diagnostic agrégé. Après examen de ce résultat, l'utilisateur a demandé les cinq délais ci-dessus et leur évaluation séparée. Le changement est donc un amendement de développement, pas une règle annoncée avant observation. Les seuils numériques de gain, de risque et de support sont conservés ; seule la condition de stabilité entre deux budgets est retirée au profit de décisions indépendantes.

La version v2_service_by_deadline utilise un nouveau schéma et un nouveau gel. Aucun résultat, modèle ni gel antérieur n'est réécrit. Les nouveaux tests et résultats n'ont pas été exécutés par l'assistant. Un succès à un délai ne sera annoncé qu'après production et lecture du nouveau diagnostic ; la validation sur des scénarios indépendants reste distincte.
