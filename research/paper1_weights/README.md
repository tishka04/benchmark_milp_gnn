# Papier 1 — variantes de pondération prédictive, avec CO₂

Ce module indépendant prépare une régression des performances opérationnelles sur les **22 composantes normalisées existantes**. Il réutilise les 600 scénarios historiques HiGHS, répétition principale, aux cinq délais indépendants 5, 10, 30, 60 et 120 s. Il n'importe ni le routeur, ni le code d'analyse figé, ni le modèle MILP.

**État de validation : les 27 tests précédents ont réussi lors du lancement utilisateur du 11 novembre 2026, puis l'extraction a échoué sur la forme des imports réseau. Le lecteur et son exemple synthétique sont corrigés ; la suite comporte désormais 29 tests et n'a pas été relancée pendant cette correction. Aucune régression n'avait commencé.**

## Deux familles de poids à distinguer

Les poids ENS/coût/temps/CO₂ définissent une préférence **choisie**, donc la cible à prédire. Ils ne sont pas estimés à partir des données. Les coefficients des composantes A1–A12 et B1–B10 sont ensuite ajustés pour prédire cette cible.

Six préférences sont préparées : les deux variantes initiales favorables à l'ENS, une pondération égale des quatre critères et des variantes favorables au temps, au coût ou au CO₂. Les priorités restent **relatives**, avec compensation possible. Les conventions, modifiables dans `protocol.json` avant la première préparation, sont :

| Cible | ENS | Coût | Temps | CO₂ |
|---|---:|---:|---:|---:|
| `service3` | 0,60 | 0,20 | 0,20 | 0 |
| `service4_co2` | 0,60 | 2/15 | 2/15 | 2/15 |
| `equal4` | 0,25 | 0,25 | 0,25 | 0,25 |
| `time_priority` | 2/15 | 2/15 | 0,60 | 2/15 |
| `cost_priority` | 2/15 | 0,60 | 2/15 | 2/15 |
| `co2_priority` | 2/15 | 2/15 | 2/15 | 0,60 |

Pour chacune des variantes prioritaires à quatre critères, le critère favorisé reçoit 60 % et chacun des trois autres reçoit 2/15, soit 13⅓ %. La variante égale affecte 25 % à chaque critère. Les deux profils initiaux sont conservés. Ces nombres sont des paramètres de préférence, pas une pondération scientifiquement optimale. Les erreurs entre des cibles différentes ne mesurent pas directement la supériorité d'une préférence sur l'autre : elles changent l'objet prédit.

Les six variantes utilisent les mêmes composantes, les mêmes normalisations et les mêmes plis de validation. Les rééchantillonnages des scénarios sont également communs. L'extraction des dispatchs et le calcul CO₂ sont partagés : ils ne sont pas répétés pour chaque préférence.

## Cible et unités

Pour une solution valide livrée à son délai propre b :

- t = `exit_s / b` : temps observé de sortie du programme dans le protocole historique. Ce n'est ni le temps de preuve d'optimalité, ni la somme des temps de cinq essais. La frontière de mesure historique est conservée.
- e = `max(ENS - 0.001 MWh, 0) / demande_MWh` : fraction non servie au-delà de la tolérance historique.
- g = `max((coût - référence) / max(abs(référence), 1 EUR), 0)`. L'écart signé est également conservé dans les données préparées. Un résultat meilleur que la référence n'est pas pénalisé.
- q = `CO2_direct_t / (demande_MWh - max(ENS,0))` : émissions territoriales directes rapportées à l'énergie servie. Si aucune énergie n'est servie, le terme carbone normalisé vaut 1 ; il ne reçoit pas une récompense de sobriété fictive.

La transformation `h(x;s) = x/(x+s)` pour x positif, et 0 à x=0, est bornée sans écrêter les grandes valeurs. Les ancres fixes sont : e₀=1 % de demande, g₀=5 % de coût, q₀=0,72 tCO₂/MWh servi. Chaque ancre donne un terme normalisé de 0,5. Ce sont des conventions explicites de mise à l'échelle ; les changer peut changer les poids appris. Aucune normalisation n'est choisie à partir des données de test.

```text
Y_service3    = 0.60 h(e;0.01) + 0.20 h(g;0.05) + 0.20 t
Y_service4_co2= 0.60 h(e;0.01) + (2/15) [h(g;0.05) + t + h(q;0.72)]
Y_p = alpha_ENS,p h(e;0.01) + alpha_coût,p h(g;0.05)
      + alpha_temps,p t + alpha_CO2,p h(q;0.72)
# Les alpha sont les coefficients de la ligne p du tableau ci-dessus.
```

**Sans solution valide livrée**, la cible vaut 1 par convention explicite de non-livraison. Les ENS, coûts et émissions indisponibles restent absents. On ne les remplace pas par zéro, ni par des résultats d'un essai plus long. Cette cible est un score de non-performance opérationnelle comprenant la disponibilité d'une solution ; ce n'est pas une régression du temps latent de convergence des essais interrompus.

## CO₂ calculé à partir des dispatchs existants

```text
CO2_direct_t = 0.72 * somme_z,t(p_thermal_MW[z,t] * dt_heures)
```

Le facteur 0,72 tCO₂/MWh provient du modèle historique, `src/milp/defaults.py`, et son empreinte est vérifiée. Il ne s'agit pas d'une nouvelle estimation du mix thermique réel. Les fichiers de dispatch sont sélectionnés par leur identité exacte (scénario, méthode, délai, répétition, graine, version du chronométrage), puis vérifiés contre le gel de la campagne. L'extraction lit uniquement ces fichiers ; elle ne reconstruit pas de dispatch et n'exécute aucun solveur.

Le lecteur respecte les formes du dispatch historique : la production thermique est une matrice zone × période, tandis que les imports et exports sont chacun une série temporelle unique pour tout le réseau. Les échanges ne sont donc pas multipliés par le nombre de zones.

Le périmètre est la **production thermique interne, en émissions directes**. Les autres productions et le déstockage ne reçoivent pas un facteur supplémentaire ; cela évite de compter deux fois la production déjà utilisée pour charger un stockage. Les émissions de cycle de vie et les émissions propres aux démarrages ne sont pas calculées. Les émissions importées sont inconnues, explicitement signalées comme telles ; les volumes importés et exportés sont exportés pour diagnostic. Aucun crédit carbone n'est attribué aux exports. Le ratio territorial par MWh servi n'est donc pas l'empreinte de consommation d'un réseau ouvert.

Les émissions du dispatch de référence sont également exportées, à titre descriptif. Le coût de référence n'est **pas** un minimum carbone. Un essai interrompu n'a pas de dispatch accepté et garde un CO₂ inconnu. À chaque délai, `support.csv` indique combien de dispatchs contribuent réellement au calcul.

Le coût économique historique inclut déjà `VOLL × ENS` et le prix du CO₂ appliqué à la production thermique. Ajouter des dimensions ENS et CO₂ leur donne volontairement une importance supplémentaire ; les quatre critères ne sont pas indépendants. Le module conserve le coût historique pour rester comparable à sa référence, et exporte les deux charges monétaires séparément. Les soustraire uniquement du résultat sans reconstruire une référence cohérente ne donnerait pas un écart à l'optimum du nouveau critère.

## Régression et conversion en poids de l'indice

Modèle principal, sans variable de famille et sans contrôle supplémentaire de taille (la taille est déjà représentée parmi les composantes) :

```text
Y_s,b = a_b + somme_j(beta_j * z_s,j) + erreur_s,b
beta_j >= 0
minimiser MSE + lambda * somme_j(beta_j²)
w_j = beta_j / somme_k(beta_k)
I_w(s) = somme_j(w_j * z_s,j)
Y_prédit(s,b) = a_b + [somme_j(beta_j)] * I_w(s)
```

Les intercepts sont propres aux délais ; les pentes sont communes dans le modèle principal. Chaque scénario contribue avec ses cinq observations, de même importance. On ne traite jamais les 3 000 lignes comme 3 000 scénarios indépendants. Les composantes restent sur leur échelle normalisée historique [0,1] : aucun changement du calcul de criticité original n'est appliqué.

La régression positive utilise les moindres carrés bornés de [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html) avec une pénalisation quadratique. La pénalité est choisie parmi les valeurs déclarées par validation interne. La conversion des pentes en poids est exacte **pour ce modèle additif et cette cible** ; elle n'est pas une conversion des anciennes importances par retrait de variables. Si toutes les pentes sont nulles, aucun indice n'est produit par substitution de poids uniformes.

B2 (horizon) et B5 (proxy constant) ne varient pas dans cette cohorte. Leur coefficient est fixé à zéro et absorbé dans les intercepts ; leur poids scientifique reste **non identifiable**. Les CSV distinguent `applied_index_weight` (zéro imposé) de `estimated_index_weight` (absent pour les constantes). Il y a 22 lignes de composantes, mais seulement 20 composantes potentiellement estimables ici.

Les très fortes corrélations A3–A5, A7–A8 et A2–A4 limitent l'identification individuelle. Ridge stabilise l'ajustement sans transformer ces allocations en effets causaux ou uniques. Les anciens résultats d'interactions ne sont pas réinterprétés comme des poids additifs.

Comparateurs prévus pour chaque cible :

- le même modèle positif ajusté séparément à chacun des cinq délais ;
- une régression commune autorisant des coefficients signés, pour mesurer le coût prédictif de la contrainte de positivité ;
- l'indice historique commun (0,5 × moyenne des 12 A + 0,5 × moyenne des 10 B), avec pente positive et intercepts réajustés ;
- des intercepts de délai seuls.

## Validation et incertitude préparées

Cinq plis externes par scénario, trois plis internes pour choisir la pénalisation, avec les mêmes séparations pour les comparaisons. La famille sert uniquement à équilibrer les plis ; elle n'entre pas dans la régression. Les cinq observations d'un scénario restent toujours ensemble. Les comparaisons par délai ne mélangent pas leurs mesures de temps.

Les sorties comprennent MSE, RMSE, MAE, R², corrélation de rang et proportion de prédictions hors [0,1]. Les prédictions linéaires ne sont pas écrêtées afin de conserver l'équation exacte de l'indice. Une lecture descriptive des scores sur les références certifiées est fournie ; elle ne constitue pas un réentraînement limité à ce sous-ensemble.

Pour les coefficients et les poids : **500 rééchantillonnages des scénarios avec réajustement complet des coefficients**, pour les modèles positifs communs et par délai. La pénalité sélectionnée sur le développement complet reste fixe dans ces réplications. Les intervalles sont donc conditionnels à ce choix, ponctuels et sans correction pour les multiples composantes/délais. Les répétitions partagent les mêmes tirages entre cibles et modèles. Les rangs sont décrits avec leurs intervalles ; une valeur moyenne ne constitue pas un classement universel.

Les gains de MSE entre comparateurs utilisent séparément 2 000 rééchantillonnages appariés des pertes hors pli, sans réajustement. Les deux types d'incertitude sont explicitement distingués dans les CSV.

L'ensemble des 600 cas reste un échantillon de développement déjà exploré. Ces analyses ne sont ni un test externe intact, ni une preuve de stabilité sur de nouveaux systèmes. La priorité donnée à un critère, ou le choix de pondérations égales, reste une convention de cible même si les coefficients des composantes sont estimés pour cette cible.

## Commandes PowerShell

Après l'erreur `Wrong dispatch shape: net_import`, le dossier observé ne contenait que le fichier de verrouillage : aucune donnée préparée ni aucun manifeste n'avait été publié. Le même nom de dossier peut donc être réutilisé pour ce lancement interrompu. Il n'est pas nécessaire de supprimer le verrou ni de relancer la campagne historique.

Le dossier par défaut est désormais `weights_v2_variants`. Un éventuel ancien dossier `weights_v1` est conservé ; il ne faut pas le réutiliser après ce changement de code et de protocole. Le lancement couvre les six préférences, avec des poids communs et cinq jeux de poids par délai pour chacune.

Depuis n'importe quel dossier, pour les tests synthétiques seuls :

```powershell
& 'C:\Users\coudr\OneDrive\projects\benchmark\benchmark_milp_gnn\research\paper1_weights\run_paper1_weights.ps1' -TestsOnly
```

Puis, si les tests réussissent, extraction des dispatchs/CO₂, régressions, bootstrap et rapports :

```powershell
& 'C:\Users\coudr\OneDrive\projects\benchmark\benchmark_milp_gnn\research\paper1_weights\run_paper1_weights.ps1' -Workers 8 -RunName 'weights_v2_variants' -SkipTests
```

Sans `-TestsOnly` ni `-SkipTests`, le lanceur exécute d'abord les tests puis toutes les étapes. Il arrête la suite au premier échec. Il n'appelle ni l'ancien module papier 1, ni la campagne historique, ni l'entraînement du routeur. Aucune installation n'est lancée. Le même environnement NumPy/SciPy est utilisé.

Chaque étape peut aussi être appelée explicitement :

```powershell
$Project = 'C:\Users\coudr\OneDrive\projects\benchmark\benchmark_milp_gnn'
Set-Location -LiteralPath $Project
$Python = Join-Path $Project '.venv\Scripts\python.exe'
$Run = Join-Path $Project 'research\paper1_weights\runs\weights_v2_variants'

function Invoke-WeightStage {
    & $Python -B -m research.paper1_weights --run $Run @args
    if ($LASTEXITCODE -ne 0) { throw "Etape interrompue : code $LASTEXITCODE." }
}
Invoke-WeightStage prepare --workers 8
Invoke-WeightStage fit --workers 8
Invoke-WeightStage bootstrap --workers 8
Invoke-WeightStage report
```

Le lanceur complet est préférable : il contrôle automatiquement chaque code de sortie. Le nombre de processus est modifiable à la reprise. Les travailleurs n'écrivent aucun fichier ; le parent publie des sorties immuables dans `research/paper1_weights/runs`. Les points déjà terminés sont réutilisés après contrôle de provenance. Ne modifier ni les sources ni `protocol.json` après la première préparation d'un dossier : une modification demande un nouveau `-RunName`. Les anciens résultats ne sont jamais supprimés.

L'extraction CO₂ peut être dominée par la lecture des dispatchs sur disque/OneDrive. Aucune durée n'a été mesurée et aucune accélération n'est promise.

## Sorties attendues

Sous `research/paper1_weights/runs/weights_v2_variants/report` :

| Fichier | Contenu |
|---|---|
| `policy_weights.csv` | Pondérations choisies des critères pour les six variantes, séparées des poids appris |
| `weights_common.csv` | 22 lignes par cible, poids communs, coefficients, intervalles, rangs et constantes |
| `weights_by_deadline.csv` | Poids séparés à 5, 10, 30, 60 et 120 s |
| `models.json` | Équations exactes, intercepts, échelles et paramètres gelés de la cible |
| `prediction_scores.csv` | Validation hors pli, par délai et sous-ensemble de certification |
| `paired_predictive_gains.csv` | Gains du modèle commun et coût éventuel des contraintes communes/positives |
| `fold_coefficients.csv` | Variation des coefficients et des pénalités entre plis |
| `carbon_dispatch.csv` | Émissions, énergie servie, imports/exports et charges ENS/CO₂ des dispatchs acceptés |
| `targets.csv` | Chaque terme normalisé et la pénalité distincte de non-livraison |
| `support.csv` | Scénarios, livraisons, échecs, références certifiées et CO₂ disponible par délai |
| `REPORT.md` | Synthèse avec limites et tableau des poids |

Les manifestes et reçus gardent les empreintes des sources, métadonnées et sorties. Les dispatchs sont vérifiés à l'extraction ; les étapes suivantes utilisent le snapshot préparé scellé plutôt que de recalculer les résultats historiques.
