# Commandes PowerShell — version délai–coût–ENS

Mise à jour du 11 novembre 2026. **Code et tests préparés, sans exécution par l'assistant.** Aucun solveur, entraînement, test Python, analyse de campagne ou compilation n'a été lancé pendant cette modification.

Cette version utilise **research/three_papers/runs/v2_service_by_deadline**. Ne pas réutiliser les gels v1 ou v2_service : le diagnostic et la sélection sont désormais séparés pour 5, 10, 30, 60 et 120 s. Les seuils numériques sont conservés. Cet amendement intervient après lecture du premier diagnostic, et ne constitue pas une confirmation indépendante. Les résultats historiques, les sorties v1 et le diagnostic v2_service restent intacts. Les références et essais de la campagne principale sont relus, pas recalculés.

Pour lancer seulement les tests, analyses conjointes, diagnostic et éventuel entraînement, le fichier [run_service_by_deadline.ps1](run_service_by_deadline.ps1) regroupe les étapes dans cet ordre. Il ne contient ni lancement de campagne historique, ni génération de nouveaux scénarios, ni essai opérationnel de routeur. Les sections 5 à 8 du présent guide sont des expériences supplémentaires distinctes.

## 1. Session et prérequis

Ouvrir PowerShell puis exécuter :

```powershell
Set-Location -LiteralPath 'C:\Users\coudr\OneDrive\projects\benchmark\benchmark_milp_gnn'
$ErrorActionPreference = 'Stop'
$Project = (Get-Location).Path
$Python = Join-Path $Project '.venv\Scripts\python.exe'
$Run = Join-Path $Project 'research\three_papers\runs\v2_service_by_deadline'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Environnement Python absent.' }

$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

function Invoke-ThreePapers {
    & $Python -B -m research.three_papers --run $Run @args
    if ($LASTEXITCODE -ne 0) { throw "Étape interrompue : code $LASTEXITCODE." }
}

Invoke-ThreePapers prerequisites
Invoke-ThreePapers profiles
Invoke-ThreePapers inventory
```

Les modèles appris existants utilisent la RTX 4050. L'inventaire est une lecture des JSON ; le gel ultérieur vérifie les 18 480 identités attendues et les étapes 10/11. Une déclaration « completed » ne remplace pas cette vérification.

Si des dépendances d'analyse manquent, les installer avant le gel, en préservant les versions du runtime de campagne :

```powershell
& $Python -m pip install --no-deps -r .\research\three_papers\requirements-analysis.txt
if ($LASTEXITCODE -ne 0) { throw 'Installation à examiner.' }
```

Si la comparaison Gurobi est retenue, préparer également son interface et sa licence avant le gel. L'initialisation ci-dessous ne résout aucun modèle :

```powershell
Invoke-ThreePapers prerequisites --gurobi-license
```

Ne pas mettre à jour automatiquement les dépendances du runtime historique. Si Gurobi manque, utiliser une version compatible avec l'environnement installé ; le gel enregistrera sa version exacte.

## 2. Contrôler les profils et exécuter les tests préparés

Les quatre profils sont dans [config.json](config.json), sous service_routing :

| Profil | Qualité économique | ENS maximale | Tolérance d'échec | Priorité parmi méthodes admissibles |
| --- | --- | --- | --- | --- |
| urgent_continuity | Pas de seuil relatif | 0,001 MWh | 5 % | Coût |
| balanced_service | 5 % de la référence | 0,001 MWh | 5 % | Temps |
| economic_precision | 1 % de la référence | Non contrainte | 5 % | Temps |
| precision_continuity | 1 % de la référence | 0,001 MWh | 5 % | Temps |

Ces valeurs sont des choix explicites, modifiables **avant le gel**, et non des garanties mesurées. La tolérance ENS reprend celle des drafts actuels. Chaque profil est évalué séparément à **5, 10, 30, 60 et 120 s**. Les données à 1 et 2 s restent archivées mais sont exclues des analyses conjointes et de l'apprentissage du routeur. Les futures suites utilisent aussi ces cinq allocations. Aucune observation à 120 s n'est relabellisée à 30 s.

Les tests dédiés ci-dessous utilisent des données synthétiques et des interfaces simulées, sans campagne, solveur ni GPU :

```powershell
& $Python -B -m unittest research.three_papers.tests.test_service_routing research.three_papers.tests.test_service_deadlines -v
if ($LASTEXITCODE -ne 0) { throw 'Corriger les contrats délai/coût/ENS avant de poursuivre.' }
```

Ou directement le groupe complet, qui inclut ces tests et de petits MILP synthétiques :

```powershell
& $Python -B -m unittest discover -s .\research\three_papers\tests -p 'test_*.py' -v
if ($LASTEXITCODE -ne 0) { throw 'Tests préparés en échec.' }
```

Le test du worker complet est distinct et explicite :

```powershell
$env:THREE_PAPERS_INTEGRATION = '1'
try {
    & $Python -B -m unittest discover -s .\research\three_papers\tests -p 'test_integration.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Échec du worker complet.' }
} finally {
    Remove-Item Env:THREE_PAPERS_INTEGRATION -ErrorAction SilentlyContinue
}
```

Pour tester aussi l'adaptateur Gurobi sur un petit modèle, ajouter temporairement THREE_PAPERS_GUROBI_TEST=1 au même bloc. Cela ne garantit pas que la licence autorise les grandes instances.

Après réussite des contrôles :

```powershell
Invoke-ThreePapers freeze
```

## 3. Analyses de la campagne existante — papiers 1 et 2

Pas de nouvelle résolution. L'audit construit des modèles pour compter leurs dimensions ; l'extraction revalide les dispatchs ; l'analyse ajuste les prédicteurs du papier 1 et produit les comparaisons statistiques.

```powershell
Invoke-ThreePapers audit
Invoke-ThreePapers extract
Invoke-ThreePapers analyze
```

analyze produit désormais aussi les tables conjointes dans artifacts/main/tables :

- joint_coverage.csv : succès/échecs/inconnus, supports et intervalles par profil, famille, méthode et budget ;
- joint_violations.csv : validité/sortie, délai, coût et ENS ; les violations peuvent se recouvrir ;
- joint_successful_quality.csv : coût, temps total et ENS conditionnels au succès conjoint, avec supports ;
- joint_paired.csv : différences de couverture et différences coût/temps/ENS sur paires acceptables ;
- joint_paired_conditions.csv : différences entre variantes et bases ;
- joint_ens_sensitivity.csv : plafonds de 0,001, 1, 10 et 100 MWh, avec ou sans seuil économique ;
- joint_cells.csv et joint_protocol.json : observations, profils et conventions pour audit.

Pour produire uniquement ces analyses conjointes après extract, sans réajuster les prédicteurs de criticité, utiliser cette alternative :

```powershell
Invoke-ThreePapers analyze-joint
```

Elle écrit sous artifacts/main_joint. Inutile de lancer les deux commandes pour obtenir les mêmes tables conjointes.

## 4. Diagnostic et apprentissage du routeur — papier 3

```powershell
Invoke-ThreePapers oracle
Get-Content -LiteralPath (Join-Path $Run 'routing\diagnostic.json') -Raw
```

Le diagnostic est propre à chaque **couple profil–délai**. Il compare l'oracle à une méthode fixe choisie hors des groupes évalués, et distingue :

- un gain de satisfaction conjointe ;
- un gain de coût ou de temps sur support commun acceptable, avec couverture préservée.

Les seuils de continuation sont explicités dans [SERVICE_ROUTING.md](SERVICE_ROUTING.md). requests contient les 20 décisions ; enabled_requests contient les couples profil–délai autorisés à poursuivre. enabled_profiles n'est qu'un résumé des profils. Un couple sans potentiel suffisant reste un résultat négatif ; il n'est pas autorisé parce qu'un autre délai a réussi. Le tableau continuation_by_request.csv et oracle_gain.csv conservent cette séparation.

```powershell
Invoke-ThreePapers train-router
```

Cette commande s'arrête sans entraînement si aucun couple profil–délai ne passe le diagnostic. Sinon, elle prépare les modèles logistiques et arbres, les régressions conditionnelles de coût/temps, leur calibration groupée et les règles de comparaison : HiGHS, HiGHS/RH, méthode fixe, famille et indice. Elle publie les prédictions hors pli, scores, courbes de calibration tabulées et reçus dans routing/.

Option à choisir avant le premier entraînement, en remplacement de la commande précédente :

```powershell
Invoke-ThreePapers train-router --allow-flexible
```

Le choix de modèle, les scores et les tables de calibration sont séparés par couple profil–délai, sur les données de développement. Les prédicteurs peuvent partager un apprentissage avec le délai en entrée, sans moyenner les évaluations entre délais. Ces scores ne constituent pas une validation finale. Les pondérations apprises de l'indice et les balayages de normalisation évoqués comme alternatives dans les drafts ne sont pas implémentés par cette modification.

## 5. Nouveaux cas et expériences de développement

Cette partie lance réellement des résolutions. Ne pas l'exécuter en parallèle d'une autre campagne.

```powershell
Invoke-ThreePapers cases
Invoke-ThreePapers references --partition development
Invoke-ThreePapers experiments --partition development --suite four
Invoke-ThreePapers experiments --partition development --suite ablation
```

120 nouveaux scénarios ; 2 400 essais four et 1 200 essais d'ablation, aux allocations indépendantes de 5/10/30/60/120 s. Chaque référence reçoit 600 s solveur et un plafond global de 900 s hors préchargement. Les nouveaux cas utilisent la graine v2 ; ne pas copier les tests finaux v1 comme s'ils étaient neufs.

Après achèvement de chaque suite :

```powershell
Invoke-ThreePapers extract-extension --partition development --suite four
Invoke-ThreePapers analyze-extension --partition development --suite four
Invoke-ThreePapers extract-extension --partition development --suite ablation
Invoke-ThreePapers analyze-extension --partition development --suite ablation
```

Les analyses incluent automatiquement les nouveaux tableaux conjoints. L'ablation compare pipeline et perturb16 ; ne pas fusionner sa seconde observation pipeline avec la suite four.

Généralisation structurelle :

```powershell
Invoke-ThreePapers references --partition controlled_development
Invoke-ThreePapers experiments --partition controlled_development --suite four
Invoke-ThreePapers extract-extension --partition controlled_development --suite four
Invoke-ThreePapers analyze-extension --partition controlled_development --suite four
```

Le plan comporte 30 bases et leurs variantes, soit 3 600 essais four. Les modèles appris 24 h ne sont pas évalués à 48/72 h ; ces cellules sont inapplicables, pas des échecs observés.

## 6. Extensions facultatives

Gurobi : 120 cas historiques, 600 nouveaux essais, mêmes références historiques.

```powershell
Invoke-ThreePapers experiments --partition gurobi120 --suite gurobi
Invoke-ThreePapers extract-extension --partition gurobi120 --suite gurobi
Invoke-ThreePapers analyze-extension --partition gurobi120 --suite gurobi
Invoke-ThreePapers match-gurobi
Invoke-ThreePapers combine --partition gurobi120 --suites four gurobi
Invoke-ThreePapers transfer --partition gurobi120 --suite gurobi
```

Multi-horizons : 540 cas de labellisation et de nouveaux modèles AE/JEPA/EBM, sans remplacer les poids historiques.

```powershell
Invoke-ThreePapers references --partition mh_training
Invoke-ThreePapers dataset
Invoke-ThreePapers train --stage ae
Invoke-ThreePapers train --stage jepa
Invoke-ThreePapers train --stage ebm
Invoke-ThreePapers experiments --partition controlled_development --suite multi
Invoke-ThreePapers extract-extension --partition controlled_development --suite multi
Invoke-ThreePapers analyze-extension --partition controlled_development --suite multi
Invoke-ThreePapers combine --partition controlled_development --suites four multi
```

Plafonds conservés : AE 60 époques, JEPA 50, EBM 20 ; horizons d'apprentissage 12/24/48 h et séparation par base. Les 72 h restent hors apprentissage.

## 7. Verrouillage puis test final indépendant

Seulement après les choix de profils, de modèles et d'extensions :

```powershell
Invoke-ThreePapers lock-design
Invoke-ThreePapers references --partition final
Invoke-ThreePapers experiments --partition final --suite four
Invoke-ThreePapers references --partition controlled_test
Invoke-ThreePapers experiments --partition controlled_test --suite four
```

Le verrou vérifie les modèles si le diagnostic a autorisé un entraînement. Il contient les profils, conventions, paramètres de routage et empreintes des artefacts. Aucun entraînement ultérieur dans cette version.

Si un routeur a été publié :

```powershell
if (Test-Path -LiteralPath (Join-Path $Run 'routing\model_receipt.json')) {
    Invoke-ThreePapers experiments --partition final --suite router
    Invoke-ThreePapers experiments --partition final --suite routing_baselines
    Invoke-ThreePapers experiments --partition controlled_test --suite router
    Invoke-ThreePapers experiments --partition controlled_test --suite routing_baselines
}
```

Chaque couple profil–délai autorisé possède ses propres identités et exécutions. Pour Q couples autorisés : 120 × Q essais router et 840 × Q essais routing_baselines sur final ; 150 × Q et 1 050 × Q sur les conditions structurelles compatibles. Q est la longueur de enabled_requests, pas le nombre de profils multiplié automatiquement par cinq. Ces nombres sont des plans, pas des durées mesurées.

Si les modèles multi-horizons ont été entraînés et verrouillés :

```powershell
Invoke-ThreePapers experiments --partition controlled_test --suite multi
```

## 8. Exports finaux

Attendre la fin des suites retenues : les exports sont immuables.

```powershell
$FinalSuites = @('four')
$StructuralSuites = @('four')

if (Test-Path -LiteralPath (Join-Path $Run 'routing\model_receipt.json')) {
    $FinalSuites += @('router', 'routing_baselines')
    $StructuralSuites += @('router', 'routing_baselines')
}
if (Test-Path -LiteralPath (Join-Path $Run 'multi_horizon\ebm_receipt.json')) {
    $StructuralSuites += 'multi'
}

foreach ($Suite in $FinalSuites) {
    Invoke-ThreePapers extract-extension --partition final --suite $Suite
    Invoke-ThreePapers analyze-extension --partition final --suite $Suite
}
foreach ($Suite in $StructuralSuites) {
    Invoke-ThreePapers extract-extension --partition controlled_test --suite $Suite
    Invoke-ThreePapers analyze-extension --partition controlled_test --suite $Suite
}
Invoke-ThreePapers combine --partition final --suites $FinalSuites
Invoke-ThreePapers combine --partition controlled_test --suites $StructuralSuites
Invoke-ThreePapers transfer --partition final --suite four
Invoke-ThreePapers transfer --partition controlled_test --suite four
```

Les suites avec routeur utilisent les exports joint_* et timed_request_rows.csv, afin de ne pas agréger des exécutions destinées à des demandes différentes dans les anciens tableaux économiques. Les sorties fixes sont évaluées contre chaque profil ; une exécution du routeur n'est évaluée que contre sa demande effective. Les colonnes routing_* exposent le repli, les probabilités prévues et les temps restants.

Une analyse conjointe seule d'une table déjà extraite est également disponible :

```powershell
Invoke-ThreePapers analyze-joint --partition final --suite combined
```

## 9. Compilation, quand les résultats ont été relus

```powershell
Get-Command latexmk, pdflatex, bibtex -ErrorAction Stop
Push-Location .\research\three_papers\manuscripts
try {
    foreach ($Draft in @('paper2_methods_generalization', 'paper1_criticality', 'paper3_routing')) {
        & latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build "$Draft.tex"
        if ($LASTEXITCODE -ne 0) { throw "Compilation interrompue : $Draft" }
    }
} finally {
    Pop-Location
}
```

Inspecter les PDF, leurs chiffres, les supports, les figures et les références. La présence d'une commande ne constitue ni une exécution réussie ni un résultat validé.

## Reprise

- Même commande et même version : les identités terminales, échecs compris, ne sont pas rejouées.
- Une interruption sans résultat terminal reste distincte d'un échec observé.
- Ne pas supprimer le fichier de verrou pour contourner un processus actif.
- Ne pas modifier code, profils, dépendances ou configuration après freeze. Une correction exige une nouvelle version et une traçabilité du réemploi.
- Aucun fichier de la campagne historique n'est réparé ni réécrit par ces analyses.
