[CmdletBinding()]
param(
    [ValidateRange(1, 61)] [int]$Workers = 8,
    [ValidatePattern('^[a-zA-Z0-9_-]+$')] [string]$RunName = 'weights_v2_variants',
    [switch]$TestsOnly,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$Python = Join-Path $Project '.venv\Scripts\python.exe'
$Run = Join-Path $PSScriptRoot ('runs\' + $RunName)
$Source = Join-Path $Project 'research\three_papers\runs\v2_service_by_deadline'
if (-not (Test-Path -LiteralPath $Python)) { throw "Python absent : $Python" }
if ($TestsOnly -and $SkipTests) { throw 'TestsOnly et SkipTests sont incompatibles.' }

Push-Location -LiteralPath $Project
try {
    $env:PYTHONDONTWRITEBYTECODE = '1'
    foreach ($Variable in @('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')) {
        [Environment]::SetEnvironmentVariable($Variable, '1', 'Process')
    }
    function Invoke-WeightsPython {
        & $Python -B @args
        if ($LASTEXITCODE -ne 0) { throw "Etape interrompue : code $LASTEXITCODE." }
    }
    if (-not $SkipTests) {
        Invoke-WeightsPython -m unittest discover -s .\research\paper1_weights\tests -p 'test_*.py' -v
    }
    if ($TestsOnly) { return }
    foreach ($Required in @('main_rows.json', 'freeze.json', 'routing\model_receipt.json')) {
        $Path = Join-Path $Source $Required
        if (-not (Test-Path -LiteralPath $Path)) { throw "Source historique absente : $Path" }
    }
    # Lecture des dispatchs et calcul du CO2 seulement : aucun MILP ni routeur lance.
    Invoke-WeightsPython -m research.paper1_weights --run $Run prepare --source-run $Source --workers $Workers
    # Regressions groupees, coefficients finaux, puis incertitude sur les poids.
    Invoke-WeightsPython -m research.paper1_weights --run $Run fit --workers $Workers
    Invoke-WeightsPython -m research.paper1_weights --run $Run bootstrap --workers $Workers
    Invoke-WeightsPython -m research.paper1_weights --run $Run report
    Write-Host 'Ponderations des criteres :' (Join-Path $Run 'report\policy_weights.csv')
    Write-Host 'Poids communs :' (Join-Path $Run 'report\weights_common.csv')
    Write-Host 'Poids par delai :' (Join-Path $Run 'report\weights_by_deadline.csv')
    Write-Host 'Emissions par dispatch :' (Join-Path $Run 'report\carbon_dispatch.csv')
    Write-Host 'Rapport :' (Join-Path $Run 'report\REPORT.md')
}
finally { Pop-Location }
