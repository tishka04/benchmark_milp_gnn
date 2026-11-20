[CmdletBinding()]
param(
    [ValidateRange(1, 61)]
    [int]$Workers = 8
)

& {
    $ErrorActionPreference = 'Stop'
    $Project = 'C:\Users\coudr\OneDrive\projects\benchmark\benchmark_milp_gnn'
    Set-Location -LiteralPath $Project
    $Python = Join-Path $Project '.venv\Scripts\python.exe'
    $Run = Join-Path $Project 'research\paper1_components\runs\components_v1'
    $Source = Join-Path $Project 'research\three_papers\runs\v2_service_by_deadline'

    foreach ($Required in @(
        $Python,
        (Join-Path $Source 'main_rows.json'),
        (Join-Path $Source 'freeze.json'),
        (Join-Path $Source 'routing\model_receipt.json')
    )) {
        if (-not (Test-Path -LiteralPath $Required)) {
            throw "Fichier requis absent : $Required"
        }
    }

    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:NUMEXPR_NUM_THREADS = '1'
    $env:VECLIB_MAXIMUM_THREADS = '1'
    $env:BLIS_NUM_THREADS = '1'

    function Invoke-Paper1Python {
        & $Python -B @args
        if ($LASTEXITCODE -ne 0) {
            throw "Etape interrompue : code $LASTEXITCODE."
        }
    }
    function Invoke-Paper1 {
        Invoke-Paper1Python -m research.paper1_components --run $Run @args
    }

    # 1. Tests synthetiques du module isole ; aucun solveur.
    Invoke-Paper1Python -m unittest discover -s .\research\paper1_components\tests -p 'test_*.py' -v

    # 2. Controle de provenance et selection des observations historiques.
    Invoke-Paper1 prepare --source-run $Source

    # 3. Audit et sensibilite arithmetique de l'indice.
    Invoke-Paper1 audit

    # 4. Ajustements independants sur plusieurs processus, un fil natif chacun.
    Write-Host "Analyses du papier 1 : $Workers processus maximum."
    Invoke-Paper1 fit --workers $Workers

    # 5. Calculs d'incertitude en parallele ; exports par le processus principal.
    Invoke-Paper1 report --workers $Workers

    Write-Host 'Audit des composantes :'
    Write-Host (Join-Path $Run 'audit\tables')
    Write-Host 'Resultats predictifs :'
    Write-Host (Join-Path $Run 'report\tables')
    Write-Host 'Figures :'
    Write-Host (Join-Path $Run 'report\figures')
}
