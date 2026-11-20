& {
    $ErrorActionPreference = 'Stop'
    $Project = 'C:\Users\coudr\OneDrive\projects\benchmark\benchmark_milp_gnn'
    Set-Location -LiteralPath $Project
    $Python = Join-Path $Project '.venv\Scripts\python.exe'
    $Run = Join-Path $Project 'research\three_papers\runs\v2_service_by_deadline'
    if (-not (Test-Path -LiteralPath $Python)) { throw 'Environnement Python absent.' }

    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:NUMEXPR_NUM_THREADS = '1'
    $env:THREE_PAPERS_INTEGRATION = '0'
    $env:THREE_PAPERS_GUROBI_TEST = '0'

    function Invoke-ProjectPython {
        & $Python -B @args
        if ($LASTEXITCODE -ne 0) { throw "Etape interrompue : code $LASTEXITCODE." }
    }
    function Invoke-ThreePapers {
        Invoke-ProjectPython -m research.three_papers --run $Run @args
    }

    # 1. Environnement et grille : 5, 10, 30, 60, 120 secondes.
    Invoke-ThreePapers prerequisites
    Invoke-ThreePapers profiles

    # 2. Tests, dont petits MILP synthetiques. Aucune campagne historique.
    Invoke-ProjectPython -m unittest discover -s .\research\three_papers\tests -p 'test_*.py' -v

    # 3. Nouveau gel ; relecture des sorties historiques, sans les recalculer.
    Invoke-ThreePapers inventory
    Invoke-ThreePapers freeze
    Invoke-ThreePapers extract

    # 4. Analyses et diagnostic separes pour chaque profil et chaque delai.
    Invoke-ThreePapers analyze-joint
    Invoke-ThreePapers oracle | Out-Null
    $DiagnosticPath = Join-Path $Run 'routing\diagnostic.json'
    $Diagnostic = Get-Content -LiteralPath $DiagnosticPath -Raw | ConvertFrom-Json
    $Diagnostic.requests |
        Select-Object profile_id, deadline_s, coverage_gate, secondary_gate, proceed |
        Format-Table -AutoSize

    # 5. Apprentissage uniquement pour les couples autorises par le diagnostic.
    if ($Diagnostic.proceed) {
        Write-Host 'Couples profil-delai autorises :'
        $Diagnostic.enabled_requests | Format-Table -AutoSize
        Invoke-ThreePapers train-router
    } else {
        Write-Host 'Aucun couple autorise : aucun routeur entraine.'
    }

    Write-Host 'Analyses conjointes :'
    Write-Host (Join-Path $Run 'artifacts\main_joint\tables')
    Write-Host 'Decisions separees :'
    Write-Host (Join-Path $Run 'artifacts\routing_diagnostic\tables\continuation_by_request.csv')
    Write-Host 'Diagnostic et eventuels modeles :'
    Write-Host (Join-Path $Run 'routing')
}
