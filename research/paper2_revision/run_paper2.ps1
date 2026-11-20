[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Check','Prepare','AnalyzeExisting','Estimate','ReferencesDevelopment','Development',
        'TrainingLabels','TrainMultiHorizon','ValidateMultiHorizon','Lock','ReferencesFinal','Final',
        'Gurobi','Analyze','Manuscript','Resume')]
    [string]$Stage,
    [string]$Run,
    [string]$Python
)
$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $Python) { $Python = Join-Path $Project '.venv/Scripts/python.exe' }
if (-not $Run) { $Run = Join-Path $PSScriptRoot 'runs/methods_v1' }
if (-not (Test-Path -LiteralPath $Python)) { throw "Environnement Python absent : $Python" }
$PreviousEnv = @{}
$Names = @('PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')
foreach ($Name in $Names) {
    $PreviousEnv[$Name] = [Environment]::GetEnvironmentVariable($Name, 'Process')
    [Environment]::SetEnvironmentVariable($Name, '1', 'Process')
}
Push-Location -LiteralPath $Project
try {
    if ($Stage -eq 'Resume') {
        $Stages = @('Check')
        if (-not (Test-Path -LiteralPath (Join-Path $Run 'design_lock.json'))) {
            $Stages += @(
                'Prepare', 'ReferencesDevelopment', 'Development', 'TrainingLabels',
                'TrainMultiHorizon', 'ValidateMultiHorizon', 'Lock'
            )
        }
        $Stages += @('ReferencesFinal', 'Final', 'Gurobi', 'Analyze', 'Manuscript')
        foreach ($NextStage in $Stages) {
            Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] $NextStage"
            & $PSCommandPath -Stage $NextStage -Run $Run -Python $Python
        }
        return
    }
    & $Python -B -m research.paper2_revision --run $Run --stage $Stage
    if ($LASTEXITCODE -ne 0) { throw "Étape $Stage interrompue (code $LASTEXITCODE)." }
} finally {
    Pop-Location
    foreach ($Name in $Names) { [Environment]::SetEnvironmentVariable($Name, $PreviousEnv[$Name], 'Process') }
}
