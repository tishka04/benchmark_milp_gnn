#requires -Version 5.1
<#
.SYNOPSIS
Compare MILP, GPU batch K=16 plus 16 CPU repairs, and JEPA at equal wall budgets.
.DESCRIPTION
Independent from run_load_shifting_pipeline.ps1. Never starts training.
The Python parent shares the legacy campaign lock and refuses concurrent runs.
Use -Preview to display the plan without workers or output writes.
#>
[CmdletBinding()]
param(
    [switch]$Resume,
    [switch]$Preview,
    [switch]$AnalyzeOnly,
    [double[]]$Budgets = @(1, 2, 5, 10, 60, 120),
    [ValidateRange(1, 1024)][int]$HighsThreads = 16,
    [ValidateRange(1, 1024)][int]$K = 16,
    [ValidateRange(1, 1000000)][int]$LimitPerFamily,
    [string]$OutputDir = "",
    [string]$PythonExe = ""
)
$ErrorActionPreference = "Stop"
if ($K -gt $HighsThreads) { throw "K ne doit pas depasser le plafond de threads CPU." }
$repo = Split-Path -Parent $PSScriptRoot
if (-not $PythonExe) { $PythonExe = Join-Path $repo ".venv/Scripts/python.exe" }
if (-not $OutputDir) { $OutputDir = Join-Path $repo "outputs/load_shifting_v1/evaluation/budgets_v1" }
if (-not (Test-Path -LiteralPath $PythonExe)) { throw "Python introuvable : $PythonExe" }
$budgetText = ($Budgets | ForEach-Object {
    if ([double]::IsNaN($_) -or [double]::IsInfinity($_) -or $_ -le 0) { throw "Budget invalide : $_" }
    $_.ToString([Globalization.CultureInfo]::InvariantCulture)
}) -join ","
$arguments = @("-u", "-B", "-m", "scripts.benchmark_matched_budgets",
    "--root", (Join-Path $repo "outputs/load_shifting_v1"), "--output-dir", $OutputDir,
    "--budgets", $budgetText, "--threads", "$HighsThreads", "--k", "$K")
if ($Resume) { $arguments += "--resume" }
if ($Preview) { $arguments += "--preview" }
if ($AnalyzeOnly) { $arguments += "--analyze-only" }
if ($PSBoundParameters.ContainsKey("LimitPerFamily")) { $arguments += @("--limit-per-family", "$LimitPerFamily") }
Push-Location $repo
$watch = [Diagnostics.Stopwatch]::StartNew()
try {
    $previous = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $PythonExe @arguments
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previous }
    if ($code -ne 0) { throw "Comparaison interrompue (code $code). Reprendre avec -Resume et les memes options." }
} finally {
    $watch.Stop()
    Write-Host ("Temps reel de cette invocation : {0:N1} min." -f $watch.Elapsed.TotalMinutes)
    Pop-Location
}
