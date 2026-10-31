#requires -Version 5.1
<#
.SYNOPSIS
Prepare, run or resume the isolated operational_exit_v1 protocol.
.DESCRIPTION
-Preview only reads the inventory. -EndStep 1 prepares without real computation.
Steps 2 through 11 perform real work only on an explicit later invocation.
See docs/operational_exit_v1.md for the protocol, gates and full commands.
#>
[CmdletBinding()]
param(
    [switch]$Preview,
    [switch]$Resume,
    [ValidateRange(1, 12)][int]$StartStep = 1,
    [ValidateRange(1, 12)][int]$EndStep = 12,
    [switch]$AnalyzeOnly,
    [switch]$WithLabelRelaxation,
    [string]$SourceRoot = "",
    [string]$OutputDir = "",
    [string]$PythonExe = ""
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
if (-not $PythonExe) { $PythonExe = Join-Path $repo ".venv/Scripts/python.exe" }
if (-not $SourceRoot) { $SourceRoot = Join-Path $repo "outputs/load_shifting_v1" }
if (-not $OutputDir) { $OutputDir = Join-Path $repo "outputs/thermal_transitions_v1/operational_exit_v1" }
if ($StartStep -gt $EndStep) { throw "StartStep doit être <= EndStep." }
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Python introuvable : $PythonExe" }
$arguments = @("-B", "-u", "-m", "src.operational_exit.campaign", "--source-root", $SourceRoot,
    "--output-dir", $OutputDir, "--start-step", "$StartStep", "--end-step", "$EndStep")
if ($Preview) { $arguments += "--preview" }
if ($Resume) { $arguments += "--resume" }
if ($AnalyzeOnly) { $arguments += "--analyze-only" }
if ($WithLabelRelaxation) { $arguments += "--with-label-relaxation" }
$timer = [Diagnostics.Stopwatch]::StartNew()
$oldEncoding = $env:PYTHONIOENCODING
Push-Location $repo
try {
    $env:PYTHONIOENCODING = "utf-8"
    $previous = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $PythonExe @arguments | ForEach-Object {
            $line = [string]$_
            if ($line.StartsWith("PROGRESS|")) {
                $fields = $line -split '\|', 6
                $stage = [int]$fields[1]
                $fraction = [int]$fields[2] / [double]$fields[3]
                $seconds = [double]::Parse($fields[4], [Globalization.CultureInfo]::InvariantCulture)
                Write-Progress -Id 0 -Activity "Operational exit v1" -Status "Etape $stage/12" `
                    -PercentComplete (100 * (($stage - 1 + $fraction) / 12))
                Write-Progress -Id 1 -ParentId 0 -Activity $fields[5] `
                    -Status ("{0}/{1} ; temps réel {2:N1} min" -f $fields[2], $fields[3], ($seconds / 60)) `
                    -PercentComplete (100 * $fraction)
            } else { Write-Host $line }
        }
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previous }
    if ($code -ne 0) { throw "Arrêt (code $code). Les résultats complets, succès et échecs, sont conservés. Reprendre avec -Resume." }
} finally {
    $timer.Stop()
    Write-Host ("Temps réel de cette invocation : {0:N1} min." -f $timer.Elapsed.TotalMinutes)
    Write-Progress -Id 1 -Activity "Etape" -Completed
    Write-Progress -Id 0 -Activity "Operational exit v1" -Completed
    $env:PYTHONIOENCODING = $oldEncoding
    Pop-Location
}
