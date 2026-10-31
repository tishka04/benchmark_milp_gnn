#requires -Version 5.1
<#
.SYNOPSIS
Recalculate the 600 corrected MILP references, then audit criticality/stress/hardness.
.DESCRIPTION
Independent from the training/inference campaign. Stage 1: references, sequential
HiGHS with 16 threads and 600 s per MILP, followed by the reference LP relaxation.
Stage 2: paired before/after statistical diagnostics and HTML/PNG/SVG reporting.
-Resume keeps validated references. -StartStep 2 forces only the statistical audit.
-Preview checks the read-only inventory and prints the plan; no output is created.
#>
[CmdletBinding()]
param(
    [switch]$Resume,
    [switch]$Preview,
    [ValidateSet(1, 2)][int]$StartStep = 1,
    [ValidateRange(1, 100)][int]$Repeats = 10,
    [string]$SourceRoot = "",
    [string]$OutputDir = "",
    [string]$PythonExe = ""
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
if (-not $PythonExe) { $PythonExe = Join-Path $repo ".venv/Scripts/python.exe" }
if (-not $SourceRoot) { $SourceRoot = Join-Path $repo "outputs/load_shifting_v1" }
if (-not $OutputDir) { $OutputDir = Join-Path $repo "outputs/thermal_transitions_v1/reference_criticality" }
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Python introuvable : $PythonExe" }
$arguments = @("-B", "-u", "-m", "scripts.recompute_thermal_references",
    "--source-root", $SourceRoot, "--output-dir", $OutputDir,
    "--start-step", "$StartStep", "--repeats", "$Repeats")
if ($Resume) { $arguments += "--resume" }
if ($Preview) { $arguments += "--preview" }
$watch = [Diagnostics.Stopwatch]::StartNew()
$oldProgress = $ProgressPreference
$oldEncoding = $env:PYTHONIOENCODING
Push-Location $repo
try {
    $ProgressPreference = "Continue"
    $env:PYTHONIOENCODING = "utf-8"
    if (-not $Preview) {
        Write-Host "Etape 1/2 : 600 references, 1 scenario x 16 threads HiGHS, MILP 600 s."
        Write-Host "Etape 2/2 : analyse avant/apres et graphiques. Aucun entrainement ni inference."
        Write-Host "Sortie : $OutputDir"
    }
    $previous = $ErrorActionPreference
    try {
        # Native stderr must not hide the actual nonzero exit code on PS 5.1.
        $ErrorActionPreference = "Continue"
        & $PythonExe @arguments | ForEach-Object {
            $line = [string]$_
            if ($line.StartsWith("PROGRESS|")) {
                $fields = $line -split '\|', 6
                if ($fields.Length -eq 6) {
                    $stage = [int]$fields[1]
                    $count = [int]$fields[2]
                    $total = [int]$fields[3]
                    $fraction = $count / [double]$total
                    $seconds = [double]::Parse($fields[4], [Globalization.CultureInfo]::InvariantCulture)
                    Write-Progress -Id 0 -Activity "References corrigees et criticite" `
                        -Status "Etape $stage/2" -PercentComplete (50 * ($stage - 1 + $fraction))
                    Write-Progress -Id 1 -ParentId 0 -Activity "Etape $stage/2" `
                        -Status ("{0}/{1} ; temps reel {2:N1} min" -f $count, $total, ($seconds / 60)) `
                        -CurrentOperation $fields[5] -PercentComplete (100 * $fraction)
                }
            } else { Write-Host $line }
        }
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previous }
    if ($code -ne 0) {
        throw "Arret (code $code). Reprendre avec -Resume et le meme -OutputDir. Les references validees seront conservees."
    }
} finally {
    $watch.Stop()
    if (-not $Preview) { Write-Host ("Temps reel de cette invocation : {0:N1} min." -f $watch.Elapsed.TotalMinutes) }
    Write-Progress -Id 1 -Activity "Etape" -Completed
    Write-Progress -Id 0 -Activity "References corrigees et criticite" -Completed
    $ProgressPreference = $oldProgress
    $env:PYTHONIOENCODING = $oldEncoding
    Pop-Location
}
