# No campaign execution: parse the launcher and load only isolated functions.
$ErrorActionPreference = "Stop"
$path = Join-Path $PSScriptRoot "../scripts/run_load_shifting_pipeline.ps1"
$tokens = $null
$errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
foreach ($name in @("TrainMilpWorkers", "TrainMilpThreads")) {
    $parameter = $ast.ParamBlock.Parameters | Where-Object { $_.Name.VariablePath.UserPath -eq $name }
    if ($parameter.DefaultValue.SafeGetValue() -ne 4) { throw "Wrong default: $name" }
}
foreach ($name in @("Invoke-MilpBatch", "Record-StepTiming")) {
    $function = $ast.FindAll({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] }, $true) |
        Where-Object { $_.Name -eq $name }
    Invoke-Expression $function.Extent.Text
}
function Archive-Output([string]$Path) { $script:Archives += $Path }  # No filesystem mutation.
function Invoke-Module([string]$Module, [string[]]$Arguments) {
    $script:Calls += [pscustomobject]@{
        module = $Module; arguments = $Arguments
        highs = $env:BENCHMARK_HIGHS_THREADS; blas = $env:OPENBLAS_NUM_THREADS
    }
    if ($script:FailModule) { throw "Simulated Python failure" }
}
$R = "unused"
$HighsThreads = 16
$TrainMilpWorkers = 4
$TrainMilpThreads = 4
$Calls = @()
$Archives = @()
$Resume = $false
$FailModule = $false
$names = @("BENCHMARK_HIGHS_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
$previous = @{}
foreach ($name in $names) { $previous[$name] = [Environment]::GetEnvironmentVariable($name, "Process") }
try {
    $env:BENCHMARK_HIGHS_THREADS = "16"
    $env:OPENBLAS_NUM_THREADS = "7"
    Invoke-MilpBatch "scenarios_v3"
    Invoke-MilpBatch "low_criticality_scenarios"
    if ($Calls[0].highs -ne "4" -or $Calls[1].highs -ne "16") { throw "Wrong solver thread allocation" }
    foreach ($i in 0,1) {
        $workerIndex = [Array]::IndexOf($Calls[$i].arguments, "--workers")
        $expected = @(4,1)[$i]
        if ([int]$Calls[$i].arguments[$workerIndex + 1] -ne $expected) { throw "Wrong worker count" }
        if ($Calls[$i].blas -ne "1") { throw "BLAS not capped" }
        if ($Calls[$i].arguments -contains "--resume") { throw "Fresh run must not reuse reports" }
    }
    if ($Archives.Count -ne 4 -or $Archives[0] -ne "$R/scenarios_v3/reports") {
        throw "Fresh run must archive previous reports and summary"
    }
    $Resume = $true
    $Archives = @()
    Invoke-MilpBatch "scenarios_v3"
    Invoke-MilpBatch "low_criticality_scenarios"
    foreach ($i in 2,3) {
        if ($Calls[$i].arguments -notcontains "--resume") { throw "Selective resume was not forwarded" }
    }
    if ($Archives.Count -ne 2 -or ($Archives | Where-Object { $_ -notlike "*/batch_summary.json" })) {
        throw "Resume must preserve reports and archive only per-attempt summaries"
    }
    $FailModule = $true
    $caught = $false
    try { Invoke-MilpBatch "scenarios_v3" } catch { $caught = $_.Exception.Message -eq "Simulated Python failure" }
    if (-not $caught) { throw "Failure did not propagate" }
    if ($env:BENCHMARK_HIGHS_THREADS -ne "16" -or $env:OPENBLAS_NUM_THREADS -ne "7") {
        throw "Training environment leaked after failure"
    }
    $State = [ordered]@{ stage_timings = @() }
    $RunId = "test"
    $ActiveStep = 2
    $StepStartedAt = [DateTime]::UtcNow.ToString("o")
    foreach ($status in @("completed", "failed")) {
        $script:StepTimer = [Diagnostics.Stopwatch]::StartNew()
        Record-StepTiming $status
    }
    if ($State.stage_timings.Count -ne 2 -or $State.stage_timings[1].status -ne "failed") {
        throw "Timing attempts not retained"
    }
    if ($null -ne $StepTimer -or $State.stage_timings[0].elapsed_wall_seconds -lt 0) {
        throw "Invalid wall timer"
    }
    Write-Host "PASS: launcher syntax, selective MILP resume, explicit fresh restart, 4x4 train / 1x16 test, environment restoration, failure propagation, stage timings."
} finally {
    foreach ($name in $previous.Keys) { [Environment]::SetEnvironmentVariable($name, $previous[$name], "Process") }
}
