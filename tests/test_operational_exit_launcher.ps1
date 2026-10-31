# The interpreter is replaced by a tiny recorder: no campaign can run here.
$ErrorActionPreference = "Stop"
$launcher = Join-Path $PSScriptRoot "../scripts/run_operational_exit.ps1"
$tokens = $null
$errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($launcher, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
$names = @($ast.ParamBlock.Parameters | ForEach-Object { $_.Name.VariablePath.UserPath })
foreach ($name in @("Preview", "Resume", "StartStep", "EndStep", "AnalyzeOnly", "WithLabelRelaxation")) {
    if ($names -notcontains $name) { throw "Missing launcher parameter: $name" }
}
$directory = Join-Path (Join-Path $PSScriptRoot "../tmp") ("launcher-test-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $directory | Out-Null
$mock = Join-Path $directory "fake-python.ps1"
$capture = Join-Path $directory "arguments.json"
@'
$args | ConvertTo-Json | Set-Content -LiteralPath $env:OP_TEST_CAPTURE -Encoding UTF8
$global:LASTEXITCODE = [int]$env:OP_TEST_CODE
'@ | Set-Content -LiteralPath $mock -Encoding UTF8
$previousCapture = $env:OP_TEST_CAPTURE
$previousCode = $env:OP_TEST_CODE
try {
    $env:OP_TEST_CAPTURE = $capture
    $env:OP_TEST_CODE = "0"
    & $launcher -PythonExe $mock -Resume -StartStep 10 -EndStep 11 -WithLabelRelaxation -SourceRoot "source with spaces" -OutputDir "output with spaces"
    $received = Get-Content -LiteralPath $capture -Raw | ConvertFrom-Json
    foreach ($arg in @("--resume", "--with-label-relaxation", "source with spaces", "output with spaces")) {
        if ($received -notcontains $arg) { throw "Argument not forwarded: $arg" }
    }
    if ($received[[Array]::IndexOf($received, "--start-step") + 1] -ne "10") { throw "Wrong start step" }
    if ($received[[Array]::IndexOf($received, "--end-step") + 1] -ne "11") { throw "Wrong end step" }
    & $launcher -PythonExe $mock -Preview -AnalyzeOnly
    $received = Get-Content -LiteralPath $capture -Raw | ConvertFrom-Json
    if ($received -notcontains "--preview" -or $received -notcontains "--analyze-only") { throw "Read-only flags missing" }
    $env:OP_TEST_CODE = "7"
    $caught = $false
    try { & $launcher -PythonExe $mock -Resume } catch { $caught = $_.Exception.Message -like "*code 7*" }
    if (-not $caught) { throw "Nonzero interpreter exit was swallowed" }
    Write-Host "Operational launcher: parsing, arguments and failure propagation passed."
} finally {
    $env:OP_TEST_CAPTURE = $previousCapture
    $env:OP_TEST_CODE = $previousCode
}
