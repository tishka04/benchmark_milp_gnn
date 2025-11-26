# Build Temporal Heterogeneous Graphs
# This script demonstrates how to generate temporal graphs in both modes

param(
    [string]$ScenarioDir = "outputs/scenarios_v1",
    [string]$ReportsDir = "outputs/scenarios_v1",
    [string]$Mode = "supra",  # "supra" or "sequence"
    [int]$MaxScenarios = -1   # -1 for all
)

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Temporal Heterogeneous Graph Builder" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Determine output directory based on mode
if ($Mode -eq "sequence") {
    $OutputDir = "outputs/temporal_graphs/sequence"
} elseif ($Mode -eq "supra") {
    $OutputDir = "outputs/temporal_graphs/supra"
} else {
    Write-Host "Error: Mode must be 'sequence' or 'supra'" -ForegroundColor Red
    exit 1
}

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Scenario Dir: $ScenarioDir"
Write-Host "  Reports Dir:  $ReportsDir"
Write-Host "  Output Dir:   $OutputDir"
Write-Host "  Mode:         $Mode"
Write-Host ""

# Create output directory if it doesn't exist
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# Build base command
$cmd = "python -m src.gnn.build_hetero_graph_dataset"
$cmd += " `"$ScenarioDir`""
$cmd += " `"$ReportsDir`""
$cmd += " `"$OutputDir`""
$cmd += " --temporal"
$cmd += " --temporal-mode $Mode"

if ($Mode -eq "supra") {
    Write-Host "Building supra-graph (time-expanded)..." -ForegroundColor Green
    Write-Host "  - Temporal edges: soc, ramp, dr"
    Write-Host "  - Time encoding: sinusoidal"
    $cmd += " --temporal-edges soc,ramp,dr"
    $cmd += " --time-enc sinusoidal"
} else {
    Write-Host "Building sequence (snapshots)..." -ForegroundColor Green
    Write-Host "  - Time encoding: cyclic-hod"
    $cmd += " --time-enc cyclic-hod"
}

Write-Host ""
Write-Host "Running command:" -ForegroundColor Cyan
Write-Host $cmd -ForegroundColor Gray
Write-Host ""

# Execute command
Invoke-Expression $cmd

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Generation complete!" -ForegroundColor Green
Write-Host "  Output: $OutputDir"
Write-Host "  Index:  $OutputDir/dataset_index.json"
Write-Host "================================================" -ForegroundColor Cyan
