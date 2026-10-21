#requires -Version 5.1
<#
.SYNOPSIS
Rebuild the load-shifting HTE / UC-JEPA / latent EBM benchmark, in order.
.DESCRIPTION
Run from any directory. -ListSteps only displays the plan.
-Resume resumes at the first unfinished step saved in _pipeline/state.json.
Within step 2, completed compatible MILP reports are retained per scenario.
-StartStep N explicitly restarts at N; -EndStep N stops after N.
Training restarts from epoch 1 within an interrupted training step.
Replaced stage outputs are archived, never silently mixed with a previous fit.
The parallel benchmark resumes at completed (scenario, K) experiments.
Candidates are generated in one CUDA batch, then repaired in K CPU processes.
Default K grid: 1,2,5,8,12,16, keeping values within the HiGHS thread count.
One deadline covers both phases; the budget is never divided by K.
Standard inference and JEPA each have their own strict 120-second deadline.
Training references use 4 processes x 4 HiGHS threads by default.
Test references and the final MILP baseline retain -HighsThreads.
Progress bars count completed steps and sub-operations, not elapsed compute time.
.EXAMPLE
.\scripts\run_load_shifting_pipeline.ps1 -ListSteps
.EXAMPLE
.\scripts\run_load_shifting_pipeline.ps1 -HighsThreads 16
.EXAMPLE
.\scripts\run_load_shifting_pipeline.ps1 -Resume
.EXAMPLE
.\scripts\run_load_shifting_pipeline.ps1 -StartStep 5
.EXAMPLE
.\scripts\run_load_shifting_pipeline.ps1 -StartStep 12 -BestOfK 1,2,5,8,12,16
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 13)][int]$StartStep = 1,
    [ValidateRange(1, 13)][int]$EndStep = 13,
    [switch]$Resume,
    [switch]$ListSteps,
    [string]$PythonExe = "",
    [ValidateRange(0, 1024)][int]$HighsThreads = 0,
    [ValidateRange(1, 1024)][int]$TrainMilpWorkers = 4,
    [ValidateRange(1, 1024)][int]$TrainMilpThreads = 4,
    [ValidateRange(1, 1024)][int]$InferenceK = 5,
    [ValidateRange(0.1, 86400)][double]$GlobalTimeBudgetSeconds = 120,
    [ValidateRange(0.1, 86400)][double]$JepaTimeBudgetSeconds = 120,
    [ValidateRange(0.1, 86400)][double]$ParallelBudgetSeconds = 120,
    [int[]]$BestOfK = @()
)

$ErrorActionPreference = "Stop"
$Steps = @{
    1 = "Copier et verifier les scenarios (24 pas)"
    2 = "Recalculer les MILP et classer gold/silver"
    3 = "Construire les graphes temporels"
    4 = "Entrainer l'encodeur HTE"
    5 = "Generer les embeddings avec le meilleur HTE"
    6 = "Extraire les labels UC train/test"
    7 = "Entrainer l'autoencodeur UC"
    8 = "Entrainer JEPA"
    9 = "Entrainer le Latent EBM"
    10 = "Inferer le pipeline standard et JEPA + LP"
    11 = "Calculer les metriques standard"
    12 = "Comparer HiGHS (P threads) au batch GPU + K reparations CPU"
    13 = "Calculer les metriques du best-of-K parallele"
}
if ($ListSteps) {
    $Steps.GetEnumerator() | Sort-Object { [int]$_.Key } | ForEach-Object {
        [pscustomobject]@{ Etape = $_.Key; Action = $_.Value }
    } | Format-Table -AutoSize
    return
}
if ($Resume -and $PSBoundParameters.ContainsKey("StartStep")) {
    throw "Utiliser -Resume OU -StartStep."
}

$Repo = Split-Path -Parent $PSScriptRoot
$R = Join-Path $Repo "outputs/load_shifting_v1"
$Control = Join-Path $R "_pipeline"
$StatePath = Join-Path $Control "state.json"
$ParallelProtocol = "gpu_batch_cpu_repairs_v1"
$StandardProtocol = "strict_standard_separate_v1"
$RunId = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$TestFamilies = @(
    "low_criticality_scenarios", "medium_criticality_scenarios",
    "high_criticality_scenarios", "persona_vre_battery_scenarios",
    "persona_network_operator_scenarios", "persona_mathematician_scenarios"
)
$AllFamilies = @("scenarios_v3") + $TestFamilies
$StepWorkUnits = @{
    1 = $AllFamilies.Count + 1       # Copy each family, then validate
    2 = $AllFamilies.Count + 2       # Validate, solve each family, classify
    3 = $AllFamilies.Count
    4 = 1
    5 = 2
    6 = $TestFamilies.Count + 2      # Check embeddings, train labels, test labels
    7 = 2; 8 = 2; 9 = 2             # Validate, then train
    10 = 1; 11 = 1; 12 = 1; 13 = 1
}
$ActiveStep = 0
$CompletedWorkUnits = 0
$StepTimer = $null
$StepStartedAt = $null
$ConfigPaths = @(
    "src/gnn/embeddings/config.py", "configs/uc_jepa/autoencoder.yaml",
    "configs/uc_jepa/jepa.yaml", "configs/uc_jepa/latent_ebm.yaml"
)

function Assert-Path([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Fichier/dossier absent : $Path" }
}

function Show-PipelineProgress([switch]$StepCompleted) {
    $completed = $ActiveStep - 1
    if ($StepCompleted) { $completed = $ActiveStep }
    Write-Progress -Id 0 -Activity "Pipeline load shifting - $($Steps.Count) etapes" `
        -Status "$completed/$($Steps.Count) etapes franchies - $($Steps[$ActiveStep])" `
        -CurrentOperation "Execution des etapes $StartStep a $EndStep" `
        -PercentComplete ([int][Math]::Floor(100.0 * $completed / $Steps.Count))
}

function Show-StepProgress([string]$Operation, [switch]$Advance, [switch]$Finished) {
    if ($ActiveStep -lt 1) { return }
    if ($Advance) { $script:CompletedWorkUnits++ }
    $total = [int]$StepWorkUnits[$ActiveStep]
    # Keep 100% for the final checks and successful state save.
    $percent = [int][Math]::Min(99, [Math]::Floor(100.0 * $CompletedWorkUnits / $total))
    $status = "$CompletedWorkUnits/$total sous-operations terminees"
    if ($Finished) { $percent = 100; $status = "Etape terminee et sauvegardee" }
    Write-Progress -Id 1 -ParentId 0 -Activity "Etape $ActiveStep : $($Steps[$ActiveStep])" `
        -Status $status -CurrentOperation $Operation -PercentComplete $percent
}

function Save-State {
    $State.updated_at = [DateTime]::UtcNow.ToString("o")
    $temp = "$StatePath.tmp"
    [IO.File]::WriteAllText($temp, ($State | ConvertTo-Json -Depth 8),
        [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temp -Destination $StatePath -Force
}

function Record-StepTiming([string]$Status) {
    if ($null -eq $script:StepTimer) { return }
    $script:StepTimer.Stop()
    $elapsed = [Math]::Round($script:StepTimer.Elapsed.TotalSeconds, 3)
    $State.stage_timings = @($State.stage_timings) + [pscustomobject]@{
        run_id = $RunId; step = $ActiveStep; status = $Status
        started_at = $StepStartedAt; finished_at = [DateTime]::UtcNow.ToString("o")
        elapsed_wall_seconds = $elapsed
    }
    Write-Host ("Temps reel etape {0}: {1:N1} min ({2:N2} h), statut {3}." -f `
        $ActiveStep, ($elapsed / 60), ($elapsed / 3600), $Status)
    $script:StepTimer = $null
}

function Archive-Output([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return }
    # Resolve and check both endpoints before moving a directory.
    $rootFull = [IO.Path]::GetFullPath($R).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $sourceFull = [IO.Path]::GetFullPath($Path)
    if (-not $sourceFull.StartsWith($rootFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Archivage hors du dossier de sorties refuse : $sourceFull"
    }
    $relative = $sourceFull.Substring($rootFull.Length)
    $destination = [IO.Path]::GetFullPath((Join-Path "$R/_archive/$RunId" $relative))
    if (-not $destination.StartsWith($rootFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Destination d'archive invalide : $destination"
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Move-Item -LiteralPath $sourceFull -Destination $destination
    Write-Host "Archive : $destination"
}

function Invoke-Module([string]$Module, [string[]]$Arguments) {
    Show-StepProgress -Operation $Module
    Write-Host ("python -m {0} {1}" -f $Module, ($Arguments -join " ")) -ForegroundColor DarkCyan
    # pandas must precede torch in this Windows environment.
    $previousPreference = $ErrorActionPreference
    try {
        # Windows PowerShell can expose ordinary native stderr (e.g. tqdm) as
        # error records. Judge Python by its exit code, not its progress stream.
        $ErrorActionPreference = "Continue"
        & $PythonExe -u -B -c "import pandas; import runpy,sys; m=sys.argv.pop(1); runpy.run_module(m,run_name='__main__',alter_sys=True)" $Module @Arguments
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
    if ($code -ne 0) { throw "$Module a echoue (code $code)." }
    Show-StepProgress -Operation "$Module termine" -Advance
}

function Invoke-MilpBatch([string]$Family) {
    $workers = 1
    $threads = $HighsThreads
    if ($Family -eq "scenarios_v3") {
        $workers = $TrainMilpWorkers
        $threads = $TrainMilpThreads
    }
    if (-not $Resume) { Archive-Output "$R/$Family/reports" }
    # Keep per-attempt timing summaries, even when the reports are reused.
    Archive-Output "$R/$Family/batch_summary.json"
    $blasNames = @("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    $previous = @{}
    foreach ($name in (@("BENCHMARK_HIGHS_THREADS") + $blasNames)) {
        $previous[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
    }
    try {
        # Each fresh Python pool inherits this allocation. Restore it even on error,
        # so training settings cannot leak into test references or the benchmark.
        $env:BENCHMARK_HIGHS_THREADS = [string]$threads
        foreach ($name in $blasNames) { [Environment]::SetEnvironmentVariable($name, "1", "Process") }
        Write-Host "MILP $Family : $workers processus x $threads threads HiGHS."
        $batchArguments = @("$R/$Family", "--solver", "highs",
            "--workers", "$workers", "--reports-dir", "$R/$Family/reports",
            "--summary-json", "$R/$Family/batch_summary.json", "--fail-fast")
        if ($Resume) { $batchArguments += "--resume" }
        Invoke-Module "src.milp.batch_runner" $batchArguments
    } finally {
        foreach ($name in $previous.Keys) {
            [Environment]::SetEnvironmentVariable($name, $previous[$name], "Process")
        }
    }
}

function Invoke-Source([string]$Source, [string[]]$Arguments) {
    Show-StepProgress -Operation "Traitement Python en cours"
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $Source | & $PythonExe -u -B - @Arguments
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
    if ($code -ne 0) { throw "Le bloc Python a echoue (code $code)." }
    Show-StepProgress -Operation "Traitement Python termine" -Advance
}

$ValidateCode = @'
import json
import sys
from pathlib import Path
import yaml

root = Path(sys.argv[1]).resolve()
configs = [yaml.safe_load(Path(f"configs/uc_jepa/{name}.yaml").read_text(encoding="utf-8"))
           for name in ("autoencoder", "jepa", "latent_ebm")]
expected_dirs = ("autoencoder", "jepa", "latent_ebm")
for cfg, name in zip(configs, expected_dirs):
    if [Path(p).resolve() for p in cfg["data"]["roots"]] != [root / "uc_jepa/dataset_train"]:
        raise ValueError(f"{name}: unexpected training data roots")
    if Path(cfg["logging"]["output_dir"]).resolve() != root / "uc_jepa" / name:
        raise ValueError(f"{name}: unexpected output directory")
    for component, filename in (("ae", "autoencoder/uc_ae_best.pt"), ("jepa", "jepa/jepa_best.pt")):
        if component in cfg and Path(cfg[component]["checkpoint"]).resolve() != root / "uc_jepa" / filename:
            raise ValueError(f"{name}: unexpected {component} checkpoint")
if configs[0]["data"]["n_timesteps"] != 24 or configs[0]["model"]["n_timesteps_max"] != 24:
    raise ValueError("This runner requires 24 time steps")
for family in sys.argv[2:]:
    files = sorted((root / family).glob("scenario_*.json"))
    if not files:
        raise ValueError(f"No scenarios: {family}")
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if abs(float(data["horizon_hours"]) * 60 / float(data["dt_minutes"]) - 24) > 1e-8:
            raise ValueError(f"{path}: horizon must contain exactly 24 steps; do not truncate")
print("Scenario horizons and training paths validated.")
'@

$HteCode = @'
import pandas
import random
import sys
from pathlib import Path
import numpy as np
import torch
from src.demand_response import require_current_formulation
from src.gnn.embeddings.config import TrainingConfig
from src.gnn.embeddings.dataset import load_dataset_v3, compute_norm_stats
from src.gnn.embeddings.train import run_training
from src.gnn.embeddings.generate import generate_embeddings
from src.gnn.models.hierarchical_temporal_encoder import HierarchicalTemporalEncoder

mode, root = sys.argv[1], Path(sys.argv[2]).resolve()
cfg = TrainingConfig(
    repo_path=str(Path.cwd()), data_dir=str(root / "graphs/scenarios_v3"),
    scenario_dir=str(root / "scenarios_v3"),
    save_dir=str(root / "encoders/hierarchical_temporal_v3"), num_workers=0,
)
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if mode == "train":
    dataset, train_loader, val_loader, _ = load_dataset_v3(cfg)
    n_features = dataset[0].x.shape[1]
else:
    checkpoint = torch.load(Path(cfg.save_dir) / "best_encoder.pt",
                            map_location=cfg.device, weights_only=False)
    require_current_formulation(checkpoint)
    cfg = checkpoint["config"]
    cfg.repo_path = str(Path.cwd())
    n_features = checkpoint["model_state_dict"]["input_proj.weight"].shape[1]
model = HierarchicalTemporalEncoder(
    node_feature_dim=n_features, hidden_dim=cfg.hidden_dim, num_heads=cfg.num_heads,
    num_spatial_layers=cfg.num_spatial_layers, num_temporal_layers=cfg.num_temporal_layers,
    dropout=cfg.dropout, max_seq_len=cfg.max_seq_len,
).to(cfg.device)
if mode == "train":
    mean, std = compute_norm_stats(train_loader.dataset, device=cfg.device)
    run_training(model, train_loader, val_loader, cfg, mean, std)
else:
    model.load_state_dict(checkpoint["model_state_dict"])
    mean, std = checkpoint["train_mean"].to(cfg.device), checkpoint["train_std"].to(cfg.device)
    for family in sys.argv[3:]:
        cfg.data_dir, cfg.scenario_dir = str(root / "graphs" / family), str(root / family)
        generate_embeddings(model, cfg, mean, std, output_dir=str(root / "embeddings" / family))
'@

$LabelCheckCode = @'
import sys
from pathlib import Path
import numpy as np
from src.demand_response import require_current_formulation
root = Path(sys.argv[1])
for family in sys.argv[2:]:
    inputs = list((root / family).glob("scenario_*.json"))
    directory = root / "embeddings" / family
    for path in inputs:
        with np.load(directory / f"{path.stem}.npz", allow_pickle=True) as data:
            require_current_formulation(data, str(path))
            if data["zones"].ndim != 3 or data["zones"].shape[1:] != (24, 128):
                raise ValueError(f"Invalid embedding dimensions: {path}")
print("All scenario embeddings are present and current.")
'@

$lock = $null
$State = $null
$transcriptStarted = $false
$oldThreadEnv = $env:BENCHMARK_HIGHS_THREADS
$oldEncoding = $OutputEncoding
$oldProgressPreference = $ProgressPreference
Push-Location $Repo
try {
    $ProgressPreference = "Continue"
    if (-not $PythonExe) { $PythonExe = Join-Path $Repo ".venv/Scripts/python.exe" }
    $PythonExe = (Resolve-Path -LiteralPath $PythonExe).Path
    New-Item -ItemType Directory -Path $Control -Force | Out-Null
    $lock = [IO.File]::Open("$Control/run.lock", [IO.FileMode]::OpenOrCreate,
        [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    $oldState = $null
    if (Test-Path -LiteralPath $StatePath) {
        $oldState = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
        if ($oldState.schema_version -ne 1 -or $oldState.dr_formulation -ne "load_shifting_v1") {
            throw "Etat de campagne incompatible : $StatePath"
        }
        if (-not $Resume -and -not $PSBoundParameters.ContainsKey("StartStep")) {
            throw "Une campagne existe. Utiliser -Resume ou -StartStep N."
        }
    } elseif ($Resume) {
        throw "Aucun etat de reprise : $StatePath"
    }
    if ($oldState) {
        # Before/within stage 2 use 4x4; old reports without matching provenance
        # will be recomputed by the batch runner during selective resume.
        # Beyond it, retain the old allocation as provenance for existing labels.
        $legacyTrainWorkers = 4
        $legacyTrainThreads = 4
        if ([int]$oldState.next_step -gt 2) {
            $legacyTrainWorkers = 1
            $legacyTrainThreads = [int]$oldState.settings.HighsThreads
        }
        foreach ($entry in @(@("TrainMilpWorkers", $legacyTrainWorkers), @("TrainMilpThreads", $legacyTrainThreads))) {
            if (-not $oldState.settings.PSObject.Properties[$entry[0]]) {
                $oldState.settings | Add-Member -NotePropertyName $entry[0] -NotePropertyValue $entry[1]
            }
        }
        # Upgrade old settings before stage 10 without repeating training.
        # Already-computed standard outputs must be recomputed explicitly.
        if ($oldState.standard_protocol -ne $StandardProtocol) {
            if ($Resume -and [int]$oldState.next_step -ge 11) {
                throw "Limites standard/JEPA modifiees. Utiliser -StartStep 10 pour recalculer ces inferences."
            }
            if ($oldState.settings.GlobalTimeBudgetSeconds -le 0) {
                $oldState.settings.GlobalTimeBudgetSeconds = 120
            }
            if (-not $oldState.settings.PSObject.Properties["JepaTimeBudgetSeconds"]) {
                $oldState.settings | Add-Member -NotePropertyName JepaTimeBudgetSeconds -NotePropertyValue 120
            }
        }
        foreach ($name in @("HighsThreads", "TrainMilpWorkers", "TrainMilpThreads", "InferenceK", "GlobalTimeBudgetSeconds",
                "JepaTimeBudgetSeconds", "ParallelBudgetSeconds", "BestOfK")) {
            if (-not $PSBoundParameters.ContainsKey($name)) {
                Set-Variable -Name $name -Value $oldState.settings.$name
            }
        }
    }
    if ($HighsThreads -eq 0) { $HighsThreads = [Environment]::ProcessorCount }
    if (-not $BestOfK.Count) {
        $BestOfK = @(1, 2, 5, 8, 12, 16) | Where-Object { $_ -le $HighsThreads }
    }
    $BestOfK = @($BestOfK | Sort-Object -Unique)
    if (@($BestOfK | Where-Object { $_ -lt 1 -or $_ -gt $HighsThreads }).Count) {
        throw "Chaque K doit etre compris entre 1 et HighsThreads=$HighsThreads."
    }
    $settings = [ordered]@{
        HighsThreads = $HighsThreads; InferenceK = $InferenceK
        TrainMilpWorkers = $TrainMilpWorkers; TrainMilpThreads = $TrainMilpThreads
        GlobalTimeBudgetSeconds = $GlobalTimeBudgetSeconds
        JepaTimeBudgetSeconds = $JepaTimeBudgetSeconds
        ParallelBudgetSeconds = $ParallelBudgetSeconds; BestOfK = $BestOfK
    }
    $configHashes = [ordered]@{}
    foreach ($path in $ConfigPaths) { $configHashes[$path] = (Get-FileHash -LiteralPath $path).Hash }
    if ($Resume) {
        $settingsChanged = $false
        foreach ($name in $settings.Keys) {
            if (($settings[$name] | ConvertTo-Json -Compress) -ne
                ($oldState.settings.$name | ConvertTo-Json -Compress)) { $settingsChanged = $true }
        }
        if ($settingsChanged -or ($configHashes | ConvertTo-Json -Compress) -ne
                ($oldState.config_hashes | ConvertTo-Json -Compress)) {
            throw "Parametres/configurations modifies. Utiliser -StartStep a l'etape concernee."
        }
        $resumeStep = [int]$oldState.next_step
        if ($resumeStep -ge 12 -and $oldState.parallel_protocol -ne $ParallelProtocol) {
            throw "Protocole parallele modifie (batch GPU). Utiliser -StartStep 12."
        }
        if ($resumeStep -gt 13) { Write-Host "Toutes les etapes sont terminees."; return }
        $StartStep = $resumeStep
    }
    if ($StartStep -gt $EndStep) { throw "StartStep doit etre <= EndStep." }
    if ($StartStep -eq 13 -and $oldState -and $oldState.parallel_protocol -ne $ParallelProtocol) {
        throw "Les anciens resultats CPU ne sont pas ceux du batch GPU. Utiliser -StartStep 12."
    }
    $State = [ordered]@{
        schema_version = 1; dr_formulation = "load_shifting_v1"
        parallel_protocol = $ParallelProtocol
        standard_protocol = $(if ($StartStep -le 10) { $StandardProtocol } else { $oldState.standard_protocol })
        settings = $settings; config_hashes = $configHashes
        next_step = $StartStep; status = "ready"; updated_at = ""; error = ""
        parallel_reset_pending = ($StartStep -le 12)
        stage_timings = @()
    }
    if ($oldState -and $oldState.PSObject.Properties["stage_timings"]) {
        $State.stage_timings = @($oldState.stage_timings)
    }
    if ($Resume) { $State.parallel_reset_pending = [bool]$oldState.parallel_reset_pending }
    Save-State
    $OutputEncoding = [Text.UTF8Encoding]::new($false)
    $env:BENCHMARK_HIGHS_THREADS = [string]$HighsThreads
    $EvalDir = "$R/evaluation/latent_k$InferenceK"
    $ParallelDir = "$R/evaluation/parallel_best_of_k"
    $ScenarioArgs = @()
    foreach ($f in $TestFamilies) { $ScenarioArgs += @("--scenario-root", "$R/$f") }
    $standardKs = (@(1,2,3,5,$InferenceK) | Where-Object { $_ -le $InferenceK } |
        Sort-Object -Unique) -join ","
    Start-Transcript -Path "$Control/run-$RunId.log" | Out-Null
    $transcriptStarted = $true
    Write-Host "MILP entrainement : $TrainMilpWorkers processus x $TrainMilpThreads threads ; references test : 1 x $HighsThreads."
    Write-Host "HiGHS: $HighsThreads threads ; batch CUDA + reparations CPU pour K=$($BestOfK -join ',')."
    Write-Host "Budget parallele commun: $ParallelBudgetSeconds s, sans division par K."
    Write-Host "Limites strictes par scenario : standard $GlobalTimeBudgetSeconds s ; JEPA $JepaTimeBudgetSeconds s (separees)."

    for ($step = $StartStep; $step -le $EndStep; $step++) {
        $ActiveStep = $step
        $CompletedWorkUnits = 0
        $StepStartedAt = [DateTime]::UtcNow.ToString("o")
        $StepTimer = [Diagnostics.Stopwatch]::StartNew()
        $State.next_step = $step
        $State.status = "running"
        Save-State
        Show-PipelineProgress
        Show-StepProgress -Operation "Demarrage"
        Write-Host "`n[$step/13] $($Steps[$step])" -ForegroundColor Cyan
        switch ($step) {
            1 {
                foreach ($f in $AllFamilies) {
                    Show-StepProgress -Operation "Copie et verification : $f"
                    $files = @(Get-ChildItem -LiteralPath "$Repo/outputs/$f" -Filter "scenario_*.json" -File)
                    if (-not $files.Count) { throw "Aucun scenario dans outputs/$f" }
                    New-Item -ItemType Directory -Path "$R/$f" -Force | Out-Null
                    $names = @($files | Select-Object -ExpandProperty Name)
                    $extra = @(Get-ChildItem -LiteralPath "$R/$f" -Filter "scenario_*.json" -File |
                        Where-Object { $_.Name -notin $names })
                    if ($extra.Count) { throw "Scenarios supplementaires dans la copie : $R/$f" }
                    foreach ($file in $files) {
                        $dest = Join-Path "$R/$f" $file.Name
                        if (Test-Path -LiteralPath $dest) {
                            if ((Get-FileHash -LiteralPath $file.FullName).Hash -ne
                                (Get-FileHash -LiteralPath $dest).Hash) {
                                throw "Le scenario copie a change : $dest"
                            }
                        } else { Copy-Item -LiteralPath $file.FullName -Destination $dest }
                    }
                    Show-StepProgress -Operation "Copie terminee : $f" -Advance
                }
                Invoke-Source $ValidateCode (@($R) + $AllFamilies)
            }
            2 {
                Invoke-Source $ValidateCode (@($R) + $AllFamilies)
                foreach ($f in $AllFamilies) {
                    Invoke-MilpBatch $f
                }
                # Rebuild derived classification only. Old tier copies and the
                # index must not survive a retry or be classified as reports.
                foreach ($name in @("gold", "silver", "bronze", "classification_index.json")) {
                    Archive-Output "$R/scenarios_v3/reports/$name"
                }
                Invoke-Module "scripts.classify_milp_reports" @(
                    "--reports-dir", "$R/scenarios_v3/reports", "--save-index")
                $index = Get-Content "$R/scenarios_v3/reports/classification_index.json" -Raw | ConvertFrom-Json
                if ($index.summary.error_count -gt 0) { throw "Classification MILP incomplete." }
            }
            3 {
                Archive-Output "$R/graphs"
                foreach ($f in $AllFamilies) {
                    Invoke-Module "src.gnn.build_hetero_graph_dataset" @("$R/$f", "$R/graphs/$f",
                        "--temporal", "--temporal-mode", "supra", "--temporal-edges", "soc,ramp,dr")
                    $index = Get-Content "$R/graphs/$f/dataset_index.json" -Raw | ConvertFrom-Json
                    $expected = @(Get-ChildItem "$R/$f" -Filter "scenario_*.json" -File).Count
                    if ($index.entries.Count -ne $expected) { throw "Graphes incomplets : $f" }
                }
            }
            4 {
                Archive-Output "$R/encoders/hierarchical_temporal_v3"
                Invoke-Source $HteCode @("train", $R)
                Assert-Path "$R/encoders/hierarchical_temporal_v3/best_encoder.pt"
            }
            5 {
                Assert-Path "$R/encoders/hierarchical_temporal_v3/best_encoder.pt"
                Archive-Output "$R/embeddings"
                Invoke-Source $HteCode (@("embeddings", $R) + $AllFamilies)
                Invoke-Source $LabelCheckCode (@($R) + $AllFamilies)
            }
            6 {
                Invoke-Source $LabelCheckCode (@($R) + $AllFamilies)
                Archive-Output "$R/uc_jepa/dataset_train"
                Archive-Output "$R/uc_jepa/dataset_eval"
                Invoke-Module "src.uc_jepa.data.extract_uc_labels" @("--reports",
                    "$R/scenarios_v3/reports/gold", "$R/scenarios_v3/reports/silver",
                    "--embeddings", "$R/embeddings/scenarios_v3",
                    "--out", "$R/uc_jepa/dataset_train", "--n-timesteps", "24")
                foreach ($f in $TestFamilies) {
                    Invoke-Module "src.uc_jepa.data.extract_uc_labels" @("--reports", "$R/$f/reports",
                        "--embeddings", "$R/embeddings/$f", "--out", "$R/uc_jepa/dataset_eval",
                        "--n-timesteps", "24")
                    $expected = @(Get-ChildItem "$R/$f" -Filter "scenario_*.json" -File).Count
                    $actual = @(Get-ChildItem "$R/uc_jepa/dataset_eval/$f" -Filter "*_uc.npz" -File).Count
                    if ($actual -ne $expected) {
                        throw "Labels de test incomplets : $f ($actual/$expected). Ne pas reduire silencieusement la cohorte."
                    }
                }
            }
            { $_ -in 7,8,9 } {
                $name = @{7="autoencoder"; 8="jepa"; 9="latent_ebm"}[$step]
                $checkpoint = @{7="uc_ae_best.pt"; 8="jepa_best.pt"; 9="latent_ebm_best.pt"}[$step]
                Assert-Path "$R/uc_jepa/dataset_train"
                Invoke-Source $ValidateCode (@($R) + $AllFamilies)
                Archive-Output "$R/uc_jepa/$name"
                Invoke-Module "src.uc_jepa.training.train_$name" @("--config", "configs/uc_jepa/$name.yaml")
                Assert-Path "$R/uc_jepa/$name/$checkpoint"
            }
            10 {
                Assert-Path "$R/uc_jepa/latent_ebm/latent_ebm_best.pt"
                Archive-Output $EvalDir
                Invoke-Module "scripts.run_strict_standard_inference" (@(
                    "--root", $R, "--output-dir", $EvalDir,
                    "--k", "$InferenceK", "--threads", "$HighsThreads",
                    "--budget", $GlobalTimeBudgetSeconds.ToString([Globalization.CultureInfo]::InvariantCulture),
                    "--jepa-budget", $JepaTimeBudgetSeconds.ToString([Globalization.CultureInfo]::InvariantCulture),
                    "--families") + $TestFamilies)
            }
            11 {
                Assert-Path "$EvalDir/pipeline_eval_all_families.pkl"
                Assert-Path "$EvalDir/manifest.json"
                $standardManifest = Get-Content -LiteralPath "$EvalDir/manifest.json" -Raw | ConvertFrom-Json
                if ($standardManifest.protocol -ne $StandardProtocol) {
                    throw "Resultats standard sans limites strictes separees. Utiliser -StartStep 10."
                }
                Invoke-Module "src.uc_jepa.evaluation.evaluate_latent_pipeline" (@(
                    "--skip-eval", "--results-pkl", "$EvalDir/pipeline_eval_all_families.pkl",
                    "--output-dir", $EvalDir, "--best-of-k-values", $standardKs) + $ScenarioArgs)
            }
            12 {
                if ($State.parallel_reset_pending) {
                    Archive-Output $ParallelDir
                    $State.parallel_reset_pending = $false
                    Save-State
                }
                Invoke-Module "scripts.benchmark_parallel_best_of_k" (@(
                    "--root", $R, "--output-dir", $ParallelDir, "--threads", "$HighsThreads",
                    "--budget", $ParallelBudgetSeconds.ToString([Globalization.CultureInfo]::InvariantCulture),
                    "--k-values", ($BestOfK -join ","), "--resume", "--families") + $TestFamilies)
            }
            13 {
                Invoke-Module "scripts.benchmark_parallel_best_of_k" @(
                    "--output-dir", $ParallelDir, "--analyze-only")
            }
        }
        Record-StepTiming "completed"
        $State.next_step = $step + 1
        $State.status = "completed_through_step_$step"
        Save-State
        Show-StepProgress -Operation $Steps[$step] -Finished
        Show-PipelineProgress -StepCompleted
        Write-Host "[OK] Etape $step/13 terminee : $($Steps[$step])" -ForegroundColor Green
        Write-Progress -Id 1 -ParentId 0 -Activity "Etape $step" -Completed
    }
    Write-Host "`nTermine jusqu'a l'etape $EndStep. Etat : $StatePath" -ForegroundColor Green
} catch {
    if ($null -ne $State -and $null -ne $lock) {
        $State.status = "failed"
        $State.error = $_.Exception.Message
        Record-StepTiming "failed"
        Save-State
    }
    Write-Host "Arret. Reprise : .\scripts\run_load_shifting_pipeline.ps1 -Resume" -ForegroundColor Yellow
    throw
} finally {
    if ($ActiveStep -gt 0) {
        Write-Progress -Id 1 -ParentId 0 -Activity "Etape $ActiveStep" -Completed
        Write-Progress -Id 0 -Activity "Pipeline load shifting" -Completed
    }
    $ProgressPreference = $oldProgressPreference
    if ($transcriptStarted) { Stop-Transcript | Out-Null }
    if ($null -ne $lock) { $lock.Dispose() }
    $env:BENCHMARK_HIGHS_THREADS = $oldThreadEnv
    $OutputEncoding = $oldEncoding
    Pop-Location
}
