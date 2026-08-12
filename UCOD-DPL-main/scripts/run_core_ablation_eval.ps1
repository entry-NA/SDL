param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("a0", "a1")]
    [string]$Group,
    [string]$Checkpoint = "",
    [string]$PythonExecutable = "C:\Anaconda\envs\test01\python.exe",
    [string]$WorkDir = "work_dir",
    [int]$Port = 11148,
    [switch]$PreflightOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$experiments = @{
    a0 = @{
        Name = "A0 UCOD-DPL original coarse-label baseline"
        Config = "configs\uscod\UCOD-DPL_dinov2_ablation_a0_baseline.py"
        ConfigStem = "UCOD-DPL_dinov2_ablation_a0_baseline"
        ExpName = "UCOD-DPL_dinov2_ablation_a0_baseline_20260725_v1"
    }
    a1 = @{
        Name = "A1 naive frozen-SAM2 refinement"
        Config = "configs\uscod\UCOD-DPL_dinov2_ablation_a1_naive_sam2.py"
        ConfigStem = "UCOD-DPL_dinov2_ablation_a1_naive_sam2"
        ExpName = "UCOD-DPL_dinov2_ablation_a1_naive_sam2_20260725_v2"
    }
}

$experiment = $experiments[$Group]
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "Python executable not found: $PythonExecutable"
}
if (-not (Test-Path -LiteralPath $experiment.Config -PathType Leaf)) {
    throw "Experiment config not found: $($experiment.Config)"
}

if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
    $experimentRoot = Join-Path $WorkDir (
        "uscod\$($experiment.ConfigStem)\$($experiment.ExpName)"
    )
    $Checkpoint = Join-Path $experimentRoot "ckp\epoch25.pth"
}
if (-not (Test-Path -LiteralPath $Checkpoint)) {
    if ($PreflightOnly) {
        Write-Output "Preflight passed except checkpoint is not present yet (expected before training): $Checkpoint"
        exit 0
    }
    throw "Checkpoint not found: $Checkpoint"
}

Write-Output "Evaluation preflight passed: $($experiment.Name)"
Write-Output "Checkpoint: $Checkpoint"
Write-Output "Datasets: CHAMELEON, TE-CAMO, TE-COD10K, NC4K"

if ($PreflightOnly) {
    Write-Output "Preflight only; evaluation was not started."
    exit 0
}

$env:PYTHONPATH = "."
$env:WANDB_DISABLED = "True"
$env:TF_CPP_MIN_LOG_LEVEL = "3"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:PYTHONIOENCODING = "utf-8"

& $PythonExecutable -m accelerate.commands.launch `
    --mixed_precision fp16 `
    --machine_rank 0 `
    --num_machines 1 `
    --main_process_port $Port `
    --num_processes 1 `
    scripts/eval.py `
    --config $experiment.Config `
    --work_dir $WorkDir `
    --load_from $Checkpoint

if ($LASTEXITCODE -ne 0) {
    throw "Evaluation failed with exit code $LASTEXITCODE"
}

Write-Output "Evaluation completed for all four datasets."
