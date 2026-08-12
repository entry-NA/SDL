param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("a0", "a1")]
    [string]$Group,
    [string]$PythonExecutable = "C:\Anaconda\envs\test01\python.exe",
    [string]$WorkDir = "work_dir",
    [int]$Seed = 42,
    [int]$Port = 11147,
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

$coarseCache = "datasets\cache\pseudo_label_cache\TR-CAMO+TR-COD10K"
$coarseCount = @(Get-ChildItem -LiteralPath $coarseCache -Filter "*.pkl" -File).Count
if ($coarseCount -ne 4040) {
    throw "Expected 4040 original pkl labels, found $coarseCount in $coarseCache"
}

if ($Group -eq "a0") {
    $disabledPath = "artifacts\core_ablation\a0_baseline_20260725_v1\NO_REFINED_LABELS"
    if (Test-Path -LiteralPath $disabledPath) {
        throw "A0 safety path must not exist, otherwise PNG labels may leak into the baseline: $disabledPath"
    }
}

if ($Group -eq "a1") {
    $naivePath = "datasets\cache\naive_sam2_labels"
    $naiveCount = @(Get-ChildItem -LiteralPath $naivePath -Filter "*.png" -File).Count
    if ($naiveCount -ne 4028) {
        throw "Expected 4028 naive SAM2 labels plus 12 pkl fallbacks, found $naiveCount in $naivePath"
    }
}

$experimentRoot = Join-Path $WorkDir (
    "uscod\$($experiment.ConfigStem)\$($experiment.ExpName)"
)
if (Test-Path -LiteralPath $experimentRoot) {
    throw "Experiment directory already exists; refusing to overwrite it: $experimentRoot"
}

Write-Output "Preflight passed: $($experiment.Name)"
Write-Output "Config: $($experiment.Config)"
Write-Output "Output: $experimentRoot"
Write-Output "Protocol: seed=$Seed, fp16, batch=16, epochs=25, DINOv2-base, 518x518"

if ($PreflightOnly) {
    Write-Output "Preflight only; training was not started."
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
    scripts/train.py `
    --config $experiment.Config `
    --work_dir $WorkDir `
    --seed $Seed

if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE"
}

$checkpoint = Join-Path $experimentRoot "ckp\epoch25.pth"
Write-Output "Training completed. Checkpoint: $checkpoint"
