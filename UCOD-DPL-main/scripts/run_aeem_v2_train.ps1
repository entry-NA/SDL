param(
    [string]$ExperimentId = "m2_full4040_structure_20260724_v1",
    [string]$PythonExecutable = "C:\Anaconda\envs\test01\python.exe",
    [string]$WorkDir = "work_dir",
    [int]$Seed = 42,
    [int]$Port = 11145
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "Python executable not found: $PythonExecutable"
}

$artifactPath = Join-Path "artifacts\aeem_v2" $ExperimentId
$manifestPath = Join-Path $artifactPath "manifest.json"
$refinedPath = Join-Path $artifactPath "refined_pseudo_labels"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Artifact manifest not found: $manifestPath"
}
$manifest = Get-Content -LiteralPath $manifestPath -Encoding UTF8 -Raw | ConvertFrom-Json
if ($manifest.status -ne "complete") {
    throw "Artifact is not complete: $artifactPath"
}
if ($manifest.output_count -ne 4040) {
    throw "Expected 4040 refined labels, found $($manifest.output_count)"
}
if (@(Get-ChildItem -LiteralPath $refinedPath -Filter "*.png" -File).Count -ne 4040) {
    throw "Refined label directory does not contain 4040 PNG files: $refinedPath"
}

$env:AEEM_EXPERIMENT_ID = $ExperimentId
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
    --config configs/uscod/UCOD-DPL_dinov2_aeem_v2_full4040.py `
    --work_dir $WorkDir `
    --seed $Seed

if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE"
}

$checkpointRoot = Join-Path $WorkDir (
    "uscod\UCOD-DPL_dinov2_aeem_v2_full4040\" +
    "UCOD-DPL_dinov2_aeem_v2_$ExperimentId\ckp"
)
Write-Output "Training completed with seed=$Seed. Checkpoints: $checkpointRoot"
