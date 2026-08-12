param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [string]$ExperimentId = "m2_full4040_structure_20260724_v1",
    [string]$PythonExecutable = "C:\Anaconda\envs\test01\python.exe",
    [string]$WorkDir = "work_dir",
    [int]$Port = 11146
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "Python executable not found: $PythonExecutable"
}
if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "Checkpoint not found: $Checkpoint"
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
    scripts/eval.py `
    --config configs/uscod/UCOD-DPL_dinov2_aeem_v2_full4040.py `
    --work_dir $WorkDir `
    --load_from $Checkpoint

if ($LASTEXITCODE -ne 0) {
    throw "Evaluation failed with exit code $LASTEXITCODE"
}

Write-Output "Evaluation completed for CHAMELEON, TE-CAMO, TE-COD10K, and NC4K."
