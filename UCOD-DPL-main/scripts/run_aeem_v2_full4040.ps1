param(
    [string]$ExperimentId = "m2_full4040_structure_20260724_v1",
    [string]$PythonExecutable = "C:\Anaconda\envs\test01\python.exe",
    [string]$Checkpoint = "C:\Users\23991\.cache\huggingface\hub\models--facebook--sam2.1-hiera-tiny\snapshots\de431c4043854a71d8101e17995dfe596bf101a5\sam2.1_hiera_tiny.pt",
    [string]$Cohort = "experiments\aeem_v2_m2_full4040.json",
    [int]$PostprocessWorkers = 2,
    [int]$PipelineBuffer = 4
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "Python executable not found: $PythonExecutable"
}
if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
    throw "SAM2 checkpoint not found: $Checkpoint"
}

$artifactPath = Join-Path "artifacts\aeem_v2" $ExperimentId
if (Test-Path -LiteralPath $artifactPath) {
    throw "Experiment artifact already exists: $artifactPath"
}

if (-not (Test-Path -LiteralPath $Cohort -PathType Leaf)) {
    & $PythonExecutable -u scripts/build_aeem_v2_cohort.py `
        --output $Cohort `
        --all-samples
    if ($LASTEXITCODE -ne 0) {
        throw "Full cohort generation failed with exit code $LASTEXITCODE"
    }
}

$cohortPayload = Get-Content -LiteralPath $Cohort -Encoding UTF8 -Raw | ConvertFrom-Json
if ($cohortPayload.generated_without_gt -ne $true) {
    throw "Cohort must declare generated_without_gt=true"
}
if ($cohortPayload.cohort_size -ne 4040) {
    throw "Expected 4040 samples, found $($cohortPayload.cohort_size)"
}

& $PythonExecutable -u scripts/run_aeem_v2_mvp.py `
    --experiment-id $ExperimentId `
    --cohort $Cohort `
    --checkpoint $Checkpoint `
    --structure-calibration `
    --postprocess-workers $PostprocessWorkers `
    --pipeline-buffer $PipelineBuffer

if ($LASTEXITCODE -ne 0) {
    throw "AEEM v2 generation failed with exit code $LASTEXITCODE"
}

Write-Output "Completed artifact: $artifactPath"
