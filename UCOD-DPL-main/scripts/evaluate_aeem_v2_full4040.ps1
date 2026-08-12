param(
    [string]$ExperimentId = "m2_full4040_structure_20260724_v1",
    [string]$PythonExecutable = "C:\Anaconda\envs\test01\python.exe",
    [string]$Cohort = "experiments\aeem_v2_m2_full4040.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "Python executable not found: $PythonExecutable"
}
if (-not (Test-Path -LiteralPath $Cohort -PathType Leaf)) {
    throw "Full cohort not found: $Cohort"
}

$predictionPath = "artifacts\aeem_v2\$ExperimentId\refined_pseudo_labels"
if (@(Get-ChildItem -LiteralPath $predictionPath -Filter "*.png" -File).Count -ne 4040) {
    throw "Expected 4040 refined labels: $predictionPath"
}

$outputPath = "artifacts\aeem_v2\evaluations\${ExperimentId}_gt_diag"
if (Test-Path -LiteralPath $outputPath) {
    throw "Evaluation artifact already exists: $outputPath"
}

& $PythonExecutable experiments/evaluate_aeem_labels.py `
    --gt-set "TR-CAMO=datasets/RefCOD/TR-CAMO/gt" `
    --gt-set "TR-COD10K=datasets/RefCOD/TR-COD10K/gt" `
    --prediction "soft=artifacts/aeem_v2/m0_controls_20260724_v1/controls/soft_coarse/refined_pseudo_labels" `
    --prediction "structure=$predictionPath" `
    --baseline soft `
    --cohort $Cohort `
    --output-dir $outputPath

if ($LASTEXITCODE -ne 0) {
    throw "Label evaluation failed with exit code $LASTEXITCODE"
}

Write-Output "Label evaluation completed: $outputPath\report.md"
