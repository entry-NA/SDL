param(
    [string]$AeemExperimentId = "m2_full4040_structure_20260724_v1",
    [string]$ControlExperimentId = "m0_controls_20260724_v1",
    [string]$CamoExperimentId = "m4_camo_all_cod10k_qsemantic25_20260724_v1",
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
    throw "Cohort not found: $Cohort"
}

$aeemDir = "artifacts\aeem_v2\$AeemExperimentId\refined_pseudo_labels"
$softDir = (
    "artifacts\aeem_v2\$ControlExperimentId\controls\" +
    "soft_coarse\refined_pseudo_labels"
)
$auditPath = "artifacts\aeem_v2\$AeemExperimentId\audit.jsonl"
foreach ($sourcePath in @($aeemDir, $softDir)) {
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Container)) {
        throw "Source label directory not found: $sourcePath"
    }
    $sourceCount = @(Get-ChildItem -LiteralPath $sourcePath -Filter "*.png" -File).Count
    if ($sourceCount -ne 4040) {
        throw "Expected 4040 source labels, found $sourceCount in $sourcePath"
    }
}
if (-not (Test-Path -LiteralPath $auditPath -PathType Leaf)) {
    throw "AEEM audit file not found: $auditPath"
}

$artifactPath = "artifacts\aeem_v2\$CamoExperimentId"
if (Test-Path -LiteralPath $artifactPath) {
    $manifestPath = Join-Path $artifactPath "manifest.json"
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Existing artifact has no manifest and will not be overwritten: $artifactPath"
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($manifest.status -ne "complete" -or $manifest.output_count -ne 4040) {
        throw "Existing artifact is incomplete and will not be overwritten: $artifactPath"
    }
    Write-Output "Already complete, skipping: $artifactPath"
    exit 0
}

& $PythonExecutable -u scripts/compose_aeem_v2_labels.py `
    --experiment-id $CamoExperimentId `
    --cohort $Cohort `
    --aeem-dir $aeemDir `
    --soft-dir $softDir `
    --aeem-dataset "TR-CAMO" `
    --ranked-audit $auditPath `
    --ranked-dataset "TR-COD10K" `
    --ranked-field "selected.q_semantic" `
    --ranked-fraction 0.25
if ($LASTEXITCODE -ne 0) {
    throw "q_semantic composition failed with exit code $LASTEXITCODE"
}

Write-Output "q_semantic top-25% artifact is ready: $artifactPath"
Write-Output "Train it with:"
Write-Output (
    ".\scripts\run_aeem_v2_train.ps1 -ExperimentId " +
    "$CamoExperimentId -Port 11151"
)
