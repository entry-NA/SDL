param(
    [string]$AeemExperimentId = "m2_full4040_structure_20260724_v1",
    [string]$ControlExperimentId = "m0_controls_20260724_v1",
    [string]$CamoExperimentId = "m3_isolate_camo_20260724_v1",
    [string]$Cod10kExperimentId = "m3_isolate_cod10k_20260724_v1",
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
foreach ($sourceDir in @($aeemDir, $softDir)) {
    if (-not (Test-Path -LiteralPath $sourceDir -PathType Container)) {
        throw "Source label directory not found: $sourceDir"
    }
    $sourceCount = @(Get-ChildItem -LiteralPath $sourceDir -Filter "*.png" -File).Count
    if ($sourceCount -ne 4040) {
        throw "Expected 4040 source labels, found $sourceCount in $sourceDir"
    }
}

function New-IsolationArtifact {
    param(
        [string]$ExperimentId,
        [string]$AeemDataset
    )

    $artifactPath = "artifacts\aeem_v2\$ExperimentId"
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
        return
    }

    & $PythonExecutable -u scripts/compose_aeem_v2_labels.py `
        --experiment-id $ExperimentId `
        --cohort $Cohort `
        --aeem-dir $aeemDir `
        --soft-dir $softDir `
        --aeem-dataset $AeemDataset
    if ($LASTEXITCODE -ne 0) {
        throw "Composition failed for $ExperimentId with exit code $LASTEXITCODE"
    }
}

New-IsolationArtifact -ExperimentId $CamoExperimentId -AeemDataset "TR-CAMO"
New-IsolationArtifact -ExperimentId $Cod10kExperimentId -AeemDataset "TR-COD10K"

Write-Output "Source-isolation artifacts are ready."
Write-Output "Run CAMO isolation first:"
Write-Output (
    ".\scripts\run_aeem_v2_train.ps1 -ExperimentId " +
    "$CamoExperimentId -Port 11147"
)
