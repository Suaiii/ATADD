param(
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$TrainManifest,
    [Parameter(Mandatory = $true)][string]$ValManifest,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string]$Device = "cuda",
    [int]$Seed = 42,
    [int]$Epochs = 0,
    [int]$BatchSize = 0,
    [double]$LearningRate = 0.0,
    [switch]$EnableAugment
)

$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "src"
if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $env:PYTHONPATH = $src
} else {
    $env:PYTHONPATH = "$src;$env:PYTHONPATH"
}

$args = @(
    "-m", "atadd.train",
    "--config", $Config,
    "--train-manifest", $TrainManifest,
    "--val-manifest", $ValManifest,
    "--output-dir", $OutputDir,
    "--device", $Device,
    "--seed", $Seed
)

if ($Epochs -gt 0) { $args += @("--epochs", $Epochs) }
if ($BatchSize -gt 0) { $args += @("--batch-size", $BatchSize) }
if ($LearningRate -gt 0) { $args += @("--lr", $LearningRate) }
if ($EnableAugment) { $args += @("--enable-augment") }

python @args

