param(
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [Parameter(Mandatory = $true)][string]$Manifest,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string]$Device = "cuda",
    [int]$BatchSize = 0
)

$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "src"
if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $env:PYTHONPATH = $src
} else {
    $env:PYTHONPATH = "$src;$env:PYTHONPATH"
}

$args = @(
    "-m", "atadd.eval",
    "--config", $Config,
    "--checkpoint", $Checkpoint,
    "--manifest", $Manifest,
    "--output-dir", $OutputDir,
    "--device", $Device
)

if ($BatchSize -gt 0) { $args += @("--batch-size", $BatchSize) }

python @args

