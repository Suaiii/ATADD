param(
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [Parameter(Mandatory = $true)][string]$InputPath,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string]$Device = "cuda",
    [ValidateSet("manifest", "audio-dir")][string]$Mode = "manifest",
    [int]$BatchSize = 0,
    [switch]$Offline
)

$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "src"
if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $env:PYTHONPATH = $src
} else {
    $env:PYTHONPATH = "$src;$env:PYTHONPATH"
}

$args = @(
    "-m", "atadd.predict",
    "--config", $Config,
    "--checkpoint", $Checkpoint,
    "--output-dir", $OutputDir,
    "--device", $Device
)

if ($Mode -eq "audio-dir") {
    $args += @("--audio-dir", $InputPath)
} else {
    $args += @("--manifest", $InputPath)
}

if ($BatchSize -gt 0) { $args += @("--batch-size", $BatchSize) }
if ($Offline) { $args += "--offline" }

python @args
