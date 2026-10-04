# Starts the backend the same way every time, regardless of the caller's
# working directory or which name someone remembers the venv folder by.
#
# Usage:
#   .\run.ps1        # emulator on this machine, or backend + Flutter on the same machine
#   .\run.ps1 -Lan    # also reachable from a physical device on the LAN

param(
    [switch]$Lan
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Error ".venv not found at $python. Set it up first -- see README.md 'Backend Setup'."
    exit 1
}

$uvicornArgs = @('-m', 'uvicorn', 'main:app', '--reload')
if ($Lan) {
    $uvicornArgs += @('--host', '0.0.0.0', '--port', '8000')
}

& $python @uvicornArgs
