# Start ML API and optional localtunnel link
# Usage:
#   .\start_api_with_tunnel.ps1           # start API only
#   .\start_api_with_tunnel.ps1 -Tunnel    # start API + localtunnel and set TUNNEL_URL

param([switch]$Tunnel)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Tunnel) {
    Write-Host "Starting localtunnel in background (port 8001)..."
    $job = Start-Job -ScriptBlock {
        Set-Location $using:PSScriptRoot
        npx --yes localtunnel --port 8001
    }
    Start-Sleep -Seconds 10
    $out = Receive-Job $job
    if ($out -match "your url is:\s*(https://[^\s]+)") {
        $url = $Matches[1]
        $env:TUNNEL_URL = $url
        Write-Host "Tunnel URL: $url"
    }
}

Write-Host "Starting ML API on http://0.0.0.0:8001 ..."
python -m uvicorn src.ml_api:app --host 0.0.0.0 --port 8001
