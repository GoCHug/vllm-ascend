# install.ps1 - Sync all skills in this library to workspace .trae/skills/
# Usage: powershell -ExecutionPolicy Bypass -File "<this script>"
$ErrorActionPreference = "Stop"

# skill library = the folder containing this script
$skillLib = $PSScriptRoot
# workspace root = skill -> precision -> 0_topic -> vllm-ascend -> <workspace>
$workspaceRoot = Split-Path (Split-Path (Split-Path (Split-Path $skillLib)))
$target = Join-Path $workspaceRoot ".trae\skills"

if (-not (Test-Path $target)) {
    New-Item -ItemType Directory -Path $target | Out-Null
}

$installed = @()
Get-ChildItem $skillLib -Directory | ForEach-Object {
    if (Test-Path (Join-Path $_.FullName "SKILL.md")) {
        Copy-Item $_.FullName -Destination $target -Recurse -Force
        $installed += $_.Name
        Write-Host "[OK] $($_.Name)"
    } else {
        Write-Host "[SKIP] $($_.Name) (missing SKILL.md)"
    }
}

Write-Host ""
Write-Host "Installed $($installed.Count) skill(s): $($installed -join ', ')"
Write-Host "Target: $target"
Write-Host "Take effect in a new TRAE session."
