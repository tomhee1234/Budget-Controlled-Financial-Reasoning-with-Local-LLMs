$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$activePath = Join-Path $projectRoot 'results\active_servers.json'
$active = Get-Content -LiteralPath $activePath -Raw | ConvertFrom-Json
foreach ($serverEntry in $active.servers) {
    $serverProcessId = [int]$serverEntry.pid
    $serverProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $serverProcessId"
    if (-not $serverProcess) { continue }
    $expectedExecutable = [System.IO.Path]::GetFullPath($serverEntry.command[0])
    $expectedModel = [string]$serverEntry.command[2]
    if ($serverProcess.ExecutablePath -ne $expectedExecutable -or -not $serverProcess.CommandLine.Contains($expectedModel)) {
        throw "PID $serverProcessId no longer matches the recorded model server; not stopping it."
    }
    Stop-Process -Id $serverProcessId
    Write-Output "Stopped $($serverEntry.label), PID $serverProcessId"
}
