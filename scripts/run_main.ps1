param([string]$RunDirectory = '', [switch]$Resume)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonExe = Join-Path $projectRoot '.venv-run\Scripts\python.exe'
if (-not $RunDirectory) {
    if ($Resume) { throw 'Resume requires an existing RunDirectory.' }
    $RunDirectory = Join-Path 'results' ('main_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
}
$runArguments = @('run_experiment.py', '--tasks', 'tasks_main.json', '--run-dir', $RunDirectory)
if ($Resume) { $runArguments += '--resume' }
& $pythonExe @runArguments
if ($LASTEXITCODE -ne 0) { throw "Experiment failed. Resume with -RunDirectory '$RunDirectory' -Resume after resolving the error." }
& $pythonExe scripts/analyze_run.py $RunDirectory
if ($LASTEXITCODE -ne 0) { throw 'Experiment complete, but analysis failed.' }
