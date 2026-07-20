$ErrorActionPreference = 'Continue'
$env:PYTHONUNBUFFERED = '1'

$suiteRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$runDirectory = Join-Path $suiteRoot 'output\benchmarks\elasticity\full_run_remaining'
New-Item -ItemType Directory -Force -Path $runDirectory | Out-Null

$stdoutPath = Join-Path $runDirectory 'runner.stdout.log'
$stderrPath = Join-Path $runDirectory 'runner.stderr.log'
$startedPath = Join-Path $runDirectory 'started.txt'
$finishedPath = Join-Path $runDirectory 'finished.txt'
$exitCodePath = Join-Path $runDirectory 'exit_code.txt'

Set-Content -LiteralPath $startedPath -Value (Get-Date -Format o)
Set-Location -LiteralPath $suiteRoot

try {
    & python -u dataset\benchmarks\elasticity\validate_all.py train --models fno gino transolver 1>> $stdoutPath 2>> $stderrPath
    $runExitCode = $LASTEXITCODE
    if ($runExitCode -eq 0) {
        & python -u dataset\benchmarks\elasticity\validate_all.py infer 1>> $stdoutPath 2>> $stderrPath
        $runExitCode = $LASTEXITCODE
    }
    if ($runExitCode -eq 0) {
        & python -u dataset\benchmarks\elasticity\validate_all.py evaluate 1>> $stdoutPath 2>> $stderrPath
        $runExitCode = $LASTEXITCODE
    }
    if ($runExitCode -eq 0) {
        & python -u dataset\benchmarks\elasticity\audit_results.py 1>> $stdoutPath 2>> $stderrPath
        $runExitCode = $LASTEXITCODE
    }
} catch {
    $_ | Out-String | Add-Content -LiteralPath $stderrPath
    $runExitCode = 1
}

Set-Content -LiteralPath $exitCodePath -Value $runExitCode
Set-Content -LiteralPath $finishedPath -Value (Get-Date -Format o)
exit $runExitCode
