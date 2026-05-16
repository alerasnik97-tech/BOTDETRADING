param(
    [switch]$ListOnly,
    [ValidateSet("OPEN_POSITION_CONFIRMED", "OPEN_POSITION_UNKNOWN", "NO_OPEN_POSITION_CONFIRMED")]
    [string]$OpenPositionStatus = "NO_OPEN_POSITION_CONFIRMED"
)

$projectPath = "C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
$scriptsToKill = @(
    "phase37_ftmo_trial_bot_runner.py",
    "phase45b_runner_recovery.py",
    "phase37ze_quick_status_panel.py",
    "STATUS_MANIPULANTE.bat",
    "STATUS_FTMO_TRIAL_AUTO.bat",
    "START_MANIPULANTE.bat"
)

Write-Host "MANIPULANTE safe stop process scan" -ForegroundColor Cyan
Write-Host "OpenPositionStatus: $OpenPositionStatus" -ForegroundColor Cyan

$processes = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and $_.CommandLine -like "*$projectPath*"
}

$candidates = @()
foreach ($proc in $processes) {
    if ($proc.ProcessId -eq $PID) { continue }

    $name = [string]$proc.Name
    $nameLower = $name.ToLowerInvariant()
    $isCandidate = $false
    foreach ($script in $scriptsToKill) {
        $isPythonScript = $script.ToLowerInvariant().EndsWith(".py")
        $isBatchScript = $script.ToLowerInvariant().EndsWith(".bat")
        $nameMatches = (
            ($isPythonScript -and $nameLower -in @("python.exe", "pythonw.exe")) -or
            ($isBatchScript -and $nameLower -eq "cmd.exe")
        )
        if ($nameMatches -and $proc.CommandLine -like "*$script*") {
            $isCandidate = $true
            break
        }
    }

    if ($isCandidate) {
        $candidates += $proc
    }
}

if ($candidates.Count -eq 0) {
    Write-Host "No project process candidates found." -ForegroundColor Green
    exit 0
}

Write-Host "Project process candidates:" -ForegroundColor Yellow
foreach ($proc in $candidates) {
    Write-Host "PID $($proc.ProcessId): $($proc.Name) - $($proc.CommandLine)" -ForegroundColor Yellow
}

if ($ListOnly -or $OpenPositionStatus -ne "NO_OPEN_POSITION_CONFIRMED") {
    Write-Host "List-only/no aggressive cleanup due to position status." -ForegroundColor Cyan
    exit 0
}

foreach ($proc in $candidates) {
    try {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped PID $($proc.ProcessId): $($proc.Name)" -ForegroundColor Green
    } catch {
        Write-Host "Could not stop PID $($proc.ProcessId)" -ForegroundColor Red
    }
}

Write-Host "Safe stop cleanup completed." -ForegroundColor Green
