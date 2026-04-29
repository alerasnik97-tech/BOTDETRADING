@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: ======================================================================
:: MANIPULANTE FTMO TRIAL START LAUNCHER (PHASE 37ZD)
:: ======================================================================

set "ROOT=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
set "SRC=%ROOT%\BOT_V2_DAYTIME_LAB\src"
set "QUICK_STATUS=%ROOT%\MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\quick_status.txt"

cd /d "%ROOT%"
set "PYTHONPATH=%SRC%;%PYTHONPATH%"

TITLE MANIPULANTE FTMO TRIAL - START

cls
echo ======================================================================
echo MANIPULANTE FTMO TRIAL -- START
echo ======================================================================

:: 1. Check for active runner (Idempotency)
set RUNNER_PID=0
for /f "tokens=*" %%i in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*phase37_ftmo_trial_bot_runner.py*' } | Select-Object -ExpandProperty ProcessId -ErrorAction SilentlyContinue"') do (
    set RUNNER_PID=%%i
)

if NOT "!RUNNER_PID!"=="0" (
    echo STATUS: BOT ALREADY RUNNING
    echo.
    echo A runner is already active ^(PID: !RUNNER_PID!^).
    echo No new instance was started.
    echo.
    if exist "%QUICK_STATUS%" (
        echo Quick Status:
        type "%QUICK_STATUS%"
        echo.
    )
    echo What to do now:
    echo 1. Close this window.
    echo 2. Open STATUS_MANIPULANTE.bat to check health.
    echo 3. Do not open START multiple times.
    echo.
    echo Safety:
    echo - Duplicate runner blocked.
    echo - No order was sent.
    echo - Strategy unchanged.
    echo ======================================================================
    pause
    exit /b 0
)

echo [INFO] Validating account and environment...
python -c "import sys; sys.path.insert(0, r'%SRC%'); from phase37_ftmo_trial_support import account_gate; res = account_gate(); sys.exit(0 if res.get('ftmo_demo_trial_confirmed') else 1)"
if errorlevel 1 (
    echo.
    echo STATUS: BLOCKED
    echo.
    echo Reason:
    echo FTMO Demo account was not confirmed.
    echo.
    echo No runner was started.
    echo No order was sent.
    echo.
    echo Check:
    echo 1. MT5 is open.
    echo 2. Account is FTMO-Demo.
    echo 3. Exness is not active in this terminal.
    echo ======================================================================
    pause
    exit /b 1
)

echo.
echo STATUS: BOT STARTED
echo.
echo Keep this window open while the bot is running.
echo.
echo Use STATUS_MANIPULANTE.bat anytime to check:
echo - account
echo - runner health
echo - news gate
echo - safe to turn off PC
echo ======================================================================
echo.

python -u "%SRC%\phase37_ftmo_trial_bot_runner.py" --ftmo-trial --risk 0.005 --no-real --i-understand-demo-automation --interval-seconds 60

echo.
echo [INFO] Runner stopped safely.
pause
