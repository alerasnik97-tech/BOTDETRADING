@echo off
setlocal EnableExtensions EnableDelayedExpansion
title MANIPULANTE FTMO TRIAL STATUS

:: ======================================================================
:: MANIPULANTE FTMO TRIAL STATUS (PHASE 37ZB)
:: ======================================================================

set "ROOT=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
set "SRC=%ROOT%\BOT_V2_DAYTIME_LAB\src"
set "HEARTBEAT_JSON=%ROOT%\MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\heartbeat.json"
set "LOCK_FILE=%ROOT%\MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\runner.lock"

cd /d "%ROOT%"
set "PYTHONPATH=%SRC%;%PYTHONPATH%"

cls
echo ======================================================================
echo           MANIPULANTE FTMO TRIAL - CONTROL PANEL (PHASE 37ZB)
echo ======================================================================

:: 1. Check Duplicate Runners
set RUNNER_COUNT=0
for /f "tokens=2 delims=," %%a in ('tasklist /FI "IMAGENAME eq python.exe" /NH /FO CSV') do (
    set /a RUNNER_COUNT+=1
)
:: We subtract 1 or 2 depending on how many pythons are usually there (VSCode, etc)
:: Better to use powershell to count exact runners
for /f %%i in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*phase37_ftmo_trial_bot_runner.py*' }).Count"') do set RUNNER_COUNT=%%i

if "%RUNNER_COUNT%"=="" set RUNNER_COUNT=0
if %RUNNER_COUNT% GTR 1 (
    powershell -Command "Write-Host \" [ALERTA] SE DETECTARON %RUNNER_COUNT% RUNNERS ACTIVOS (DUPLICADOS)\" -ForegroundColor Red"
)

:: 2. Check Account Gate
echo [ACCOUNT] Validando MT5...
python -c "import sys; sys.path.insert(0, r'%SRC%'); from phase37_ftmo_trial_support import account_gate; res = account_gate(); print(f' Company: {res.get(\"company\")} | Server: {res.get(\"server\")}'); sys.exit(0 if res.get('ftmo_demo_trial_confirmed') else 1)"
if errorlevel 1 (
    echo [ERROR] No se detecto cuenta FTMO Demo activa.
)

:: 3. Check Runner Process
if exist "%LOCK_FILE%" (
    set /p PID=<"%LOCK_FILE%"
    tasklist /FI "PID eq !PID!" 2>NUL | find /I "!PID!" >NUL
    if !ERRORLEVEL! == 0 (
        echo [STATUS]  RUNNER: ACTIVE (PID: !PID!)
    ) else (
        echo [STATUS]  RUNNER: STALE LOCK (CRITICAL)
    )
) else (
    echo [STATUS]  RUNNER: NOT RUNNING
)

:: 4. Parse Heartbeat
if exist "%HEARTBEAT_JSON%" (
    echo [STATUS]  HEARTBEAT: FOUND
    
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$hb = Get-Content '%HEARTBEAT_JSON%' | ConvertFrom-Json; \
    echo \"----------------------------------------------------------------------\"; \
    echo \" NY TIME:      $($hb.timestamp_ny)\"; \
    echo \" SESSION:      $($hb.session_state)\"; \
    echo \" POSITION:     $($hb.position_state)\"; \
    echo \" ATTEMPTS:     $($hb.forced_close_attempts)\"; \
    echo \"----------------------------------------------------------------------\"; \
    if($hb.safe_to_turn_off_pc -eq $true){ \
        Write-Host \" [VERDICT]     SAFE_TO_TURN_OFF_PC (FLAT CONFIRMED)\" -ForegroundColor Green; \
    } elseif($hb.manual_intervention_required -eq $true){ \
        Write-Host \" [VERDICT]     MANUAL_CLOSE_REQUIRED (STILL OPEN)\" -ForegroundColor Red; \
    } else { \
        Write-Host \" [VERDICT]     NOT_SAFE_YET (WAITING OR MANAGING)\" -ForegroundColor Yellow; \
    }"
) else (
    echo [STATUS]  HEARTBEAT: MISSING
)

echo ======================================================================
echo  [C] Cerrar Panel  [R] Reiniciar MT5 (Manual)  [S] Stop Bot
echo ======================================================================
pause
