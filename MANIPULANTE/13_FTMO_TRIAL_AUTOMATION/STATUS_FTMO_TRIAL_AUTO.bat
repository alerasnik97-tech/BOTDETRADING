@echo off
setlocal enabledelayedexpansion

:: ======================================================================
:: MANIPULANTE FTMO TRIAL AUTO-RUNNER STATUS PANEL (PHASE 37Y)
:: ======================================================================

set "ROOT=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
set "SRC=%ROOT%\BOT_V2_DAYTIME_LAB\src"
set "HEARTBEAT_JSON=%ROOT%\MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\heartbeat.json"
set "LOCK_FILE=%ROOT%\MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\runner.lock"

cd /d "%ROOT%"
set "PYTHONPATH=%SRC%;%PYTHONPATH%"

cls
echo ======================================================================
echo           MANIPULANTE FTMO TRIAL - CONTROL PANEL (PHASE 37Y)
echo ======================================================================

:: 1. Check Account Gate
echo [ACCOUNT] Validando MT5...
python -c "import sys; sys.path.insert(0, r'%SRC%'); from phase37_ftmo_trial_support import account_gate; res = account_gate(); print(f' Company: {res.get(\"company\")} | Server: {res.get(\"server\")}'); sys.exit(0 if res.get('ftmo_demo_trial_confirmed') else 1)"
if errorlevel 1 (
    echo [ERROR] No se detecto cuenta FTMO Demo activa.
)

:: 2. Check Runner Process
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

:: 3. Parse Heartbeat
if exist "%HEARTBEAT_JSON%" (
    echo [STATUS]  HEARTBEAT: FOUND
    
    powershell -Command "$hb = Get-Content '%HEARTBEAT_JSON%' | ConvertFrom-Json; \
    echo \"----------------------------------------------------------------------\"; \
    echo \" NY TIME:      $($hb.timestamp_ny)\"; \
    echo \" SESSION:      $($hb.session_state)\"; \
    echo \" ENTRIES:      $(if($hb.can_open_new_trades){'ALLOWED'}else{'BLOCKED'})\"; \
    echo \" POSITION:     $($hb.position_state)\"; \
    echo \" TICKETS:      $($hb.position_ticket)\"; \
    echo \" ATTEMPTS:     $($hb.forced_close_attempts)\"; \
    echo \"----------------------------------------------------------------------\"; \
    echo \" DEADLINE NY:  $($hb.pc_off_deadline_ny)\"; \
    echo \" FLAT 19:50:   $($hb.flat_confirmed_1950)\"; \
    echo \" FLAT 19:55:   $($hb.flat_confirmed_1955)\"; \
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
