@echo off
setlocal enabledelayedexpansion

:: ======================================================================
:: MANIPULANTE FTMO TRIAL AUTO-RUNNER STATUS PANEL (PHASE 37X-C)
:: ======================================================================

set HEARTBEAT_JSON="c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\heartbeat.json"
set LOCK_FILE="c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\runner.lock"

cls
echo ======================================================================
echo           MANIPULANTE FTMO TRIAL - CONTROL PANEL (PHASE 37X-C)
echo ======================================================================

:: 1. Check Runner Process
if exist %LOCK_FILE% (
    set /p PID=<%LOCK_FILE%
    tasklist /FI "PID eq !PID!" 2>NUL | find /I "!PID!" >NUL
    if !ERRORLEVEL! == 0 (
        echo [STATUS]  RUNNER: ACTIVE (PID: !PID!)
    ) else (
        echo [STATUS]  RUNNER: STALE LOCK (CRITICAL)
    )
) else (
    echo [STATUS]  RUNNER: NOT RUNNING
)

:: 2. Parse Heartbeat
if exist %HEARTBEAT_JSON% (
    echo [STATUS]  HEARTBEAT: FOUND
    
    :: Basic extraction using powershell for the status dashboard
    powershell -Command "$hb = Get-Content %HEARTBEAT_JSON% | ConvertFrom-Json; \
    echo \"----------------------------------------------------------------------\"; \
    echo \" NY TIME:      $($hb.timestamp_ny)\"; \
    echo \" SESSION:      $($hb.session_state)\"; \
    echo \" ENTRIES:      $(if($hb.can_open_new_trades){'ALLOWED'}else{'BLOCKED'})\"; \
    echo \" POSITION:     $($hb.position_state)\"; \
    echo \" TICKETS:      $($hb.position_ticket)\"; \
    echo \" PROX NOTICIA: $($hb.next_news_block)\"; \
    echo \"----------------------------------------------------------------------\"; \
    echo \" DEADLINE NY:  $($hb.pc_off_deadline_ny)\"; \
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
