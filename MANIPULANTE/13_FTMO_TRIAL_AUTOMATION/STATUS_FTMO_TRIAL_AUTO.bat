@echo off
setlocal EnableExtensions EnableDelayedExpansion
title MANIPULANTE FTMO TRIAL - STATUS

:: ======================================================================
:: MANIPULANTE FTMO TRIAL STATUS PANEL (PHASE 37ZD)
:: ======================================================================

set "ROOT=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
set "SRC=%ROOT%\BOT_V2_DAYTIME_LAB\src"
set "HEARTBEAT=%ROOT%\MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\heartbeat.json"
set "DECISIONS=%ROOT%\MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\decisions.csv"

cd /d "%ROOT%"
set "PYTHONPATH=%SRC%;%PYTHONPATH%"

cls
echo ======================================================================
echo MANIPULANTE FTMO TRIAL -- STATUS PANEL
echo ======================================================================

:: 1. Check Runners
set RUNNER_COUNT=0
for /f %%i in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*phase37_ftmo_trial_bot_runner.py*' }).Count"') do set RUNNER_COUNT=%%i

:: 2. Check MT5
set MT5_STATUS=CERRADO
tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I /N "terminal64.exe" >NUL
if "%ERRORLEVEL%"=="0" set MT5_STATUS=ABIERTO

:: 3. Traffic Light Logic
if %RUNNER_COUNT% GTR 1 (
    powershell -Command "Write-Host \"STATUS: [PURPLE] REVIEW -- DUPLICATE RUNNERS DETECTED\" -ForegroundColor Cyan"
) else if %RUNNER_COUNT% EQU 0 (
    powershell -Command "Write-Host \"STATUS: [RED] BOT IS NOT RUNNING\" -ForegroundColor Red"
) else (
    if exist "%HEARTBEAT%" (
        powershell -Command "$hb = Get-Content '%HEARTBEAT%' | ConvertFrom-Json; \
        if($hb.manual_intervention_required -eq $true -or $hb.critical_position_still_open -eq $true){ \
            Write-Host \"STATUS: [RED] DO NOT TURN OFF PC -- ACTIVE RISK\" -ForegroundColor White -BackgroundColor Red; \
        } elseif($hb.news_gate -ne 'ALLOW' -or $hb.can_open_new_trades -eq $false){ \
            Write-Host \"STATUS: [YELLOW] BOT ACTIVE BUT NOT TRADING (GATES)\" -ForegroundColor Yellow; \
        } else { \
            Write-Host \"STATUS: [GREEN] BOT ACTIVE AND HEALTHY\" -ForegroundColor Green; \
        }"
    ) else (
        echo STATUS: [RED] NO HEARTBEAT FOUND
    )
)

:: 4. Summary Table
if exist "%HEARTBEAT%" (
    powershell -Command "$hb = Get-Content '%HEARTBEAT%' | ConvertFrom-Json; \
    echo \"ACCOUNT: $($hb.account_company) / $($hb.account_mode)\"; \
    echo \"MT5:     %MT5_STATUS%\"; \
    echo \"NEWS:    $($hb.news_gate)\"; \
    echo \"DECISION: $($hb.last_decision)\"; \
    if($hb.safe_to_turn_off_pc -eq $true){ \
        Write-Host \"SAFE TO TURN OFF PC: YES\" -ForegroundColor Green; \
    } else { \
        Write-Host \"SAFE TO TURN OFF PC: NO\" -ForegroundColor Red; \
    }"
)

echo ======================================================================
echo Last Decision Details:
if exist "%DECISIONS%" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content '%DECISIONS%' -Tail 3" 2>&1
)
echo ======================================================================
echo.
echo [INFO] This window is for monitoring only. You can close it anytime.
echo [INFO] The main bot engine runs in the START window.
echo.
pause
