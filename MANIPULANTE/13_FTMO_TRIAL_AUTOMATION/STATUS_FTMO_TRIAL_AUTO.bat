@echo off
setlocal EnableExtensions EnableDelayedExpansion
title MANIPULANTE FTMO TRIAL STATUS

:: ======================================================================
:: MANIPULANTE FTMO TRIAL STATUS (PHASE 37ZC)
:: ======================================================================

set "ROOT=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
set "SRC=%ROOT%\BOT_V2_DAYTIME_LAB\src"
set "HEARTBEAT=%ROOT%\MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\heartbeat.json"
set "QUICK_STATUS=%ROOT%\MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\quick_status.txt"
set "LOCK_FILE=%ROOT%\MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\runner.lock"

cd /d "%ROOT%"
set "PYTHONPATH=%SRC%;%PYTHONPATH%"

cls
echo ======================================================================
echo           MANIPULANTE — ESTADO RÁPIDO (5 SEG)
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
    powershell -Command "Write-Host \"ESTADO: 🟣 REVISAR — RUNNERS DUPLICADOS\" -ForegroundColor Cyan"
) else if %RUNNER_COUNT% EQU 0 (
    powershell -Command "Write-Host \"ESTADO: 🔴 BOT NO ESTÁ CORRIENDO\" -ForegroundColor Red"
) else (
    :: We have exactly 1 runner, check heartbeat
    if exist "%HEARTBEAT%" (
        powershell -Command "$hb = Get-Content '%HEARTBEAT%' | ConvertFrom-Json; \
        $diff = (New-TimeSpan -Start (Get-Date $hb.timestamp_ny) -End (Get-Date (Get-Date).ToUniversalTime().AddHours(-4))).TotalSeconds; \
        if($hb.manual_intervention_required -eq $true -or $hb.critical_position_still_open -eq $true){ \
            Write-Host \"ESTADO: 🚨 NO APAGAR PC (RIESGO ACTIVO)\" -ForegroundColor White -BackgroundColor Red; \
        } elseif($hb.news_gate -ne 'ALLOW' -or $hb.can_open_new_trades -eq $false){ \
            Write-Host \"ESTADO: 🟡 BOT ACTIVO PERO NO OPERA (GATE)\" -ForegroundColor Yellow; \
        } else { \
            Write-Host \"ESTADO: 🟢 BOT ACTIVO Y SEGURO\" -ForegroundColor Green; \
        }"
    ) else (
        echo ESTADO: 🔴 SIN HEARTBEAT
    )
)

:: 4. Summary Table
if exist "%HEARTBEAT%" (
    powershell -Command "$hb = Get-Content '%HEARTBEAT%' | ConvertFrom-Json; \
    echo \"CUENTA: $($hb.account_company) / $($hb.account_mode)\"; \
    echo \"RUNNER: ACTIVO / PID $($hb.pid)\"; \
    echo \"MT5:    %MT5_STATUS%\"; \
    echo \"NEWS:   $($hb.news_gate)\"; \
    echo \"ULTIMA DECISION: $($hb.last_decision)\"; \
    if($hb.safe_to_turn_off_pc -eq $true){ \
        Write-Host \"SEGURO APAGAR PC: SÍ\" -ForegroundColor Green; \
    } else { \
        Write-Host \"SEGURO APAGAR PC: NO\" -ForegroundColor Red; \
    }"
)

echo ======================================================================
echo.
echo [INFO] Esta ventana es solo de consulta y se puede cerrar.
echo [INFO] El bot principal corre en la ventana de START.
echo.
pause
