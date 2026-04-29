@echo off
setlocal EnableExtensions EnableDelayedExpansion
title MANIPULANTE FTMO TRIAL STATUS

:: ======================================================================
:: MANIPULANTE FTMO TRIAL STATUS (PHASE 37Z)
:: ======================================================================

set "ROOT=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
set "SRC=%ROOT%\BOT_V2_DAYTIME_LAB\src"
set "LOG=%ROOT%\MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\status_debug.log"
set "HEARTBEAT_TXT=%ROOT%\MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\heartbeat.txt"
set "DECISIONS=%ROOT%\MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\decisions.csv"

cd /d "%ROOT%"
set "PYTHONPATH=%SRC%;%PYTHONPATH%"

echo ======================================================================
echo MANIPULANTE FTMO TRIAL STATUS
echo ======================================================================
echo ROOT: %ROOT%
echo SRC : %SRC%
echo.

echo [INFO] Fecha/hora:
date /t
time /t
echo.

echo [INFO] Probando import Python...
python -c "import sys; sys.path.insert(0, r'%SRC%'); import phase37_ftmo_trial_support; print('IMPORT_OK')" 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Fallo importando modulos del bot.
    echo Revisar PYTHONPATH o instalacion Python.
    echo.
    pause
    exit /b 1
)

echo.
echo [INFO] Validando cuenta MT5...
python -c "import sys; sys.path.insert(0, r'%SRC%'); from phase37_ftmo_trial_support import account_gate; res=account_gate(); print(res)" 2>&1

echo.
echo [INFO] Buscando runner Python activo...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*phase37_ftmo_trial_bot_runner.py*' } | Select-Object ProcessId, CommandLine" 2>&1

echo.
echo [INFO] Buscando MT5 activo...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Process terminal64 -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, StartTime" 2>&1

echo.
echo [INFO] Heartbeat:
if exist "%HEARTBEAT_TXT%" (
    type "%HEARTBEAT_TXT%"
) else (
    echo [WARNING] No existe heartbeat.txt
)

echo.
echo [INFO] Ultimas decisiones:
if exist "%DECISIONS%" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content '%DECISIONS%' -Tail 10" 2>&1
) else (
    echo [WARNING] No existe decisions.csv
)

echo.
echo ======================================================================
echo INTERPRETACION RAPIDA
echo ======================================================================
echo SAFE_TO_TURN_OFF_PC  = podes apagar PC (FLAT CONFIRMED)
echo NOT_SAFE_YET         = esperar y revisar de nuevo (ACTIVE RISK)
echo MANUAL_CLOSE_REQUIRED= cerrar manualmente antes de apagar (ERROR)
echo ======================================================================
echo.

echo [INFO] Status finalizado. Esta ventana debe quedar abierta.
pause
exit /b 0
