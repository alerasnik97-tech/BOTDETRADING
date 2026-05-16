@echo off
setlocal EnableExtensions
set "ROOT=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
set "SRC=%ROOT%\BOT_V2_DAYTIME_LAB\src"
set "PY_EXE=C:\Users\alera\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if not exist "%PY_EXE%" set "PY_EXE=python"
set "ALERTS_STATUS_TMP=%TEMP%\manipulante_alerts_status.txt"

cd /d "%ROOT%"
set "PYTHONPATH=%SRC%;%PYTHONPATH%"

echo Iniciando loop de alertas MANIPULANTE en segundo plano...
"%PY_EXE%" "%SRC%\phase45_run_alert_check.py" --status-line > "%ALERTS_STATUS_TMP%"
type "%ALERTS_STATUS_TMP%"
findstr /C:"ALERTS_RUNNING" "%ALERTS_STATUS_TMP%" > nul
if %errorlevel% equ 0 (
    echo ALERTS_LOOP_ALREADY_RUNNING
    exit /b 0
)

set "ALERTS_DRY_RUN_ARG="
if /I "%MANIPULANTE_ALERTS_DRY_RUN%"=="1" set "ALERTS_DRY_RUN_ARG=--dry-run"

start "MANIPULANTE_ALERTS" /min "%PY_EXE%" "%SRC%\phase45_run_alert_check.py" --loop --interval-seconds 60 %ALERTS_DRY_RUN_ARG%
powershell -NoProfile -Command "Start-Sleep -Seconds 4" > nul
echo.
"%PY_EXE%" "%SRC%\phase45_run_alert_check.py" --status-line
echo.
echo Proceso de inicio de alertas completado.
