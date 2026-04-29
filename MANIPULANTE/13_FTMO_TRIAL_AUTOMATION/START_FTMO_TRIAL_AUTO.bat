@echo off
setlocal EnableExtensions

set "ROOT=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
set "SRC=%ROOT%\BOT_V2_DAYTIME_LAB\src"
set "COUNT_FILE=%TEMP%\manipulante_runner_count.txt"
set "PY_EXE=C:\Users\alera\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if not exist "%PY_EXE%" set "PY_EXE=python"

cd /d "%ROOT%"
set "PYTHONPATH=%SRC%;%PYTHONPATH%"
title MANIPULANTE - INICIO

"%PY_EXE%" "%SRC%\phase37ze_quick_status_panel.py" --runner-count > "%COUNT_FILE%" 2>nul
set "RUNNER_COUNT=0"
if exist "%COUNT_FILE%" set /p RUNNER_COUNT=<"%COUNT_FILE%"
if "%RUNNER_COUNT%"=="" set "RUNNER_COUNT=0"

cls
echo ============================================================
echo MANIPULANTE - INICIO
echo ============================================================
echo.

if not "%RUNNER_COUNT%"=="0" (
    echo ESTADO: BOT YA ESTA PRENDIDO
    echo.
    echo No se inicio otro bot.
    echo No hay riesgo duplicado.
    echo.
    echo Use STATUS_MANIPULANTE.bat para ver el estado.
    echo.
    echo ============================================================
    echo.
    pause
    exit /b 0
)

set "STOP_FILE=%ROOT%\MANIPULANTE\13_FTMO_TRIAL_AUTOMATION\STOP_BOT.txt"
if not exist "%STOP_FILE%" goto :start_bot

echo ESTADO: BOT DETENIDO POR STOP_BOT
echo.
echo Se encontro un archivo de parada activa.
echo.
echo Presione una tecla para REINICIAR el bot (se borrara el bloqueo).
echo O cierre esta ventana para mantener el bot apagado.
echo.
echo ============================================================
pause > nul
del "%STOP_FILE%"
cls
echo ============================================================
echo MANIPULANTE - REINICIANDO
echo ============================================================
echo.

:start_bot
echo ESTADO: INICIANDO BOT
echo.
"%PY_EXE%" -c "import sys; sys.path.insert(0, r'%SRC%'); from phase37_ftmo_trial_support import account_gate; r=account_gate(); sys.exit(0 if r.get('ftmo_demo_trial_confirmed') else 1)"
if errorlevel 1 (
    cls
    echo ============================================================
    echo MANIPULANTE - INICIO
    echo ============================================================
    echo.
    echo ESTADO: BLOQUEADO
    echo.
    echo Cuenta FTMO Demo no confirmada.
    echo No se inicio el bot.
    echo No se envio ninguna orden.
    echo.
    echo Revise MT5 y la cuenta FTMO-Demo.
    echo ============================================================
    echo.
    pause
    exit /b 1
)

echo Cuenta: FTMO Demo OK
echo Riesgo: 0.50%%
echo Modo: Trial Demo
echo Estrategia: MANIPULANTE
echo.
echo Deje esta ventana abierta.
echo.
echo ============================================================
echo.

"%PY_EXE%" -u "%SRC%\phase37_ftmo_trial_bot_runner.py" --ftmo-trial --risk 0.005 --no-real --i-understand-demo-automation --interval-seconds 60

echo.
echo El bot se detuvo.
echo.
pause
