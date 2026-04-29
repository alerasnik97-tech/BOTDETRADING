@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: ======================================================================
:: MANIPULANTE FTMO TRIAL AUTO-RUNNER (PHASE 37ZC)
:: ======================================================================

set "ROOT=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
set "SRC=%ROOT%\BOT_V2_DAYTIME_LAB\src"

cd /d "%ROOT%"
set "PYTHONPATH=%SRC%;%PYTHONPATH%"

TITLE MANIPULANTE FTMO TRIAL AUTO-RUNNER

echo ======================================================================
echo MANIPULANTE FTMO TRIAL AUTO-RUNNER [INICIO]
echo ======================================================================

:: 1. Check for active runner
set RUNNER_COUNT=0
for /f %%i in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*phase37_ftmo_trial_bot_runner.py*' }).Count"') do set RUNNER_COUNT=%%i

if %RUNNER_COUNT% GTR 0 (
    echo.
    echo [AVISO] BOT YA ESTÁ CORRIENDO - NO SE INICIÓ OTRO.
    echo [INFO] No abra esta ventana varias veces.
    echo [INFO] Use STATUS_MANIPULANTE.bat para ver el estado.
    echo.
    pause
    exit /b 0
)

echo [INFO] Validando cuenta MT5 antes de iniciar...
python -c "import sys; sys.path.insert(0, r'%SRC%'); from phase37_ftmo_trial_support import account_gate; res = account_gate(); print(f'Company: {res.get(\"company\")} | Server: {res.get(\"server\")}'); sys.exit(0 if res.get('ftmo_demo_trial_confirmed') else 1)"
if errorlevel 1 (
    echo [ERROR] No se detecto cuenta FTMO Demo activa o conexion fallida.
    pause
    exit /b 1
)

echo.
echo [IMPORTANTE] ESTA VENTANA DEBE QUEDAR ABIERTA MIENTRAS EL BOT TRABAJA.
echo.
echo [INFO] Iniciando runner visible...
python -u "%SRC%\phase37_ftmo_trial_bot_runner.py" --ftmo-trial --risk 0.005 --no-real --i-understand-demo-automation --interval-seconds 60

echo.
echo ======================================================================
echo El runner se ha detenido de forma segura.
pause
