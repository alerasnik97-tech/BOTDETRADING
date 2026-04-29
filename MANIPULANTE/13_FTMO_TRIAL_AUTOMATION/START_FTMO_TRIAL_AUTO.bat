@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: ======================================================================
:: MANIPULANTE START LAUNCHER (PHASE 37ZE)
:: ======================================================================

set "ROOT=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
set "SRC=%ROOT%\BOT_V2_DAYTIME_LAB\src"

cd /d "%ROOT%"
set "PYTHONPATH=%SRC%;%PYTHONPATH%"

TITLE MANIPULANTE — INICIO

:: Set console to UTF-8
chcp 65001 > nul

cls
echo ======================================================================
echo MANIPULANTE — INICIO
echo ======================================================================

:: 1. Check for active runner (Idempotency)
set RUNNER_PID=0
for /f "tokens=*" %%i in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*phase37_ftmo_trial_bot_runner.py*' } | Select-Object -ExpandProperty ProcessId -ErrorAction SilentlyContinue"') do (
    set RUNNER_PID=%%i
)

if NOT "!RUNNER_PID!"=="0" (
    echo ESTADO: BOT YA ESTÁ PRENDIDO
    echo.
    echo No se inició otro bot.
    echo No hay riesgo duplicado ^(PID activo: !RUNNER_PID!^).
    echo.
    echo Abrí STATUS_MANIPULANTE.bat para ver el estado.
    echo.
    echo ======================================================================
    pause
    exit /b 0
)

:: 2. Pre-flight Checks (New Instance)
echo ESTADO: INICIANDO BOT
echo.
python -c "import sys; sys.path.insert(0, r'%SRC%'); from phase37_ftmo_trial_support import account_gate; res = account_gate(); print(f'Cuenta: {res.get(\"company\")} OK'); sys.exit(0 if res.get('ftmo_demo_trial_confirmed') else 1)"
if errorlevel 1 (
    cls
    echo ======================================================================
    echo MANIPULANTE — BLOQUEADO
    echo ======================================================================
    echo.
    echo Motivo: cuenta FTMO Demo no confirmada
    echo.
    echo No se inició el bot.
    echo No se envió ninguna orden.
    echo.
    echo Revisa:
    echo 1. MT5 abierto
    echo 2. Cuenta FTMO-Demo
    echo 3. No Exness / No real
    echo ======================================================================
    pause
    exit /b 1
)

echo Riesgo: 0.50%%
echo Modo: Trial / Demo
echo Estrategia: MANIPULANTE
echo.
echo IMPORTANTE:
echo Deja esta ventana abierta mientras el bot trabaja.
echo Para ver el estado usa STATUS_MANIPULANTE.bat
echo ======================================================================
echo.

python -u "%SRC%\phase37_ftmo_trial_bot_runner.py" --ftmo-trial --risk 0.005 --no-real --i-understand-demo-automation --interval-seconds 60

echo.
echo [INFO] El bot se ha detenido.
pause
