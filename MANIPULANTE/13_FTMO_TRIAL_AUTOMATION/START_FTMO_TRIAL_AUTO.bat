@echo off
TITLE MANIPULANTE FTMO TRIAL AUTO-RUNNER
cd /d "%~dp0..\.."

echo ======================================================================
echo MANIPULANTE FTMO TRIAL AUTO-RUNNER [LAUNCHER]
echo ======================================================================
echo.

:: Check if already running
if exist "MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\runner.lock" (
    echo [WARNING] Detectado runner.lock. Verificando proceso...
    :: Simple check: we rely on the python script's own lock mechanism to be 100% sure
)

echo [INFO] Validando cuenta MT5 antes de iniciar...
python -c "from phase37_ftmo_trial_support import account_gate; res = account_gate(); print(f'Company: {res.get(\"company\")} | Server: {res.get(\"server\")}'); exit(0 if res.get('ftmo_demo_trial_confirmed') else 1)"
if errorlevel 1 (
    echo [ERROR] No se detectó cuenta FTMO Demo activa. Abortando.
    pause
    exit /b 1
)

echo [INFO] Iniciando runner en modo continuo...
echo [INFO] Riesgo: 0.50%% | Intervalo: 60s
echo ======================================================================

:: Use python -u for unbuffered output to see logs in real time
python -u BOT_V2_DAYTIME_LAB\src\phase37_ftmo_trial_bot_runner.py --ftmo-trial --risk 0.005 --no-real --i-understand-demo-automation --interval-seconds 60

echo.
echo ======================================================================
echo El runner se ha detenido.
pause
