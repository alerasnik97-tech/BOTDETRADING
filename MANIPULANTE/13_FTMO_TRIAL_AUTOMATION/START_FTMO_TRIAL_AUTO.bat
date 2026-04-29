@echo off
TITLE MANIPULANTE FTMO TRIAL AUTO-RUNNER
cd /d "%~dp0..\.."

echo ======================================================================
echo MANIPULANTE FTMO TRIAL AUTO-RUNNER [LIFECYCLE]
echo ======================================================================
echo.

echo [INFO] Validando cuenta MT5 antes de iniciar...
python -c "from phase37_ftmo_trial_support import account_gate; res = account_gate(); print(f'Company: {res.get(\"company\")} | Server: {res.get(\"server\")}'); exit(0 if res.get('ftmo_demo_trial_confirmed') else 1)"
if errorlevel 1 (
    echo [ERROR] No se detectó cuenta FTMO Demo activa o conexión fallida.
    pause
    exit /b 1
)

echo [INFO] Politica Horaria:
echo   - 07:00 NY: Inicio Sesion
echo   - 16:30 NY: Fin nuevas entradas
echo   - 19:45 NY: Cierre forzado obligatorio
echo   - 20:00 NY: Auto-shutdown (solo si FLAT)
echo.
echo [INFO] Iniciando runner...
python -u BOT_V2_DAYTIME_LAB\src\phase37_ftmo_trial_bot_runner.py --ftmo-trial --risk 0.005 --no-real --i-understand-demo-automation --interval-seconds 60

echo.
echo ======================================================================
echo El runner se ha detenido de forma segura.
pause
