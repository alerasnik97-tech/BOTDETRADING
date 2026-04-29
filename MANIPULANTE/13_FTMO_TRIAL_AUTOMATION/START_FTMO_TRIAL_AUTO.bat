@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: ======================================================================
:: MANIPULANTE FTMO TRIAL AUTO-RUNNER (PHASE 37Y)
:: ======================================================================

set "ROOT=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
set "SRC=%ROOT%\BOT_V2_DAYTIME_LAB\src"

cd /d "%ROOT%"
set "PYTHONPATH=%SRC%;%PYTHONPATH%"

TITLE MANIPULANTE FTMO TRIAL AUTO-RUNNER

echo ======================================================================
echo MANIPULANTE FTMO TRIAL AUTO-RUNNER [LIFECYCLE]
echo ======================================================================

echo [INFO] Validando imports y entorno...
python -c "import sys; sys.path.insert(0, r'%SRC%'); import phase37_ftmo_trial_support; print('IMPORT_OK')"
if errorlevel 1 (
    echo [ERROR] No se pudo importar los modulos de Phase 37. Verifique PYTHONPATH.
    pause
    exit /b 1
)

echo [INFO] Validando cuenta MT5 antes de iniciar...
python -c "import sys; sys.path.insert(0, r'%SRC%'); from phase37_ftmo_trial_support import account_gate; res = account_gate(); print(f'Company: {res.get(\"company\")} | Server: {res.get(\"server\")}'); sys.exit(0 if res.get('ftmo_demo_trial_confirmed') else 1)"
if errorlevel 1 (
    echo [ERROR] No se detecto cuenta FTMO Demo activa o conexion fallida.
    echo [AVISO] Prohibido operar en cuentas reales o servidores no autorizados.
    pause
    exit /b 1
)

echo [INFO] Politica Horaria:
echo   - 07:00 NY: Inicio Sesion
echo   - 16:30 NY: Fin nuevas entradas
echo   - 19:45 NY: Cierre forzado obligatorio
echo   - 20:00 NY: Auto-shutdown (solo si FLAT)
echo.
echo [INFO] Iniciando runner visible...
python -u "%SRC%\phase37_ftmo_trial_bot_runner.py" --ftmo-trial --risk 0.005 --no-real --i-understand-demo-automation --interval-seconds 60

echo.
echo ======================================================================
echo El runner se ha detenido de forma segura.
pause
