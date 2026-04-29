@echo off
TITLE MANIPULANTE FTMO TRIAL AUTO-RUNNER
cd /d "c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"

echo ======================================================================
echo MANIPULANTE FTMO TRIAL AUTO-RUNNER LAUNCHER
echo ======================================================================
echo.
echo Entorno: FTMO Demo/Trial 10k
echo Riesgo: 0.50%%
echo Política: News Gate Required / Fail-Closed
echo.
echo Presione CTRL+C para detener o cree STOP_BOT.txt en 13_FTMO_TRIAL_AUTOMATION
echo ======================================================================

:loop
python BOT_V2_DAYTIME_LAB\src\phase37_ftmo_trial_bot_runner.py --ftmo-trial --risk 0.005 --no-real --i-understand-demo-automation --interval-seconds 60
if errorlevel 1 (
    echo [ERROR] El bot se detuvo inesperadamente. Reintentando en 10 segundos...
    timeout /t 10
    goto loop
)

echo [INFO] Ciclo finalizado (STOP_BOT detectado o final normal).
pause
