@echo off
TITLE STOP MANIPULANTE FTMO TRIAL
cd /d "%~dp0"

echo ======================================================================
echo DETENIENDO MANIPULANTE FTMO TRIAL (SAFE SHUTDOWN)
echo ======================================================================
echo.
echo Creando STOP_BOT.txt...
echo STOP_BOT > STOP_BOT.txt

echo [INFO] El runner detectara este archivo en el proximo ciclo (max 60s)
echo [INFO] y se detendra de forma segura.
echo.
echo Espere a que la ventana del runner se cierre sola.
echo ======================================================================
pause
