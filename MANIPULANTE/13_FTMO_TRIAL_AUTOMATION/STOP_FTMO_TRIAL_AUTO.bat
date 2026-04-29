@echo off
TITLE STOP MANIPULANTE FTMO TRIAL
cd /d "%~dp0"

echo ======================================================================
echo DETENIENDO MANIPULANTE FTMO TRIAL (SAFE SHUTDOWN REQUEST)
echo ======================================================================
echo.
echo Creando STOP_BOT.txt...
echo STOP_BOT > STOP_BOT.txt

echo.
echo [ADVERTENCIA]
echo Si hay una posicion abierta, el bot NO se apagara hasta que se cierre.
echo Verifique el estado con STATUS_FTMO_TRIAL_AUTO.bat.
echo NO APAGUE LA PC si hay una posicion activa.
echo ======================================================================
pause
