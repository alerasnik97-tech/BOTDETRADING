@echo off
setlocal

:: ======================================================================
:: STOP MANIPULANTE FTMO TRIAL (PHASE 37Y)
:: ======================================================================

set "ROOT=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
set "STOP_FILE=%ROOT%\MANIPULANTE\13_FTMO_TRIAL_AUTOMATION\STOP_BOT.txt"

TITLE STOP MANIPULANTE FTMO TRIAL

echo ======================================================================
echo DETENIENDO MANIPULANTE FTMO TRIAL (SAFE SHUTDOWN REQUEST)
echo ======================================================================
echo.
echo Creando STOP_BOT.txt en:
echo %STOP_FILE%
echo STOP_BOT > "%STOP_FILE%"

echo.
echo Verificando procesos activos y estado de la cuenta...
"C:\Users\alera\AppData\Local\Python\pythoncore-3.14-64\python.exe" "C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\src\phase45b_runner_recovery.py" --stop-runner-safe

echo.
echo [ADVERTENCIA]
echo Si hay una posicion abierta, el bot NO se apagara a la fuerza.
echo Verifique el estado con STATUS_FTMO_TRIAL_AUTO.bat.
echo NO APAGUE LA PC si hay una posicion activa.
echo ======================================================================
pause
