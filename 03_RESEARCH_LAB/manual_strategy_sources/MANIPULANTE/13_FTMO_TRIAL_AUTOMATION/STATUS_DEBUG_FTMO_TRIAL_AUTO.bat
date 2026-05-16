@echo off
setlocal
title STATUS DEBUG MANIPULANTE FTMO TRIAL

set "ROOT=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
set "STATUS_BAT=%ROOT%\MANIPULANTE\13_FTMO_TRIAL_AUTOMATION\STATUS_FTMO_TRIAL_AUTO.bat"
set "LOG=%ROOT%\MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\status_debug.log"

echo ======================================================================
echo INICIANDO STATUS CON LOGGING DE DEPURACION
echo ======================================================================
echo Bat: %STATUS_BAT%
echo Log: %LOG%
echo.

:: Execute status and capture everything
call "%STATUS_BAT%" > "%LOG%" 2>&1

echo [INFO] El proceso ha terminado.
echo Contenido del log:
echo ----------------------------------------------------------------------
type "%LOG%"
echo ----------------------------------------------------------------------
echo.
echo [INFO] El log completo esta en: %LOG%
pause
