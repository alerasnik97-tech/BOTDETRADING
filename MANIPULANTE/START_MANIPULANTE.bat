@echo off
setlocal
set "TARGET=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE\13_FTMO_TRIAL_AUTOMATION\START_FTMO_TRIAL_AUTO.bat"
if not exist "%TARGET%" (
  echo [ERROR] No se encontro el launcher oficial:
  echo %TARGET%
  pause
  exit /b 1
)
call "%TARGET%"
