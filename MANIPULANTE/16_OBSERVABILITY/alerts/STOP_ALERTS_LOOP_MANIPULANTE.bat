@echo off
setlocal EnableExtensions
set "ROOT=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
set "SRC=%ROOT%\BOT_V2_DAYTIME_LAB\src"
set "PY_EXE=C:\Users\alera\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if not exist "%PY_EXE%" set "PY_EXE=python"

echo Deteniendo loop de alertas MANIPULANTE...

cd /d "%ROOT%"
set "PYTHONPATH=%SRC%;%PYTHONPATH%"
"%PY_EXE%" "%SRC%\phase45_run_alert_check.py" --stop-loop
set "STOP_EXIT=%ERRORLEVEL%"

exit /b %STOP_EXIT%
