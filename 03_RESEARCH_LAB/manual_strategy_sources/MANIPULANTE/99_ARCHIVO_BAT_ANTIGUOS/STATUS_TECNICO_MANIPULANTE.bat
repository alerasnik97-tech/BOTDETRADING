@echo off
setlocal EnableExtensions

set "ROOT=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
set "SRC=%ROOT%\BOT_V2_DAYTIME_LAB\src"
set "PY_EXE=C:\Users\alera\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if not exist "%PY_EXE%" set "PY_EXE=python"

cd /d "%ROOT%"
set "PYTHONPATH=%SRC%;%PYTHONPATH%"
title MANIPULANTE - PANEL TECNICO

:loop
cls
"%PY_EXE%" "%SRC%\phase37ze_quick_status_panel.py" --mode technical
echo.
timeout /t 30 /nobreak > nul
goto loop
