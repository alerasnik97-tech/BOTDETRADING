@echo off
setlocal EnableExtensions

set "ROOT=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
set "SRC=%ROOT%\BOT_V2_DAYTIME_LAB\src"

cd /d "%ROOT%"
set "PYTHONPATH=%SRC%;%PYTHONPATH%"
title MANIPULANTE - PANEL DE ESTADO

:loop
cls
python "%SRC%\phase37ze_quick_status_panel.py"
echo.
timeout /t 30 /nobreak > nul
goto loop
