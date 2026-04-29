@echo off
setlocal EnableExtensions EnableDelayedExpansion
title MANIPULANTE — PANEL DE ESTADO

:: ======================================================================
:: MANIPULANTE STATUS PANEL (PHASE 37ZE)
:: ======================================================================

set "ROOT=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
set "SRC=%ROOT%\BOT_V2_DAYTIME_LAB\src"

cd /d "%ROOT%"
set "PYTHONPATH=%SRC%;%PYTHONPATH%"

:: Set console to UTF-8
chcp 65001 > nul

:loop
cls
python "%SRC%\phase37ze_quick_status_panel.py"

:: Wait 30 seconds before refresh
timeout /t 30 /nobreak
goto loop
