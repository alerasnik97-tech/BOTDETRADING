@echo off
setlocal EnableExtensions

set "TARGET=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE\13_FTMO_TRIAL_AUTOMATION\STATUS_FTMO_TRIAL_AUTO.bat"

if not exist "%TARGET%" (
    echo ============================================================
    echo MANIPULANTE - PANEL DE ESTADO
    echo ============================================================
    echo.
    echo ESTADO GENERAL: ROJO
    echo.
    echo No se encontro el panel oficial.
    echo %TARGET%
    echo.
    echo ============================================================
    pause
    exit /b 1
)

call "%TARGET%"
