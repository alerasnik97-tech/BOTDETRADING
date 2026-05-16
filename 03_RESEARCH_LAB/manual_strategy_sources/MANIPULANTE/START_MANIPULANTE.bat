@echo off
setlocal EnableExtensions

set "TARGET=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE\13_FTMO_TRIAL_AUTOMATION\START_FTMO_TRIAL_AUTO.bat"

if not exist "%TARGET%" (
    echo ============================================================
    echo MANIPULANTE - INICIO
    echo ============================================================
    echo.
    echo ESTADO: ROJO
    echo.
    echo No se encontro el lanzador oficial.
    echo %TARGET%
    echo.
    echo ============================================================
    pause
    exit /b 1
)

echo.
set "ALERTS_START=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE\16_OBSERVABILITY\alerts\START_ALERTS_LOOP_MANIPULANTE.bat"
if exist "%ALERTS_START%" (
    echo Preparando sistema de alertas...
    call "%ALERTS_START%"
) else (
    echo [ADVERTENCIA] No se encontro el script de inicio de alertas.
)

echo.
call "%TARGET%"
