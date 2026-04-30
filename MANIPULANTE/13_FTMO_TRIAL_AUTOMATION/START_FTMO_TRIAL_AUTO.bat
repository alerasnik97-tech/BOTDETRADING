@echo off
setlocal EnableExtensions

set "ROOT=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
set "SRC=%ROOT%\BOT_V2_DAYTIME_LAB\src"
set "STOP_FILE=%ROOT%\MANIPULANTE\13_FTMO_TRIAL_AUTOMATION\STOP_BOT.txt"
set "COUNT_FILE=%TEMP%\manipulante_runner_count.txt"
set "GUARD_FILE=%TEMP%\manipulante_start_guard_%RANDOM%.txt"
set "LOCK_FILE=%TEMP%\manipulante_start_lock_%RANDOM%.txt"
set "START_LOCK=%TEMP%\MANIPULANTE_FTMO_TRIAL_START_LOCK"
set "PY_EXE=C:\Users\alera\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if not exist "%PY_EXE%" set "PY_EXE=python"

cd /d "%ROOT%"
set "PYTHONPATH=%SRC%;%PYTHONPATH%"
set "RUNNER_LOCK_FILE=%ROOT%\MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\runner.lock"
title MANIPULANTE - INICIO

"%PY_EXE%" "%SRC%\phase37ze_quick_status_panel.py" --runner-count > "%COUNT_FILE%" 2>nul
set "RUNNER_COUNT=0"
if exist "%COUNT_FILE%" set /p RUNNER_COUNT=<"%COUNT_FILE%"
if "%RUNNER_COUNT%"=="" set "RUNNER_COUNT=0"

if not "%RUNNER_COUNT%"=="0" goto :already_running

"%PY_EXE%" "%SRC%\phase37_ftmo_trial_support.py" --acquire-start-lock-kv "%LOCK_FILE%" --lock-dir "%START_LOCK%" >nul 2>nul
call :load_kv "%LOCK_FILE%" "L_"
if not "%L_ACQUIRED%"=="SI" goto :already_running

"%PY_EXE%" "%SRC%\phase37_ftmo_trial_support.py" --start-guard-kv "%GUARD_FILE%" >nul 2>nul
call :load_kv "%GUARD_FILE%" "G_"

if "%G_DECISION%"=="ALREADY_RUNNING" (
    call :release_lock
    goto :already_running
)

if "%G_DECISION%"=="BLOCKED_POSITION_OPEN" (
    call :release_lock
    goto :position_open
)

if "%G_DECISION%"=="BLOCKED_POSITION_UNKNOWN" (
    call :release_lock
    goto :position_unknown
)

if "%G_DECISION%"=="EMERGENCY_ABORT_REAL_OR_EXNESS" (
    call :release_lock
    goto :emergency_abort
)

if not "%G_DECISION%"=="START_ALLOWED" (
    call :release_lock
    goto :blocked_account
)

set "STOP_CLEANED=NO"
if "%G_CAN_CLEAR_STOP_BOT%"=="SI" (
    del /f /q "%STOP_FILE%" >nul 2>nul
    if exist "%STOP_FILE%" (
        call :release_lock
        goto :stop_delete_failed
    )
    set "STOP_CLEANED=SI"
)

set "LOCK_CLEANED=NO"
if "%G_CAN_CLEAR_RUNNER_LOCK%"=="SI" (
    del /f /q "%RUNNER_LOCK_FILE%" >nul 2>nul
    if not exist "%RUNNER_LOCK_FILE%" set "LOCK_CLEANED=SI"
)

cls
echo ============================================================
echo MANIPULANTE - INICIO
echo ============================================================
echo.
echo ESTADO: BOT INICIADO
echo.
if "%STOP_CLEANED%"=="SI" echo Se limpio STOP_BOT de forma segura.
if "%LOCK_CLEANED%"=="SI" echo Se limpio lock viejo de forma segura.
echo Cuenta: FTMO-Demo
echo Modo: DEMO
echo.
echo Deje esta ventana abierta mientras el bot trabaja.
echo.
echo ============================================================
echo.

"%PY_EXE%" -u "%SRC%\phase37_ftmo_trial_bot_runner.py" --ftmo-trial --risk 0.005 --no-real --i-understand-demo-automation --interval-seconds 60
set "RUNNER_RC=%ERRORLEVEL%"
call :release_lock

echo.
echo El bot se detuvo.
echo.
pause
exit /b %RUNNER_RC%

:already_running
cls
echo ============================================================
echo MANIPULANTE - INICIO
echo ============================================================
echo.
echo ESTADO: BOT YA ESTA CORRIENDO
echo.
echo PID: %RUNNER_COUNT% (o multiples)
echo No se inicio otro bot.
echo No hay riesgo duplicado.
echo.
echo Use STATUS_MANIPULANTE.bat para ver el estado.
echo.
echo ============================================================
echo.
pause
exit /b 0

:position_open
cls
echo ============================================================
echo MANIPULANTE - INICIO BLOQUEADO
echo ============================================================
echo.
echo ESTADO: PELIGRO - OPERACION ABIERTA
echo.
echo No se reinicio el bot.
echo Revise MT5 y STATUS antes de continuar.
echo.
echo ============================================================
echo.
pause
exit /b 2

:position_unknown
cls
echo ============================================================
echo MANIPULANTE - INICIO BLOQUEADO
echo ============================================================
echo.
echo ESTADO: BLOQUEADO - NO SE PUDO CONFIRMAR CUENTA FLAT
echo.
echo No se limpio STOP_BOT.
echo No se inicio el bot.
echo Causa: %G_REASON%
echo.
echo ============================================================
echo.
pause
exit /b 3

:emergency_abort
cls
echo ============================================================
echo MANIPULANTE - INICIO BLOQUEADO
echo ============================================================
echo.
echo ESTADO: EMERGENCY ABORT - CUENTA NO PERMITIDA
echo.
echo No se inicio el bot.
echo No se limpio STOP_BOT.
echo No se envio ninguna orden.
echo.
echo FTMO Demo: %G_FTMO_DEMO%
echo Real detectado: %G_REAL_DETECTED%
echo Exness detectado: %G_EXNESS_DETECTED%
echo Causa: %G_REASON%
echo.
echo ============================================================
echo.
pause
exit /b 4

:blocked_account
cls
echo ============================================================
echo MANIPULANTE - INICIO BLOQUEADO
echo ============================================================
echo.
echo ESTADO: BLOQUEADO - CUENTA NO FTMO DEMO
echo.
echo No se inicio el bot.
echo No se limpio STOP_BOT.
echo No se envio ninguna orden.
echo.
echo Cuenta: %G_ACCOUNT_LABEL%
echo Modo: %G_MODE%
echo Causa: %G_REASON%
echo.
echo ============================================================
echo.
pause
exit /b 5

:stop_delete_failed
cls
echo ============================================================
echo MANIPULANTE - INICIO BLOQUEADO
echo ============================================================
echo.
echo ESTADO: BLOQUEADO - NO SE PUDO LIMPIAR STOP_BOT
echo.
echo No se inicio el bot.
echo Revise el archivo:
echo %STOP_FILE%
echo.
echo ============================================================
echo.
pause
exit /b 6

:release_lock
"%PY_EXE%" "%SRC%\phase37_ftmo_trial_support.py" --release-start-lock --lock-dir "%START_LOCK%" >nul 2>nul
exit /b 0

:load_kv
if not exist "%~1" exit /b 0
for /f "usebackq tokens=1,* delims==" %%A in ("%~1") do set "%~2%%A=%%B"
exit /b 0
