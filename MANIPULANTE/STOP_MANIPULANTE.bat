@echo off
setlocal EnableExtensions

set "ROOT=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
set "SRC=%ROOT%\BOT_V2_DAYTIME_LAB\src"
set "PY_EXE=C:\Users\alera\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if not exist "%PY_EXE%" set "PY_EXE=python"
set "STOP_FILE=%ROOT%\MANIPULANTE\13_FTMO_TRIAL_AUTOMATION\STOP_BOT.txt"
set "STATUS_TMP=%TEMP%\manipulante_stop_status.json"
set "OPEN_STATUS_TMP=%TEMP%\manipulante_stop_open_position_status.txt"
set "SAFE_STOP_SCRIPT=%ROOT%\MANIPULANTE\13_FTMO_TRIAL_AUTOMATION\safe_stop_manipulante_processes.ps1"

title MANIPULANTE - DETENER BOT

cls
echo ============================================================
echo MANIPULANTE - DETENER BOT
echo ============================================================
echo.
echo ESTADO: SOLICITANDO DETENCION SEGURA
echo.
echo No se cierra MT5 a la fuerza.
echo No se borra ningun log.
echo No se toca la estrategia.
echo.
echo ============================================================
echo.

echo Creando senal de parada...
echo STOP_BOT > "%STOP_FILE%"
echo.

:: Obtener estado actual
"%PY_EXE%" "%SRC%\phase37ze_quick_status_panel.py" --json > "%STATUS_TMP%"

:: Validar estado de posicion sin relanzar MT5
"%PY_EXE%" "%SRC%\phase37ze_quick_status_panel.py" --open-position-status > "%OPEN_STATUS_TMP%"
set /p OPEN_POSITION_STATUS=<"%OPEN_STATUS_TMP%"
if "%OPEN_POSITION_STATUS%"=="" set "OPEN_POSITION_STATUS=OPEN_POSITION_UNKNOWN"

echo OPEN_POSITION_STATUS: %OPEN_POSITION_STATUS%
echo.

if /I "%OPEN_POSITION_STATUS%"=="OPEN_POSITION_CONFIRMED" (
    echo OPEN_POSITION_CONFIRMED
    echo No apagues la PC hasta cerrar o verificar la posicion.
    echo.
    echo STOP_BOT queda activo, pero no se hara limpieza agresiva.
    echo Revise su terminal MT5 o el panel de STATUS.
    echo.
    echo ============================================================
    pause
    exit /b 1
)

if /I "%OPEN_POSITION_STATUS%"=="OPEN_POSITION_UNKNOWN" (
    echo OPEN_POSITION_UNKNOWN
    echo STOP_BOT queda activo.
    echo No se relanza MT5.
    echo No se hara limpieza agresiva por seguridad.
    echo.
    if exist "%SAFE_STOP_SCRIPT%" (
        powershell -NoProfile -ExecutionPolicy Bypass -File "%SAFE_STOP_SCRIPT%" -ListOnly -OpenPositionStatus OPEN_POSITION_UNKNOWN
    )
    echo.
    echo ============================================================
    echo STOP PARCIAL SEGURO - REVISION MANUAL REQUERIDA
    echo ============================================================
    pause
    exit /b 2
)

echo NO_OPEN_POSITION_CONFIRMED
echo Esperando a que el bot cierre de forma ordenada (20 segundos)...
echo.

timeout /t 20 /nobreak > nul

:: Verificar si el runner sigue activo
"%PY_EXE%" "%SRC%\phase37ze_quick_status_panel.py" --runner-count > "%STATUS_TMP%"
set /p RUNNER_COUNT=<"%STATUS_TMP%"
if "%RUNNER_COUNT%"=="" set "RUNNER_COUNT=0"

if not "%RUNNER_COUNT%"=="0" (
    echo El bot no cerro solo. Forzando cierre del proceso runner...
    "%PY_EXE%" "%SRC%\phase45b_runner_recovery.py" --stop-runner-safe
)

:: Limpiar lock si quedo huerfano o stale
echo Verificando limpieza de locks...
"%PY_EXE%" "%SRC%\phase45b_runner_recovery.py" --clean-stale-lock

echo.
echo Realizando limpieza profunda de procesos huerfanos...
if exist "%SAFE_STOP_SCRIPT%" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%SAFE_STOP_SCRIPT%" -OpenPositionStatus NO_OPEN_POSITION_CONFIRMED
) else (
    echo [ADVERTENCIA] No se encontro safe_stop_manipulante_processes.ps1
)

echo.
echo ============================================================
echo BOT DETENIDO (PROCESOS LIMPIOS)
echo Ahora podes cerrar MT5 si queres.
echo ============================================================
echo.
pause
exit /b 0
