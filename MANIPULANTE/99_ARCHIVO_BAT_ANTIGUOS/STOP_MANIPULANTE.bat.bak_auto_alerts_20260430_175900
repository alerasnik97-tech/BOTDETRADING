@echo off
setlocal EnableExtensions

set "ROOT=C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
set "SRC=%ROOT%\BOT_V2_DAYTIME_LAB\src"
set "PY_EXE=C:\Users\alera\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if not exist "%PY_EXE%" set "PY_EXE=python"
set "STOP_FILE=%ROOT%\MANIPULANTE\13_FTMO_TRIAL_AUTOMATION\STOP_BOT.txt"
set "STATUS_TMP=%TEMP%\manipulante_stop_status.json"

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

:: Obtener estado actual
"%PY_EXE%" "%SRC%\phase37ze_quick_status_panel.py" --json > "%STATUS_TMP%"

:: Validar si hay operacion abierta
findstr /C:"\"OPERACION_ABIERTA\": \"SI\"" "%STATUS_TMP%" > nul
if %errorlevel% equ 0 (
    echo PELIGRO - HAY OPERACION ABIERTA
    echo No apagues la PC hasta cerrar o verificar la posicion.
    echo.
    echo El bot NO se detendra automaticamente mientras haya riesgo.
    echo Revise su terminal MT5 o el panel de STATUS.
    echo.
    echo ============================================================
    pause
    exit /b 1
)

:: No hay operacion abierta, proceder a detener
echo Creando señal de parada...
echo STOP_BOT > "%STOP_FILE%"
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
echo ============================================================
echo BOT DETENIDO
echo Ahora podes cerrar MT5 si queres.
echo ============================================================
echo.
pause
exit /b 0
