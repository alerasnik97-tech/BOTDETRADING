@echo off
TITLE STATUS MANIPULANTE FTMO TRIAL
cd /d "%~dp0..\.."

echo ======================================================================
echo MANIPULANTE FTMO TRIAL - CONTROL PANEL STATUS [LIFECYCLE]
echo ======================================================================
echo.

echo [PROCESS CHECK]
wmic process where "name='python.exe' and commandline like '%%phase37_ftmo_trial_bot_runner.py%%'" get processid,commandline /format:list

echo.
echo [LOCK FILE]
if exist "MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\runner.lock" (
    echo runner.lock PRESENTE (PID: )
    type "MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\runner.lock"
) else (
    echo runner.lock AUSENTE (Bot no activo)
)

echo.
echo [HEARTBEAT]
if exist "MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\heartbeat.txt" (
    type "MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\heartbeat.txt"
) else (
    echo Heartbeat no encontrado.
)

echo.
echo [LAST 5 DECISIONS]
powershell -Command "if (Test-Path 'MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\decisions.csv') { Get-Content 'MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\decisions.csv' -Tail 5 } else { 'No log found' }"

echo.
echo ======================================================================
echo NOTA: Si ves CRITICAL_DO_NOT_TURN_OFF_PC, NO APAGUES LA PC.
echo El bot intentara cerrar la posicion a las 19:45 NY.
echo ======================================================================
pause
