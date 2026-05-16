@echo off
set ROOT_DIR=%~dp0..\..\..
cd /d "%ROOT_DIR%"
echo Ejecutando chequeo de alertas MANIPULANTE (Una vez)...
python BOT_V2_DAYTIME_LAB\src\phase45_run_alert_check.py --once
pause
