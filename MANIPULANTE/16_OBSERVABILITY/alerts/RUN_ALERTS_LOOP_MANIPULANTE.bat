@echo off
set ROOT_DIR=%~dp0..\..\..
cd /d "%ROOT_DIR%"
echo Iniciando loop de alertas MANIPULANTE (60s)...
python BOT_V2_DAYTIME_LAB\src\phase45_run_alert_check.py --loop --interval-seconds 60
pause
