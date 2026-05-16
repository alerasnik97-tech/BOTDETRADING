@echo off
TITLE MT5 DEMO STARTUP
echo Iniciando entorno MT5 DEMO + Python...
powershell -ExecutionPolicy Bypass -File "%~dp0START_MT5_DEMO_LOCAL.ps1"
pause
