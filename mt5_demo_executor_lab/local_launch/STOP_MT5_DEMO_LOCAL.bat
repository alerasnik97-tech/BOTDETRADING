@echo off
TITLE MT5 DEMO STOP
echo ====================================================
echo      DETENCION DE ENTORNO DEMO (Python)
echo ====================================================
echo.
echo Intentando cerrar procesos de mt5_demo_executor.py de forma segura...

:: Buscar procesos de PowerShell que esten corriendo el executor y cerrarlos
powershell -Command "Get-Process | Where-Object { $_.MainWindowTitle -like '*mt5_demo_executor*' -or $_.CommandLine -like '*mt5_demo_executor*' } | Stop-Process -Force -ErrorAction SilentlyContinue"

echo.
echo [OK] Procesos detenidos.
echo [!] REVISA TU TERMINAL MT5 antes de apagar la PC.
echo.
pause
