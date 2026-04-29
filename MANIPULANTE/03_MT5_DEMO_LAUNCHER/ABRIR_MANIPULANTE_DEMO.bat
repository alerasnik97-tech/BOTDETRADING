@echo off
echo ==============================================================
echo MANIPULANTE MT5 DEMO LAUNCHER
echo ==============================================================
echo WARNING: DEMO/PAPER ONLY. NO REAL TRADING.
echo NO AUTO ORDERS.
echo GLOBAL HARD CLOSE FRIDAY 16:55 NY.
echo ==============================================================
echo.
echo Presione cualquier tecla para confirmar que entiende que esto es SOLO DEMO...
pause >nul

echo Abriendo Checklists y Runbook...
start "" "..\04_OPERACION_DIARIA\MANIPULANTE_DAILY_RUNBOOK.md"
start "" "..\06_TEMPLATES\MANIPULANTE_DAILY_TRADE_LOG.csv"

echo.
echo Para abrir MT5 debe configurar la ruta en mt5_path_config.json
echo.
pause
