Write-Host "==============================================================" -ForegroundColor Yellow
Write-Host "MANIPULANTE MT5 DEMO LAUNCHER" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Yellow
Write-Host "WARNING: DEMO/PAPER ONLY. NO REAL TRADING." -ForegroundColor Red
Write-Host "NO AUTO ORDERS." -ForegroundColor Red
Write-Host "GLOBAL HARD CLOSE FRIDAY 16:55 NY." -ForegroundColor Red
Write-Host "==============================================================" -ForegroundColor Yellow
Write-Host ""
Read-Host "Presione Enter para confirmar que entiende que esto es SOLO DEMO..."

Write-Host "Abriendo Checklists y Runbook..."
Start-Process "..\04_OPERACION_DIARIA\MANIPULANTE_DAILY_RUNBOOK.md"
Start-Process "..\06_TEMPLATES\MANIPULANTE_DAILY_TRADE_LOG.csv"

Write-Host ""
Write-Host "Para abrir MT5 debe configurar la ruta en mt5_path_config.json"
Read-Host "Presione Enter para salir"
