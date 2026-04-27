
# Script para arrancar el monitoreo en la VPS
Write-Host "Arrancando monitoreo FORWARD DEMO..." -ForegroundColor Cyan

# Ejecutar preflight
.\VPS_READINESS\scripts\vps_preflight_check.ps1
if ($LASTEXITCODE -ne 0) { exit 1 }

# Ejecutar check de conexión MT5
python .\VPS_READINESS\scripts\vps_mt5_connection_check.py
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Entorno validado. Iniciando loop de ejecución..." -ForegroundColor Green
# Aquí iría la llamada al runner principal de forward
# python .\BOT_V2_DAYTIME_LAB\src\run_forward_demo.py
