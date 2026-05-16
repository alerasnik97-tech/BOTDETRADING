# START_MT5_DEMO_LOCAL.ps1
# Script de lanzamiento para MetaTrader 5 Demo + Python Executor

$ErrorActionPreference = "Continue"
$configPath = Join-Path $PSScriptRoot "mt5_local_config.json"

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   MT5 DEMO LOCAL LAUNCHER (Python + MT5)   " -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan

# 1. Verificar Configuracion
if (-not (Test-Path $configPath)) {
    Write-Host "ERROR: No se encuentra 'mt5_local_config.json'." -ForegroundColor Red
    Write-Host "Copia 'mt5_local_config.json.example' a 'mt5_local_config.json' y editalo."
    Read-Host "Presiona Enter para salir"
    exit
}

$config = Get-Content $configPath | ConvertFrom-Json

# 2. Iniciar MT5
if (Test-Path $config.mt5_terminal_path) {
    Write-Host "Abriendo MT5 DEMO desde: $($config.mt5_terminal_path)" -ForegroundColor Green
    Start-Process -FilePath $config.mt5_terminal_path
} else {
    Write-Host "ADVERTENCIA: No se encontro el terminal de MT5 en la ruta especificada." -ForegroundColor Yellow
    Write-Host "Asegurate de que MT5 ya este abierto antes de continuar."
}

# 3. Esperar carga
Write-Host "Esperando $($config.startup_delay_seconds) segundos para la carga de MT5..." -ForegroundColor Gray
Start-Sleep -Seconds $config.startup_delay_seconds

# 4. Lanzar Executor Python
Write-Host "Lanzando mt5_demo_executor.py..." -ForegroundColor Cyan
Set-Location $config.project_root

$pythonScript = "mt5_demo_executor_lab\mt5_demo_executor.py"

if (Test-Path $pythonScript) {
    Write-Host "Iniciando proceso Python en nueva ventana..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "& $($config.python_command) $pythonScript"
    Write-Host "PROCESO INICIADO CORRECTAMENTE" -ForegroundColor Green
} else {
    Write-Host "ERROR: No se encontro el script $pythonScript" -ForegroundColor Red
}

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "Audita los logs en mt5_demo_executor_lab\outputs\" -ForegroundColor Gray
Write-Host "====================================================" -ForegroundColor Cyan
Start-Sleep -Seconds 3
