
# VPS PREFLIGHT CHECK SCRIPT
$ErrorActionPreference = "Stop"

Write-Host "--- INICIANDO VPS PREFLIGHT CHECK ---" -ForegroundColor Cyan

# 1. Verificar Python
try {
    $pyVer = python --version
    Write-Host "[OK] Python detectado: $pyVer"
} catch {
    Write-Error "Python no encontrado en el PATH."
}

# 2. Verificar Entorno Virtual
if (Test-Path ".venv") {
    Write-Host "[OK] Entorno virtual .venv detectado."
} else {
    Write-Warning "Entorno virtual .venv no encontrado."
}

# 3. Verificar Dependencias (Imports)
Write-Host "Verificando imports críticos..."
python -c "import pandas; import numpy; import pytz; import MetaTrader5; print('Imports OK')"
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Librerías instaladas correctamente."
} else {
    Write-Error "Faltan librerías. Ejecute pip install -r requirements.txt"
}

# 4. Verificar Config Local
if (Test-Path "mt5_local_config.json") {
    $config = Get-Content "mt5_local_config.json" | ConvertFrom-Json
    if ($config.account_mode -eq "DEMO_ONLY" -and $config.allow_live -eq $false) {
        Write-Host "[OK] Configuración segura detectada (DEMO_ONLY, allow_live=false)."
    } else {
        Write-Error "¡RIESGO DETECTADO! La configuración permite trading real o no es DEMO_ONLY."
    }
} else {
    Write-Warning "mt5_local_config.json no encontrado. Use las plantillas en VPS_READINESS\config_templates\"
}

# 5. Verificar Git
$branch = git rev-parse --abbrev-ref HEAD
Write-Host "[OK] Rama Git actual: $branch"
if ($branch -ne "chore/github-clean-sync") {
    Write-Warning "No estás en la rama segura recomendada (chore/github-clean-sync)."
}

Write-Host "--- PREFLIGHT COMPLETADO ---" -ForegroundColor Green
