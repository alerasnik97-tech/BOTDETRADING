
# Validar integridad local antes de sincronizar
Write-Host "Verificando higiene local de la VPS..."

if (Test-Path "000_PARA_CHATGPT.zip") {
    Write-Host "[OK] ZIP canónico detectado."
} else {
    Write-Warning "Falta ZIP canónico 000_PARA_CHATGPT.zip"
}

$dupZips = Get-ChildItem -Path . -Filter "*.zip" -Recurse | Where-Object { $_.Name -ne "000_PARA_CHATGPT.zip" }
if ($dupZips) {
    Write-Warning "Se detectaron ZIPs duplicados que podrían causar polución: $($dupZips.Name)"
}

if (Test-Path ".env") {
    Write-Error "¡RIESGO! Archivo .env detectado en raíz. Debe ser ignorado."
}
