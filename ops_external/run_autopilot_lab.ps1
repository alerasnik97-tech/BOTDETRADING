# FEEDER-AUTOPILOT ALIGNMENT SCRIPT
# Ejecuta autopilot con fecha de ayer (día completo recién cerrado)
$now = Get-Date
$targetDate = $now.AddDays(-1).ToString("yyyy-MM-dd")

Write-Host "=== AUTOPILOT LAB ALIGNMENT ===" -ForegroundColor Cyan
Write-Host "Fecha objetivo (ayer completo): $targetDate" -ForegroundColor Yellow

cd "C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
python "scratch\run_lab_autopilot_v1.py" --date $targetDate

exit $LASTEXITCODE
