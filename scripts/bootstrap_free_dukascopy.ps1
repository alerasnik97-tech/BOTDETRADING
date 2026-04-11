param(
    [string[]]$Pairs = @("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "GBPJPY"),
    [int]$StartYear = 2020,
    [int]$EndYear = 2025,
    [string]$DataDir = "data_free_full",
    [string]$ReportDir = "reports_free_full",
    [double]$RiskPct = 0.75,
    [switch]$RunBacktest,
    [switch]$RunOptimize
)

$ErrorActionPreference = "Stop"
$scriptPath = Join-Path $PSScriptRoot "fx_multi_timeframe_backtester.py"
$failures = @()

Write-Host ""
Write-Host "=== Bootstrap Dukascopy Gratis ==="
Write-Host "Pairs: $($Pairs -join ', ')"
Write-Host "Period: $StartYear-$EndYear"
Write-Host "DataDir: $DataDir"
Write-Host "ReportDir: $ReportDir"

for ($year = $StartYear; $year -le $EndYear; $year++) {
    $start = "{0}-01-01" -f $year
    $end = "{0}-12-31" -f $year

    foreach ($pair in $Pairs) {
        Write-Host ""
        Write-Host ">>> Caching $pair for $year"

        try {
            & python $scriptPath cache-data `
                --pairs $pair `
                --start $start `
                --end $end `
                --source dukascopy `
                --download-missing `
                --data-dir $DataDir `
                --report-dir $ReportDir

            if ($LASTEXITCODE -ne 0) {
                throw "Python exited with code $LASTEXITCODE"
            }
        }
        catch {
            $failures += "$pair $year :: $($_.Exception.Message)"
            Write-Warning "Cache load failed for $pair $year. Rerunning the script will reuse any completed monthly files."
        }
    }
}

if ($failures.Count -gt 0) {
    Write-Host ""
    Write-Warning "Cache population had failures. Prepared files were not rebuilt to avoid using an incomplete dataset."
    $failures | ForEach-Object { Write-Host " - $_" }
    exit 1
}

Write-Host ""
Write-Host ">>> Building prepared M5/M15/H1 files for the full range"
& python $scriptPath prepare-data `
    --pairs $Pairs `
    --start "$StartYear-01-01" `
    --end "$EndYear-12-31" `
    --source dukascopy `
    --download-missing `
    --data-dir $DataDir `
    --report-dir $ReportDir

Write-Host ""
Write-Host ">>> Validating data quality for the full range"
& python $scriptPath validate-data `
    --pairs $Pairs `
    --start "$StartYear-01-01" `
    --end "$EndYear-12-31" `
    --source local `
    --data-dir (Join-Path $DataDir "prepared") `
    --report-dir $ReportDir

if ($RunBacktest) {
    Write-Host ""
    Write-Host ">>> Running full backtest"
    & python $scriptPath run `
        --strict-data-quality `
        --pairs $Pairs `
        --start "$StartYear-01-01" `
        --end "$EndYear-12-31" `
        --source local `
        --data-dir (Join-Path $DataDir "prepared") `
        --report-dir $ReportDir `
        --risk-pct $RiskPct
}

if ($RunOptimize) {
    Write-Host ""
    Write-Host ">>> Running robust optimization"
    & python $scriptPath optimize `
        --strict-data-quality `
        --pairs $Pairs `
        --start "$StartYear-01-01" `
        --end "$EndYear-12-31" `
        --source local `
        --data-dir (Join-Path $DataDir "prepared") `
        --report-dir $ReportDir `
        --risk-pct $RiskPct `
        --max-combinations 12
}

Write-Host ""
Write-Host "Free bootstrap finished."
