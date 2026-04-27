$ZipPath = "000_PARA_CHATGPT.zip"
$FilesToZip = @(
    "H6_FORWARD_ONLY_FREEZE.md",
    "H6_FORWARD_ONLY_GATE_PLAN.md",
    "H6_PAPER_EXECUTION_FREEZE.md",
    "H6_PAPER_STAGE_RUNBOOK.md",
    "H6_PAPER_STAGE_METRICS.md",
    "H6_PAPER_STAGE_OBSERVATION_PLAN.md",
    "H6_PAPER_STAGE_CHECKPOINT_20_SIGNALS.md",
    "H6_SPREAD_SLIPPAGE_CALIBRATION_AUDIT.md",
    "scripts/h6_paper_shadow_runner.py",
    "scripts/h6_provenance_tagger.py",
    "results/H6_SHADOW_LEDGER.csv",
    "results/H6_SHADOW_LEDGER_OFFICIAL.csv",
    "results/H6_SHADOW_LEDGER_OBSERVED.csv",
    "results/H6_RESEARCH_VS_SHADOW_OFFICIAL.csv",
    "results/H6_RESEARCH_VS_SHADOW_OBSERVED.csv"
)

if (Test-Path $ZipPath) {
    Remove-Item $ZipPath -Force
}

$ValidFiles = @()
foreach ($file in $FilesToZip) {
    if (Test-Path $file) {
        $ValidFiles += $file
    }
    else {
        Write-Warning "Archivo ausente (no se incluirá en el zip): $file"
    }
}

Compress-Archive -Path $ValidFiles -DestinationPath $ZipPath -Force
Write-Output "ZIP de checkpoint generado exitosamente en: $(Resolve-Path $ZipPath)"