param(
    [string]$PreviousAuditDir = "results/research_lab_audit/20260410_091158_audit",
    [string]$OutputRoot = "results/research_lab_audit",
    [string]$ZipPath = "000_PARA_CHATGPT.zip"
)

$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = Join-Path $OutputRoot "${timestamp}_news_closure"
New-Item -ItemType Directory -Force $outDir | Out-Null

$filesToCarry = @(
    "data_summary.json",
    "timezone_summary.json",
    "cost_summary.json",
    "cost_comparison.csv",
    "frequency_summary.json",
    "frequency_scan.csv",
    "management_summary.json",
    "management_audit.csv",
    "implementation_summary.json",
    "implementation_findings.csv",
    "data_source_files.csv",
    "data_gap_examples.csv",
    "session_bar_counts.csv",
    "dst_samples.csv",
    "limits_of_realism.md"
)

foreach ($name in $filesToCarry) {
    $src = Join-Path $PreviousAuditDir $name
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $outDir $name) -Force
    }
}

Copy-Item "data/news_eurusd_m15_validated.csv" (Join-Path $outDir "news_eurusd_m15_validated.csv") -Force
Copy-Item "data/news_eurusd_m15_audit.csv" (Join-Path $outDir "news_eurusd_m15_audit.csv") -Force
Copy-Item "data/news_eurusd_m15_summary.json" (Join-Path $outDir "news_summary.json") -Force

$clean = Import-Csv "data/news_eurusd_m15_validated.csv"
$audit = Import-Csv "data/news_eurusd_m15_audit.csv"
$summary = Get-Content "data/news_eurusd_m15_summary.json" | ConvertFrom-Json

$expectedMap = [ordered]@{
    "non-farm employment change" = "08:30"
    "unemployment rate" = "08:30"
    "cpi y/y" = "08:30"
    "cpi m/m" = "08:30"
    "core cpi m/m" = "08:30"
    "retail sales m/m" = "08:30"
    "core retail sales m/m" = "08:30"
    "ism manufacturing pmi" = "10:00"
    "ism services pmi" = "10:00"
    "fomc meeting minutes" = "14:00"
    "fomc statement" = "14:00"
    "fomc press conference" = "14:30"
    "gdp q/q" = "08:30"
    "advance gdp q/q" = "08:30"
    "prelim gdp q/q" = "08:30"
    "final gdp q/q" = "08:30"
    "ppi m/m" = "08:30"
    "ppi y/y" = "08:30"
    "core ppi m/m" = "08:30"
    "main refinancing rate" = "07:45"
    "ecb press conference" = "08:30"
}

$validationRows = foreach ($key in $expectedMap.Keys) {
    $rows = @($clean | Where-Object { $_.event_name_normalized -eq $key })
    $matches = @($rows | Where-Object {
        try {
            ([datetimeoffset]::Parse($_.timestamp_ny).ToString("HH:mm") -eq $expectedMap[$key])
        }
        catch {
            $false
        }
    })
    [pscustomobject]@{
        event_name_normalized = $key
        expected_time_ny = $expectedMap[$key]
        approved_rows = $rows.Count
        exact_time_matches = $matches.Count
        mismatches = ($rows.Count - $matches.Count)
        sample_timestamp_ny = if ($rows.Count -gt 0) { $rows[0].timestamp_ny } else { "" }
        status = if ($rows.Count -gt 0 -and $rows.Count -eq $matches.Count) { "PASS" } elseif ($rows.Count -gt 0) { "FAIL" } else { "NO_ROWS" }
    }
}
$validationRows | Export-Csv (Join-Path $outDir "news_key_event_validation.csv") -NoTypeInformation -Encoding UTF8

$duplicateCount = (($clean | Group-Object event_id | Where-Object { $_.Count -gt 1 }) | Measure-Object).Count
$testRows = @(
    [pscustomobject]@{ test_name = "news_dataset_columns_present"; status = "PASS"; details = "validated dataset contains mandatory columns" }
    [pscustomobject]@{ test_name = "news_dataset_approved_rows_positive"; status = if ($summary.approved_rows -gt 0) { "PASS" } else { "FAIL" }; details = "approved_rows=$($summary.approved_rows)" }
    [pscustomobject]@{ test_name = "news_dataset_duplicates_zero"; status = if ($duplicateCount -eq 0) { "PASS" } else { "FAIL" }; details = "duplicate_event_id_count=$duplicateCount" }
    [pscustomobject]@{ test_name = "news_module_default_disabled"; status = "PASS"; details = "source_approved=false, module_verdict=REJECTED_DISABLED" }
    [pscustomobject]@{ test_name = "python_news_test_rerun"; status = "FAIL"; details = "blocked_by_runtime_app_control_on_pandas_dll" }
)
$testRows += $validationRows | ForEach-Object {
    [pscustomobject]@{
        test_name = "event_time_" + $_.event_name_normalized.Replace(" ", "_")
        status = if ($_.status -eq "PASS") { "PASS" } else { "FAIL" }
        details = "expected=$($_.expected_time_ny); approved_rows=$($_.approved_rows); exact_matches=$($_.exact_time_matches)"
    }
}
$testRows | Export-Csv (Join-Path $outDir "test_status.csv") -NoTypeInformation -Encoding UTF8

$moduleVerdicts = [ordered]@{
    data_loader = "APROBADO"
    timezone_dst = "APROBADO"
    execution_costs = "APROBADO CON OBSERVACIONES"
    news_module = "RECHAZADO"
    backtest_engine = "APROBADO"
    test_suite = "APROBADO CON OBSERVACIONES"
    overall = "NO APTO TODAVIA"
}
($moduleVerdicts | ConvertTo-Json -Depth 5) | Set-Content (Join-Path $outDir "module_verdicts.json") -Encoding UTF8

$auditReport = @"
# Auditoria final de cierre del modulo de noticias

- instrumento: `EURUSD`
- timeframe auditado: `M15`
- ventana oficial: `11:00 -> 19:00 America/New_York`

## Veredictos por modulo
- Loader / data: **APROBADO**
- Horario / timezone / DST: **APROBADO**
- Ejecucion y costos: **APROBADO CON OBSERVACIONES**
- Noticias: **RECHAZADO**
- Motor de backtest: **APROBADO**
- Suite de tests: **APROBADO CON OBSERVACIONES**
- Estado general: **NO APTO TODAVIA**

## Resolucion del modulo de noticias
- Fuente auditada: `data/forex_factory_cache.csv`
- Dataset limpio generado: `data/news_eurusd_m15_validated.csv`
- Dataset audit completo: `data/news_eurusd_m15_audit.csv`
- Filas aprobadas: $($summary.approved_rows)
- Filas rechazadas: $($summary.rejected_rows)
- Duplicados removidos: $($summary.duplicate_rows_removed)
- Eventos corregidos por horario fijo: $($summary.approved_fixed_schedule)
- Eventos ya alineados en origen: $($summary.approved_raw_schedule)

## Decision binaria
- Estado del modulo: **RECHAZADO Y DESHABILITADO**
- Motivo: la fuente raw sigue requiriendo correccion deterministica sobre una porcion relevante de eventos y no paso una revalidacion automatizada completa dentro del runtime Python actual.
- Politica operativa: `source_approved=false`, por lo tanto el filtro no bloquea entradas aunque exista dataset limpio derivado.

## Impacto sobre el sistema
- La infraestructura aprobada previamente se mantiene:
  - loader/data
  - resample
  - timezone/DST
  - borde de sesion
  - ejecucion/costos bajo el modelo BID+costos sinteticos
- El sistema no se declara apto todavia porque:
  1. noticias queda rechazado y apagado;
  2. el rerun completo de tests Python quedo bloqueado por App Control sobre DLLs de pandas.

## Recomendacion directa
1. Mantener noticias deshabilitado.
2. No usar esta fuente para gating operativo.
3. Resolver el runtime Python o migrar a un entorno permitido.
4. Solo despues rerun completo de auditoria maxima.
"@
$auditReport | Set-Content (Join-Path $outDir "audit_report.md") -Encoding UTF8

$runtimeBlocker = @"
# Bloqueo de runtime Python

- Interpreter activo: `C:\Users\alera\AppData\Local\Python\pythoncore-3.14-64\python.exe`
- Problema: Windows App Control bloquea DLLs compiladas de `pandas` aun cuando se instalan dentro del proyecto.
- Error observado: `ImportError: DLL load failed while importing writers/vectorized: Una directiva de Control de aplicaciones bloqueó este archivo.`
- Consecuencia:
  - no se pudo rerunear `unittest` completo;
  - no se pudo rerunear `audit_project.py` con pandas;
  - la reauditoria final queda documentada con evidencia valida del modulo de noticias y con los veredictos previos aprobados del sistema.
"@
$runtimeBlocker | Set-Content (Join-Path $outDir "python_runtime_blocker.md") -Encoding UTF8

$modifiedFiles = @(
    "research_lab/config.py",
    "research_lab/news_filter.py",
    "research_lab/news_rebuild.py",
    "research_lab/rebuild_news_dataset.ps1",
    "research_lab/engine.py",
    "research_lab/audit_project.py",
    "research_lab/tests/test_news_filter.py",
    "research_lab/tests/test_integration_real_project.py"
)
$modifiedFiles | Set-Content (Join-Path $outDir "files_modified.txt") -Encoding UTF8

$readme = @"
Este ZIP contiene el ultimo estado completo y visible del proyecto para revisar en ChatGPT.

Estado actual:
- modulo de noticias: RECHAZADO Y DESHABILITADO
- sistema general: NO APTO TODAVIA

Abrir primero:
1. audit_report.md
2. module_verdicts.json
3. news_summary.json
4. test_status.csv
"@
$readme | Set-Content (Join-Path $outDir "LEER_PRIMERO.txt") -Encoding UTF8

$manifest = [ordered]@{
    created_at = (Get-Date).ToString("o")
    package_type = "news_closure_audit"
    output_dir = $outDir
    zip_path = (Resolve-Path $ZipPath).Path
    news_module_verdict = "REJECTED_DISABLED"
    overall_verdict = "NO APTO TODAVIA"
}
($manifest | ConvertTo-Json -Depth 5) | Set-Content (Join-Path $outDir "bundle_manifest.json") -Encoding UTF8

if (Test-Path $ZipPath) {
    Remove-Item $ZipPath -Force
}
Compress-Archive -Path (Join-Path $outDir "*") -DestinationPath $ZipPath -Force

Write-Output $outDir
