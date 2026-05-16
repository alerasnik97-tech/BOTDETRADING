# NEXT PROMPT: CLAUDE EURUSD DATA FOUNDATION AUDIT

Actua como Claude Code en modo institutional quant data-foundation auditor, Python test architecture auditor, timezone/DST auditor, leakage prevention specialist y Git safety officer.

Objetivo:
Auditar la fundacion EURUSD prepared OHLCV construida localmente desde raw tick, sin ejecutar backtest ni estrategias.

Repo:
`C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`

Branch:
`governance/phase-d-reconciliation-20260516`

Archivos a auditar:

- `03_RESEARCH_LAB/research_lab/data_preparation/eurusd_prepared_ohlcv_builder.py`
- `04_INFRASTRUCTURE_ENGINEERING/data_builders/build_eurusd_prepared_ohlcv.py`
- `03_RESEARCH_LAB/research_lab/config.py`
- `03_RESEARCH_LAB/research_lab/tests/test_eurusd_prepared_ohlcv_builder.py`
- `03_RESEARCH_LAB/research_lab/tests/test_integration_real_project.py`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/EURUSD_PREPARED_OHLCV_BUILD_REPORT.md`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/EURUSD_PREPARED_OHLCV_MANIFEST.csv`
- local ignored output folder: `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared`

Do not run:

- backtest
- strategy
- F06 real
- optimization
- sweep
- validation
- holdout
- news rebuild
- downloads
- scraping

Audit checklist:

1. Confirm no prepared output contains timestamp `>= 2025-01-01T00:00:00Z`.
2. Confirm raw builder excludes 2025/2026 monthly files by filename and filters max timestamp before resampling.
3. Confirm builder constructs OHLC causally from current/past tick values only.
4. Confirm no forward-fill, interpolation, empty-bar fabrication, or synthetic price creation.
5. Confirm `volume` is observed tick count per bar.
6. Confirm loader contract: filenames, OHLCV columns, timezone-explicit index, monotonic index, no duplicates.
7. Confirm generated CSVs and local manifests are gitignored and not committed.
8. Confirm governance manifest rowcounts and hashes match local files.
9. Confirm news is fail-closed and not required for core loader.
10. Confirm F06 pipeline remains PASS and broader research_lab failures are not hidden.

Required decision:

- EURUSD_DATA_FOUNDATION_ACCEPTED_FOR_FINAL_PRELAB_AUDIT
- EURUSD_DATA_FOUNDATION_BLOCKED_LEAKAGE
- EURUSD_DATA_FOUNDATION_BLOCKED_SCHEMA
- EURUSD_DATA_FOUNDATION_BLOCKED_LOADER_CONTRACT
- EURUSD_DATA_FOUNDATION_BLOCKED_GIT_SAFETY
