# NEXT PROMPT: CLAUDE EURUSD 2015-2026 DATA FOUNDATION AUDIT

Actua como Claude Code en modo institutional quant data-foundation auditor, Python test architecture auditor, timezone/DST auditor, leakage prevention specialist, holdout governance officer y Git safety officer.

Objetivo: auditar la fundacion EURUSD 2015-2026 sin ejecutar backtest ni estrategias.

Repo: `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`
Branch: `governance/phase-d-reconciliation-20260516`

No ejecutar:
- backtest
- strategy
- F06 real
- optimization
- sweep
- validation
- holdout research
- news rebuild
- downloads
- scraping
- performance metrics

Auditar:
1. Raw coverage 2015-01 through 2026-04 from `EURUSD_RAW_2015_2026_SOURCE_INVENTORY.csv`.
2. Builder code and partition logic in `eurusd_prepared_ohlcv_builder.py`.
3. Train prepared OHLCV 2015-2024 and no 2025/2026 leakage.
4. Sealed holdout 2025-2026 and seal manifests.
5. `DEFAULT_DATA_DIRS` excludes sealed holdout.
6. Master manifest hashes, rowcounts, timestamps, and loader contract.
7. Tests and remaining broader research_lab failures.
8. Git safety: generated OHLCV data is not committed.

Required decision:
- EURUSD_2015_2026_DATA_FOUNDATION_ACCEPTED_FOR_FINAL_PRELAB_GATE
- EURUSD_2015_2026_DATA_FOUNDATION_BLOCKED_RAW_COVERAGE
- EURUSD_2015_2026_DATA_FOUNDATION_BLOCKED_TRAIN_QUALITY
- EURUSD_2015_2026_DATA_FOUNDATION_BLOCKED_HOLDOUT_SEAL
- EURUSD_2015_2026_DATA_FOUNDATION_BLOCKED_GIT_SAFETY

Do not authorize strategies directly. Only decide whether this can move to final pre-lab gate.
