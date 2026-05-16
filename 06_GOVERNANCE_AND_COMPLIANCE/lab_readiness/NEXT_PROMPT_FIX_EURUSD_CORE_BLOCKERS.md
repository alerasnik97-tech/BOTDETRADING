# NEXT PROMPT - FIX EURUSD CORE BLOCKERS

Act as institutional quant lab readiness auditor, Python test architecture engineer, data dependency mapper, and repo integrity officer.

Repository:
`C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`

Branch:
`governance/phase-d-reconciliation-20260516`

Current Phase E verdict:
`PHASE_E_BURNDOWN_PARTIAL_OWNER_REVIEW_REQUIRED`

Do not run backtests, strategy research, optimization, sweeps, validation, holdout, F06 real runs, data download, data regeneration, scraping, or synthetic data creation.

## Objective

Resolve only the remaining pre-lab blockers that prevent a clean EURUSD train-only lab audit.

## Blocker 1 - EURUSD Prepared OHLCV Source Authority

Evidence:

- `research_lab.config.DEFAULT_DATA_DIRS` points to:
  - `05_MARKET_DATA_VAULT/eurusd_data/data_free_2020/prepared`
  - `05_MARKET_DATA_VAULT/eurusd_data/data_candidates_2022_2025/prepared`
- Those prepared folders are empty.
- `research_lab.data_loader.load_prepared_ohlcv` reads prepared CSV files named `EURUSD_<TF>.csv`.
- Raw tick material exists under `05_MARKET_DATA_VAULT/BOT_MARKET_DATA/tick/EURUSD`, but it is not wired into the loader.

Required owner/Claude decision:

Choose one path and document it before implementation:

1. Materialize approved prepared OHLCV CSVs into the configured `DEFAULT_DATA_DIRS`, with manifest, hashes, rowcounts, timezone certification, 2025/2026 exclusion rules, and no invented data.
2. Approve a loader bridge/repoint from the existing raw tick vault to a prepared-OHLCV contract, with tests and fail-closed guards.
3. Reject both and keep lab blocked.

No silent loader repointing.

## Blocker 2 - Broader research_lab Non-Green Tests

Post-Phase-E compact inventory:

- 164 tests run
- 16 failures
- 9 errors
- 13 skipped

Remaining groups:

- Engine/stop-entry/level2/high-precision synthetic behavior assertions after test stubs now expose `generate_signal`.
- AM news builder coverage expectations.
- Legacy `NewsConfig().enabled` expectation.

Required next action:

Classify each remaining failure as:

- REAL_ENGINE_REGRESSION
- LEGACY_EXPECTATION
- TEST_CONTRACT_NEEDS_UPDATE
- OPTIONAL_MODULE_BLOCKED
- NOT_REQUIRED_FOR_EURUSD_CORE

Fix only test contracts that are objectively stale. Do not rewrite expected trading behavior just to pass. Do not touch engine/trading logic without explicit owner approval.

## Must Preserve

- No main.
- No force push.
- No ZIP workflow.
- No data mutation without manifest.
- No 2025/2026 analysis.
- No strategy/backtest/validation/holdout.
- No fake stubs or synthetic datasets.

## Expected Output

Create/update governance report under:
`06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/`

End with one of:

- `EURUSD_CORE_LAB_BLOCKED_BY_REQUIRED_DATA`
- `EURUSD_CORE_LAB_BLOCKED_BY_IMPORT_PATH_TESTS`
- `EURUSD_CORE_LAB_SCOPE_READY_FOR_CLAUDE_AUDIT`
- `PHASE_E_BURNDOWN_PARTIAL_OWNER_REVIEW_REQUIRED`
