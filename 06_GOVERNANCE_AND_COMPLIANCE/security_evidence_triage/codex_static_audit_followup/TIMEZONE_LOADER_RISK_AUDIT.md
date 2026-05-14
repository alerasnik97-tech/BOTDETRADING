# TIMEZONE / LOADER RISK AUDIT

Status: COMPLETE_STATIC_READ
Decision class: BLOCKS_PAPER / BLOCKS_DEMO_FUNDED

## Scope read
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v7_engine/engine.py`
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v6_utils/data_loader.py`
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v6_utils/bars.py`
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v6_utils/execution.py`
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/v50b_family_preflight_runner.py`

## Findings

### TZ-001 - Aware timestamp offset can be dropped
- Severity: HIGH
- Status: CONFIRMED_ACTIVE
- Evidence: `engine.py` converts aware timestamps with `.replace(tzinfo=None)` after only checking `tzinfo`, without `astimezone(UTC)`.
- Impact: safe only if every upstream timestamp is already UTC. A non-UTC aware timestamp would be relabeled as UTC and could shift schedule/news/FTMO checks.
- Classification: BLOCKS_PAPER, BLOCKS_DEMO_FUNDED
- Required correction: convert aware timestamps to UTC first, then strip tzinfo only at the engine boundary; add a non-UTC-aware regression test.

### LOADER-001 - Legacy parquet root does not point to canonical vault
- Severity: HIGH
- Status: CONFIRMED_ACTIVE
- Evidence: `data_loader.py` hardcodes `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA\tick\EURUSD\monthly`; physical vault is under `05_MARKET_DATA_VAULT/BOT_MARKET_DATA/tick/EURUSD/monthly`.
- Impact: any caller using the generic loader can fail or silently depend on a stale local path.
- Classification: BLOCKS_PAPER, BLOCKS_DEMO_FUNDED
- Required correction: route all loaders through an explicit vault config and fail closed if the canonical path is missing.

### LOADER-002 - Price columns downcast to float32 by default
- Severity: MEDIUM
- Status: CONFIRMED_ACTIVE
- Evidence: `data_loader.py` downcasts bid/ask and volumes to `float32`.
- Impact: acceptable for memory probes only; unsafe as a default for audited tick fills and price precision.
- Classification: FUTURE_HARDENING before paper/demo
- Required correction: default to float64 for trading-critical paths; allow float32 only under explicit memory-safe research mode.

### V50B-LOADER-001 - Current V50B preflight does not use real loader/execution
- Severity: CRITICAL
- Status: CONFIRMED_ACTIVE
- Evidence: `v50b_family_preflight_runner.py` logs month completion but does not load ticks, build bars, detect signals, or execute trades.
- Impact: V50B result files cannot be treated as backtest evidence.
- Classification: BLOCKS_CURRENT_RESEARCH
- Required correction: invalidate generated V50B results and rebuild from real engine outputs only.
