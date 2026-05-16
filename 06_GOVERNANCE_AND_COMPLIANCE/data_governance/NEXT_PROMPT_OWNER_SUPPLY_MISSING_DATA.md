# NEXT PROMPT — OWNER: SUPPLY MISSING DATA

**Priority: hard lab blocker (B4). Owner action required — agent cannot
recover these (absent locally; scrape/download not authorized).**

## What is missing and where to place it

| # | File / bundle | Place EXACTLY at | Used by |
|---|---|---|---|
| 1 | `forex_factory_cache.csv` | `05_MARKET_DATA_VAULT/data/forex_factory_cache.csv` | `build_usdjpy_news_fortress_dataset.py`, `news_filter.py`, news_phase3, `test_usdjpy_readiness` |
| 2 | `news_eurusd_v2_utc.csv` | `05_MARKET_DATA_VAULT/data/news_eurusd_v2_utc.csv` | `build_am_grade_news_dataset.py`, config `DEFAULT_NEWS_V2_UTC_FILE` |
| 3 | hi-precision **M1 dukascopy bundle** (EURUSD) | populate `05_MARKET_DATA_VAULT/legacy_data/data_precision/dukascopy/` | high-precision data loader (`test_data_loader_high_precision_timeframe`) |

All three are confirmed absent from project, Desktop, Downloads, Documents,
`07_BACKUPS`, `ARCHIVO_HISTORICO`, and quarantine. Every dukascopy directory
is empty.

## What the owner must do
1. Provide each file/bundle from an owner-held backup or external archive
   (USB, cloud drive, original acquisition machine).
2. Drop them at the EXACT canonical paths above (do not rename; preserve
   original content — no edits).
3. For each, record provenance in
   `06_GOVERNANCE_AND_COMPLIANCE/data_governance/`: origin, original
   acquisition date, sha256, row/file count, date range, source.
4. Recover `forex_factory_cache.csv` first — `news_eurusd_v2_utc.csv` rebuild
   may depend on it.
5. Then re-run the data-completeness audit + safe tests (no fabrication).

## Hard rules
- The agent will NOT scrape/download/regenerate/synthesize to fill these
  (default = audit only). No stubs to fake green tests. No 2025/2026 analysis.
- No backtest/strategy/F06-real/optimization/sweep/validation/holdout.
- No `main`, no force push, no ZIP. Heavy data stays gitignored/local — never
  force-add to git.
- If a file genuinely cannot be supplied, that scope (e.g. USDJPY / hi-precision
  research) stays formally BLOCKED — do not work around it.
