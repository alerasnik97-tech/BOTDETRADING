# NEXT PROMPT — DATA RECOVERY: MISSING NEWS / PRECISION FILES

**Priority: B4 — owner action required. Parallelizable with B1.**

## Problem (evidence from Phase E audit, 2026-05-16)
Recursive existence checks (no data loaded) confirm these required assets are
**absent everywhere** in the repo:

| File / asset | Expected canonical path | Required by |
|---|---|---|
| `forex_factory_cache.csv` | `05_MARKET_DATA_VAULT/data/forex_factory_cache.csv` | USDJPY news-fortress builder (`build_usdjpy_news_fortress_dataset.py:18`), `news_filter.py`, `news_phase3`, `test_usdjpy_readiness` |
| `news_eurusd_v2_utc.csv` | `05_MARKET_DATA_VAULT/data/news_eurusd_v2_utc.csv` | `build_am_grade_news_dataset.py:23`, config `DEFAULT_NEWS_V2_UTC_FILE` |
| high-precision M1 dukascopy bundle | `05_MARKET_DATA_VAULT/legacy_data/data_precision/dukascopy` | high-precision data loader (`test_data_loader_high_precision_timeframe`) |

Present and OK: `canonical_anchor_events.csv` (68,915 B, regenerated).

## Objective
Restore the missing assets via an **owner-authorized** path. This is a data
governance / recovery task, NOT an automated scrape.

## Scope of the next prompt
1. For each missing asset, the owner decides ONE recovery route:
   - provide the file from an owner-held backup / external archive; OR
   - authorize a controlled, logged regeneration **only if** a safe,
     pre-existing pipeline exists and the owner explicitly approves running it
     (not in audit scope); OR
   - authorize a scoped scrape (explicit owner approval; out of audit scope).
2. Record provenance + integrity (sha256, row count, date range, source) in
   `06_GOVERNANCE_AND_COMPLIANCE/data_governance/`.
3. Note dependency order: `news_eurusd_v2_utc.csv` rebuild may depend on
   `forex_factory_cache.csv` — recover the FF cache first.
4. After restore, re-run the Phase E audit (data + research_lab tests) to
   confirm B4 (and the data-driven subset of B5) clears.

## Hard rules
- **No scrape / download / internet / dataset generation performed by the
  agent without explicit per-asset owner authorization.** Default = audit only.
- No 2025/2026 period analysis. No validation/holdout process. No backtest /
  strategy / F06-real / optimization / sweep. No data deletion or overwrite of
  existing files. No ZIP workflow. No `main`, no force push.
- Missing data must never be silently synthesized or stubbed to fake a green
  test.
