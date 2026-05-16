# EURUSD 2015-2026 DATA FOUNDATION REPORT

## 1. Status

EURUSD_2015_2026_DATA_FOUNDATION_READY_FOR_CLAUDE_AUDIT

## 2. Executive Summary

EURUSD raw coverage from 2015-01 through 2026-04 is confirmed from local monthly parquet metadata. Train prepared OHLCV 2015-2024 is certified against hashes, rowcounts, loader schema, timezone, duplicates, OHLC integrity, NaN checks, and no 2025/2026 leakage. A separate sealed holdout partition for 2025-2026 was built locally and is not included in `DEFAULT_DATA_DIRS`.

No strategy, backtest, optimization, validation, holdout research, F06 real run, news rebuild, download, scraping, or performance metric was executed.

## 3. Raw Coverage Audit

- status: `RAW_EURUSD_2015_2026_COVERAGE_CONFIRMED`
- files_found: `3548`
- total_size_bytes: `9210754686`
- monthly_parquet_files: `136`
- expected_months: `136`
- missing_months: `[]`
- schema_consistent: `True`
- inventory: `06_GOVERNANCE_AND_COMPLIANCE/data_governance/EURUSD_RAW_2015_2026_SOURCE_INVENTORY.csv`

## 4. Train Prepared Audit

- status: `EURUSD_TRAIN_PREPARED_CERTIFIED_FOR_PRELAB`
- prepared_dir: `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared`
- default loader access: `YES`
- no 2025/2026 rows: `YES`
- audit: `06_GOVERNANCE_AND_COMPLIANCE/data_governance/EURUSD_PREPARED_TRAIN_2015_2024_AUDIT.md`

| timeframe | rows | min_timestamp_utc | max_timestamp_utc | sha256 |
|---|---:|---|---|---|
| M1 | 3634609 | 2015-01-01 22:01:00+00:00 | 2024-12-31 22:00:00+00:00 | 93b74bb65b794b8ee20fd8fd1a46e55ccd1329cb1e3d5de59794b9f6b9078f23 |
| M5 | 729382 | 2015-01-01 22:05:00+00:00 | 2024-12-31 22:00:00+00:00 | 386ab589d14e52236581201b03aa7d8e6c5d2c9771bc59eea00d34abc1afa625 |
| M15 | 243169 | 2015-01-01 22:15:00+00:00 | 2024-12-31 22:00:00+00:00 | ebe7ade5b77850286435715755ad114e67efc797b0150574a43041e0bbdbc04e |
| H1 | 60800 | 2015-01-01 23:00:00+00:00 | 2024-12-31 22:00:00+00:00 | ef1bc7156e6fc4938c73e8ca277c38231605631c416d2d298a6bbe23de60f852 |

## 5. Sealed Holdout Build

- seal: `SEALED_NOT_FOR_RESEARCH_SELECTION`
- prepared_dir: `05_MARKET_DATA_VAULT/eurusd_data/sealed_holdout_2025_2026/prepared`
- manifests: `05_MARKET_DATA_VAULT/eurusd_data/sealed_holdout_2025_2026/manifests` (local ignored)
- source period: `2025-01` through `2026-04`
- source files used: `16`
- raw rows read: `29839765`
- default loader access: `NO`
- note: output timestamps are right-edge bar labels; `2026-05-01 00:00:00+00:00` labels bars formed from final April 2026 ticks, not a strategy use authorization.

| timeframe | rows | min_timestamp_utc | max_timestamp_utc | sha256 | default_loader_access |
|---|---:|---|---|---|---|
| M1 | 474482 | 2025-01-01 22:01:00+00:00 | 2026-05-01 00:00:00+00:00 | b82c4fd7a6da734562d1ec9252ab61b780e5261aa2ff9ffaf0e38c57f40e365c | NO |
| M5 | 95219 | 2025-01-01 22:05:00+00:00 | 2026-05-01 00:00:00+00:00 | 91de8cc576518d6831d64d5a38d7a964c89e9b7490be85252ed08f70cd6ec77e | NO |
| M15 | 31748 | 2025-01-01 22:15:00+00:00 | 2026-05-01 00:00:00+00:00 | de928f1bbecbaef9bf6df5c58f85735a268b2cc1dc6546ffe70d7f28f3460952 | NO |
| H1 | 7939 | 2025-01-01 23:00:00+00:00 | 2026-05-01 00:00:00+00:00 | cdaf31a549102aa0e4e343a0e6b069144cee8b280b2c44c99bfee0c408fc523a | NO |

## 6. No-Leakage Guards

- `DEFAULT_DATA_DIRS` contains only `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared`.
- `sealed_holdout_2025_2026` is not in default loader config.
- `test_eurusd_holdout_seal.py` verifies holdout exclusion and partition boundaries.
- `test_integration_real_project.py` verifies train M5 no-leakage when local data exists.

## 7. Loader Contract

- required files: `EURUSD_M1.csv`, `EURUSD_M5.csv`, `EURUSD_M15.csv`, `EURUSD_H1.csv`
- required columns: `open`, `high`, `low`, `close`, `volume`
- timestamp: timezone-explicit UTC CSV index, parsed by loader and converted to `America/New_York`
- no forward fill, no interpolation, no fabricated empty bars.

## 8. News Scope

- news_enabled_for_core: `False`
- `forex_factory_cache.csv`: missing
- `news_eurusd_v2_utc.csv`: missing
- `canonical_anchor_events.csv`: provenance unverified; not used in this data foundation phase
- news rebuild remains a separate phase.

## 9. Tests

- research_lab import: PASS.
- strategy registry: PASS, 63 strategies.
- engine import: PASS.
- F06 pipeline tests: PASS, 119/119.
- Data foundation targeted tests: PASS, 15 run / 0 failures / 6 skipped optional missing data.
- Broader research_lab suite remains red and not hidden: 171 run / 15 failures / 9 errors / 12 skipped.

## 10. Safety Verification

- raw_data_modified: NO
- data_deleted: NO
- data_synthesized: NO
- data_downloaded: NO
- scraping_run: NO
- backtest_run: NO
- strategy_run: NO
- f06_real_run: NO
- validation_process_run: NO
- holdout_research_process_run: NO
- 2025_2026_used_for_strategy: NO

## 11. Remaining Blockers

- Full lab is not authorized until Claude audits this data foundation.
- Broader research_lab tests remain red outside this data-foundation scope.
- News provenance/rebuild remains separate and fail-closed.

## 12. Copy-Paste Summary for ChatGPT

EURUSD raw 2015-01 through 2026-04 coverage is confirmed with 136 monthly parquet files, no missing months, homogeneous UTC schema, and no metadata blockers. Train 2015-2024 prepared OHLCV is certified with exact hashes/rowcounts, no 2025/2026 rows, valid OHLC, no duplicate timestamps, and loader consumability. Sealed holdout 2025-2026 was built locally from 16 monthly raw files, marked `SEALED_NOT_FOR_RESEARCH_SELECTION`, gitignored, and not included in `DEFAULT_DATA_DIRS`. No backtest, strategy, optimization, validation, F06 real, news rebuild, download, scraping, or performance metric was run. Next step is Claude EURUSD 2015-2026 Data Foundation Audit.
