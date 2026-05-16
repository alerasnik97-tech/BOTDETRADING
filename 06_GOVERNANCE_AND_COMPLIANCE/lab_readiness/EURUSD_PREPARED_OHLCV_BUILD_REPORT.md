# EURUSD PREPARED OHLCV BUILD REPORT

## 1. Status

EURUSD_PREPARED_OHLCV_BUILT_READY_FOR_CLAUDE_AUDIT

Scope note: this approves only a Claude data-foundation audit. It does not approve backtest, strategy search, optimization, validation, holdout, F06 real execution, or live/paper/demo use.

## 2. Executive Summary

The hard blocker `EURUSD prepared OHLCV missing at configured DEFAULT_DATA_DIRS` was resolved locally from existing raw tick data only.

The builder read monthly raw EURUSD parquet files from `05_MARKET_DATA_VAULT/BOT_MARKET_DATA/tick/EURUSD`, included only 2015-01 through 2024-12 monthly files, excluded every 2025/2026 monthly file by filename, and wrote local train-only OHLCV CSVs under `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared`.

Generated prepared CSVs and local manifests are gitignored and must not be force-added. Governance metadata is tracked in `EURUSD_PREPARED_OHLCV_MANIFEST.csv`.

## 3. Loader Contract Audit

| loader_contract_field | expected_value | source_file | risk | decision |
|---|---|---|---|---|
| Data directories | `DEFAULT_DATA_DIRS` | `03_RESEARCH_LAB/research_lab/config.py` | previously pointed at empty prepared folders | updated path-only to train-only prepared folder |
| File names | `EURUSD_M1.csv`, `EURUSD_M5.csv`, `EURUSD_M15.csv`, `EURUSD_H1.csv` | `03_RESEARCH_LAB/research_lab/data_loader.py` | missing files blocked loader | generated exactly these names |
| Required columns | `open`, `high`, `low`, `close`, `volume` | `OHLCV_COLUMNS` | missing volume would break loader/features | volume is observed tick count per bar |
| Index | first CSV column, timezone-explicit timestamp | `parse_prepared_index` | naive timestamps rejected | output writes UTC offset |
| Timezone | parse as UTC, convert to `America/New_York` in loader | `NY_TZ`, `parse_prepared_index` | DST handling must be explicit | source timestamp is UTC; output UTC explicit |
| Duplicates | loader drops duplicate bar timestamps after concat; validator rejects remaining duplicates | `load_prepared_ohlcv`, `validate_price_frame` | duplicate bars could alter candles | builder collapses duplicate bars and verification found 0 duplicates |
| Gaps | no fill in loader; no fill in builder | `resample_ohlcv_to_timeframe` | fabricated empty bars would contaminate research | no forward fill, no interpolation, no empty bar fabrication |
| Bid/ask requirement | prepared core only requires OHLCV | `load_prepared_ohlcv` | hi-precision BID/ASK is separate | core uses mid OHLC from bid/ask; hi-precision remains separate |

## 4. Raw EURUSD Audit

Raw path: `05_MARKET_DATA_VAULT/BOT_MARKET_DATA/tick/EURUSD`

| field | value |
|---|---|
| total files | 3548 |
| total size bytes | 9210754686 |
| monthly parquet files | 136 |
| included monthly train files | 120 |
| excluded monthly 2025/2026 files | 16 |
| min source period by monthly filename | 2015-01 |
| max source period by monthly filename | 2026-04 |
| included period | 2015-01 through 2024-12 |
| raw rows read | 274002003 |
| raw rows kept | 274002003 |
| exact duplicate ticks removed | 0 |
| rows filtered by max timestamp | 0 |
| raw min timestamp UTC | 2015-01-01 22:00:01.413000+00:00 |
| raw max timestamp UTC | 2024-12-31 21:59:58.249000+00:00 |

Sampled schema from monthly parquet:

`timestamp_utc`, `bid`, `ask`, `bid_volume`, `ask_volume`, `timestamp_ny`, `spread`, `spread_pips`, `source`, `symbol`.

`timestamp_utc` is parquet `timestamp[us, tz=UTC]`. Bid/ask are present and sufficient to construct causal mid-price OHLC without inventing prices.

## 5. Train-Only Policy

- Max exclusive timestamp: `2025-01-01 00:00:00+00:00`.
- 2025 and 2026 monthly parquet files are excluded by filename.
- Any included row at or after max timestamp is filtered before resampling.
- Output verification found `0` rows at or after `2025-01-01T00:00:00Z`.
- No holdout process was opened.
- No validation process was opened.
- No strategy labels, future outcome features, or news features were created.

## 6. Builder Implementation

Implemented:

- `03_RESEARCH_LAB/research_lab/data_preparation/eurusd_prepared_ohlcv_builder.py`
- `04_INFRASTRUCTURE_ENGINEERING/data_builders/build_eurusd_prepared_ohlcv.py`

Builder behavior:

- Discovers only monthly files matching `EURUSD_ticks_YYYY_MM.parquet`.
- Excludes `YYYY >= 2025`.
- Reads only local parquet data.
- Requires `timestamp_utc`, `bid`, `ask`.
- Constructs `mid=(bid+ask)/2`.
- Uses observed tick count as `volume`.
- Resamples with `label="right", closed="right"` for loader compatibility.
- Writes CSVs atomically.
- Writes local manifests in the ignored prepared folder.
- Writes tracked governance manifest CSV under `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness`.

## 7. Dry Run Result

Command:

```powershell
python 04_INFRASTRUCTURE_ENGINEERING\data_builders\build_eurusd_prepared_ohlcv.py --dry-run
```

Result:

- status: `DRY_RUN_OK`
- files found: `3548`
- monthly files found: `136`
- included files: `120`
- excluded 2025/2026 monthly files: `16`
- target files: M1, M5, M15, H1 CSVs
- files written: `0`

## 8. Build Result

Command:

```powershell
python 04_INFRASTRUCTURE_ENGINEERING\data_builders\build_eurusd_prepared_ohlcv.py --max-date 2024-12-31 --output-dir 05_MARKET_DATA_VAULT\eurusd_data\prepared_train_2015_2024\prepared --write-governance-manifest
```

Result:

- status: `BUILT_OK`
- source files used: `120`
- source files excluded 2025/2026: `16`
- raw rows read: `274002003`
- exact duplicate ticks removed: `0`
- output dir: `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared`

## 9. Data Quality Checks

| timeframe | rows | min_timestamp_utc | max_timestamp_utc | sha256 | rows_2025_plus | duplicate_timestamps | invalid_ohlc |
|---|---:|---|---|---|---:|---:|---:|
| M1 | 3634609 | 2015-01-01 22:01:00+00:00 | 2024-12-31 22:00:00+00:00 | 93b74bb65b794b8ee20fd8fd1a46e55ccd1329cb1e3d5de59794b9f6b9078f23 | 0 | 0 | 0 |
| M5 | 729382 | 2015-01-01 22:05:00+00:00 | 2024-12-31 22:00:00+00:00 | 386ab589d14e52236581201b03aa7d8e6c5d2c9771bc59eea00d34abc1afa625 | 0 | 0 | 0 |
| M15 | 243169 | 2015-01-01 22:15:00+00:00 | 2024-12-31 22:00:00+00:00 | ebe7ade5b77850286435715755ad114e67efc797b0150574a43041e0bbdbc04e | 0 | 0 | 0 |
| H1 | 60800 | 2015-01-01 23:00:00+00:00 | 2024-12-31 22:00:00+00:00 | ef1bc7156e6fc4938c73e8ca277c38231605631c416d2d298a6bbe23de60f852 | 0 | 0 | 0 |

All output indexes are UTC, timezone-explicit, monotonic, duplicate-free, and compatible with `load_prepared_ohlcv`.

## 10. Config / Loader Wiring

Applied a path-only config update:

- `DEFAULT_DATA_DIRS` now points to `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared`.
- `DEFAULT_NEWS_ENABLED` is `False` for fail-closed news policy.
- No engine, signal, strategy, backtest, or trading logic was modified.

Loader check:

```text
loader M5 OK 729382 2015-01-01 17:05:00-05:00 2024-12-31 17:00:00-05:00 news_enabled False
```

## 11. News Module Policy

News remains disabled for EURUSD core.

| asset | status | core lab dependency | decision |
|---|---|---|---|
| `05_MARKET_DATA_VAULT/data/forex_factory_cache.csv` | missing | no, if news disabled | separate news rebuild phase |
| `05_MARKET_DATA_VAULT/data/news_eurusd_v2_utc.csv` | missing | no, if news disabled | separate news rebuild phase |
| `05_MARKET_DATA_VAULT/data/official_anchors/out/canonical_anchor_events.csv` | present, provenance unverified | no, if news disabled | do not use for lab until provenance audit |

Created next prompt: `NEXT_PROMPT_NEWS_DATA_PROVENANCE_AND_REBUILD.md`.

## 12. Tests

| command | result |
|---|---|
| `python -c "import research_lab"` | PASS |
| `python -c "from research_lab.strategies import STRATEGY_REGISTRY; print(len(STRATEGY_REGISTRY))"` | PASS, `63` |
| `python -c "import research_lab.engine"` | PASS |
| `python -c "import research_lab.light_runner"` | PASS |
| `python -m unittest 03_RESEARCH_LAB.research_lab.tests.test_eurusd_prepared_ohlcv_builder` | PASS, 3 tests |
| `python -m unittest 03_RESEARCH_LAB.research_lab.tests.test_integration_real_project` | PASS, 9 tests, 6 skipped missing optional data |
| F06 pipeline unittest discover | PASS, 119 tests |
| broader `research_lab/tests` discover | FAIL, 168 run, 15 failures, 9 errors, 12 skipped |

Remaining broader failures are not hidden. They are legacy/news/engine-level contract failures outside this data foundation change. Engine logic was not changed.

## 13. Safety Verification

- data_deleted: NO
- raw_data_modified: NO
- data_synthesized: NO
- data_downloaded: NO
- scraping_run: NO
- backtest_run: NO
- strategy_run: NO
- f06_real_run: NO
- validation_process_run: NO
- holdout_process_run: NO
- 2025_2026_analysis: NO
- 2025_2026_in_output: NO

## 14. Remaining Blockers

| blocker | classification | blocks EURUSD prepared OHLCV foundation | next action |
|---|---|---|---|
| broader research_lab suite red | LEGACY/ENGINE/NEWS TEST CONTRACT BLOCKER | no for data-foundation audit, yes for full lab authorization | separate owner-approved cleanup |
| missing news files | SOFT_BLOCKER_OPTIONAL_MODULE | no if news disabled | owner-supplied files or explicit rebuild authorization |
| canonical anchor provenance | GOVERNANCE_BLOCKER for news authority | no if news disabled | provenance audit before any news use |
| hi-precision BID/ASK package | SOFT_BLOCKER_OPTIONAL_MODULE | no for prepared OHLCV core | keep high-precision tests skipped missing data |

## 15. Copy-Paste Summary for ChatGPT

EURUSD prepared OHLCV foundation was built locally from raw tick parquet only. The builder used `timestamp_utc`, `bid`, and `ask`, constructed mid-price OHLC, used observed tick count as volume, excluded all 2025/2026 monthly files, wrote M1/M5/M15/H1 CSVs under the ignored local prepared train folder, and created governance manifest hashes/rowcounts. Output max timestamp is `2024-12-31 22:00:00+00:00`; rows at or after 2025-01-01 are zero for every timeframe. Loader can load M5 from `DEFAULT_DATA_DIRS`. News is disabled by default and missing news files remain a separate soft blocker. F06 tests pass 119/119. Builder and integration tests pass. Broader research_lab suite remains red with 168 run / 15 failures / 9 errors / 12 skipped, classified outside this data foundation change and not hidden.
