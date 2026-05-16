# EURUSD RAW 2015-2026 COVERAGE AUDIT

## Status

RAW_EURUSD_2015_2026_COVERAGE_CONFIRMED

## Summary

- raw_path: `05_MARKET_DATA_VAULT\BOT_MARKET_DATA\tick\EURUSD`
- files_found: `3548`
- total_size_bytes: `9210754686`
- monthly_parquet_files: `136`
- expected_months_2015_01_to_2026_04: `136`
- missing_months: `[]`
- duplicate_expected_filenames: `[]`
- schema_consistent: `True`
- metadata_errors: `[]`
- suspicious_size_months: `[]`
- extensions: `{".csv": 3, ".json": 147, ".parquet": 3398}`

## Schema

```json
{
  "ask": "double",
  "ask_volume": "double",
  "bid": "double",
  "bid_volume": "double",
  "source": "large_string",
  "spread": "double",
  "spread_pips": "double",
  "symbol": "large_string",
  "timestamp_ny": "timestamp[us, tz=America/New_York]",
  "timestamp_utc": "timestamp[us, tz=UTC]"
}
```

## Sampled Monthly Metadata

| period | size_bytes | schema_ok | timestamp_utc_schema_ok | min_timestamp_sample | max_timestamp_sample | bid_ask_valid_by_metadata | symbol_eurusd_by_metadata | notes |
|---|---:|---|---|---|---|---|---|---|
| 2015-01 | 38646598 | YES | YES | 2015-01-01 22:00:01.413000+00:00 | 2015-01-30 21:59:54.471000+00:00 | YES | YES |  |
| 2016-01 | 39155451 | YES | YES | 2016-01-03 22:00:01.446000+00:00 | 2016-01-31 23:59:59.429000+00:00 | YES | YES |  |
| 2017-01 | 40402253 | YES | YES | 2017-01-01 22:00:20.786000+00:00 | 2017-01-31 23:59:53.808000+00:00 | YES | YES |  |
| 2018-01 | 44682471 | YES | YES | 2018-01-01 22:00:08.661000+00:00 | 2018-01-31 23:59:47.019000+00:00 | YES | YES |  |
| 2019-01 | 64797089 | YES | YES | 2019-01-01 22:02:37.254000+00:00 | 2019-01-31 23:59:58.297000+00:00 | YES | YES |  |
| 2020-01 | 30285980 | YES | YES | 2020-01-01 22:01:12.821000+00:00 | 2020-01-31 21:59:56.198000+00:00 | YES | YES |  |
| 2021-01 | 36260962 | YES | YES | 2021-01-03 22:00:00.040000+00:00 | 2021-01-31 23:59:53.519000+00:00 | YES | YES |  |
| 2022-01 | 25646833 | YES | YES | 2022-01-02 22:03:54.650000+00:00 | 2022-01-31 23:59:58.963000+00:00 | YES | YES |  |
| 2023-01 | 55493287 | YES | YES | 2023-01-01 22:04:01.067000+00:00 | 2023-01-31 23:59:58.334000+00:00 | YES | YES |  |
| 2024-01 | 38361696 | YES | YES | 2024-01-01 22:00:12.108000+00:00 | 2024-01-31 23:59:57.610000+00:00 | YES | YES |  |
| 2024-12 | 36392479 | YES | YES | 2024-12-01 22:00:48.121000+00:00 | 2024-12-31 21:59:58.249000+00:00 | YES | YES |  |
| 2025-01 | 40945266 | YES | YES | 2025-01-01 22:00:14.647000+00:00 | 2025-01-31 21:59:57.318000+00:00 | YES | YES |  |
| 2026-04 | 34159379 | YES | YES | 2026-04-01 00:00:00.203000+00:00 | 2026-04-30 23:59:59.486000+00:00 | YES | YES |  |

## Decision

Raw EURUSD monthly parquet coverage is suitable as source material for train and sealed holdout preparation if downstream builders preserve the train/holdout partition and do not use holdout for research selection.
