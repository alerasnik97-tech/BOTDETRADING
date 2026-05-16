# EURUSD PREPARED TRAIN 2015-2024 AUDIT

## Status

EURUSD_TRAIN_PREPARED_CERTIFIED_FOR_PRELAB

## Summary

- prepared_dir: `05_MARKET_DATA_VAULT\eurusd_data\prepared_train_2015_2024\prepared`
- manifest: `06_GOVERNANCE_AND_COMPLIANCE\lab_readiness\EURUSD_PREPARED_OHLCV_MANIFEST.csv`
- error_timeframes: `[]`
- verification: hashes, rowcounts, no 2025/2026, monotonic index, no duplicate timestamps, OHLC validity, no NaN OHLC, non-negative volume, exact loader schema, explicit timezone, loader consumability.

## Results

| timeframe | rows | min_timestamp_utc | max_timestamp_utc | hash_match | rows_2025_2026 | duplicate_timestamps | ohlc_valid | no_nan_ohlc | volume_non_negative | loader_consumable |
|---|---:|---|---|---|---:|---:|---|---|---|---|
| M1 | 3634609 | 2015-01-01 22:01:00+00:00 | 2024-12-31 22:00:00+00:00 | True | 0 | 0 | True | True | True | True |
| M5 | 729382 | 2015-01-01 22:05:00+00:00 | 2024-12-31 22:00:00+00:00 | True | 0 | 0 | True | True | True | True |
| M15 | 243169 | 2015-01-01 22:15:00+00:00 | 2024-12-31 22:00:00+00:00 | True | 0 | 0 | True | True | True | True |
| H1 | 60800 | 2015-01-01 23:00:00+00:00 | 2024-12-31 22:00:00+00:00 | True | 0 | 0 | True | True | True | True |

## Decision

Prepared train is certified for pre-lab audit only. It is not a strategy authorization.
