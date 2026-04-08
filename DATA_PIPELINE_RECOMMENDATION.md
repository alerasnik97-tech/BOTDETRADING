# Recommended pipeline for a solid FX system

## Goal

Build a data workflow that is reproducible, inspectable, and strict enough to reject weak datasets before they poison research.

## Best free setup right now

1. Use Dukascopy as the free historical source.
2. Cache data locally by pair and by month.
3. Rebuild prepared M5/M15/H1 files only after cache population succeeds.
4. Run data validation before any backtest or optimization.

## Quality hierarchy

1. Tier A:
   premium tick or M1 FX data, frozen locally in Parquet with dataset versioning.
2. Tier B:
   Dukascopy M1 data, cached locally and validated before use.
3. Tier C:
   external CSV files only if they pass coverage, schedule, and OHLC sanity checks.

## Operational rules

1. Never optimize on unfrozen data.
2. Never mix vendors in the same period unless it is explicitly recorded.
3. Always resample from the finest available timeframe.
4. Always validate data quality before backtest and again before optimization.
5. Export reports with timestamps.
6. Treat high-impact news protection as part of the execution pipeline, not as an optional overlay.

## Recommended structure

```text
data/
  cache/
    dukascopy/
      EURUSD/
        EURUSD_2020_01_M5.csv.gz
  prepared/
    EURUSD_M5.csv
    EURUSD_M15.csv
    EURUSD_H1.csv
reports/
```

## Project commands

```bash
python fx_multi_timeframe_backtester.py cache-data --download-missing
python fx_multi_timeframe_backtester.py prepare-data --download-missing
python fx_multi_timeframe_backtester.py validate-data --strict-data-quality
python fx_multi_timeframe_backtester.py run --strict-data-quality
python fx_multi_timeframe_backtester.py optimize --strict-data-quality --max-combinations 12
python fx_multi_timeframe_backtester.py screen-pairs --strict-data-quality --max-combinations 12
python fx_multi_timeframe_backtester.py fetch-news --news-source tradingeconomics --news-min-importance 3 --news-output data/news_events.csv
```

## Recommended free bootstrap flow

1. Populate cache first:

```bash
python fx_multi_timeframe_backtester.py cache-data --pairs EURUSD GBPUSD USDJPY --start 2020-01-01 --end 2020-12-31 --source dukascopy --download-missing --data-dir data_free_2020
```

2. Build prepared M5/M15/H1 files:

```bash
python fx_multi_timeframe_backtester.py prepare-data --pairs EURUSD GBPUSD USDJPY --start 2020-01-01 --end 2020-12-31 --source dukascopy --download-missing --data-dir data_free_2020
```

3. Validate prepared files:

```bash
python fx_multi_timeframe_backtester.py validate-data --pairs EURUSD GBPUSD USDJPY --start 2020-01-01 --end 2020-12-31 --source local --data-dir data_free_2020/prepared --report-dir reports_free_2020_validate
```

4. Run research:

```bash
python fx_multi_timeframe_backtester.py optimize --strict-data-quality --pairs EURUSD GBPUSD USDJPY --start 2020-01-01 --end 2020-12-31 --source local --data-dir data_free_2020/prepared --report-dir reports_free_2020_opt --strategy-family adaptive_session_reversion --optimization-profile consistency --max-combinations 12 --news-source csv --news-file data/news_events.csv
```

5. For the long free bootstrap:

```powershell
.\bootstrap_free_dukascopy.ps1
```

That script now works pair by pair and year by year, so it is much easier to resume after timeouts or temporary Dukascopy failures.

## Practical principle

If the goal is real robustness, prefer:

- fewer parameter combinations
- better and cleaner data
- walk-forward style validation
- conservative execution costs
- calendar blackout plus pre-news flatten
- hard veto windows for NFP, CPI, FOMC, rate decisions, and similar event names
- a volatility shock circuit breaker for unscheduled events

That usually beats chasing the prettiest in-sample equity curve.
