# BOTDETRADING

Research-focused FX backtesting workspace for multi-timeframe intraday strategies.

## What is included

- `fx_multi_timeframe_backtester.py`
  - event-driven FX backtester
  - M5 / M15 / H1 synchronization
  - Dukascopy cache pipeline
  - data validation
  - backtest reporting
  - parameter optimization
  - context-aware strategy diagnostics
  - fixed-risk sizing over initial capital or current equity
  - layered news protection and experimental strategy families
  - setup-level summaries for playbook research
- `candidate_block_lab.py`
  - optimize and validate a new candidate block by pair
- `bootstrap_free_dukascopy.ps1`
  - year-by-year, pair-by-pair free data bootstrap
- `DATA_PIPELINE_RECOMMENDATION.md`
  - data workflow guidance
- `RESEARCH_NOTES_2020.md`
  - current research findings on real 2020 data

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Current default risk model:

- `1%` risk per trade
- based on `initial_capital`

You can override it with:

- `--risk-pct`
- `--risk-budget-mode initial_capital|equity`

Populate free Dukascopy cache:

```bash
python fx_multi_timeframe_backtester.py cache-data --download-missing
```

Build prepared M5 / M15 / H1 files:

```bash
python fx_multi_timeframe_backtester.py prepare-data --download-missing
```

Validate data:

```bash
python fx_multi_timeframe_backtester.py validate-data --strict-data-quality
```

Run a backtest:

```bash
python fx_multi_timeframe_backtester.py run --strict-data-quality
```

Export or normalize a high-impact news calendar:

```bash
python fx_multi_timeframe_backtester.py fetch-news --news-source tradingeconomics --news-min-importance 3 --news-output data/news_events.csv
```

Run with news guard and volatility circuit breaker enabled:

```bash
python fx_multi_timeframe_backtester.py run --strict-data-quality --news-source csv --news-file data/news_events.csv --news-min-importance 3 --news-no-entry-pre-minutes 45 --news-no-entry-post-minutes 30 --news-flatten-minutes-before 15 --news-hard-no-entry-pre-minutes 90 --news-hard-no-entry-post-minutes 60 --news-hard-flatten-minutes-before 30 --shock-no-entry-atr-multiple 2.5 --shock-flatten-atr-multiple 3.0 --shock-cooldown-minutes 30
```

Run a grid search:

```bash
python fx_multi_timeframe_backtester.py optimize --strict-data-quality --max-combinations 12
```

Screen pairs independently:

```bash
python fx_multi_timeframe_backtester.py screen-pairs --strict-data-quality --strategy-family adaptive_session_reversion --optimization-profile consistency --max-combinations 12
```

Compare unprotected vs calendar-only vs layered news protection:

```bash
python compare_news_guard.py --summary-json reports_free_2020_2021_opt_usdjpy_winrate/20260408_090129_optimize/summary.json --pair USDJPY --start 2020-01-01 --end 2021-12-31 --source local --data-dir data_free_2020/prepared --news-file data/forex_factory_cache.csv --report-dir reports_usdjpy_news_guard_ab --strict-data-quality
```

Run the setup lab for the USDJPY playbook:

```bash
python playbook_setup_lab.py --pair USDJPY --design-start 2020-01-01 --design-end 2021-12-31 --design-data-dir data_free_2020/prepared --oos-start 2022-01-01 --oos-end 2025-12-31 --oos-data-dir data_usdjpy_2022_2025/prepared --source local --news-file data/forex_factory_cache.csv --optimization-profile consistency --max-combinations 24 --report-dir reports_playbook_setup_lab --strict-data-quality --disable-shock-guard
```

Run the context whitelist lab for the surviving `core_reversion` setup:

```bash
python core_reversion_context_lab.py --pair USDJPY --base-summary reports_playbook_setup_lab/core_reversion/design/summary.json --design-start 2020-01-01 --design-end 2021-12-31 --design-data-dir data_free_2020/prepared --oos-start 2022-01-01 --oos-end 2025-12-31 --oos-data-dir data_usdjpy_2022_2025/prepared --source local --news-file data/forex_factory_cache.csv --report-dir reports_core_reversion_context_lab --strict-data-quality --disable-shock-guard
```

The same context whitelist flags are also available in the main CLI:

- `--context-whitelist-weekdays`
- `--context-whitelist-times`
- `--context-whitelist-regimes`
- `--context-whitelist-extensions`
- `--context-whitelist-directions`

Stress test the surviving line under tougher execution assumptions:

```bash
python survivor_stress_test.py --pair USDJPY --base-summary reports_playbook_setup_lab/core_reversion/design/summary.json --design-start 2016-01-01 --design-end 2021-12-31 --design-data-dir data_usdjpy_2016_2021/prepared --oos-start 2022-01-01 --oos-end 2025-12-31 --oos-data-dir data_usdjpy_2022_2025/prepared --source local --news-file data/forex_factory_cache.csv --report-dir reports_survivor_stress_test --strict-data-quality
```

Screen candidate pairs by transplanting the same surviving line:

```bash
python survivor_pair_screen.py --pairs USDJPY EURUSD USDCAD USDCHF --base-summary reports_playbook_setup_lab/core_reversion/design/summary.json --design-start 2020-01-01 --design-end 2021-12-31 --design-data-dir data_free_2020/prepared --oos-start 2022-01-01 --oos-end 2025-12-31 --oos-data-dir data_candidates_2022_2025/prepared --source local --news-file data/forex_factory_cache.csv --report-dir reports_survivor_pair_screen --strict-data-quality
```

Optimize and validate a second candidate block:

```bash
python candidate_block_lab.py --strategy-family session_trend_reclaim --pairs USDJPY EURUSD --design-start 2020-01-01 --design-end 2021-12-31 --design-data-dir data_free_2020/prepared --oos-start 2022-01-01 --oos-end 2025-12-31 --oos-data-dir data_candidates_2022_2025/prepared --source local --news-file data/forex_factory_cache.csv --report-dir reports_candidate_block_lab_trend_reclaim --strict-data-quality --optimization-profile consistency --max-combinations 24
```

## Notes

- Large raw datasets and generated reports are intentionally ignored in version control.
- The current research direction favors robustness, conservative execution costs, and walk-forward style validation over aggressive in-sample tuning.
- The safest execution path is layered: economic calendar blackout, pre-news forced flatten, post-news cooldown, and a volatility shock circuit breaker for unscheduled events.
- Hard-news veto is enabled by default for event names such as NFP, CPI, FOMC, rate decisions, and other central-bank style releases, even if the provider labels them below the minimum impact threshold.
