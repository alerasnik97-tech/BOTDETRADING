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

Run a grid search:

```bash
python fx_multi_timeframe_backtester.py optimize --strict-data-quality --max-combinations 12
```

## Notes

- Large raw datasets and generated reports are intentionally ignored in version control.
- The current research direction favors robustness, conservative execution costs, and walk-forward style validation over aggressive in-sample tuning.
