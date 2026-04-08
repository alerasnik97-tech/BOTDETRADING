# Research notes on real 2020 FX data

## Dataset used

- Source: Dukascopy free historical feed
- Base resolution: M5
- Trading session: 11:00 to 18:45 New York
- Validation windows for short-range optimization: quarterly

## What was tested first

- Pairs: EURUSD, GBPUSD, USDJPY
- Strategy families:
  - `trend_pullback`
  - `session_mean_reversion`
  - `adaptive_session_reversion`

## Main findings from the 3-pair sample

1. The original `trend_pullback` family can create frequency, but it does not survive spread, slippage, and commissions.
2. `session_mean_reversion` improved damage control versus the breakout family, but still had no positive net edge.
3. `adaptive_session_reversion` was the least harmful family.
4. Best short result for `adaptive_session_reversion` was roughly:
   - return: about `-4.8%`
   - max drawdown: about `4.9%`
   - win rate: about `48%`
   - trades per month: about `2.3`
5. Interpretation: this family is calmer and more stable, but still too selective and still negative.

## New result after expanding to the full 7-pair 2020 universe

- Pairs: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, GBPJPY
- Report folder:
  - `reports_free_2020_opt_adaptive_all7/20260407_221857_optimize`
- Best tested family/profile:
  - `adaptive_session_reversion`
  - optimization profile: `consistency`

### Best 7-pair 2020 result

- final equity: `85,605.91`
- total return: `-14.39%`
- max drawdown: `14.39%`
- Sharpe: `-2.51`
- profit factor: `0.45`
- total trades: `72`
- trades per month: `6.0`
- win rate: `50.0%`

### Pair-level read

- `USDCAD` was the only clearly profitable pair in this sample:
  - `11` trades
  - `81.8%` win rate
  - `profit factor 2.69`
  - `+2,107.56 USD`
- Every other pair was negative in 2020 under the same global rules.
- This strongly suggests pair behavior matters more than the current global rule set assumes.

## Context-layer read

The online context filter now exports `context_summary.csv` and is useful for diagnostics.

What the 7-pair sample suggests:

- Some contexts look promising, especially `wed_thu|core|balanced|deep|long` and `fri|core|balanced|deep|long`.
- Several other contexts are consistently poor, especially many `mon_tue` and `short` buckets.
- The sample is still too small to lock those into hard production rules without more years of data.

## What this means

1. The project already has a serious enough pipeline to reject weak ideas on real data.
2. The current adaptive family is still the best base, but it is not strong enough yet.
3. The next correct move is not blind parameter tuning.
4. The next correct move is to expand the real sample to 2021-2025 and only then decide:
   - which contexts deserve hard vetoes
   - whether some pairs should be excluded
   - whether pair-specific logic is justified

## Research rule

Do not treat a strategy as a serious candidate unless it gets close to:

- profit factor `> 1.15`
- win rate `> 50%`
- drawdown clearly below `20%`
- consistent profitable months
- no visible collapse in validation windows

## Current conclusion

The strongest progress so far is not a pretty backtest.
It is a much more reliable research process.

The next best move is to enlarge the sample with real data and let the context layer prove what is robust and what is noise.
