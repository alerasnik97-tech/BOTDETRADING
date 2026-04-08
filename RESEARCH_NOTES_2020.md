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

## Extension to 2020-2021

After extending the free Dukascopy dataset to 2021 and rerunning the adaptive family on all 7 pairs, the global portfolio still failed:

- best 7-pair adaptive run:
  - return: `-20.15%`
  - max drawdown: `20.34%`
  - win rate: `52.69%`
  - profit factor: `0.34`
  - trades per month: `3.88`

This confirmed that a decent win rate alone is not enough and that the global portfolio is still structurally weak.

## Pair screening result

A new `screen-pairs` workflow was added to optimize each pair independently.

The first 2020-2021 screen with `adaptive_session_reversion` showed:

- `USDJPY` is the strongest candidate:
  - return: `+0.32%`
  - max drawdown: `2.65%`
  - profit factor: `1.05`
  - win rate: `62.07%`
- `USDCAD` is the second-best candidate:
  - return: `-2.25%`
  - max drawdown: `4.50%`
  - win rate: `59.38%`
- The rest of the pairs remain clearly weak.

## Narrowed subset test

Testing only `USDJPY + USDCAD` improved the portfolio a lot versus the 7-pair basket:

- return: `-1.46%`
- max drawdown: `5.40%`
- profit factor: `0.89`
- win rate: `57.41%`

This is still not strong enough, but it is much closer to tradable than the full basket.

## USDJPY-focused optimization

The most promising result so far came from narrowing to `USDJPY` only and optimizing for `winrate`:

- period: `2020-01-01` to `2021-12-31`
- family: `adaptive_session_reversion`
- best profile tested: `winrate`
- result:
  - return: `+1.17%`
  - max drawdown: `1.28%`
  - Sharpe: `0.31`
  - profit factor: `1.28`
  - win rate: `66.67%`
  - trades per month: `1.4`

Interpretation:

- This is the first configuration in the project that starts to look genuinely defensible.
- The weakness is frequency: the edge is cleaner, but the system is still too selective.
- The next research direction should focus on increasing trade count around this `USDJPY` regime without destroying the drawdown and profit factor profile.

## Execution safety upgrade

Because live losses around red-folder news are a real operational risk, the engine now includes a layered protection model:

- pair-aware economic calendar filter by currency
- no new entries before and after high-impact events
- forced flatten before the event time
- hard-event veto by keyword for releases like NFP, CPI, FOMC, and rate decisions, even when the vendor importance field is imperfect
- volatility shock circuit breaker for unscheduled events or bad calendar coverage

This changes the project direction in an important way: execution safety is now treated as part of the core system design, not as a discretionary operational step.

## USDJPY news-guard comparison

Using the strongest `USDJPY` configuration found so far, a direct comparison over `2020-01-01` to `2021-12-31` showed that news protection helps rather than hurts:

- unprotected:
  - return: `+1.17%`
  - max drawdown: `1.28%`
  - profit factor: `1.28`
  - win rate: `66.67%`
  - trades: `21`
- calendar only:
  - return: `+1.50%`
  - max drawdown: `0.98%`
  - profit factor: `1.46`
  - win rate: `70.59%`
  - trades: `17`
- layered guard:
  - return: `+1.30%`
  - max drawdown: `1.05%`
  - profit factor: `1.40`
  - win rate: `68.75%`
  - trades: `16`

Interpretation:

- This is exactly the kind of trade-off we want: slightly fewer trades, but better quality.
- For the current project state, `calendar_only` looks best on this sample.
- `layered_guard` remains attractive as the safer live-trading default because it adds protection against unscheduled shocks.

## USDJPY 2022-2025 re-optimization

The real out-of-sample test from `2022-01-01` to `2025-12-31` changed the picture materially.

Re-optimizing `adaptive_session_reversion` with news protection active improved the strategy versus the original frozen parameter set, but it still failed the robustness threshold:

- best `adaptive_session_reversion` consistency result:
  - return: `-3.13%`
  - max drawdown: `3.51%`
  - profit factor: `0.56`
  - win rate: `60.71%`
  - trades: `28`
- best `adaptive_session_reversion` winrate result:
  - return: `-5.12%`
  - max drawdown: `5.69%`
  - profit factor: `0.27`
  - win rate: `41.18%`
  - trades: `17`

Interpretation:

- News protection still helps relative to trading unprotected.
- Re-optimization reduces damage, but this family still does not recover a tradable edge on the new regime.
- The issue is no longer parameter tuning. The issue is the family itself.

## Core-session reversion prototype

A new experimental family, `core_session_reversion`, was added to reflect the out-of-sample findings:

- avoids the early session
- trades only in the core part of New York
- demands a stronger M5 reversal body
- keeps the same hard news protection layer

First 2022-2025 optimization result:

- return: `-0.58%`
- max drawdown: `0.70%`
- profit factor: `0.17`
- trades: `2`

Interpretation:

- This is safer, but still not investable.
- The prototype proves that stricter timing can control damage.
- The weakness is now the opposite of before: quality is not terrible, but the setup is too rare to form a robust system.

## USDJPY session playbook attempt

The next architectural step was to build a `usdjp_session_playbook` with three micro-setups:

- `core_reversion`
- `late_continuation`
- `compression_breakout`

The goal was correct: stop forcing one single logic into every context.
However, the first implementation failed immediately on the 2020-2021 design sample:

- return: `-33.42%`
- max drawdown: `33.64%`
- profit factor: `0.20`
- trades: `166`

Setup decomposition:

- `compression_breakout`: `126` trades, `-24.4k USD`, `PF 0.20`
- `core_reversion`: `11` trades, `-2.8k USD`, `PF 0.23`
- `late_continuation`: `29` trades, `-6.2k USD`, `PF 0.19`

Interpretation:

- The architectural direction is still right.
- The first selector implementation was too permissive and overtraded.
- This is not ready for out-of-sample validation. It should be rejected in its current form.
- The next iteration must start from stricter discovery rules, not from a broad combined playbook.

## Setup lab result

The next professional step was to isolate every setup and test it alone with a dedicated lab.

Results:

- `core_reversion`
  - design `2020-2021`: `-1.34%`, `1.50%` DD, `7` trades
  - OOS `2022-2025`: `+1.16%`, `0.56%` DD, `6` trades, `PF 3.05`
- `late_continuation`
  - design `2020-2021`: `-6.76%`, `6.82%` DD
  - OOS `2022-2025`: `-6.63%`, `7.68%` DD
- `compression_breakout`
  - design `2020-2021`: `-12.75%`, `13.01%` DD
  - OOS `2022-2025`: `-18.34%`, `18.53%` DD

Interpretation:

- `late_continuation` and `compression_breakout` should be rejected for now.
- `core_reversion` is the first setup that shows a potentially real safety profile in the new regime.
- The weakness is sample size: it is too sparse to trust yet.
- The next iteration should focus only on `core_reversion`, trying to increase sample count without damaging the OOS profile.

## Core-reversion context whitelist lab

The next iteration stayed disciplined: no looser entry logic, no new setup family.
Instead, a static context whitelist was added to the engine so we can test a
small set of explicit session filters directly inside the backtest.

Candidate results:

- `base`
  - design `2020-2021`: `-1.34%`, `1.50%` DD, `7` trades
  - OOS `2022-2025`: `+1.16%`, `0.56%` DD, `6` trades, `PF 3.05`
- `core_only`
  - design `2020-2021`: `-0.34%`, `0.73%` DD, `4` trades
  - OOS `2022-2025`: `+0.88%`, `0.56%` DD, `5` trades, `PF 2.56`
- `wed_thu_fri_core`
  - design `2020-2021`: `-0.34%`, `0.73%` DD, `4` trades
  - OOS `2022-2025`: `+0.96%`, `0.29%` DD, `3` trades, `PF inf`
- `core_short_only`
  - design `2020-2021`: `+0.00%`, `0.00%` DD, `1` trade
  - OOS `2022-2025`: `+0.61%`, `0.26%` DD, `2` trades, `PF inf`

Interpretation:

- The important signal is structural: context pruning helps more than forcing extra frequency.
- The best practical filter so far is `wed_thu_fri + core`.
- `core_short_only` looks even cleaner on paper, but the sample is too tiny to treat as real evidence.
- The surviving research line is now a narrow `USDJPY core_reversion` book, guarded by news veto and restricted to selected context buckets.
- The next professional priority is not to loosen rules. It is to expand the sample size on this surviving line with more historical years before promoting it further.

## Expanded design window: 2016-2021

The next step was exactly that expansion. USDJPY `2016-2019` was downloaded,
prepared, and merged with the existing `2020-2021` window so the surviving
core-reversion line could be tested on a much broader design sample.

Key results:

- `base`
  - design `2016-2021`: `-0.32%`, `2.34%` DD, `16` trades, `PF 0.88`
  - OOS `2022-2025`: `+1.16%`, `0.56%` DD, `PF 3.05`
- `core_only`
  - design `2016-2021`: `+0.39%`, `1.47%` DD, `12` trades, `PF 1.26`
  - OOS `2022-2025`: `+0.88%`, `0.56%` DD, `PF 2.56`
- `wed_thu_fri_core`
  - design `2016-2021`: `+0.47%`, `0.97%` DD, `8` trades, `PF 1.47`
  - OOS `2022-2025`: `+0.96%`, `0.29%` DD, `PF inf`
- `core_short_only`
  - design `2016-2021`: `+0.48%`, `0.50%` DD, `5` trades, `PF 1.95`
  - OOS `2022-2025`: `+0.61%`, `0.26%` DD, `PF inf`

Interpretation:

- This is the first serious sign that the surviving line may have a real core.
- The setup remains low-frequency, but now the broader design sample also turns positive under disciplined context pruning.
- `wed_thu_fri + core` is still the best balanced candidate because it stays positive in design and OOS while keeping drawdown very low.
- `core_short_only` is interesting, but still too sparse to promote over the more balanced filter.
- The research priority now shifts from "find any surviving setup" to "stress-test the surviving setup harder" without relaxing it.

## Survivor stress test

The surviving line was then stress-tested under tougher execution assumptions.
The tested candidate was:

- `USDJPY`
- setup family `usdjp_session_playbook`
- only `core_reversion` enabled
- context filter `wed_thu_fri + core`
- economic calendar active

Scenarios:

- `baseline_no_shock`
  - design `2016-2021`: `+0.47%`, `0.97%` DD, `PF 1.47`, `8` trades
  - OOS `2022-2025`: `+0.96%`, `0.29%` DD, `PF inf`, `3` trades
- `production_layered`
  - design `2016-2021`: `+0.00%`, `0.97%` DD, `PF 1.00`, `7` trades
  - OOS `2022-2025`: `+0.96%`, `0.29%` DD, `PF inf`, `3` trades
- `high_costs_layered`
  - design `2016-2021`: `-0.79%`, `1.57%` DD, `PF 0.50`, `7` trades
  - OOS `2022-2025`: `+0.78%`, `0.35%` DD, `PF inf`, `3` trades
- `extreme_costs_layered`
  - design `2016-2021`: `-1.26%`, `1.83%` DD, `PF 0.32`, `7` trades
  - OOS `2022-2025`: `+0.60%`, `0.41%` DD, `PF inf`, `3` trades

Interpretation:

- The line survives under the normal layered production model.
- The line is still fragile under aggressive cost assumptions.
- This means the setup is not ready to carry a portfolio alone, but it is good enough to keep as a candidate building block.
- The correct next step is not to loosen it for more trades. The correct next step is to look for the same quality profile in additional pairs or complementary setups so the portfolio reaches its trade-frequency target without diluting quality.

Portfolio design note:

- The current portfolio target is `15-25 trades per month` in total across all active pairs and accepted setups.
- That target should be reached by stacking multiple validated building blocks, not by forcing a single setup to overtrade.

## Risk model update

The execution model was aligned with the current portfolio mandate:

- default risk per trade: `1%`
- risk base: `initial capital`, not current equity

Interpretation:

- This keeps the per-trade dollar risk stable while we are still in research mode.
- It also makes comparisons between windows and candidate pairs cleaner.
- If compounding is desired later, the engine now supports switching back to equity-based sizing explicitly.

## Survivor pair transplant screen

The next step was to transplant the surviving line to candidate pairs without reoptimizing it.
Tested candidate:

- family `usdjp_session_playbook`
- only `core_reversion` enabled
- context filter `wed_thu_fri + core`
- layered news protection active
- risk `1%` of initial capital per trade

Pairs screened:

- `USDJPY`
  - design `2020-2021`: `-0.34%`, `0.73%` DD, `4` trades
  - OOS `2022-2025`: `+0.96%`, `0.29%` DD, `3` trades
- `EURUSD`
  - design `2020-2021`: `+0.19%`, `0.37%` DD, `PF 1.63`, `4` trades
  - OOS `2022-2025`: `-0.79%`, `1.69%` DD, `PF 0.63`, `10` trades
- `USDCAD`
  - design `2020-2021`: `-1.25%`, `1.34%` DD, `5` trades
  - OOS `2022-2025`: `-2.57%`, `2.57%` DD, `6` trades
- `USDCHF`
  - design `2020-2021`: `-1.11%`, `1.18%` DD, `3` trades
  - OOS `2022-2025`: `-1.05%`, `1.77%` DD, `5` trades

Interpretation:

- `USDJPY` remains the only pair that survives the transplant screen cleanly.
- `EURUSD` looked acceptable in design but failed out of sample.
- `USDCAD` and `USDCHF` should be rejected for this exact line.
- This is a strong signal that the path to `15-25 trades per month` will not come from cloning this same setup across many pairs.
- The portfolio will need multiple validated building blocks, not one building block copied everywhere.

## Second block candidate: session_trend_reclaim

To start building that second block, a new family `session_trend_reclaim` was added.
The idea was to capture a more frequent continuation pattern:

- H1 aligned trend and positive slope
- M15 pullback back toward the EMA
- M5 RSI reclaim and breakout trigger
- same layered news protection
- same fixed `1%` risk on initial capital

This family was optimized and tested on `USDJPY` and `EURUSD`.

Results:

- `USDJPY`
  - design `2020-2021`: `-1.10%`, `1.21%` DD, `11` trades, `1.22` trades/month
  - OOS `2022-2025`: `-5.22%`, `7.05%` DD, `29` trades, `1.21` trades/month
- `EURUSD`
  - design `2020-2021`: `-5.07%`, `5.41%` DD, `25` trades, `1.56` trades/month
  - OOS `2022-2025`: `-5.67%`, `7.22%` DD, `43` trades, `1.54` trades/month

Interpretation:

- This second block candidate should be rejected.
- It does not survive in either pair.
- It also does not add enough frequency to justify deeper tuning.
- That means the next valid step is not to tweak this continuation idea further. The next valid step is to test a different behavior class for block two.
