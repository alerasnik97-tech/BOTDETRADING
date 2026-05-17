# FIRST BATCH PRE-REGISTRATION V1

This document registers the frozen quantitative hypotheses, parameter boundaries, and testing contracts for the first batch of strategy candidates. **NO code is currently written, and NO backtests are authorized.**

---

## 1. Selected Candidates Summary

1.  **`BO01`** (London Breakout Continuation) - Family: `LBC`
2.  **`MR02`** (London Breakout Failure) - Family: `LBF`
3.  **`MR03`** (NY Open Exhaustion Reversal) - Family: `NER`
4.  **`LS01`** (Prior Day High/Low Failed Auction) - Family: `DAF`
5.  **`LS02`** (Liquidity Alternative No-Manipulante) - Family: `LAN`

---

## 2. Strategy Pre-Registration Templates

### Candidate 1: `BO01` (London Breakout Continuation)
-   **Strategy ID:** `BO01`
-   **Family ID:** `LBC`
-   **Working Name:** `bo01_london_breakout_continuation`
-   **Hypothesis:** The London opening session volume breaks the Asian range consolidation and drives a persistent trend that continues through early European hours.
-   **What Would Invalidate This Hypothesis:** A high fakeout rate where breakouts close back inside the range within 3 bars, leading to a net loss under standard spreads.
-   **Session NY:** London (07:00 - 12:00 GMT).
-   **Entry Logic Sketch:** Enter long on the close of the first M5 bar that breaks above the Asian Range high (00:00 - 06:30 GMT) by $\ge 0.5$ ATR(14), provided the M5 close is above the EMA(20) and EMA(200) on M15.
-   **Exit Logic Sketch:** Hard stop loss at the Asian Range mid-point. Fixed take profit at $2.0 \times$ risk (2R target).
-   **Risk Management Initial:** $0.5\%$ risk per trade. Maximum 1 open position.
-   **Max Trades Per Day:** 1.
-   **Filters Allowed:** Time-of-day filter (strictly between 07:00 and 10:00 GMT), minimum Asian range width of 8 pips.
-   **Filters Forbidden:** News filters (strictly forbidden during train-only), high-precision ticks.
-   **Parameters Allowed Initial:** Asian range time bounds, ATR breakout multiplier ($0.3 - 1.0$), EMA trend period ($10 - 30$).
-   **Parameters Forbidden:** Optimization of entry minutes or dynamic take profit multipliers.
-   **Data Scope Train-Only:** 2015–2024.
-   **Validation Status:** **LOCKED** (Contaminated unsealing is blocked).
-   **Holdout Status:** **SEALED** (Completely closed).
-   **Expected Frequency:** $8-12$ trades per month.
-   **Minimum Survival Metrics:** Base Profit Factor $\ge 1.15$, Stress Profit Factor $\ge 1.00$, Expectancy $\ge 0.15$ R.
-   **Hard Reject Conditions:** PF $< 1.15$, $< 30$ total trades, zero trades logged since 2019.
-   **Audit Requirements:** Full contract test suite, sealed manifest verification.
-   **Correlation Expectation:** Low correlation with daily trend portfolios.
-   **Implementation Complexity:** Low.
-   **Owner Approval Needed Before Code:** **YES** (Mandatory).

---

### Candidate 2: `MR02` (London Breakout Failure)
-   **Strategy ID:** `MR02`
-   **Family ID:** `LBF`
-   **Working Name:** `mr02_london_breakout_failure`
-   **Hypothesis:** The early London breakout is an institutional liquidity sweep that traps breakout traders before reversing to range fair value.
-   **What Would Invalidate This Hypothesis:** Strong trend days where the breakout runs continuously without pullbacks.
-   **Session NY:** London (07:00 - 12:00 GMT).
-   **Entry Logic Sketch:** Enter contrarian short if price breaks above the Asian Range high by $< 0.5$ ATR(14) and a M5 candle closes back inside the Asian range with a bearish engulfing pattern.
-   **Exit Logic Sketch:** Stop loss 2 pips above the fakeout high. Take profit at the opposite Asian Range boundary or $1.5 \times$ risk.
-   **Risk Management Initial:** $0.5\%$ risk per trade. Maximum 1 open position.
-   **Max Trades Per Day:** 1.
-   **Filters Allowed:** Max Asian range width of 22 pips.
-   **Filters Forbidden:** Technical indicators containing lookahead features.
-   **Parameters Allowed Initial:** Breakout threshold ($0.1 - 0.5$ ATR), engulfing bar criteria.
-   **Parameters Forbidden:** Multi-variable price channel overrides.
-   **Data Scope Train-Only:** 2015–2024.
-   **Validation Status:** **LOCKED**.
-   **Holdout Status:** **SEALED**.
-   **Expected Frequency:** $10-14$ trades per month.
-   **Minimum Survival Metrics:** Base Profit Factor $\ge 1.15$, Stress Profit Factor $\ge 1.00$, Expectancy $\ge 0.15$ R.
-   **Hard Reject Conditions:** PF $< 1.15$, Max drawdown $> 15\%$.
-   **Audit Requirements:** Sealed runner manifest, standard contract tests.
-   **Correlation Expectation:** Low.
-   **Implementation Complexity:** Medium.
-   **Owner Approval Needed Before Code:** **YES** (Mandatory).

---

### Candidate 3: `MR03` (NY Open Exhaustion Reversal)
-   **Strategy ID:** `MR03`
-   **Family ID:** `NER`
-   **Working Name:** `mr03_ny_exhaustion_reversal`
-   **Hypothesis:** Intraday morning sweeps on NY open exhaust their volatility once retail stop-losses are fully consumed.
-   **What Would Invalidate This Hypothesis:** Strong, news-driven momentum trends that drift linearly for several hours.
-   **Session NY:** Midday NY (14:30 - 17:30 GMT / 09:30 - 12:30 EST).
-   **Entry Logic Sketch:** Enter short/long contrarian if price moves $> 2.0$ ATR(14) away from its 20-bar M15 VWAP between 15:00 and 16:30 GMT, and prints a M5 pinbar reversal.
-   **Exit Logic Sketch:** Stop loss 1 pip outside the exhaustion high/low. Take profit at a daily VWAP touch.
-   **Risk Management Initial:** $0.5\%$ risk per trade. Max 1 open position.
-   **Max Trades Per Day:** 2.
-   **Filters Allowed:** Daily Trend Filter (EMA(20) must be horizontal on H1).
-   **Filters Forbidden:** Dynamic optimization during execution.
-   **Parameters Allowed Initial:** VWAP deviation multiplier ($1.5 - 2.5$), pinbar nose ratio.
-   **Parameters Forbidden:** Dynamic trailing stops.
-   **Data Scope Train-Only:** 2015–2024.
-   **Validation Status:** **LOCKED**.
-   **Holdout Status:** **SEALED**.
-   **Expected Frequency:** $8-12$ trades per month.
-   **Minimum Survival Metrics:** Base Profit Factor $\ge 1.15$, Stress Profit Factor $\ge 1.00$, Expectancy $\ge 0.15$ R.
-   **Hard Reject Conditions:** PF $< 1.15$, $< 30$ trades.
-   **Audit Requirements:** Metric reconciliation and monotonicity verification.
-   **Correlation Expectation:** Low.
-   **Implementation Complexity:** Medium.
-   **Owner Approval Needed Before Code:** **YES** (Mandatory).

---

### Candidate 4: `LS01` (Prior Day High/Low Failed Auction)
-   **Strategy ID:** `LS01`
-   **Family ID:** `DAF`
-   **Working Name:** `ls01_prior_day_failed_auction`
-   **Hypothesis:** The prior day's high or low acts as a major liquidity magnet. Sweeping these levels without immediate continuation triggers powerful reversals back to range center.
-   **What Would Invalidate This Hypothesis:** Strong breakout trends where prior day levels are shattered with high volume continuation.
-   **Session NY:** London / NY (07:00 - 19:00 GMT).
-   **Entry Logic Sketch:** Enter short if price moves above the prior day's daily high by $< 3.0$ pips, then closes back below the level on the M15 timeframe.
-   **Exit Logic Sketch:** Stop loss 2 pips above the sweep swing high. Take profit at the M15 VWAP or $2.0 \times$ risk.
-   **Risk Management Initial:** $0.5\%$ risk per trade.
-   **Max Trades Per Day:** 1.
-   **Filters Allowed:** Dynamic daily volatility filter (ATR(14) on Daily must be in the bottom 80%).
-   **Filters Forbidden:** News sweeps.
-   **Parameters Allowed Initial:** Sweep threshold in pips ($1.0 - 5.0$ pips), M15 engulfing criteria.
-   **Parameters Forbidden:** Parameter optimizations.
-   **Data Scope Train-Only:** 2015–2024.
-   **Validation Status:** **LOCKED**.
-   **Holdout Status:** **SEALED**.
-   **Expected Frequency:** $5-8$ trades per month.
-   **Minimum Survival Metrics:** Base Profit Factor $\ge 1.15$, Stress Profit Factor $\ge 1.00$, Expectancy $\ge 0.15$ R.
-   **Hard Reject Conditions:** PF $< 1.15$, Max drawdown $> 15\%$.
-   **Audit Requirements:** Contract standard validation.
-   **Correlation Expectation:** Low.
-   **Implementation Complexity:** Low.
-   **Owner Approval Needed Before Code:** **YES** (Mandatory).

---

### Candidate 5: `LS02` (Liquidity Alternative No-Manipulante)
-   **Strategy ID:** `LS02`
-   **Family ID:** `LAN`
-   **Working Name:** `ls02_liquidity_alternative_no_manipulante`
-   **Hypothesis:** Multi-session liquidity sweeps (H4 peaks/troughs) offer highly stable, low-frequency reversals completely uncorrelated to the main trend portfolios.
-   **What Would Invalidate This Hypothesis:** Multi-day linear trends that bypass key levels without pulling back.
-   **Session NY:** London / NY.
-   **Entry Logic Sketch:** Map H4 swing high/low points (20-bar lookback). Enter contrarian short/long on M15 if price sweeps the level by $< 5.0$ pips and closes back inside, provided the trade does not overlap with standard trend schedules.
-   **Exit Logic Sketch:** Stop loss 3 pips outside the sweep high/low. Take profit at $2.5 \times$ risk.
-   **Risk Management Initial:** $0.5\%$ risk per trade.
-   **Max Trades Per Day:** 1.
-   **Filters Allowed:** Disjoint schedule filter.
-   **Filters Forbidden:** News parameters.
-   **Parameters Allowed Initial:** Lookback swing length ($10 - 30$ H4 bars).
-   **Parameters Forbidden:** Any parameter sweeping.
-   **Data Scope Train-Only:** 2015–2024.
-   **Validation Status:** **LOCKED**.
-   **Holdout Status:** **SEALED**.
-   **Expected Frequency:** $4-6$ trades per month.
-   **Minimum Survival Metrics:** Base Profit Factor $\ge 1.15$, Stress Profit Factor $\ge 1.00$, Expectancy $\ge 0.15$ R.
-   **Hard Reject Conditions:** PF $< 1.15$, $< 15$ trades.
-   **Audit Requirements:** Full contract test verification.
-   **Correlation Expectation:** Zero (Targeted).
-   **Implementation Complexity:** Medium.
-   **Owner Approval Needed Before Code:** **YES** (Mandatory).
