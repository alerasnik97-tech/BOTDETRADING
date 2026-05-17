# MR03 IMPLEMENTATION SPEC V1

## 1. Purpose
This document provides the formal, mathematical, and fully programable technical specifications for the systematic strategy candidate `MR03`. It provides the exact design criteria necessary for future execution, verification, and testing.

## 2. Hypothesis
Intraday morning sweeps on NY open exhaust their volatility once retail stop-losses are fully consumed.

## 3. Market Logic
At the US market open (13:30 GMT), a surge of orders drives rapid directional moves. If this surge pushes prices extremely far from the intraday volume-weighted average price (VWAP) without news continuation, the price exhausts and reverts to equilibrium as lunchtime volume declines.

## 4. Invalidation Criteria
Strong, news-driven momentum trends that drift linearly for several hours without pulling back.

## 5. Session NY
The strategy monitors the pre-market US session and triggers trades strictly during the early New York session (14:30 - 17:30 GMT). No trades can be triggered after 17:30 GMT.

## 6. Data Requirements
*   Historical price data: EURUSD M5 and M15 bars.
*   Variables needed: Open, High, Low, Close, Volume.

## 7. Feature Definitions
*   `VWAP_M15`: Volume-Weighted Average Price calculated on M15 bars from the daily open (00:00 GMT).
*   `ATR_M15`: Standard Average True Range with a period of 14 on M15 bars.
*   `EMA_H1`: Exponential Moving Average with a period of 20 on H1 bars.
*   `VWAP_Deviation`: Absolute difference between the current M15 Close and the current `VWAP_M15`: `abs(Close - VWAP_M15)`.

## 8. Entry Rules
A contrarian short signal is triggered on the close of the first M5 bar after 14:30 GMT that satisfies all of the following conditions:
1.  Current time is between 14:30:00 GMT and 17:30:00 GMT (inclusive).
2.  The M15 close is at least $2.0 \times ATR_M15$ above the `VWAP_M15`.
3.  The M5 bar exhibits a pinbar reversal pattern (Upper shadow is at least 3.0 times the body, and lower shadow is less than 0.5 times the body).
4.  The H1 EMA(20) is relatively flat (The angle of EMA(20) over the past 3 H1 bars is between -5 and +5 degrees).
5.  There is no active position and the daily trade count is less than 2.

*(Symmetric opposite rules apply for long signals when price is below VWAP_M15).*

## 9. Exit Rules
*   The position is closed immediately upon hitting the `Stop_Loss` or `Take_Profit` levels.
*   Any open position is closed at the end of the session at exactly 19:30:00 GMT.

## 10. Stop Logic
*   **Stop Loss (SL):** Positioned 1.0 pip above the high of the pinbar trigger bar: `SL_Price = Pinbar_High + 0.00010`.

## 11. Target Logic
*   **Take Profit (TP):** Target price is set exactly at the current value of the `VWAP_M15` at the moment of entry. The target does not trail dynamically.

## 12. Risk Management
*   Initial risk is strictly capped at $0.5\%$ of account equity per trade.
*   Maximum number of concurrent open positions is 1.

## 13. Max Trades Per Day
Exactly 2. No new positions can be opened on the same day if the daily trade count reaches 2.

## 14. Allowed Parameters
*   `VWAP_Dev_Multiplier` (range $1.5 - 2.5$)
*   `Pinbar_Shadow_Ratio` (range $2.5 - 4.0$)
*   `Flat_EMA_Threshold` (range $3.0 - 8.0$ degrees)

## 15. Forbidden Parameters
*   Dynamic trailing stop-loss values.
*   Dynamic parameter optimization during execution.
*   Lookahead parameters that utilize future bar data.

## 16. Filters Allowed
*   H1 EMA flat filter to avoid entering during vertical trends.

## 17. Filters Forbidden
*   News filters during the training phase.
*   High-precision ticks.

## 18. Cost Assumptions
Transaction costs are modeled across three profiles:
*   **Base:** 0.8 pips spread + $20 per million commission.
*   **Conservative:** 1.2 pips spread + $30 per million commission.
*   **Stress:** 1.6 pips spread + $40 per million commission.

## 19. Guardrails Required
*   Dynamic time-of-day limits.
*   Zero-trade activity warning flags.
*   Temporal trade density scanners.

## 20. Anti-Lookahead Requirements
*   No calculation can access `i+k` bar values (future prices).
*   All M5/M15 indicators must be calculated strictly using closed bars.

## 21. Unit Tests Required
*   `test_mr03_no_future_leakage`: Verifies that shifting future rows does not affect historical signals.
*   `test_mr03_weekend_gap`: Confirms that Friday close to Monday open does not trigger signals.

## 22. Contract Tests Required
*   `test_mr03_contract_border`: Assures strict parameter schemas and data types are passed.

## 23. Micro-Run Protocol Future
*   A 10-day dry-run to verify order entries, stop/limit fills, and logging accuracy.

## 24. Formal Train Run Protocol Future
*   A sealed, one-shot sweep over the 2015–2024 training dataset.

## 25. Rejection Rules
The strategy will be permanently rejected if:
*   Base Profit Factor (PF) is $< 1.15$.
*   Total trades in train dataset are $< 30$.
*   Maximum drawdown exceeds $15\%$ under $0.5\%$ risk.

## 26. Watchlist Rules
The strategy will be placed on the watchlist if:
*   Base Profit Factor is $\ge 1.15$ but stress PF is $< 1.00$.
*   Total trades are between 15 and 30.

## 27. Advance Rules
The strategy will be approved for validation if:
*   Base PF $\ge 1.30$, stress PF $\ge 1.10$, and expectancy $\ge 0.25$ R.

## 28. Known Risks
*   Catastrophic losses if entering against an institutional US trend campaign.
*   VWAP calculation deviations if daily rollover times are mismatched.

## 29. Implementation Notes
*   Ensure that GMT timezone is strictly maintained without relying on local system time.

## 30. Owner Approval Required Before Code
**YES**. Implementation code cannot be written without explicit owner approval.
