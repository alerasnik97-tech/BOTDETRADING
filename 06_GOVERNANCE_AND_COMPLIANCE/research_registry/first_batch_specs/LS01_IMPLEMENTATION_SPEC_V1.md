# LS01 IMPLEMENTATION SPEC V1

## 1. Purpose
This document provides the formal, mathematical, and fully programable technical specifications for the systematic strategy candidate `LS01`. It provides the exact design criteria necessary for future execution, verification, and testing.

## 2. Hypothesis
The prior day's high or low acts as a major liquidity magnet. Sweeping these levels without immediate continuation triggers powerful reversals back to range center.

## 3. Market Logic
Daily highs and lows accumulate high concentrations of resting sell and buy stop-loss orders. Institutional players temporarily sweep these levels to accumulate large order blocks (e.g., buying when shorts are forced to cover) before reversing price.

## 4. Invalidation Criteria
Strong, trend campaigns where daily high or low levels are shattered with significant expansion and high volume continuation.

## 5. Session NY
The strategy monitors daily high/low levels and triggers trades strictly during London and NY sessions (07:00 - 19:00 GMT). No trades can be triggered after 19:00 GMT.

## 6. Data Requirements
*   Historical price data: EURUSD M15 and Daily bars.
*   Variables needed: Open, High, Low, Close.

## 7. Feature Definitions
*   `Prior_Day_High`: Maximum High price recorded during the previous complete daily bar (00:00:00 to 23:59:59 GMT).
*   `Prior_Day_Low`: Minimum Low price recorded during the previous complete daily bar (00:00:00 to 23:59:59 GMT).
*   `ATR_Daily`: Standard Average True Range with a period of 14 on Daily bars.
*   `M15_VWAP`: Volume-Weighted Average Price calculated on M15 bars from the daily open (00:00 GMT).

## 8. Entry Rules
A contrarian short signal is triggered on the close of the first M15 bar after 07:00 GMT that satisfies all of the following conditions:
1.  Current time is between 07:00:00 GMT and 19:00:00 GMT (inclusive).
2.  The current (or preceding) M15 bar High must have exceeded the `Prior_Day_High` by less than 3.0 pips (0.00030).
3.  The current M15 bar Close is strictly below the `Prior_Day_High`.
4.  The `ATR_Daily` is in the bottom 80% of its past 100-day distribution (ensuring the day is not an extreme breakout session).
5.  There is no active position and the daily trade count is zero.

*(Symmetric opposite rules apply for long signals on breach of Prior_Day_Low).*

## 9. Exit Rules
*   The position is closed immediately upon hitting the `Stop_Loss` or `Take_Profit` levels.
*   Any open position is closed at the end of the session at exactly 21:00:00 GMT.

## 10. Stop Logic
*   **Stop Loss (SL):** Positioned 2.0 pips above the highest price reached during the daily high sweep: `SL_Price = Sweep_High_Swing + 0.00020`.

## 11. Target Logic
*   **Take Profit (TP):** Target price is set exactly at the current value of the `M15_VWAP` at the moment of entry. The target does not trail dynamically.

## 12. Risk Management
*   Initial risk is strictly capped at $0.5\%$ of account equity per trade.
*   Maximum number of concurrent open positions is 1.

## 13. Max Trades Per Day
Exactly 1. No new positions can be opened on the same day if a position was already filled.

## 14. Allowed Parameters
*   `Sweep_Bound_Pips` (range $1.0 - 5.0$ pips)
*   `ATR_Daily_Percentile` (range $70\% - 90\%$)
*   `M15_Close_Buffer` (range $0.0 - 1.0$ pips)

## 15. Forbidden Parameters
*   Dynamic trailing stop-loss values.
*   Multi-variable daily bracket calculations.
*   Lookahead parameters that utilize future bar data.

## 16. Filters Allowed
*   Daily ATR percentile filter to restrict sweeps to standard volatility days.

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
*   All Daily/M15 indicators must be calculated strictly using closed bars.

## 21. Unit Tests Required
*   `test_ls01_no_future_leakage`: Verifies that shifting future rows does not affect historical signals.
*   `test_ls01_weekend_gap`: Confirms that Friday close to Monday open does not trigger signals.

## 22. Contract Tests Required
*   `test_ls01_contract_border`: Assures strict parameter schemas and data types are passed.

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
*   Catastrophic losses if daily breakouts have massive follow-through on high-impact announcements.
*   Mismatched daily open/close alignments across different brokers.

## 29. Implementation Notes
*   Ensure that GMT timezone is strictly maintained without relying on local system time.

## 30. Owner Approval Required Before Code
**YES**. Implementation code cannot be written without explicit owner approval.
