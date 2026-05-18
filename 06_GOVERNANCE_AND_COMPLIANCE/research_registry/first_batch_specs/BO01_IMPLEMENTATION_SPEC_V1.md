# BO01 IMPLEMENTATION SPEC V1

## 1. Purpose
This document provides the formal, mathematical, and programable technical specifications for the systematic strategy candidate `BO01`. It provides the exact design criteria necessary for future execution, verification, and testing.

## 2. Hypothesis
The London opening session volume breaks the Asian range consolidation and drives a persistent trend that continues through early European hours.

## 3. Market Logic
At the European open (07:00 GMT), a massive injection of liquidity enters the market. If this volume breaks the consolidated boundaries established overnight (Asian session), it signals a strong imbalance that is highly likely to continue in the breakout direction.

## 4. Invalidation Criteria
A high rate of stop hunts where breakouts immediately fail and close back inside the range, leading to net losses under standard transaction costs.

## 5. Session NY
The template heading is Session NY, but the strategy is tailored strictly for the London session: it monitors the Asian pre-market session (00:00 - 06:30 GMT) and triggers trades strictly during the early London session (07:00 - 10:00 GMT). No trades can be triggered after 10:00 GMT.

## 6. Data Requirements
*   Historical price data: EURUSD M5 bars.
*   Variables needed: Open, High, Low, Close, Volume.

## 7. Feature Definitions
*   `Asian_High`: Maximum High price recorded between 00:00:00 GMT and 06:30:00 GMT (inclusive).
*   `Asian_Low`: Minimum Low price recorded between 00:00:00 GMT and 06:30:00 GMT (inclusive).
*   `Asian_Range_Width`: `Asian_High` - `Asian_Low` (measured in pips).
*   `ATR_M5`: Standard Average True Range with a period of 14 on M5 bars.
*   `EMA_M5`: Exponential Moving Average with a period of 20 on M5 bars.
*   `EMA_M15`: Exponential Moving Average with a period of 200 on M15 bars.

## 8. Entry Rules
A buy signal is triggered on the close of the first M5 bar after 07:00 GMT that satisfies all of the following conditions:
1.  Current time is between 07:00:00 GMT and 10:00:00 GMT (inclusive).
2.  The M5 close is strictly above the `Asian_High` by at least $0.5 \times ATR_M5$.
3.  The M5 close is strictly above the `EMA_M5`.
4.  The current price on the M15 timeframe is strictly above the `EMA_M15`.
5.  The `Asian_Range_Width` is at least 8.0 pips.
6.  There is no active position and the daily trade count is zero.

*(Symmetric opposite rules apply for sell signals).*

## 9. Exit Rules
*   The position is closed immediately upon hitting the `Stop_Loss` or `Take_Profit` levels.
*   Any open position is closed at the end of the session at exactly 12:00:00 GMT.

## 10. Stop Logic
*   **Stop Loss (SL):** Positioned at the mid-point of the Asian Range: `SL_Price = Asian_Low + (Asian_Range_Width / 2.0)`.

## 11. Target Logic
*   **Take Profit (TP):** Fixed target calculated at $2.0 \times$ the initial risk: `TP_Price = Entry_Price + 2.0 * (Entry_Price - SL_Price)` for long trades.

## 12. Risk Management
*   Initial risk is strictly capped at $0.5\%$ of account equity per trade.
*   Maximum number of concurrent open positions is 1.

## 13. Max Trades Per Day
Exactly 1. No new positions can be opened on the same day if a position was already filled.

## 14. Allowed Parameters
*   `Asian_Start_Time` (default 00:00 GMT)
*   `Asian_End_Time` (default 06:30 GMT)
*   `ATR_Multiplier` (range $0.3 - 1.0$)
*   `EMA_Trend_Period` (range $10 - 30$)

## 15. Forbidden Parameters
*   Dynamic optimization of entry minutes.
*   Varying target ratios during execution.
*   Lookahead parameters that utilize future bar data.

## 16. Filters Allowed
*   Minimum Asian range width of 8.0 pips to avoid breakout whipsaws.

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
*   All M5 indicators must be calculated strictly using closed bars.

## 21. Unit Tests Required
*   `test_bo01_no_future_leakage`: Verifies that shifting future rows does not affect historical signals.
*   `test_bo01_weekend_gap`: Confirms that Friday close to Monday open does not trigger signals.

## 22. Contract Tests Required
*   `test_bo01_contract_border`: Assures strict parameter schemas and data types are passed.

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
The strategy will be eligible to request validation approval, subject to owner gate, if:
*   Base PF $\ge 1.30$, stress PF $\ge 1.10$, and expectancy $\ge 0.25$ R.

## 28. Known Risks
*   Whipsaws inside highly erratic opening hours.
*   Slippage during highly volatile opens.

## 29. Implementation Notes
*   Ensure that GMT timezone is strictly maintained without relying on local system time.

## 30. Owner Approval Required Before Code
**YES**. Implementation code cannot be written without explicit owner approval.
