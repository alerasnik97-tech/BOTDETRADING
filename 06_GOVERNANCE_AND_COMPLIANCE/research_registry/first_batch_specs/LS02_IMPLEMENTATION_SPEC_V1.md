# LS02 IMPLEMENTATION SPEC V1

## 1. Purpose
This document provides the formal, mathematical, and programable technical specifications for the systematic strategy candidate `LS02`. It provides the exact design criteria necessary for future execution, verification, and testing.

## 2. Hypothesis
Multi-session liquidity sweeps (H4 peaks/troughs) offer highly stable, low-frequency reversals completely uncorrelated to the main trend portfolios.

## 3. Market Logic
H4 timeframe peaks and troughs represent highly robust and visible levels of overnight and multi-day consolidations. A micro-sweep ($< 5.0$ pips) of these levels on M15 traps resting breakouts and triggers strong contrarian reversals to the opposite boundaries.

## 4. Invalidation Criteria
Strong multi-day linear trends that bypass key levels without pulling back, leading to extended drawdown.

## 5. Session NY
The template heading is Session NY, and the strategy triggers trades strictly during London and NY sessions (07:00 - 19:00 GMT). No trades can be triggered after 19:00 GMT.

## 6. Data Requirements
*   Historical price data: EURUSD M15 and H4 bars.
*   Variables needed: Open, High, Low, Close.

## 7. Feature Definitions
*   `H4_High_Peak`: Maximum High price of the past 20 closed H4 bars.
*   `H4_Low_Trough`: Minimum Low price of the past 20 closed H4 bars.
*   `ATR_H4`: Standard Average True Range with a period of 14 on H4 bars.

## 8. Entry Rules
A contrarian short signal is triggered on the close of the first M15 bar after 07:00 GMT that satisfies all of the following conditions:
1.  Current time is between 07:00:00 GMT and 19:00:00 GMT (inclusive).
2.  The current (or preceding) M15 bar High must have exceeded the `H4_High_Peak` by less than 5.0 pips (0.00050).
3.  The current M15 bar Close is strictly below the `H4_High_Peak`.
4.  The current entry schedule does not overlap with the standard daily trend strategy trigger schedules.
5.  There is no active position and the daily trade count is zero.

*(Symmetric opposite rules apply for long signals on breach of H4_Low_Trough).*

## 9. Exit Rules
*   The position is closed immediately upon hitting the `Stop_Loss` or `Take_Profit` levels.
*   Any open position is closed at the end of the session at exactly 21:00:00 GMT.

## 10. Stop Logic
*   **Stop Loss (SL):** Positioned 3.0 pips above the highest price reached during the H4 peak sweep: `SL_Price = Sweep_High_Swing + 0.00030`.

## 11. Target Logic
*   **Take Profit (TP):** Fixed target calculated at $2.5 \times$ the initial risk: `TP_Price = Entry_Price - 2.5 * (SL_Price - Entry_Price)` for short trades.

## 12. Risk Management
*   Initial risk is strictly capped at $0.5\%$ of account equity per trade.
*   Maximum number of concurrent open positions is 1.

## 13. Max Trades Per Day
Exactly 1. No new positions can be opened on the same day if a position was already filled.

## 14. Allowed Parameters
*   `H4_Lookback_Length` (range $10 - 30$ H4 bars)
*   `H4_Sweep_Bound` (range $3.0 - 8.0$ pips)
*   `TP_Target_Ratio` (range $2.0 - 3.0$)

## 15. Forbidden Parameters
*   Dynamic trailing stop-loss values.
*   Parameter sweeping grids during execution.
*   Lookahead parameters that utilize future bar data.

## 16. Filters Allowed
*   Disjoint session filter to prevent overlapping signals with core daily models.

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
*   All H4/M15 indicators must be calculated strictly using closed bars.

## 21. Unit Tests Required
*   `test_ls02_no_future_leakage`: Verifies that shifting future rows does not affect historical signals.
*   `test_ls02_weekend_gap`: Confirms that Friday close to Monday open does not trigger signals.

## 22. Contract Tests Required
*   `test_ls02_contract_border`: Assures strict parameter schemas and data types are passed.

## 23. Micro-Run Protocol Future
*   A 10-day dry-run to verify order entries, stop/limit fills, and logging accuracy.

## 24. Formal Train Run Protocol Future
*   A sealed, one-shot sweep over the 2015–2024 training dataset.

## 25. Rejection Rules
The strategy will be permanently rejected if:
*   Base Profit Factor (PF) is $< 1.15$.
*   Total trades in train dataset are $< 15$.
*   Maximum drawdown exceeds $15\%$ under $0.5\%$ risk.

## 26. Watchlist Rules
The strategy will be placed on the watchlist if:
*   Base Profit Factor is $\ge 1.15$ but stress PF is $< 1.00$.
*   Total trades are between 15 and 30.

## 27. Advance Rules
The strategy will be eligible to request validation approval, subject to owner gate, if:
*   Base PF $\ge 1.30$, stress PF $\ge 1.10$, and expectancy $\ge 0.25$ R.

## 28. Known Risks
*   Severe slippage when trading H4 levels under low liquidity hours.
*   Prolonged holding times across session closes.

## 29. Implementation Notes
*   Ensure that GMT timezone is strictly maintained without relying on local system time.

## 30. Owner Approval Required Before Code
**YES**. Implementation code cannot be written without explicit owner approval.
