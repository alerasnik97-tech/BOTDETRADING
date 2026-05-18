# BO01 FIRST TRAIN-ONLY BACKTEST FRAMEWORK DESIGN V1

## 1. Purpose
This document presents the formal technical specification for a future, controlled backtesting framework for candidate strategy **`BO01`** (London Breakout). 
> [!IMPORTANT]
> **This is a DESIGN-ONLY specification.** No code is being written or executed, no data is being loaded, and no performance curves or profitability metrics are calculated. Access to validation data, holdout data, or future dates (2025/2026) is strictly prohibited. No optimization sweeps or parameter search algorithms are defined or permitted under this protocol.

---

## 2. Why BO01
The decision to prioritize candidate strategy `BO01` for the first backtesting design is based on the physical evidence gathered during the M2 Conservative Train-Only Structural Retry execution:
- **`BO01` (London Breakout)** demonstrated highly robust operational density:
  - **`638` structural signals** generated over the 3-month train slice.
  - **`41` distinct calendar days** with active signals.
  - Crossover session alignment perfectly verified (signals occurred strictly within hours `07:00`, `08:00`, `09:00`, and `10:00` UTC).
  - **`0` execution exceptions** and **`0` fail-closed events** during the full structural countdown.
- **`MR02` (VWAP Stretch Reversion)** has been placed in **strict observation status** due to extremely low signal density (only 5 valid signals across 3 months), making it unsuitable for a standalone backtesting framework at this stage.

> [!WARNING]
> The structural signal density of `BO01` does **NOT** demonstrate or imply any directional edge or profitability. It only validates the programmatic correctness of the strategy contract under the M2 runner.

---

## 3. Future Backtest Scope
- **Strategy Candidate**: `BO01` only.
- **Asset**: `EURUSD` only.
- **Timeframe**: `M5` (5-minute bars) as the base operational cadence.
- **Context Dependency**: `M15` (15-minute bars) dataset permitted strictly for pre-computing `ema_m15_200` to satisfy the strategy contract.
- **Partition**: Train-only. Access to validation or holdout data is strictly locked.
- **Calendar Boundaries**:
  - **Phase 1 (Plumbing Test)**: `2015-01-05 00:00:00 UTC` to `2015-01-09 23:59:59 UTC` (exactly 1 trading week to verify execution simulation logic).
  - **Phase 2 (Expanded Test)**: `2015-01-01 00:00:00 UTC` to `2015-03-31 23:59:59 UTC` (full 3-month structural baseline).
- **Prohibited Dates**: Zero rows or records from years `2025` or `2026` will be loaded or processed.

---

## 4. Execution Model
To prevent lookahead bias and ensure programmatic realism, the future backtest execution engine must adhere to the following rules:
1. **Candle-by-Candle Processing**: The simulation must step through the M5 dataframe row-by-row. Signals at index $t$ can only utilize data from indices $\le t$.
2. **Strict Causality**: Stop-loss (SL) and Take-profit (TP) exits must be simulated chronologically. High and Low bounds of subsequent candles must be checked sequentially.
3. **No Overlapping Trades**: A maximum of **one active position** is permitted at any given time. 
4. **Daily Trade Limit**: A maximum of **one trade per day** is allowed.
   - If multiple valid signals occur within the same session, the engine must execute **strictly on the first valid signal** and ignore all subsequent signals for that day.
   - If a position is carried over from a previous day, new signals must be completely ignored until the position is closed.
5. **Entry Execution**: Strictly hardcoded under the **`ENTRY_NEXT_CANDLE_OPEN`** execution policy.
   - A valid BO01 signal is evaluated strictly at the close of candle $t$.
   - If there is no open position and no trade has been executed yet on that calendar day, the entry is filled at the exact `Open` price of the next candle $t+1$.
   - This next-candle-open fill is the **only** authorized entry policy for the first train-only backtesting phase.
   - **Alternative entries (breakout prices, contract boundaries, or any intrabar time-division resolutions) are strictly prohibited.**
   - *Technical Justification*: The `ENTRY_NEXT_CANDLE_OPEN` model is causal, deterministic, and 100% reproducible. It avoids the need for sub-candle/tick-level resolution within M5 bars (preventing ambiguous intrabar TP/SL vs breakout-entry ordering) and provides the highest degree of reliability for the initial plumbing verification. Future breakout-price models, if desired, must be designed under a completely separate, independent governance protocol.
6. **Stop Loss (SL)**: Defined dynamically by the strategy signal contract at index $t$.
7. **Take Profit (TP)**: Defined by the static target risk-reward ratio (`target_rr`), with no dynamic calculation.
8. **Same-Bar Resolution Policy**: If the candle High reaches the TP and the candle Low reaches the SL within the exact same M5 bar:
   - **`STOP-FIRST` conservative policy must be applied.** The trade must be registered as a full loss (-1R). No optimistic resolution is allowed.
9. **No Discretionary Modifications**: Scale-in, scale-out, trailing stops (unless explicitly registered in strategy parameters), or manual trade management are strictly prohibited.

---

## 5. Cost Model
Every trade must be penalized to simulate real market frictions. The future backtest must evaluate performance across three distinct cost profiles:

| Cost Parameter | Base Profile | Conservative Profile | Stress Profile |
|---|---|---|---|
| **Spread** | 1.2 pips | 1.62 pips | 3.0 pips |
| **Slippage** | 0.2 pips | 0.5 pips | 1.0 pip |
| **Commission (Round-Turn)** | $7.00 per Standard Lot | $7.00 per Standard Lot | $7.00 per Standard Lot |
| **Max Spread Guard** | 3.0 pips (no execution if exceeded) | 3.0 pips (no execution if exceeded) | 4.0 pips (no execution if exceeded) |

> [!TIP]
> These cost profiles are designed to test strategy resilience under varying market conditions. They are static and must not be modified or "optimized" to artificially inflate results.

---

## 6. Risk Model
- **Fixed Risk Metric**: All performance metrics must be calculated in terms of **`R-multiples`** (where 1R is the initial stop-loss distance).
- **No Compounding**: Position sizing must remain constant throughout the backtest. No account-balance compounding is permitted.
- **No Martingale/Recovery Logic**: Trade sizing must not increase after losing trades.
- **Trade Cadence Bound**: Strictly capped at a maximum of 1 trade per day.

---

## 7. Metrics Policy
The future backtest execution is authorized to compute strictly objective, R-multiple based metrics:
- **Authorized Metrics**:
  - `trade_count` (total number of trades executed)
  - `gross_R` (total R-multiples gained before costs)
  - `net_R` (total R-multiples gained after cost profile subtraction)
  - `average_R` (mean R-multiple per trade)
  - `median_R` (median R-multiple per trade)
  - `winrate` (percentage of winning trades)
  - `profit_factor_R` (sum of winning R-multiples divided by sum of losing R-multiples)
  - `max_drawdown_R` (maximum peak-to-trough decline in cumulative R)
  - `max_losing_streak` / `max_winning_streak` (consecutive loss/win counts)
  - `expectancy_R` (expected net R per trade)
  - `exposure_days` (percentage of trading days with at least one active trade)
  - `trades_by_month` / `trades_by_hour` (temporal trade distributions)
  - `stop_count` / `target_count` / `BE_count` / `timeout_count` (exit type breakdowns)
  - `cost_impact_R` (total R lost to spreads, slippage, and commissions)

- **Prohibited Metrics and Actions**:
  - NO validation or holdout partition statistics.
  - NO 2025/2026 data metrics.
  - NO parameter ranking or optimization sweeps.
  - NO curve-fitting or selection of "best parameters".
  - NO "champion strategy" declarations.
  - NO claims of FTMO, prop-firm, or live-trading readiness.

---

## 8. Abort Conditions
The future backtest execution must be immediately terminated (fail-closed) if any of the following anomalies are detected:
1. **Data Leakage**: Attempting to read validation or holdout splits, or encountering any date from `2025` or `2026` in the loaded dataset, or utilizing a data source whose train-only status cannot be mathematically proven.
2. **Missing Dependency**: Absence of the verified `prepared_train_2015_2024` M5 or M15 CSV files under the `05_MARKET_DATA_VAULT`.
3. **Lineage Modification**: Attempting to modify strategy files (`BO01Strategy.py`) or the structural runner code to "improve" backtest results.
4. **Execution Ambiguity**: 
   - Encountering multiple signals on the same day without a clear first-signal selection rule.
   - Same-bar SL/TP without stop-first conservative resolution.
   - Any attempt to execute entries at breakout boundaries, at prices other than the next-candle-open ($t+1$), or using any alternative/intrabar entry policy model.
5. **Cost Omission**: Calculating net performance without applying the full fee/slippage/spread cost profiles.
6. **Lookahead Leak**: 
   - Attempting to use future close/high/low bounds of a candle to resolve entry/exits in past timestamps.
   - Attempting to execute an entry without a valid candle $t+1$ available, or when $t+1$ falls outside the permitted boundaries of the train dataset.
7. **Optimization Attempt**: Incorporating any loop or grid search intended to evaluate multiple parameter variations or selecting entry models based on retrospective performance results.
8. **Output Violation**: Attempting to stage or commit local backtesting output files (e.g. trade logs, equity curves) to GitHub.

---

## 9. Outputs Policy
All execution outputs must be kept strictly local and never committed to GitHub.
- **Allowed Local Output Path**:
  `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_first_train_only_backtest/<RUN_ID>/`
- **Permitted Local Files**:
  - `BO01_TRAIN_ONLY_BACKTEST_REPORT.md` (local execution report)
  - `output_manifest.json` (SHA256 manifest of local outputs)
  - `command_log.txt` (exact command executed)
  - `data_access_log.txt` (records paths and dates accessed)
  - `trades_structural.csv` (detailed list of simulated trades with exit types)
  - `equity_R.csv` (cumulative R curve)
  - `monthly_summary.csv` (R-multiples grouped by calendar month)
  - `diagnostic_counts.json` (detailed programmatic counts)

- **Only Staged/Committed Governance Files**:
  - `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_FIRST_TRAIN_ONLY_BACKTEST_EXECUTION_REPORT_V1.md`
  - `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_BO01_FIRST_TRAIN_ONLY_BACKTEST_EXECUTION_V1.md`

---

## 10. Success Criteria
A future backtesting run will be deemed successful **strictly** if it meets the following programmatic standards:
- All data parsed belongs 100% to the train-only slice (no validation/holdout/2025/2026 dates).
- Spread, slippage, and commission costs are fully deducted.
- The 1 trade/day constraint and first-signal policy are verified.
- Stop-first same-bar conservative resolution is verified.
- The local execution files are gitignored and never committed.
- The execution is fully reproducible by another independent agent.

> [!CAUTION]
> Meeting these success criteria only proves the technical integrity and realism of the backtesting plumbing. It does **NOT** validate directional edge, does **NOT** certify the strategy, and does **NOT** authorize live or demo deployment.
