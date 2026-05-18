# BO01 FIRST TRAIN-ONLY REAL-DATA BACKTEST PROTOCOL DESIGN V1

## 1. Purpose

This document designs a strict methodological framework for a future execution of the first backtest of candidate strategy BO01 (London Breakout) on real market data. 

**CRITICAL LIMITATIONS**:
- **Design-Only Phase**: This is a pure design specification. No actual backtest runs, no market data loading, and no code execution is authorized during this phase.
- **Train-Only Scope**: The future protocol targets strictly the historical training dataset split.
- **Strictly Bounded Partition Access**: Access to any validation or holdout data (specifically the years 2025 and 2026) is strictly prohibited.
- **Overfitting Prevention**: No parameter sweeps, walk-forward searches, or parameter tuning of any kind are permitted. 
- **Zero Profitability Claims**: This document does not claim trading edge or readiness for live, demo, or FTMO environments.

---

## 2. Current Evidence

This protocol is built upon preceding, audited stages of quant research pipeline readiness:
- **BO01 M2 Structural Verification**: Evaluated 638 structural signals over a 3-month window. Showed zero exceptions and perfect execution contract compliance.
- **BO01 Synthetic Backtest Runner**: Verified via an external read-only audit passing with a comprehensive suite of 25 synthetic tests covering:
  - Strict date boundaries and monotonic temporal indexing.
  - Fail-closed execution of list/string and other malformed strategy signal contracts.
  - Correct tracking of skipped evaluation candle bars while active positions are open.
  - Multi-profile friction commission models calculated strictly in-memory.
- **Excluded Strategies**: MR02 is strictly excluded from this execution protocol due to low signal count and remains deferred.

---

## 3. Strategy Scope

- **Candidate Strategy**: `BO01` only.
- **Asset**: `EURUSD` FX currency pair.
- **Base Candlestick Timeframe**: M5.
- **Context Timeframe**: M15 is allowed only as read-only context if strictly required by the internal BO01 strategy logic.
- **No Portfolio Sizing**: Assumes a single-pair, single-instrument execution model.

---

## 4. Data Scope

### Authorized Future Datasets:
- **Primary Data Path**: `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared/EURUSD_M5.csv`
- **Secondary Context Path (M15 context if needed)**: `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared/EURUSD_M15.csv`

### Chronological Execution Windows:
- **Phase A (Plumbing Smoke Backtest Window)**:
  - **Start Date**: `2015-01-05 00:00:00 UTC`
  - **End Date**: `2015-01-09 23:59:59 UTC`
  - **Objective**: Plumbing verification to confirm dataloader interfaces, monotonic indexing, chronological order fill causality, cost computation, and zero worktree drifts. No qualitative conclusions of edge will be drawn from this 5-day window.
- **Phase B (M2-Compatible Backtest Window)**:
  - **Start Date**: `2015-01-01 00:00:00 UTC`
  - **End Date**: `2015-03-31 23:59:59 UTC`
  - **Objective**: The first standard train-only backtesting window to record structural statistics, matching preceding M2 structural evaluation metrics. This phase is only allowed if Phase A successfully passes a separate external audit.

---

## 5. Data Proof Requirements

Any future loader process MUST programmatically verify and report the following structural properties of the loaded dataframe before starting backtest loops:
1. **File Existence & Location**: Confirmed to reside within the designated path inside `05_MARKET_DATA_VAULT/`.
2. **Train-Only Partition Verification**: Checks metadata path structures and verifies that `validation` or `holdout` values do not exist in dataset split columns.
3. **No Year 2025/2026 Leakage**: The loaded index must satisfy `not (df.index.year.isin([2025, 2026])).any()`.
4. **Monotonicity & Frequency**: Checks that `df.index.is_monotonic_increasing` is `True`, with zero duplicated timestamps, and valid 5-minute spacing for the active segments.
5. **Data Completeness**: Checks that no NaN values exist in `open`, `high`, `low`, or `close` columns.

---

## 6. Execution Model

The future backtest execution must run exclusively on the audited runner `BO01_BACKTEST_RUNNER_SYNTHETIC_V1`. The execution logic enforces the following structural rules:
1. **Entry Policy**: `ENTRY_NEXT_CANDLE_OPEN`. A strategy signal validated at the close of candle $t$ triggers entry *only* at the Open price of candle $t+1$. Intrabar, contract boundary, or breakout price entries are strictly banned.
2. **Exit Resolution Chronology**: Evaluated row-by-row starting from entry candle index $t+1$.
3. **Same-Bar Resolution Policy**: `STOP_FIRST`. If both the stop-loss price and the target profit price are touched in the same candle bar, a loss of `-1.0 R` (plus friction commissions and spreads) is conservatively recorded.
4. **Position Constraints**: Max `1` active trade and max `1` trade per calendar day (the first valid signal of the day is executed; subsequent signals of that day are completely ignored).
5. **No Discretionary Modifications**: Bypasses any scale-in, scale-out, trailing stop updates, or qualitative manual filters.

---

## 7. Cost Profiles

To prevent overclaims and map execution frictions realistically, the future backtest must evaluate and report three distinct cost profiles:

1. **Base Profile**:
   - Spread: `1.2 pips`
   - Slippage: `0.2 pips`
   - Commission: `$7.0 USD` round-turn per standard lot (converted to R-multiples using pip sizing and stop-loss distance).
   - Max Spread Guard: `3.0 pips`

2. **Conservative Profile**:
   - Spread: `1.62 pips`
   - Slippage: `0.5 pips`
   - Commission: `$7.0 USD` round-turn per standard lot.
   - Max Spread Guard: `3.0 pips`

3. **Stress Profile**:
   - Spread: `3.0 pips`
   - Slippage: `1.0 pip`
   - Commission: `$7.0 USD` round-turn per standard lot.
   - Max Spread Guard: `4.0 pips`

---

## 8. Risk and Metrics Policy

### Risk Bounds:
- Cumulative results must be calculated strictly in terms of **R-multiples** (multiples of the initial stop distance).
- Compound risk sizing, monetary scaling, recovery models, or martingale sizing are strictly prohibited.

### Metrics Computed:
The future execution reports the following structural statistics:
- `trade_count`
- `gross_R`
- `net_R` (for all three cost profiles)
- `average_R` and `median_R`
- `winrate`
- `profit_factor_R`
- `max_drawdown_R`
- `max_losing_streak` and `max_winning_streak`
- `expectancy_R`
- `stop_count`, `target_count`, `timeout_count`, and `same_bar_stop_first_count`
- `skipped_signals_same_day`
- `skipped_signals_active_position` (candle evaluation slots bypassed while holding a trade)
- `invalid_signal_count` and `exception_count`

---

## 9. Output Policy

### Local In-Memory & Non-Committed Outputs:
Local outputs are allowed only in:
`03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_first_train_only_realdata_backtest/<RUN_ID>/`

**Allowed non-committed local files**:
- `BO01_TRAIN_ONLY_REALDATA_BACKTEST_REPORT.md`
- `output_manifest.json`
- `trades_structural.csv`
- `equity_R.csv`
- `monthly_summary.csv`

These outputs must be completely blocked by local gitignore configurations and must never be commited to the GitHub repository.

### Governance Documents (Committable):
The only committable artifacts resulting from the future execution are:
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_FIRST_TRAIN_ONLY_REALDATA_BACKTEST_EXECUTION_REPORT_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_BO01_FIRST_TRAIN_ONLY_REALDATA_BACKTEST_EXECUTION_V1.md`

---

## 10. Safety / Abort Conditions

The future execution process MUST immediately abort and fail closed under any of the following conditions:
1. **Branch Deviation**: Current branch is not `research/bo01-first-train-only-realdata-backtest-execution-*`.
2. **Worktree Drift**: Local changes detected during precheck.
3. **Multi-Agent Conflict**: Active Python processes or concurrent model actions detected.
4. **Data Contamination**: Any timestamp in the loaded dataset belongs to `2025` or `2026`, or contains partitions labeled `validation` or `holdout`.
5. **Runner Mutation**: Any source code change detected in `bo01_backtest_runner.py` compared to the audited `5bdb4bed1f829eb7e8bfe65dc30a6e2f49657d89` commit.
6. **Git Violations**: Attempting to run `git add .` or commit raw CSV, local output folders, or ZIP archives.
7. **Optimization Attempts**: Detection of sweep scripts, nested parameter loops, or automated candidate rankings.

---

## 11. Success Criteria

A future Phase A execution will be deemed successful *only* if:
1. All data proof requirements are programmatically satisfied.
2. The audited runner executes without modifying a single line of logic.
3. Local execution logs and trades CSV remain restricted to gitignored local output folders.
4. Zero validation or holdout data points are exposed.
5. All three cost profiles are fully evaluated and reported without qualitative model selection.
6. A clean, sober execution report is committed.

*Note: Success of the execution phase signifies plumbing validation; it does NOT confirm trading edge, profitability, or readiness for demo, live, or FTMO environments.*
