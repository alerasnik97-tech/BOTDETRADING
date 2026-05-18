# BO01 PHASE A EXECUTION PROMPT DESIGN REPORT V1

## 1. Status

**BO01_PHASE_A_EXECUTION_PROMPT_WARNING_PATCH_READY_FOR_EXTERNAL_AUDIT**

---

## 2. Scope

- **Activity Bound**: Specifying the detailed execution prompt for Phase A (Plumbing Smoke Backtest, 5-day window from 2015-01-05 to 2015-01-09) on EURUSD M5 train-only.
- **Rigor & Limits**: Markdown files only. No python run, no database loads, no validation, and no holdout splits.
- **Forbidden Actions**: Absolute ban on parameters searches, sweeps, or live/demo/FTMO claims.

---

## 3. Design Summary

- **Strategy**: BO01 (London Breakout) only.
- **Data Scope**: `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared/EURUSD_M5.csv`.
- **Target Window**: `2015-01-05` to `2015-01-09` (5 calendar days).
- **Data Proofs**: Validates existence, path correctness, train-only metadata, date ranges, monotonic ordering, cadences, and SHA256 hashes.
- **Execution Model**: Bounded by `ENTRY_NEXT_CANDLE_OPEN`, `STOP_FIRST`, and a maximum of 1 trade per day.
- **Frictions**: Evaluates three fixed cost profiles (Base, Conservative, Stress).
- **Outputs**: Local outputs (9 files including logs, count tables, and structural CSVs) are restricted to gitignored local folders. Commits are restricted strictly to the committed execution report.

---

## 4. Decision

**Ready for external read-only audit of the warning-patched execution prompt draft.**

---

## 5. Allowed Next Step

- **A) External read-only audit of Phase A execution prompt draft.**

---

## 6. Forbidden Next Steps

- NO loading of real market data or running backtest scripts.
- NO access to validation or holdout datasets.
- NO 2025/2026 index dates.
- NO sweeps or grid searches.
- NO live, demo, or FTMO deployment attempts.
