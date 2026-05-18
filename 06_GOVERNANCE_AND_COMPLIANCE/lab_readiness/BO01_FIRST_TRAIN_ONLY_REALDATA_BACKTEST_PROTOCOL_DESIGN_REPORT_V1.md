# BO01 FIRST TRAIN-ONLY REAL-DATA BACKTEST PROTOCOL DESIGN REPORT V1

## 1. Status

**BO01_FIRST_TRAIN_ONLY_REALDATA_BACKTEST_PROTOCOL_DESIGN_READY_FOR_EXTERNAL_AUDIT**

---

## 2. Scope

This design report defines the scope of the first real-data train-only backtesting protocol design.
- **Components Included**: Pure methodological design specification for candidate strategy BO01 on EURUSD M5 train-only.
- **Action Bounded**: Absolute zero Python execution, no data loading, no database reads, no validation, and no holdout splits.
- **Rigor & Limits**: Banned any optimization sweeps, parameter searching, or live/demo/FTMO claims. 

---

## 3. Evidence Used

- **M2 Structural Phase**: 638 compliant structural signals evaluated, with zero exceptions and 0 fail-closed triggers.
- **Audited Synthetic Runner**: Resolves W-01 (full-index date guard), W-02 (non-dict fail-closed), W-03 (active position skipped evaluation candle counter), and W-04 (EURUSD lot commission scaling) under commit `5bdb4bed1f829eb7e8bfe65dc30a6e2f49657d89` with 25 passing synthetic tests.
- **Approved Policies**: `ENTRY_NEXT_CANDLE_OPEN` and `STOP_FIRST` are fully established as base execution parameters.

---

## 4. Protocol Summary

- **Strategy**: BO01 (London Breakout) only.
- **Timeframe**: EURUSD M5.
- **Datasets**: `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared/EURUSD_M5.csv`.
- **Phased Windows**:
  - **Phase A (Plumbing Smoke)**: `2015-01-05` to `2015-01-09`.
  - **Phase B (M2-Compatible)**: `2015-01-01` to `2015-03-31` (Phase B is deferred until Phase A passes external audit).
- **Execution Constraints**: Max 1 trade active, max 1 trade per calendar day (first valid signal).
- **Cost Profiles**: Bounded by three profiles (Base, Conservative, Stress).
- **Safety**: Absolute abort triggers if any 2025/2026 dates, validation/holdout paths, or code changes are encountered.

---

## 5. Decision

**Ready for external read-only audit of protocol design.**

---

## 6. Allowed Next Step

- **A) External read-only audit of the protocol design.**

---

## 7. Forbidden Next Steps

- NO loading of real market data or running backtest scripts.
- NO access to validation or holdout datasets.
- NO 2025/2026 index dates.
- NO sweeps or grid searches.
- NO live, demo, or FTMO deployment attempts.
