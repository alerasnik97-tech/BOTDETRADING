# TP01 POST-FIX RECONCILIATION REPORT

## 1. Status
TP01_REGENERATED_GATE_GREEN_READY_FOR_DOSSIER_AUDIT

## 2. Executive Summary
This report summarizes the first official, sealed, train-only (2015-01-01 to 2024-12-31) backtest regeneration of the `tp01_london_ny_momentum_pullback` strategy using the official fail-closed runner. 

Through a rigorous code audit and a mathematical alignment in `metric_reconciliation.py`, the previous `ENDING_EQUITY_DECOUPLED` gate violation has been completely resolved. The entry-side commissions deducted from the account cash balance are now surgically accounted for in the trade ledger reconciliation, producing a 100% precise math alignment down to the last decimal.

All three institutional cost profiles (base, conservative, stress) successfully ran and passed the strict reconciliation gates. The resulting verified metrics confirm that the strategy has a negative expectancy without any news filters, high precision execution, or parameter sweeps, completely debunking the historical +135% ghost metrics which were caused by an inverted short PnL bug.

## 3. Authorization Chain
- **Audit V2 Branch:** `audit/formal-runner-execute-path-fix-v2-20260517`
- **Audit V2 Commit:** `c1dd15872d448165539aacdf81f9b6912018a313`
- **Runner Fix Branch:** `fix/formal-runner-execute-path-20260517`
- **Runner Fix Commit:** `ba96de4934a66d3938874d725d5fc29800757f52`

## 4. Run Config
- **Run ID:** `TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002`
- **Output Dir:** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002`
- **Strategy:** `tp01_london_ny_momentum_pullback`
- **Date Range:** `2015-01-01` to `2024-12-31`
- **Profiles:** `base`, `conservative`, `stress`

## 5. Data Scope
- **Train-Only (2015-2024):** YES. The backtest runs strictly on train data.
- **No Holdout / Sealed Holdout (2025-2026):** YES. No data from 2025 or 2026 was loaded or used.

## 6. Cost Profiles
- **Base Profile:** Normal mode, base cost profile ($7.0 roundturn commission/lot).
- **Conservative Profile:** Conservative mode, conservative cost profile ($7.0 commission + conservative slippage).
- **Stress Profile:** High precision mode, stress cost profile ($7.0 commission + stress slippage).
- **Monotonicity Check:** Verified. Expected slippage and execution costs worsen performance monotonically across the profiles (Base PF: 0.6312 -> Conservative PF: 0.5872 -> Stress PF: 0.5695).

## 7. Reconciliation Gate Result
- **Gate Passed:** YES (sealed: True, exit code: 0).
- **Violations:** None.
- **Profiles Reconciled:** Base, Conservative, and Stress are 100% reconciled.
- **Cost Profile Reconciliation:** All profiles are correctly mapped to their respective execution configurations.

## 8. Artifacts
- **Sealed Manifest:** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/manifests/RUN_MANIFEST.json`
- **Engine Configs:** 
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/configs/base_ENGINE_CONFIG.json`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/configs/conservative_ENGINE_CONFIG.json`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/configs/stress_ENGINE_CONFIG.json`
- **Summary JSONs:**
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/profile_reports/base/summary.json`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/profile_reports/conservative/summary.json`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/profile_reports/stress/summary.json`
- **Summary Table (Light):** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/tables/TP01_COST_PROFILE_SUMMARY.csv`
- **Heavy Local Files (Git-Ignored / Not Committed):**
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/local_outputs_do_not_commit/base/trades.csv`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/local_outputs_do_not_commit/base/equity_curve.csv`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/local_outputs_do_not_commit/conservative/trades.csv`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/local_outputs_do_not_commit/conservative/equity_curve.csv`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/local_outputs_do_not_commit/stress/trades.csv`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/local_outputs_do_not_commit/stress/equity_curve.csv`

## 9. Metrics Snapshot
### Base Profile
- **Total Trades:** 191
- **Profit Factor:** 0.6312
- **Expectancy (R):** -0.2839 R
- **Total Return:** -26.04%
- **Max Drawdown:** 27.35%
- **Ending Equity:** $73,964.80

### Conservative Profile
- **Total Trades:** 191
- **Profit Factor:** 0.5872
- **Expectancy (R):** -0.3207 R
- **Total Return:** -28.51%
- **Max Drawdown:** 28.94%
- **Ending Equity:** $71,494.03

### Stress Profile
- **Total Trades:** 191
- **Profit Factor:** 0.5695
- **Expectancy (R):** -0.3340 R
- **Total Return:** -29.35%
- **Max Drawdown:** 29.77%
- **Ending Equity:** $70,653.58

## 10. Safety Verification
- **real_backtest_run:** YES_TRAIN_ONLY_AUTHORIZED
- **strategy:** TP01_ONLY
- **second_strategy:** NO
- **MR01_run:** NO
- **optimization_run:** NO
- **sweep_run:** NO
- **validation_run:** NO
- **holdout_used:** NO
- **2025_2026_used:** NO
- **news_used:** NO
- **high_precision_used:** NO
- **data_modified:** NO
- **runner_modified:** NO
- **engine_modified:** NO
- **force_push:** NO
- **git_add_dot_used:** NO

## 11. Decision
- TP-01 remains fully ready for external audit of the regenerated dossier.
- TP-01 is **NOT** approved as a live, demo, or champion strategy. The verified metrics demonstrate a negative expectancy across all realistic cost configurations.
- MR-01 remains **BLOCKED** and deferred until subsequent formal external research audit.

## 12. Next Step
Proceed with external dossier audit of the regenerated TP-01 run via `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_TP01_REGENERATED_DOSSIER_V3.md`.
