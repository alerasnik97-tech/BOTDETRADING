# VEORB POST-RUN RECONCILIATION REPORT

## 1. Status
VEORB_REGENERATED_GATE_GREEN_READY_FOR_DOSSIER_AUDIT

## 2. Executive Summary
This report summarizes the first official, sealed, train-only (2015-01-01 to 2024-12-31) backtest execution of the `ve_orb_volatility_expansion` (VE-ORB) strategy using the official fail-closed runner `research_lab.runners.formal_train_runner`.

All three institutional cost profiles (base, conservative, stress) successfully ran and passed the strict reconciliation gates without any violations, resulting in `sealed: True`. The verified metrics confirm that the strategy has a marginally positive expectancy across all configurations.

Additionally, a temporal activity audit revealed that the strategy has logged zero trades since 2016 (active only during 2015), showing extreme regime obsolescence and sample density failure.

## 3. Authorization Chain
- **MR-01 Audit V1 Branch:** `audit/mr01-regenerated-dossier-v1-20260517`
- **MR-01 Audit V1 Commit:** `11ea9d1b68d022d8bfb76f7db75bf2214e43ee30`
- **VE-ORB Release Decision:** `MR01_CLOSED_REJECTED_VEORB_RELEASED_FOR_FORMAL_RUN`

## 4. Run Config
- **Run ID:** `VEORB_OFFICIAL_RUNNER_RUN_2015_2024_20260517_165407`
- **Output Dir:** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/ve_orb_volatility_expansion/VEORB_OFFICIAL_RUNNER_RUN_2015_2024_20260517_165407`
- **Strategy:** `ve_orb_volatility_expansion`
- **Date Range:** `2015-01-01` to `2024-12-31`
- **Profiles:** `base`, `conservative`, `stress`
- **Runner:** `research_lab.runners.formal_train_runner`

## 5. Data Scope
- **Train-Only (2015-2024):** YES. The backtest runs strictly on train data.
- **No Holdout / Sealed Holdout (2025-2026):** YES. No data from 2025 or 2026 was loaded or used.

## 6. Cost Profiles
- **Base Profile:** Normal mode, base cost profile ($7.0 roundturn commission/lot).
- **Conservative Profile:** Conservative mode, conservative cost profile ($7.0 commission + conservative slippage).
- **Stress Profile:** High precision mode, stress cost profile ($7.0 commission + stress slippage).
- **Monotonicity Check:** Verified. Expected slippage and execution costs worsen performance monotonically across the profiles (Base PF: 1.0620 -> Conservative PF: 1.0450 -> Stress PF: 1.0308).

## 7. Reconciliation Gate Result
- **Gate Passed:** YES (sealed: True, exit code: 0).
- **Violations:** None.
- **Profiles Reconciled:** Base, Conservative, and Stress are 100% reconciled.
- **Cost Profile Reconciliation:** All profiles are correctly mapped to their respective execution configurations.

## 8. Artifacts
- **Sealed Manifest:** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/ve_orb_volatility_expansion/VEORB_OFFICIAL_RUNNER_RUN_2015_2024_20260517_165407/manifests/RUN_MANIFEST.json`
- **Engine Configs:** 
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/ve_orb_volatility_expansion/VEORB_OFFICIAL_RUNNER_RUN_2015_2024_20260517_165407/configs/base_ENGINE_CONFIG.json`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/ve_orb_volatility_expansion/VEORB_OFFICIAL_RUNNER_RUN_2015_2024_20260517_165407/configs/conservative_ENGINE_CONFIG.json`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/ve_orb_volatility_expansion/VEORB_OFFICIAL_RUNNER_RUN_2015_2024_20260517_165407/configs/stress_ENGINE_CONFIG.json`
- **Summary JSONs:**
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/ve_orb_volatility_expansion/VEORB_OFFICIAL_RUNNER_RUN_2015_2024_20260517_165407/profile_reports/base/summary.json`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/ve_orb_volatility_expansion/VEORB_OFFICIAL_RUNNER_RUN_2015_2024_20260517_165407/profile_reports/conservative/summary.json`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/ve_orb_volatility_expansion/VEORB_OFFICIAL_RUNNER_RUN_2015_2024_20260517_165407/profile_reports/stress/summary.json`
- **Summary Table (Light):** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/ve_orb_volatility_expansion/VEORB_OFFICIAL_RUNNER_RUN_2015_2024_20260517_165407/tables/VEORB_COST_PROFILE_SUMMARY.csv`
- **Heavy Local Files (Git-Ignored / Not Committed):**
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/ve_orb_volatility_expansion/VEORB_OFFICIAL_RUNNER_RUN_2015_2024_20260517_165407/local_outputs_do_not_commit/base/trades.csv`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/ve_orb_volatility_expansion/VEORB_OFFICIAL_RUNNER_RUN_2015_2024_20260517_165407/local_outputs_do_not_commit/base/equity_curve.csv`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/ve_orb_volatility_expansion/VEORB_OFFICIAL_RUNNER_RUN_2015_2024_20260517_165407/local_outputs_do_not_commit/conservative/trades.csv`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/ve_orb_volatility_expansion/VEORB_OFFICIAL_RUNNER_RUN_2015_2024_20260517_165407/local_outputs_do_not_commit/conservative/equity_curve.csv`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/ve_orb_volatility_expansion/VEORB_OFFICIAL_RUNNER_RUN_2015_2024_20260517_165407/local_outputs_do_not_commit/stress/trades.csv`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/ve_orb_volatility_expansion/VEORB_OFFICIAL_RUNNER_RUN_2015_2024_20260517_165407/local_outputs_do_not_commit/stress/equity_curve.csv`

## 9. Metrics Snapshot
### Base Profile
- **Total Trades:** 15
- **Profit Factor:** 1.0620
- **Expectancy (R):** 0.0361 R
- **Total Return:** 0.1376%
- **Max Drawdown:** 1.5213%
- **Ending Equity:** Not available (standard ledger PnL USD: 243.15)
- **Win Rate:** 46.67%
- **Avg Trades/Month:** 0.1250

### Conservative Profile
- **Total Trades:** 15
- **Profit Factor:** 1.0450
- **Expectancy (R):** 0.0272 R
- **Total Return:** 0.0734%
- **Max Drawdown:** 1.5507%
- **Ending Equity:** Not available (standard ledger PnL USD: 161.43)
- **Win Rate:** 46.67%
- **Avg Trades/Month:** 0.1250

### Stress Profile
- **Total Trades:** 15
- **Profit Factor:** 1.0308
- **Expectancy (R):** 0.0198 R
- **Total Return:** 0.0199%
- **Max Drawdown:** 1.5757%
- **Ending Equity:** Not available (standard ledger PnL USD: 98.42)
- **Win Rate:** 46.67%
- **Avg Trades/Month:** 0.1250

## 10. Initial Strategy Classification
VEORB_PRELIMINARY_INTERESTING_NEEDS_AUDIT

The strategy has a marginally positive expectancy and PF > 1.0 across all profiles, but suffers from extreme regime obsolescence (logging zero trades in 2016-2024, active only during 2015) and suffers from an extremely low sample size (15 trades total).

## 11. Safety Verification
- real_backtest_run: YES_TRAIN_ONLY_AUTHORIZED
- strategy: VEORB_ONLY
- second_strategy: NO
- TP01_rerun: NO
- MR01_rerun: NO
- optimization_run: NO
- sweep_run: NO
- validation_run: NO
- holdout_used: NO
- 2025_2026_used: NO
- news_used: NO
- high_precision_used: NO
- data_modified: NO
- runner_modified: NO
- engine_modified: NO
- strategy_modified: NO
- force_push: NO
- git_add_dot_used: NO

## 12. Decision
- VE-ORB is fully ready for external dossier audit.
- VE-ORB is **NOT** approved as a live, demo, or champion strategy. The extremely low sample size and temporal inactivity require external quant audit.
- No validation, no holdout, no production/incubation.

## 13. Next Step
Proceed with external dossier audit of the VE-ORB run via `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_VEORB_REGENERATED_DOSSIER_V1.md`.
