# MR01 POST-RUN RECONCILIATION REPORT

## 1. Status
MR01_REGENERATED_GATE_GREEN_READY_FOR_DOSSIER_AUDIT

## 2. Executive Summary
This report summarizes the first official, sealed, train-only (2015-01-01 to 2024-12-31) backtest execution of the `mr01_anchor_elastic` (MR-01) strategy using the official fail-closed runner `research_lab.runners.formal_train_runner`. 

All three institutional cost profiles (base, conservative, stress) successfully ran and passed the strict reconciliation gates without any violations, resulting in `sealed: True`. The verified metrics confirm that the strategy has a negative expectancy across all configurations.

Additionally, a temporal activity audit revealed that the strategy has logged zero trades since 2016 (active only during 2015), showing extreme regime obsolescence and sample density failure.

## 3. Authorization Chain
- **TP-01 Audit V3 Branch:** `audit/tp01-regenerated-dossier-v3-20260517`
- **TP-01 Audit V3 Commit:** `8aa811832b43b16047496917ddabb670f2640e6f`
- **MR-01 Release Decision:** `TP01_CLOSED_REJECTED_MR01_RELEASED_FOR_FORMAL_RUN`

## 4. Run Config
- **Run ID:** `MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509`
- **Output Dir:** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509`
- **Strategy:** `mr01_anchor_elastic`
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
- **Monotonicity Check:** Verified. Expected slippage and execution costs worsen performance monotonically across the profiles (Base PF: 0.7122 -> Conservative PF: 0.6534 -> Stress PF: 0.6070).

## 7. Reconciliation Gate Result
- **Gate Passed:** YES (sealed: True, exit code: 0).
- **Violations:** None.
- **Profiles Reconciled:** Base, Conservative, and Stress are 100% reconciled.
- **Cost Profile Reconciliation:** All profiles are correctly mapped to their respective execution configurations.

## 8. Artifacts
- **Sealed Manifest:** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/manifests/RUN_MANIFEST.json`
- **Engine Configs:** 
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/configs/base_ENGINE_CONFIG.json`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/configs/conservative_ENGINE_CONFIG.json`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/configs/stress_ENGINE_CONFIG.json`
- **Summary JSONs:**
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/profile_reports/base/summary.json`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/profile_reports/conservative/summary.json`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/profile_reports/stress/summary.json`
- **Summary Table (Light):** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/tables/MR01_COST_PROFILE_SUMMARY.csv`
- **Heavy Local Files (Git-Ignored / Not Committed):**
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/local_outputs_do_not_commit/base/trades.csv`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/local_outputs_do_not_commit/base/equity_curve.csv`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/local_outputs_do_not_commit/conservative/trades.csv`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/local_outputs_do_not_commit/conservative/equity_curve.csv`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/local_outputs_do_not_commit/stress/trades.csv`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/local_outputs_do_not_commit/stress/equity_curve.csv`

## 9. Metrics Snapshot
### Base Profile
- **Total Trades:** 28
- **Profit Factor:** 0.7122
- **Expectancy (R):** -0.1470 R
- **Total Return:** -2.47%
- **Max Drawdown:** 3.56%
- **Win Rate:** 50.00%
- **Avg Trades/Month:** 0.2333

### Conservative Profile
- **Total Trades:** 28
- **Profit Factor:** 0.6534
- **Expectancy (R):** -0.1776 R
- **Total Return:** -2.87%
- **Max Drawdown:** 3.83%
- **Win Rate:** 50.00%
- **Avg Trades/Month:** 0.2333

### Stress Profile
- **Total Trades:** 28
- **Profit Factor:** 0.6070
- **Expectancy (R):** -0.2018 R
- **Total Return:** -3.18%
- **Max Drawdown:** 4.05%
- **Win Rate:** 50.00%
- **Avg Trades/Month:** 0.2333

## 10. Initial Strategy Classification
MR01_PRELIMINARY_REJECTED_LOW_EDGE

The strategy has a negative mathematical expectancy across all cost configurations, and shows an extreme regime obsolescence (logging zero trades in 2016-2024, active only during 2015).

## 11. Safety Verification
- real_backtest_run: YES_TRAIN_ONLY_AUTHORIZED
- strategy: MR01_ONLY
- second_strategy: NO
- TP01_rerun: NO
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
- MR-01 is fully ready for external dossier audit.
- MR-01 is **NOT** approved as a live, demo, or champion strategy. The verified metrics demonstrate a negative expectancy and severe temporal instability.
- No validation, no holdout, no production/incubation.

## 13. Next Step
Proceed with external dossier audit of the MR-01 run via `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_MR01_REGENERATED_DOSSIER_V1.md`.
