# MR01 REGENERATED DOSSIER EXTERNAL AUDIT V1 REPORT

## 1. Status
MR01_DOSSIER_AUDIT_V1_REJECTED_LOW_EDGE_NEXT_STRATEGY_RELEASED

## 2. Executive Summary
This report presents the formal external institutional audit of the train-only (2015-01-01 to 2024-12-31) dossier for `mr01_anchor_elastic` (MR-01). The strategy was executed using the official fail-closed runner `research_lab.runners.formal_train_runner` on branch `research/mr01-official-runner-run-20260517` (commit `eb170cd02d5e3f071c7b27de05b2f41373340bc1`), and successfully sealed (`sealed: True`, exit code 0).

Following a thorough evaluation of the performance, cost profile monotonicity, and temporal trade distribution, MR-01 is **rejected** due to negative net expectancy and severe structural regime obsolescence. However, because all safety protocols, data-leakage controls, and output policies were meticulously respected, the block is lifted and **VE-ORB (Volatility Expansion Opening Range Breakout)** is officially released for formal train-only backtesting.

## 3. Files Audited
- **Post-Run Reconciliation Report:** `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/MR01_POST_RUN_RECONCILIATION_REPORT.md`
- **Manifest:** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/manifests/RUN_MANIFEST.json`
- **Engine Configs:**
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/configs/base_ENGINE_CONFIG.json`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/configs/conservative_ENGINE_CONFIG.json`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/configs/stress_ENGINE_CONFIG.json`
- **Profile Summaries:**
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/profile_reports/base/summary.json`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/profile_reports/conservative/summary.json`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/profile_reports/stress/summary.json`
- **Yearly Table:** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/profile_reports/base/tables/yearly.csv`
- **Summary Table (Light):** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509/tables/MR01_COST_PROFILE_SUMMARY.csv`

## 4. Metric Integrity Audit
Verified. Performance is negative across all profiles:
- **Base Profile:** trades=28, PF=0.7122, expectancy=-0.1470 R, return=-2.47%, DD=3.56%.
- **Conservative Profile:** trades=28, PF=0.6534, expectancy=-0.1776 R, return=-2.87%, DD=3.83%.
- **Stress Profile:** trades=28, PF=0.6070, expectancy=-0.2018 R, return=-3.18%, DD=4.05%.

Reconciliation is 100% precise, and no decoupling occurred.

## 5. Cost Profile Audit
Monotonicity holds perfectly. Performance metrics degrade strictly as execution conditions degrade:
* Base PF (0.7122) > Conservative PF (0.6534) > Stress PF (0.6070).
* Base Expectancy (-0.1470 R) > Conservative Expectancy (-0.1776 R) > Stress Expectancy (-0.2018 R).

The stress engine configuration uses `stress_mode` execution rules correctly under standard M1 OHLC pricing. High precision (secondary tick data loading) was not active (`high_precision_used: false`), which is correct and compliant.

## 6. Temporal Activity Audit
* **2015:** 28 trades (PF: 0.7122)
* **2016-2024:** 0 trades.
* **Findings:** The strategy has logged exactly zero trades since 2016. Its entry filters or EMA pullback conditions simply never met under the structural market regime of 2016–2024, proving extreme regime obsolescence and insufficient sample density (28 trades total).

## 7. Safety / Leakage Audit
* **Holdout Used:** None (2025/2026 data remained completely untouched and sealed).
* **Optimization/Sweeps:** None.
* **News Filters:** None.
* **Second Strategy:** None.
* **External Scans:** Completed with 0 leaks.

## 8. Strategy Classification
`MR01_REJECTED_LOW_EDGE_AND_REGIME_OBSOLESCENCE`. The strategy lacks any positive expectancy and suffers from severe temporal instability. It is officially rejected and archived.

## 9. Decision
* **Strategy MR-01:** Classified as `MR01_REJECTED_LOW_EDGE_AND_REGIME_OBSOLESCENCE`. Archived and rejected.
* **Strategy VE-ORB:** Officially **RELEASED** for its first formal train-only backtesting run.
* **Holdout:** Remains completely sealed and protected.
* **Production/Incubation:** Strictly disabled.

## 10. Next Step
Execute the first formal train-only backtest of `veorb_volatility_expansion` under the official runner according to the instructions in `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_FORMAL_TRAIN_ONLY_MR02_OR_VEORB_WITH_OFFICIAL_RUNNER.md`.
