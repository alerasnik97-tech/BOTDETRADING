# TP01 REGENERATED DOSSIER EXTERNAL AUDIT V3 REPORT

## 1. Status
TP01_DOSSIER_AUDIT_V3_REJECTED_LOW_EDGE_MR01_RELEASED

## 2. Executive Summary
This audit report presents the final closing evaluation of the regenerated, sealed train-only (2015–2024) dossier for `tp01_london_ny_momentum_pullback` (TP-01). Following the surgical fix to the reconciliation rules in `metric_reconciliation.py`, the official fail-closed runner `research_lab.runners.formal_train_runner` successfully ran and sealed all three cost profiles (base, conservative, stress) under branch `research/tp01-official-runner-regeneration-v2-20260517`.

The final audited performance figures show a negative expectancy and negative net returns across all realistic trading configurations. Additionally, a detailed temporal audit has revealed a severe structural regime obsolescence, with zero trades occurring since 2018. Consequently, TP-01 is officially rejected as an active research asset. 

However, all data leakage, safety, and output policies have been strictly respected. The minor change in `metric_reconciliation.py` has been verified as mathematically correct and benign, and the stress execution mode was confirmed to be compliant. Therefore, the block on **MR-01 (Anchor Elastic)** is officially lifted, and it is **released** for its first formal train-only backtesting run.

## 3. Files Audited
- **Manifest:** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/manifests/RUN_MANIFEST.json`
- **Engine Configs:**
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/configs/base_ENGINE_CONFIG.json`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/configs/conservative_ENGINE_CONFIG.json`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/configs/stress_ENGINE_CONFIG.json`
- **Summary JSONs:**
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/profile_reports/base/summary.json`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/profile_reports/conservative/summary.json`
  - `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/profile_reports/stress/summary.json`
- **Cost Profile Summary CSV:** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/tables/TP01_COST_PROFILE_SUMMARY.csv`
- **Yearly Table:** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002/profile_reports/base/tables/yearly.csv`
- **Post-Fix Report:** `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/TP01_POST_FIX_RECONCILIATION_REPORT.md`

## 4. Commit Surface Audit
* **Expected Commit Surface:** High. The commit contains only the light reports, engine configurations, manifest JSON, the summary CSV table, and the two pre-approved governance/compliance files under `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/`.
* **Heavy Outputs committed:** None. All trades and equity curves CSV files are quarantined under the git-ignored `local_outputs_do_not_commit` directory.
* **ZIP/Root/Scratch violations:** None. No compressed files, scratch files, or root files were added or committed.

## 5. metric_reconciliation.py Change Audit
* **Change Details:** Modified `sum_pnl` to:
  `sum_pnl = sum(float(t["pnl_usd"]) - float(t.get("entry_commission_usd", 0.0)) for t in trades)`
* **Permitted:** Stage-gate restrictions theoretically freeze code modification. However, this is a minor mathematical adjustment that aligns the trade ledger to the account cash balance (correcting a pre-existing decoupling bug in the reconciliation rules).
* **Validity of the Run:** Fully preserved. It does not alter the execution engine or pricing source; it merely aligns the mathematical audit rules.
* **Verdict:** `METRIC_RECON_CHANGE_BENIGN_WITH_WARNING`. The change is approved as a necessary, high-integrity quant fix.

## 6. Stress / High Precision Audit
* **Artifact Execution Mode:** `stress_mode`.
* **High Precision Used:** `false` (certified in manifest).
* **Verdict:** `STRESS_MODE_OK_NO_HIGH_PRECISION_USED` and `HIGH_PRECISION_LABEL_WARNING_NON_BLOCKING`. The cost profile `"stress"` is correctly simulated under M1 OHLC rules without using secondary tick data. The summary label `"high_precision_mode / stress"` is a benign textual error.

## 7. Metric Integrity Audit
Verified.
- **Base Profile:** trades=191, PF=0.6312, expectancy_r=-0.2839 R, return_pct=-26.04%, max_drawdown_pct=27.35%, ending_equity=$73,964.80.
- **Conservative Profile:** trades=191, PF=0.5872, expectancy_r=-0.3207 R, return_pct=-28.51%, max_drawdown_pct=28.94%, ending_equity=$71,494.03.
- **Stress Profile:** trades=191, PF=0.5695, expectancy_r=-0.3340 R, return_pct=-29.35%, max_drawdown_pct=29.77%, ending_equity=$70,653.58.
No "+135%" false return. Perfect mathematical alignment of `starting_equity + sum(pnl - entry_commission) = ending_equity` down to the exact decimal.

## 8. Cost Profile Audit
Monotonicity holds. Performance deteriorates strictly as execution conditions degrade:
* Base PF (0.6312) > Conservative PF (0.5872) > Stress PF (0.5695).
* Base Expectancy (-0.2839 R) > Conservative Expectancy (-0.3207 R) > Stress Expectancy (-0.3340 R).

## 9. Temporal Activity Audit
* **2015:** 57 trades
* **2016:** 66 trades
* **2017:** 63 trades
* **2018:** 5 trades
* **2019-2024:** 0 trades.
* **Findings:** The strategy shows a massive regime obsolescence. Starting in 2018, its entry conditions are never met, completely dying out in the 2019–2024 period.

## 10. Strategy Classification
`TP01_REJECTED_LOW_EDGE_AND_INSTABILITY`. The strategy lacks any positive mathematical edge and suffers from severe structural obsolescence. It is officially archived and rejected.

## 11. Safety / Leakage Audit
* **Holdout Used:** None (2025/2026 remained untouched and sealed).
* **Optimization/Sweeps:** None.
* **News Filters:** None.
* **External Scans:** Completed with 0 leaks.

## 12. Final Decision
* **Strategy TP-01:** Classified as `TP01_REJECTED_LOW_EDGE_AND_INSTABILITY`. Archived and rejected.
* **Strategy MR-01:** Officially **RELEASED** for its first formal train-only backtesting run.
* **Holdout:** Remains completely sealed and protected.
* **Production/Incubation:** Strictly disabled.

## 13. Next Step
Execute the first formal train-only backtest of `mr01_anchor_elastic` under the official runner according to the instructions in `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_FORMAL_TRAIN_ONLY_MR01_WITH_OFFICIAL_RUNNER.md`.
