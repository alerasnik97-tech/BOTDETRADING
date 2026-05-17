# TP01 FORMAL TRAIN RUN REPORT V1

**Document Reference:** GOV-REP-TP01-V1-20260517  
**Status:** TP01_TRAIN_GATE_GREEN_READY_FOR_EXTERNAL_AUDIT  
**Date:** May 17, 2026  
**Auditor:** Institutional Quant Execution Committee  

---

## 1. Status
**`TP01_TRAIN_GATE_GREEN_READY_FOR_EXTERNAL_AUDIT`**  
The formal execution of strategy `tp01_london_ny_momentum_pullback` has completed with 100% green status. The runner was successfully executed under the strict control standard, passed the post-execution reconciliation gate, and the run was sealed.

---

## 2. Executive Summary
This report summarizes the results of the first official formal backtest of the optimized daytime strategy `tp01_london_ny_momentum_pullback`. The strategy was run under 3 institutional cost profiles (Base, Conservative, Stress) over the 2015-2024 train-only window. 

The results exhibit perfect monotonicity across cost profiles, showing expected economic degradation as transaction costs increase. The newly integrated activity sentinel successfully analyzed trade density, logging no degeneracy, zero temporal clustering, and confirming a robust sample size of 191 trades over the active period (2015–2018). The platform successfully resolved the timeframe discrepancy C10 and corrected reporting metadata cadences.

---

## 3. Authorization Chain
*   **Guardrails Diff Audit Branch:** `audit/guardrails-standardization-diff-v1-20260517`
*   **Guardrails Diff Audit Commit:** `a1643baa615ab7d9416212b1f83a0c4ed30ccf0e`
*   **Verdict:** Approved & Ready for Acceleration.

---

## 4. Run Config
*   **Run ID:** `tp01_london_ny_momentum_pullback_FORMAL`
*   **Output Directory:** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_FORMAL_TRAIN_RUN_2015_2024_20260517_202700`
*   **Strategy:** `tp01_london_ny_momentum_pullback`
*   **Date Range:** 2015-01-01 to 2024-12-31
*   **Profiles Run:** `base`, `conservative`, `stress`
*   **Runner Module:** `research_lab.runners.formal_train_runner`

---

## 5. Data Scope
*   **Train-Only Focus:** 2015–2024.
*   **Active Sub-Period:** 2015-01-02 to 2018-05-18 (based on prepared data availability).
*   **Validation:** Disabled.
*   **Holdout:** Disabled.
*   **2025/2026 Data:** Completely closed.

---

## 6. Pre-Run Tests
The following targeted unit tests were executed prior to the backtest run and passed with 100% green status:
1.  `test_engine_strategy_contract.py`: Causality and future poisoning invariance checked on TP-01 and VE-ORB.
2.  `test_engine_time_contract.py`: Timezone, EST/EDT offset bounds, and weekend gaps checked.
3.  `test_strategy_activity_gates.py`: Telemetry density classifications verified.
4.  `test_formal_train_runner_contract.py`: Runner contract schemas and paths checked.
5.  `test_formal_train_runner_execute_contract.py`: Warning flags (zero-trades, mismatch) verified.
6.  `test_metric_reconciliation.py`: Profile metric alignment verified.
7.  `test_cost_profiles.py`: Cost structures and monotonicity checked.
8.  `test_engine.py`: Core order fills, stop limits, and basic spreads verified.
9.  `test_engine_stop_entry.py`: Stop entry fill triggers checked.

**Result:** `OK (99 tests passed, 0 failed)`

---

## 7. Cost Profiles & Monotonicity Check
The simulation has been verified across three cost levels:
*   **Base:** Standard spreads + basic commission.
*   **Conservative:** Elevated spreads + commission padding.
*   **Stress:** High slippage + high spreads.

**Monotonicity Status:** **PASSED**  
Economic metrics degrade linearly as transaction costs increase, confirming execution safety:
*   Profit Factor: Base (0.6312) $\ge$ Conservative (0.5872) $\ge$ Stress (0.5695)
*   Expectancy (R): Base (-0.2839) $\ge$ Conservative (-0.3207) $\ge$ Stress (-0.3340)
*   Total Return (%): Base (-26.04%) $\ge$ Conservative (-28.51%) $\ge$ Stress (-29.35%)
*   Max Drawdown (%): Base (27.35%) $\le$ Conservative (28.94%) $\le$ Stress (29.77%)

---

## 8. Guardrails / Warnings
The automated platform guardrails reported the following warnings during the execution:
1.  **`WARN_DECLARED_TIMEFRAME_DIFFERS_FROM_EFFECTIVE_CADENCE`** (Classification: **BENIGN**)
    *   *Details:* The runner was declared with `"M1"`, but the prepared data cadence was dynamically inferred to be exactly `"M5"`. The platform dynamically resolved this discrepancy and successfully corrected the resulting metadata summaries, avoiding distortion.
2.  **`WARN_ZERO_TRADES`** (Classification: **PASSED**)
    *   *Details:* Passive check successfully passed. Total trades = 191.
3.  **`WARN_LOW_SAMPLE_SIZE`** (Classification: **PASSED**)
    *   *Details:* Passive check successfully passed. Trades $\ge 30$.
4.  **`WARN_SINGLE_ACTIVE_YEAR`** (Classification: **PASSED**)
    *   *Details:* Passive check successfully passed. Trades are distributed over 4 active years (2015-2018).
5.  **`WARN_EXTREME_TEMPORAL_CONCENTRATION`** (Classification: **PASSED**)
    *   *Details:* Passive check successfully passed. Peak monthly concentration is only 6.81%, indicating highly stable temporal distribution.

---

## 9. Reconciliation Gate Result
*   **Gate Status:** **PASSED**  
*   **Violations:** None.
*   **Sealed:** True. All metrics are mathematically reconciled and matching across base and cost-degraded files.

---

## 10. Artifacts Generated
All requested files were generated under the run directory:
*   `manifests/RUN_MANIFEST.json`
*   `configs/base_ENGINE_CONFIG.json`
*   `configs/conservative_ENGINE_CONFIG.json`
*   `configs/stress_ENGINE_CONFIG.json`
*   `profile_reports/*/summary.json`
*   `profile_reports/*/tables/monthly.csv`
*   `profile_reports/*/tables/yearly.csv`

**Heavy Outputs (Locally Stored & Ignored from Git):**
*   `local_outputs_do_not_commit/*/trades.csv`
*   `local_outputs_do_not_commit/*/equity_curve.csv`

---

## 11. Metrics Snapshot

| Metric | Base Profile | Conservative Profile | Stress Profile |
| :--- | :--- | :--- | :--- |
| **Total Trades** | 191 | 191 | 191 |
| **Profit Factor** | 0.6312 | 0.5872 | 0.5695 |
| **Expectancy (R)** | -0.2839 | -0.3207 | -0.3340 |
| **Total Return (%)** | -26.04% | -28.51% | -29.35% |
| **Max Drawdown (%)** | 27.35% | 28.94% | 29.77% |
| **Win Rate (%)** | 27.75% | 26.70% | 26.70% |
| **Avg Trades/Month** | 1.59 | 1.59 | 1.59 |
| **Positive Years** | 1 (2015) | 0 | 0 |
| **Negative Years** | 3 (2016-2018) | 4 | 4 |
| **Timeframe** | M5 | M5 | M5 |

---

## 12. Initial Strategy Classification
**`TP01_PRELIMINARY_REJECTED_LOW_EDGE`**  
*   *Rationale:* Although the platform executed the simulation perfectly, the strategy `tp01_london_ny_momentum_pullback` failed to show an edge during the train-only phase. With a profit factor of 0.63 and negative expectancy (-0.28 R) under the base profile, the strategy shows a consistent downward equity curve and high vulnerability to cost degradation. 
*   *Verdict:* This strategy is rejected for any forward staging, incubation, or further validation. It remains strictly archived in the research logs.

---

## 13. Safety Verification
*   **real_backtest_run:** YES_TRAIN_ONLY_AUTHORIZED
*   **strategy:** TP01_ONLY
*   **second_strategy:** NO
*   **VEORB_rerun:** NO
*   **Manipulante_rerun:** NO
*   **optimization_run:** NO
*   **sweep_run:** NO
*   **validation_run:** NO
*   **holdout_used:** NO
*   **2025_2026_used:** NO
*   **data_modified:** NO
*   **runner_modified:** NO
*   **engine_modified:** NO
*   **strategy_modified:** NO
*   **force_push:** NO
*   **git_add_dot_used:** NO

---

## 14. Decision
*   Strategy `tp01_london_ny_momentum_pullback` is officially **REJECTED** as an investable candidate due to negative expectancy in train data.
*   **NO** validation or holdout data was exposed.
*   The research branch is ready to be committed and submitted for final read-only external audit.

---
*End of Report (GOV-REP-TP01-V1-20260517)*
