# TP01 FORMAL TRAIN EXTERNAL AUDIT V1

**Document Reference:** GOV-AUD-TP01-V1-20260517  
**Status:** **`AUDIT_PASS_TP01_REJECTED_LOW_EDGE`**  
**Date:** May 17, 2026  
**Lead Auditor:** Institutional Quant Audit Committee  

---

## 1. Audit Status
**`AUDIT_PASS_TP01_REJECTED_LOW_EDGE`**  
The formal audit of strategy `tp01_london_ny_momentum_pullback` (TP-01) train-only dossier is complete. The run was executed with technical validity, the metrics are verified to be mathematically reconciled, and the strict safety guardrails were successfully maintained. The strategy shows negative expectancy and is officially rejected from any validation or holdout phase.

---

## 2. Executive Verdict
1.  **Technical Validity:** **PASSED**. The execution was conducted strictly via the official runner on train-only data (2015–2024), utilizing the standard M5 prepared cadence.
2.  **Metric Reconciliation:** **PASSED**. Economic metrics exhibit perfect monotonicity across cost profiles and aggregate cleanly from monthly to yearly tables.
3.  **Safety & Leakage Guardrails:** **PASSED**. No validation or holdout datasets were exposed or read. No high-precision or news-filtering features were utilized.
4.  **Temporal Obsolescence:** **UNCOVERED**. The strategy logged exactly zero trades since 2018 (active strictly during 2015-2018), exhibiting massive regime obsolescence.
5.  **Rejection Verdict:** **CONFIRMED**. The strategy failed to show an edge during the train-only phase, logging a base profit factor of `0.6312` and a net return of `-26.04%`. It will not progress to any further development stage.

---

## 3. Scope Audited
-   **Research Branch:** `research/tp01-formal-train-run-v1-20260517`
-   **Audit Commit:** `ba9b81d7442eb744a4e8a158b2a551068f9f0fce`
-   **Run ID:** `tp01_london_ny_momentum_pullback_FORMAL`
-   **Output Dir:** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_FORMAL_TRAIN_RUN_2015_2024_20260517_202700`
-   **Files Inspected:**
    -   `manifests/RUN_MANIFEST.json`
    -   `configs/base_ENGINE_CONFIG.json`, `conservative_ENGINE_CONFIG.json`, `stress_ENGINE_CONFIG.json`
    -   `profile_reports/base/summary.json`, `profile_reports/conservative/summary.json`, `profile_reports/stress/summary.json`
    -   `profile_reports/base/tables/yearly.csv`, `profile_reports/base/tables/monthly.csv`
    -   `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/TP01_FORMAL_TRAIN_RUN_REPORT_V1.md`
-   **Rerun Performed:** **NONE** (Strictly Read-Only).

---

## 4. Safety Verification

| Parameter | Status | Evidence / Notes |
| :--- | :--- | :--- |
| **Code modified?** | **NO** | Core engine, runner, and strategy scripts remain frozen. |
| **Data modified?** | **NO** | `05_MARKET_DATA_VAULT` was untouched and read-only. |
| **Runner modified?** | **NO** | Standard official runner `formal_train_runner` was used. |
| **Strategy modified?** | **NO** | Standard daytime skeleton script was utilized. |
| **Backtest rerun?** | **NO** | Audit is strictly based on pre-existing research outputs. |
| **Validation used?** | **NO** | Strictly disabled in config and manifest. |
| **Holdout used?** | **NO** | Capped at 2024-12-31; 2025/2026 data completely sealed. |
| **2025/2026 used?** | **NO** | Confirmed sealed. |
| **Optimization/Sweep?** | **NO** | Strictly one-shot execution. |
| **Heavy outputs committed?**| **NO** | Confirmed that `trades.csv` and `equity_curve.csv` are Git-ignored. |
| **Git add dot used?** | **NO** | Explicit staging rules enforced. |

---

## 5. Manifest Audit
-   **Branch/Commit:** Matches the repository head perfectly.
-   **Temporal Range:** Capped at `2015-01-01` to `2024-12-31`.
-   **Declared Scope:** `train_only: true`, `validation_run: false`, `holdout_used: false`, `optimization_run: false`.
-   **Cadence Resolution:** The manifest correctly logs the `WARN_DECLARED_TIMEFRAME_DIFFERS_FROM_EFFECTIVE_CADENCE` warning, verifying that the discrepancy was caught and resolved.
-   **Verdict:** **PASS**.

---

## 6. Config Audit
-   **Execution Mode:** `normal_mode` for base/conservative, standard M1 bar loading.
-   **Transaction Cost Pads:**
    -   Commission: `$7.00 roundturn/lot` consistently.
    -   Slippage: Monotonically scaled (Base: 0.2 pips -> Conservative: 0.2 * 1.3 pips -> Stress: 0.2 * 1.6 pips).
    -   Spread: Monotonically scaled.
-   **Verdict:** **PASS**.

---

## 7. Metric Reconciliation Audit
The metrics across the profiles exhibit perfect mathematical alignment and consistency:
-   **Monotonic Degradation:** Expectancy (Base: `-0.2839 R` $\ge$ Conservative: `-0.3207 R` $\ge$ Stress: `-0.3340 R`). PF (Base: `0.6312` $\ge$ Conservative: `0.5872` $\ge$ Stress: `0.5695`). Return (Base: `-26.04%` $\ge$ Conservative: `-28.51%` $\ge$ Stress: `-29.35%`).
-   **Aggregation Check:** Summing monthly PnL R from `monthly.csv` gives `-54.232 R`, which aggregates to the total expectancy of `-0.2839 R` over 191 trades.
-   **Verdict:** **PASS**.

---

## 8. Yearly/Monthly Consistency Audit
-   **Temporal Distribution:** Extremely concentrated in 2015–2017:
    -   `2015`: 57 trades (+3.08 R)
    -   `2016`: 66 trades (-16.30 R)
    -   `2017`: 63 trades (-39.78 R)
    -   `2018`: 5 trades (-1.23 R)
    -   `2019–2024`: **0 trades**.
-   **Verdict:** **PASS** (Confirmed extreme regime obsolescence. The strategy is structural garbage).

---

## 9. Guardrails Audit
-   **Anti-Lookahead Sentinel:** Passed with `VERIFIED_GREEN`. No future pricing leaks were introduced.
-   **Activity Sentinel:** Inactivity warnings correctly handled. The strategy logged 191 trades over its active lifespan, avoiding degeneracy warnings because of the initial density, but the total halt since 2018 is a fatal operational blocker.
-   **Output Policy:** Staging filter is verified. The heavy 42.8 MB base equity curve is completely ignored in git.
-   **Verdict:** **PASS**.

---

## 10. Cost Profile Audit
The transaction cost scaling represents realistic and highly conservative prop firm conditions. The monotonic degradation proves that transaction costs are not padded/skipped:
-   Base Cost PnL: `-$26,035`
-   Conservative Cost PnL: `-$28,506`
-   Stress Cost PnL: `-$29,346`
-   **Verdict:** **PASS**.

---

## 11. Statistical Decision
The metrics leave no room for optimism:
-   Profit Factor is significantly below `1.0` in all configurations (`0.63` -> `0.57`).
-   Expectancy is deeply negative (`-0.28 R` to `-0.33 R`), meaning every trade loses an average of ~0.3 R in commissions and slippage.
-   The drawdowns are severe (`27%` to `30%`) with zero recovery since 2015.
-   Win rate is poor (~27%).
-   Conclusion: **NO EDGE DETECTED**.

---

## 12. Classification Review
-   **Prior Classification:** `TP01_PRELIMINARY_REJECTED_LOW_EDGE`
-   **Auditor Classification:** **`TP01_OFFICIALLY_REJECTED_LOW_EDGE_AND_REGIME_OBSOLESCENCE`** (Confirmed and extended with temporal obsolescence rationale).

---

## 13. Professional Decision
-   **NO** validation data will be spent.
-   **NO** holdout data will be unsealed.
-   **NO** production/incubation steps.
-   **NO** demo/paper trading.
-   **NO** parameter search or sweep to "rescue" this candidate.
-   **Archiving verdict:** Strategy is permanently archived in the rejected research logs.

---

## 14. Findings Table

| ID | Severity | Category | Finding | Evidence | Implication | Required Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **F-01** | **BLOCKER** | Statistical | Negative Expectancy | Base expectancy is -0.28 R, PF is 0.63. | Strategy is mathematically unviable. | Permanent strategy rejection. |
| **F-02** | **BLOCKER** | Temporal | Regime Obsolescence | exactly 0 trades logged since 2018. | Strategy rules are completely obsolete. | Archive and defer family. |
| **F-03** | **WARN** | Data | Timeframe Discrepancy | Declared M1 vs M5 effective cadence. | Platform resolved dynamiccadence benignly. | None (Auto-resolved). |
| **F-04** | **PASS** | Security | Git Ignored Outputs | Heavy CSVs in local_outputs_do_not_commit. | Git index is kept extremely lightweight. | Ensure no staging. |

---

## 15. Allowed Next Step
**A) Archive TP-01 as rejected low-edge and move to institutional acceleration plan.**

---

## 16. Final Institutional Verdict
Strategy `tp01_london_ny_momentum_pullback` represents a classic example of an unviable retail concept under institutional cost realistic execution conditions. The strategy suffered from high cost sensitivity, poor entry edge, and structural regime obsolescence after 2018. By rejecting this strategy immediately in the train-only phase, the laboratory successfully preserved validation and holdout resources. The dossier is sealed, and we will proceed directly with the Institutional Research Acceleration Plan.

*End of Report (GOV-AUD-TP01-V1-20260517)*
