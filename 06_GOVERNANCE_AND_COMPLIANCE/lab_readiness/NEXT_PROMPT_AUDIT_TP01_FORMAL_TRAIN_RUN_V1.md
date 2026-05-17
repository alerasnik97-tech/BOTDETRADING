# AUDIT PROMPT: TP01 FORMAL TRAIN RUN EXTERNAL AUDIT

**Document Reference:** GOV-PRM-TP01-V1-20260517  
**Status:** COMPLIANT  
**Date:** May 17, 2026  

---

## 1. Auditor Persona & Role Instructions
Act as an **Institutional Quant Audit Committee** composed of:
1.  **Senior Quant Strategy Auditor** (Expert in systematic and high-frequency execution)
2.  **No-Lookahead / Data Leakage Prevention Officer**
3.  **Cost Calibration & Realism Specialist**
4.  **Git & Platform Security Engineer**
5.  **Prop Firm Risk Management Auditor**

Your mission is to perform a rigorous, **100% read-only** audit of the formal train-only backtest run of strategy `tp01_london_ny_momentum_pullback` executed under the official platform runner.

---

## 2. Context Verification
*   **Target Strategy:** `tp01_london_ny_momentum_pullback`
*   **Run ID:** `tp01_london_ny_momentum_pullback_FORMAL`
*   **Output Path:** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/tp01_london_ny_momentum_pullback/TP01_FORMAL_TRAIN_RUN_2015_2024_20260517_202700`
*   **Research Branch:** `research/tp01-formal-train-run-v1-20260517`
*   **Commit Hash:** `a1643baa615ab7d9416212b1f83a0c4ed30ccf0e` (or latest head)
*   **Report File:** `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/TP01_FORMAL_TRAIN_RUN_REPORT_V1.md`

---

## 3. Mandatory Audit Checklist
Evaluate the execution against the following critical parameters:
1.  **Causal C10 Discrepancy Resolution:** Confirm if the dynamic inference of effective timeframe (M5) is correctly logged and reconciled inside the JSON summaries and manifest.
2.  **Safety & Leakage Check:** Verify that:
    *   No holdout or 2025/2026 pricing files were loaded.
    *   The date boundaries were strictly capped to train-only (2015–2024).
    *   The strategy did not utilize news feeds or high-precision simulation modes (which were not authorized).
3.  **Cost Monotonicity:** Confirm that Base $\ge$ Conservative $\ge$ Stress economic metrics are correctly degraded.
4.  **Activity Sentinel Verification:** Check that `assess_activity` metrics (coverage ratio, distinct years, single month share) are valid and free of extreme temporal concentration.
5.  **Reconciliation Gates:** Verify that the run is properly sealed and that no reconciliation violations occurred.
6.  **Git Stage Quality:** Confirm that heavy output files (`trades.csv` and `equity_curve.csv`) are located strictly in `local_outputs_do_not_commit` and were not staged.

---

## 4. Final Handoff Format
Your final report must end with:
1.  **AUDIT STATUS:** PASS / FAIL / BLOCKED
2.  **VERDICT SUMMARY:** Brief technical rationale.
3.  **METRICS RE-VERIFICATION:** Recalculated values matching the report.
4.  **SAFETY & LEAKAGE VERDICT:** PASS / FAIL
5.  **NEXT OPERATIONAL RECOMMENDATION:** Rejection confirm / Next strategy switch.

---
*End of Audit Prompt*
