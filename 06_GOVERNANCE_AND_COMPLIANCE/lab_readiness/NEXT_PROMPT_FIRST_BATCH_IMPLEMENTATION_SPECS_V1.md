# NEXT PROMPT: FIRST BATCH IMPLEMENTATION SPECS

**Document Reference:** GOV-PRM-SPECS-V1-20260517  
**Status:** COMPLIANT  
**Date:** May 17, 2026  

---

## 1. Context & Rationale
The institutional Strategy Research Registry and first batch pre-registration have been officially approved and sealed under branch `planning/research-registry-and-first-batch-v1-20260517` (commit `7f76acf7ac5bda582404ff86c4fcc37a7fd0d159` or latest planning head).

To proceed systematically without coding or testing prematurely, the next phase is to write the **Technical Specifications and Contract Tests Plan** for our first priority candidate:
-   **Strategy ID:** `BO01`
-   **Strategy Name:** `bo01_london_breakout_continuation`
-   **Family:** `LBC`

---

## 2. Research Target & Branching
-   **Active Branch:** `planning/bo01-technical-specs-v1`
-   **Base Commit:** Current audited planning branch head.
-   **Scope:** Planning, Specs, and Test Design. **NO backtesting, NO code implementation, NO holdout exposure**.

---

## 3. Mandatory Tasks for Next Agent
Act as a **Senior Quant Systems Architect** and **Lead Test Engineer** to:

1.  **Draft Technical Specifications for `BO01`:**
    -   Specify the exact mathematical formulas for the Asian Range boundaries (high/low calculations over the frozen 00:00 - 06:30 GMT period).
    -   Detail the M5 bar closing triggers and ATR breakout validation logic ($ATR(14)$ multiplier calculation).
    -   Formalize the timezone conversion mechanisms to prevent EST/EDT daylight saving offsets from shifting the session windows.
2.  **Design Targeted Contract Unit Tests:**
    -   Draft the exact test skeletons and assertions for `test_strategy_contract_bo01.py` and `test_strategy_tz_bo01.py`.
    -   Ensure assertions explicitly verify future price poisoning prevention (no-lookahead) and schedule boundaries.
3.  **Define Parameter Boundaries:**
    -   Freeze the initial parameters allowed (Asian range window, ATR multiplier, EMA trend filter) and explicitly list all parameters forbidden from optimization.
4.  **Confirm Execution Guardrails:**
    -   Ensure that the strategy rules strictly restrict execution to a maximum of 1 trade per day.
5.  **Output Staging Security:**
    -   Do not modify any engine, runner, or pricing files. Create only markdown spec files under `06_GOVERNANCE_AND_COMPLIANCE/research_registry/` and `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/`.

---

## 4. Final Handoff Format
Your final report must present:
1.  **SPECS STATUS:** READY / BLOCKED
2.  **TECHNICAL FORMULAS:** Details of the Asian range and breakout equations.
3.  **TEST SCHEMAS:** Structural overview of the unit test files and assertions.
4.  **NEXT IMPLEMENTATION RECOMMENDATION:** Ready for owner review and coding authorization.

*End of Specs Prompt*
