# AUDITED PROMPT: BO01 TECHNICAL SPECIFICATION & TEST DESIGN

**Document Reference:** GOV-PRM-BO01-SPECS-V1-20260517  
**Status:** COMPLIANT  
**Date:** May 17, 2026  

---

## 1. Auditor Persona & Role Instructions
Act as a **Senior Quant Systems Architect** and **Lead Test Engineer** operating under the strict Trading BOT quant factory standards.

Your mission is to write the **Technical Specifications and Contract Tests Design** for our first priority pre-registered candidate strategy:
*   **Strategy ID:** `BO01`
*   **Strategy Name:** `bo01_london_breakout_continuation`
*   **Family:** `LBC` (London Breakout Continuation)

---

## 2. Strict Constraints
*   **NO CODE IMPLEMENTATION:** You are strictly forbidden from writing the final strategy execution code script or loading historical price files.
*   **NO BACKTESTING:** You must not run any backtests, dynamic sweeps, optimizations, or grid searches.
*   **NO DATA CONTAMINATION:** The `05_MARKET_DATA_VAULT` must remain completely untouched and read-only.
*   **NO HOLDOUT EXPOSURE:** Validation and holdout datasets remain locked.
*   **ONLY SPECS AND TESTS DESIGN:** You are authorized ONLY to write markdown specification documents and unit test skeletons.

---

## 3. Mandatory Specs Scope
You must define and document:
1.  **Mathematical Formulations:**
    *   The exact, unambiguous formulas for the Asian Range High/Low calculations. The time bounds are frozen at 00:00 - 06:30 GMT.
    *   The ATR-based breakout validation calculation ($ATR(14)$ on M5 with a frozen threshold multiplier $\ge 0.5$).
    *   The trend direction validation filters (EMA(20) and EMA(200) alignments).
2.  **Timezone Guardrails:**
    *   The exact timezone translation mechanism to protect the time windows from EST/EDT daylight saving shifts.
3.  **Targeted Contract Tests Plan:**
    *   Write the complete skeletons and assertion structures for `test_strategy_contract_bo01.py` and `test_strategy_tz_bo01.py` under `03_RESEARCH_LAB/research_lab/tests`.
    *   Ensure assertions explicitly verify future price poisoning prevention (lookahead blocking) and schedule boundaries.
4.  **Parameter Locking:**
    *   Explicitly list all parameters allowed for testing and freeze all parameters forbidden from optimization.
5.  **Risk Invariant Controls:**
    *   Ensure that the strategy design strictly enforces a maximum of 1 trade per day and 1 concurrent position.

---

## 4. Final Handoff Format
Your final report must end with:
1.  **SPECS STATUS:** READY / BLOCKED
2.  **TECHNICAL EQUATIONS:** Precise mathematical definitions.
3.  **TEST SCHEMAS:** Structural overview of test assertions.
4.  **NEXT PROGRESSION RECOMMENDATION:** Ready for owner review and coding authorization.

---
*End of Audited Specs Prompt*
