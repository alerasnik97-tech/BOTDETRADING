# NEXT PROMPT: IMPLEMENT SUB-BATCH 1A SKELETONS & TESTS

**Document Reference:** GOV-PRM-1A-IMPLEMENT-V1-20260517  
**Status:** COMPLIANT  
**Date:** May 17, 2026  

---

## 1. Persona & Context
Act as a **Senior Quant Systems Implementer** and **Lead QA Test Engineer**. 

The owner has reviewed and approved the technical specifications report for the first batch strategies under branch `specs/first-batch-implementation-specs-v1-20260517`. You are authorized to begin **Phase 2 (targeted unit tests and strategy code skeletons)** strictly for **Sub-Batch 1A**:
*   **Strategy ID:** `BO01` (London Breakout Continuation)
*   **Strategy ID:** `MR02` (London Breakout Failure)

---

## 2. Active Branching
*   **Active Branch:** `research/subbatch-1a-implementation-v1`
*   **Base Commit:** Latest audited specs branch head.
*   **Scope:** Strategy code skeletons creation and targeted unit tests implementation. **NO backtesting, NO validation unsealing, NO holdout exposure**.

---

## 3. Mandatory Safety Constraints
*   **NO CORE ENGINE MODIFICATIONS:** The core execution engine and training runners must remain strictly frozen.
*   **NO DYNAMIC OPTIMIZATIONS:** Parameter sweeps, walk-forward routines, or grid searches are strictly prohibited.
*   **NO ZIP ARCHIVES:** Creation or modification of zip files is banned.
*   **SURGICAL STAGING:** No broad `git add .` operations. Staging must proceed file-by-file.
*   **DATA VAULT QUARANTINE:** The `05_MARKET_DATA_VAULT` is read-only. Do not modify pricing data files.

---

## 4. Execution Workflow
You must implement Sub-Batch 1A through the following rigorous progression:

### Step 1: Write Skeletons and targeted unit tests
*   Create the strategy skeleton classes `BO01Strategy` and `MR02Strategy` under `03_RESEARCH_LAB/research_lab/strategies/` implementing the frozen entry/exit mathematical logic.
*   Write targeted unit/contract test files `test_strategy_contract_bo01.py`, `test_strategy_tz_bo01.py`, `test_strategy_contract_mr02.py`, and `test_strategy_tz_mr02.py` under `03_RESEARCH_LAB/research_lab/tests/`.
*   Run the test suite and confirm **100% green status (exit code 0)**.

### Step 2: Micro-Run Preflight (Phase 3)
*   Execute a fast, local 10-day dry-run to verify order telemetry, spreads, commissions, and telemetry logging accuracy.

### Step 3: Stop & Handoff
*   Once Sub-Batch 1A unit tests pass and the micro-run is green, **STOP**. Do not run the full training backtests (Phase 4) without explicit owner review and audit of the tests dossier.

---

## 5. Final Handoff Format
Your final report must end with:
1.  **IMPLEMENTATION STATUS:** READY_FOR_TRAIN / BLOCKED
2.  **TEST SUITE RESULTS:** Details of unit test executions.
3.  **MICRO-RUN VERDICT:** Telemetry verification summary.
4.  **NEXT STEP:** Ready for owner authorization to run Phase 4 sealed training backtest.

---
*End of Prompt*
