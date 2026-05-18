# DEPRECATED — DO NOT USE

This prompt is superseded by V2. It must not be used because it lacks full precheck, branch verification, process checks, staging whitelist, safety scan, and audit-grade handoff requirements.

# NEXT PROMPT: OWNER APPROVES SUB-BATCH 1A SKELETONS & TARGETED TESTS V1 (DEPRECATED)

**Document Reference:** GOV-PRM-1A-APPROVE-V1-20260517 (DEPRECATED)  
**Status:** DEPRECATED / DO NOT USE  
**Date:** May 17, 2026  

---

## 1. Persona & Context
Act as a **Senior Quant Systems Implementer** and **Lead QA Test Engineer**.

The owner has reviewed and explicitly approved the technical specifications report for the first batch strategies under branch `specs/first-batch-implementation-specs-v1-20260517`. You are authorized to begin **Phase 2 (targeted unit/contract tests and strategy code skeletons)** strictly for **Sub-Batch 1A**:
*   **Strategy ID:** `BO01` (London Breakout Continuation)
*   **Strategy ID:** `MR02` (London Breakout Failure)

---

## 2. Active Branching & Workspace Scope
*   **Active Branch:** `research/subbatch-1a-implementation-v1`
*   **Base Commit:** Latest audited specs branch head.
*   **Scope:** Strategy code skeletons creation and targeted unit/contract tests implementation.
*   **Writers:** A single writing agent is allowed. No parallel writers.

---

## 3. Mandatory Safety Constraints
*   **NO CORE ENGINE MODIFICATIONS:** The core execution engine and training runners must remain strictly frozen.
*   **NO DYNAMIC OPTIMIZATIONS:** Parameter sweeps, walk-forward routines, or grid searches are strictly prohibited.
*   **NO ZIP ARCHIVES:** Creation or modification of zip files is banned.
*   **SURGICAL STAGING:** No broad `git add .` operations. Staging must proceed file-by-file.
*   **DATA VAULT QUARANTINE:** The `05_MARKET_DATA_VAULT` is read-only. Do not modify pricing data files.
*   **NO 2025/2026 OR HOLDOUT:** Out of bounds dates must remain strictly sealed.
*   **NO DYNAMIC EXECUTION:** No micro-runs, dry-runs, training runs, or backtests are authorized under this prompt.

---

## 4. Execution Workflow
You must implement Sub-Batch 1A through the following rigorous progression:

### Step 1: Write Skeletons
*   Create the strategy skeleton classes `BO01Strategy` and `MR02Strategy` under `03_RESEARCH_LAB/research_lab/strategies/` implementing the frozen entry/exit mathematical logic.
*   The strategy entry and exit rules must be completely objective, following the GMT session windows defined in the specifications.

### Step 2: Write Targeted Unit and Contract Tests
*   Write targeted unit/contract test files `test_strategy_contract_bo01.py`, `test_strategy_tz_bo01.py`, `test_strategy_contract_mr02.py`, and `test_strategy_tz_mr02.py` under `03_RESEARCH_LAB/research_lab/tests/`.
*   Tests must cover:
    1.  **Lookahead / Poisoning Invariance:** Mutating future rows does not change the signal output at index `i`.
    2.  **DST boundaries:** Session time boundaries are verified for March/October daylight saving weekend shifts.
    3.  **Fills contract:** Spread, commission, and slip calculations are properly logged.
    4.  **Limits bounds:** Daily trade counts are strictly capped.

### Step 3: Run Test Suite
*   Run the test suite and confirm **green status (exit code 0)**.

### Step 4: STOP & Handoff
*   Once the Sub-Batch 1A strategy skeletons are written and the targeted tests pass cleanly, **STOP**.
*   Do not attempt any micro-runs, dynamic backtests, or training sweeps.
*   Prepare the tests execution dossier for an external read-only audit.

---

## 5. Final Handoff Format
Your final report must present:
1.  **IMPLEMENTATION STATUS:** `SKELETONS_AND_TESTS_COMPLETED`
2.  **TEST SUITE RESULTS:** Details of unit test executions.
3.  **TEST PLAN COMPLIANCE:** Verification of lookahead prevention and timezone anchors.
4.  **NEXT STEP PROMPT:** Draft a future separated prompt for the external audit of strategy implementation and tests.

---
*End of Prompt*
