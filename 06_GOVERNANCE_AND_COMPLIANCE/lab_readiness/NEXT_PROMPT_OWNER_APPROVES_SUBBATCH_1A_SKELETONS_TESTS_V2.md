# OWNER APPROVES SUB-BATCH 1A SKELETONS AND TARGETED TESTS V2

**Document Reference:** GOV-PRM-1A-APPROVE-V2-20260517  
**Status:** FUTURE CANDIDATE — USE ONLY AFTER EXPLICIT OWNER APPROVAL  
**Date:** May 17, 2026  

> [!IMPORTANT]
> **USAR SOLO SI EL OWNER APROBÓ EXPLÍCITAMENTE POR ESCRITO EL SIGUIENTE COMANDO DE INICIACIÓN:**  
> **"APRUEBO IMPLEMENTAR SKELETONS + UNIT/CONTRACT TESTS DE BO01/MR02, SIN MICRO-RUN NI BACKTEST."**

---

## 1. Persona & Context
Act as a **Senior Quant Systems Implementer** and **Lead QA Test Engineer**. 

If the owner has provided the explicit approval statement above, you are authorized to execute **Phase 2 (targeted unit/contract tests and strategy code skeletons)** strictly for **Sub-Batch 1A**:
*   **Strategy ID:** `BO01` (London Breakout Continuation)
*   **Strategy ID:** `MR02` (London Breakout Failure)

---

## 2. Mandatory Precheck Workflow
Before modifying any files or writing code, you **MUST** execute the following shell commands and inspect the output:

```powershell
cd "C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
git status --short
git branch --show-current
git rev-parse HEAD
git fetch origin --prune
git remote -v
```

### Process Isolation Gate
Check for any active Python execution engines, training runners, or optimization sweeps running on the system:
```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, StartTime, CPU, WorkingSet
```
If there are any active Python backtests, validation unsealings, optimization sweeps, or micro-runners running, you **MUST ABORT IMMEDIATELY** with status:
**`BLOCKED_ACTIVE_RESEARCH_PROCESS_DETECTED`**

---

## 3. Branching & Workspace Control
*   **Base Branch:** `audit/final-owner-review-hardening-v1-20260517`
*   **Target Development Branch:** `research/subbatch-1a-skeletons-tests-v1-20260517`
*   **Command to checkout branch:**
    ```powershell
    git switch audit/final-owner-review-hardening-v1-20260517
    git pull origin audit/final-owner-review-hardening-v1-20260517
    git switch -c research/subbatch-1a-skeletons-tests-v1-20260517
    ```
*   **Writers Constraint:** Only a single agent writer is authorized. No parallel agents are permitted to write code simultaneously.

---

## 4. Specs Reference Documents
You **MUST** view and completely read the following specifications files before writing strategy skeletons or tests:
1.  [BO01_IMPLEMENTATION_SPEC_V1.md](file:///C:/Users/alera/Desktop/Bot/BOT%20DE%20TRADING%20ultimo/06_GOVERNANCE_AND_COMPLIANCE/research_registry/first_batch_specs/BO01_IMPLEMENTATION_SPEC_V1.md)
2.  [MR02_IMPLEMENTATION_SPEC_V1.md](file:///C:/Users/alera/Desktop/Bot/BOT%20DE%20TRADING%20ultimo/06_GOVERNANCE_AND_COMPLIANCE/research_registry/first_batch_specs/MR02_IMPLEMENTATION_SPEC_V1.md)
3.  [FIRST_BATCH_TEST_PLAN_V1.md](file:///C:/Users/alera/Desktop/Bot/BOT%20DE%20TRADING%20ultimo/06_GOVERNANCE_AND_COMPLIANCE/research_registry/first_batch_specs/FIRST_BATCH_TEST_PLAN_V1.md)
4.  [FIRST_BATCH_SUBBATCH_DECISION_V1.md](file:///C:/Users/alera/Desktop/Bot/BOT%20DE%20TRADING%20ultimo/06_GOVERNANCE_AND_COMPLIANCE/research_registry/first_batch_specs/FIRST_BATCH_SUBBATCH_DECISION_V1.md)

---

## 5. Allowed and Prohibited File Scope Whitelist

### ALLOWED Files (Whitelist):
You are strictly limited to creating or modifying **ONLY** the following paths:
*   `03_RESEARCH_LAB/research_lab/strategies/BO01Strategy.py` (Strategy skeleton)
*   `03_RESEARCH_LAB/research_lab/strategies/MR02Strategy.py` (Strategy skeleton)
*   `03_RESEARCH_LAB/research_lab/tests/test_strategy_contract_bo01.py` (Lookahead & parameter contract)
*   `03_RESEARCH_LAB/research_lab/tests/test_strategy_tz_bo01.py` (Timezone & session window validations)
*   `03_RESEARCH_LAB/research_lab/tests/test_strategy_contract_mr02.py` (Lookahead & parameter contract)
*   `03_RESEARCH_LAB/research_lab/tests/test_strategy_tz_mr02.py` (Timezone & session window validations)
*   `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/SUBBATCH_1A_IMPLEMENTATION_REPORT_V1.md` (Implementation and test execution report)
*   `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_SUBBATCH_1A_TESTS_AUDIT_V1.md` (Future read-only external audit prompt)

### PROHIBITED Files (Blacklist):
Do **NOT** touch, modify, or create files under:
*   Core Engine files (`01_CORE_PRODUCTION/` or `04_INFRASTRUCTURE_ENGINEERING/`)
*   Official Runner files (`formal_train_runner.py`)
*   Data Vault files (`05_MARKET_DATA_VAULT/`)
*   Validation/Holdout datasets
*   Dynamic output spreadsheets (`trades.csv`, `equity_curve.csv`)
*   Temporary ZIP files
*   Root directories

---

## 6. Skeletons and Tests Execution Guide

### Step 1: Write Skeletons
Implement the frozen mathematical signals inside `BO01Strategy` and `MR02Strategy` classes strictly based on closed bars, GMT timezone rules, and capital management guidelines.

### Step 2: Write Targeted Unit and Contract Tests
Implement the test files under `03_RESEARCH_LAB/research_lab/tests/` to assert:
1.  **Future Poisoning Invariance:** Changing future rows must not affect signal outputs at the current bar index `i`.
2.  **DST boundaries:** Session time boundaries are verified for March/October daylight saving weekend shifts.
3.  **Fills contract:** Cost profile deductions are correctly applied.
4.  **Limits bounds:** Daily trade counts are strictly capped.

### Step 3: Run the Test Suite
Run the test scripts locally using a lightweight test runner and confirm that all targeted tests passed cleanly with exit code 0.

### Step 4: STOP & Handoff
Do **NOT** attempt any micro-runs, dry-runs, parameter sweeps, training runs, or backtests. Once tests pass, freeze the workspace and prepare for a read-only audit.

---

## 7. Staging, Commit & Push Controls
You **MUST** stage the files individually. Broad staging (`git add .`) is **STRICTLY PROHIBITED**.
```powershell
git status --short
# Add whitelist files individually
git add 03_RESEARCH_LAB/research_lab/strategies/BO01Strategy.py
git add 03_RESEARCH_LAB/research_lab/strategies/MR02Strategy.py
git add 03_RESEARCH_LAB/research_lab/tests/test_strategy_contract_bo01.py
git add 03_RESEARCH_LAB/research_lab/tests/test_strategy_tz_bo01.py
git add 03_RESEARCH_LAB/research_lab/tests/test_strategy_contract_mr02.py
git add 03_RESEARCH_LAB/research_lab/tests/test_strategy_tz_mr02.py
git add 06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/SUBBATCH_1A_IMPLEMENTATION_REPORT_V1.md
git add 06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_SUBBATCH_1A_TESTS_AUDIT_V1.md
```

Verify that **ONLY** whitelist files are cached:
```powershell
git diff --cached --name-only
```
If any other file is displayed, abort and unstage immediately.

Commit and push:
```powershell
git commit -m "research: implement subbatch 1a skeletons and tests"
git push -u origin research/subbatch-1a-skeletons-tests-v1-20260517
```

---

## 8. Final Handoff Format
Your final response must present exactly:

1.  **IMPLEMENTATION STATUS:** `SKELETONS_AND_TESTS_COMPLETED`
2.  **BRANCH:**
    *   base: `audit/final-owner-review-hardening-v1-20260517`
    *   dev_branch: `research/subbatch-1a-skeletons-tests-v1-20260517`
    *   head: `[COMMIT_SHA]`
3.  **TEST SUITE RESULTS:** Details of unit test executions. Use **"all targeted tests passed"** or **"tests failed/blocker"** to report results (never use "100% green" or speculative metrics).
4.  **TEST PLAN COMPLIANCE:** Verification of DST transition anchors and future poisoning immunities.
5.  **NEXT STEP PROMPT:** Draft a future separated prompt for the read-only external audit of strategy skeletons and test executions.

---
*End of Prompt V2*
