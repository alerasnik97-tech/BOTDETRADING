# NEXT PROMPT — OWNER DECIDES M0 SYNTHETIC-ONLY MICRORUN EXECUTION V1

## 0. Nature Of This Document
This is a governance-read-only decision template. It does NOT authorize any code execution, micro-runs, dry-runs, backtests, validation unsealings, or parameter sweeps. It is designed strictly to present three options to the owner regarding a future execution plan.

---

## 1. Context
The design-only micro-run protocol cleanup for strategy candidates **BO01** and **MR02** (Sub-Batch 1A) has passed external read-only audit on branch `audit/microrun-protocol-design-cleanup-review-v1-20260517` under commit `271f77d29e59150512ee42cab0c50863f9867956` (review branch HEAD: will be recorded in next session's checkout). The next step is a formal owner decision gate.

---

## 2. Mandatory Prechecks
Before selecting any option, confirm that no active Python execution engines, training runners, or optimization sweeps are running:
```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, StartTime, CPU, WorkingSet
Get-CimInstance Win32_Process -Filter "name='python.exe'" | Select-Object ProcessId, CommandLine | Format-List
```
If there are any active Python backtests, validation unsealings, optimization sweeps, or micro-runners running, you **MUST ABORT IMMEDIATELY** with status:
**`BLOCKED_ACTIVE_RESEARCH_PROCESS_DETECTED`**

---

## 3. Owner Decision Options
The owner must explicitly select and approve ONE of the following paths:

### OPTION A: Commission a future M0 synthetic-only execution prompt
*   **What this does:** Authorizes the writing of a separate, highly controlled future execution prompt template only.
*   **Safety warning:** Selection of Option A does NOT execute any code, micro-run, dry-run, backtest, or parameter sweep. It only authorizes the writing of the prompt.
*   **Preconditions:** Execution remains unauthorized until that future prompt is separately generated, audited, and owner-approved. W-01 (dirty tree) and W-02 (output debt) must remain as strict gates before any execution can be initiated.

### OPTION B: Request a minor documentation patch
*   **What this does:** Rejects the current audit report due to minor wording or traceability concerns, requiring a documentary patch before a decision can be made.
*   **Safety warning:** No execution is authorized.

### OPTION C: Abort and decommission strategy candidates
*   **What this does:** Permanently shifts strategy candidates BO01 and MR02 to `RETIRED` status.
*   **Safety warning:** No execution is authorized.

---

## 4. Forbidden Actions
Under all options:
- **NO modification of strategy code or test scripts is authorized.**
- **NO modification of datasets or the market-data vault is allowed.**
- **NO execution of any micro-runs, dry-runs, or backtests is permitted.**
- **NO unsealing of validation sets or holdout (2025/2026) exposure is allowed.**
- **NO parameter sweeps or optimization sweeps are permitted.**
- **NO parallel writing agents are permitted in the laboratory.**
- **NO use of production, demo, real, or FTMO accounts is allowed.**

---

## 5. Allowed Actions
- Presenting these three options clearly to the owner.
- Recording the owner's chosen path in a future session.

---
*End of Prompt*
