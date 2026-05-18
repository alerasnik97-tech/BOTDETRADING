# NEXT PROMPT — AUDIT MICRORUN PROTOCOL DESIGN V1

## 0. Nature Of This Document
This is a governance-locked template. It does NOT authorize any code execution, micro-runs, dry-runs, backtests, validation unsealings, or parameter sweeps. It is designed strictly for a future external read-only audit of the micro-run protocol design.

---

## 1. Context
The design-only micro-run protocol for strategy candidates **BO01** and **MR02** (Sub-Batch 1A) has been drafted on branch `research/microrun-protocol-design-v1-20260517`. The candidates are currently registered at `MICRO_RUN_PROTOCOL_DESIGN_PENDING` status. No execution exists. The next step is a read-only audit of this protocol design.

---

## 2. Mandatory Prechecks
Before starting, check for any active Python execution engines, training runners, or optimization sweeps running on the system:
```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, StartTime, CPU, WorkingSet
Get-CimInstance Win32_Process -Filter "name='python.exe'" | Select-Object ProcessId, CommandLine | Format-List
```
If there are any active Python backtests, validation unsealings, optimization sweeps, or micro-runners running, you **MUST ABORT IMMEDIATELY** with status:
**`BLOCKED_ACTIVE_RESEARCH_PROCESS_DETECTED`**

---

## 3. Audit Scope
You must audit the following items:
1.  **Git Diff Integrity:** Verify that only the authorized markdown files under `06_GOVERNANCE_AND_COMPLIANCE/` were modified or created. Confirm that no code scripts, test scripts, raw price data, or root outputs entered the commit.
2.  **Protocol Completeness:** Inspect `SUBBATCH_1A_BO01_MR02_MICRORUN_PROTOCOL_DESIGN_V1.md` and confirm that it defines all mandatory sections (Nature of Document, Preconditions, Data Policy, Command Drafts prefixed as DRAFT_DO_NOT_RUN, Output Path quarantine, and Abort conditions).
3.  **Registry Matching:** Verify that `STRATEGY_RESEARCH_REGISTRY.md` correctly shows BO01 and MR02 at `MICRO_RUN_PROTOCOL_DESIGN_PENDING` status with the correct branch name and commit SHA.
4.  **W-01/W-02 Gates:** Confirm that the protocol strictly preserves W-01 (dirty tree) and W-02 (output debt) as future execution gates, with no physical file modifications in this phase.
5.  **Adjective Sobriety:** Verify that no absolute or qualitative terms (e.g., "perfect", "guaranteed", "fully resolved", "complete auditability") exist in the new files.

---

## 4. Forbidden Actions
Under this audit phase:
- **NO modification of strategy code or test scripts is authorized.**
- **NO modification of datasets or the market-data vault is allowed.**
- **NO execution of any micro-runs, dry-runs, or backtests is permitted.**
- **NO unsealing of validation sets or holdout (2025/2026) exposure is allowed.**
- **NO parameter sweeps or optimization sweeps are permitted.**
- **NO parallel writing agents are permitted in the laboratory.**
- **NO use of production, demo, real, or FTMO accounts is allowed.**

---

## 5. Allowed Actions
- Read-only inspection of the files in the workspace.
- Standard git status, git diff, and git log command executions.
- Creating the audit report document `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/MICRORUN_PROTOCOL_DESIGN_EXTERNAL_AUDIT_V1.md`.
- Staging, committing, and pushing ONLY the created audit report file to a new audit branch (`audit/microrun-protocol-design-review-v1-20260517`).

---
*End of Prompt*
