# NEXT PROMPT — AUDIT M0 SYNTHETIC EXECUTION PROMPT V1

## 0. Nature Of This Document
This is a governance-locked template. It does NOT authorize any code execution, micro-runs, dry-runs, backtests, validation unsealings, or parameter sweeps. It is designed strictly for a future external read-only audit of the M0 synthetic execution prompt draft.

---

## 1. Context
The M0 synthetic execution prompt draft for strategy candidates **BO01** and **MR02** (Sub-Batch 1A) has been written on branch `research/draft-m0-synthetic-execution-prompt-v1-20260517`. The candidates are currently registered at `M0_SYNTHETIC_EXECUTION_PROMPT_DRAFT_PENDING_AUDIT` status. No execution exists. The next step is a read-only audit of this draft.

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
2.  **Prompt Completeness:** Inspect `NEXT_PROMPT_EXECUTE_M0_SYNTHETIC_MICRORUN_BO01_MR02_V1.md` and confirm that it defines all required sections (Activation Gate demanding the exact owner phrase, Nature, Allowed Scope, Forbidden Scope, Future Prechecks, Future Branching, Synthetic Fixture Policy, Output Policy, Future Script Policy, Safety Scan, Future Report, Future Git Policy, and Final Format).
3.  **Synthetic-Only Restrictions:** Verify that the execution prompt strictly restricts execution to temporary in-memory synthetic bar fixtures. Confirm that no file input or disc reading is permitted.
4.  **No Data Vault / Real Data:** Confirm that any access to `05_MARKET_DATA_VAULT` or real market datasets is heavily prohibited.
5.  **No Validation/Holdout/2025/2026:** Confirm that validation, holdout, and 2025/2026 datasets are completely sealed.
6.  **No Backtest / Train:** Confirm that backtests, train runs, and standard runner scripts are completely blocked.
7.  **No Optimization / Sweep:** Confirm that optimization, parameter sweep, grid search, and walk-forward sweeps are prohibited.
8.  **Output and .gitignore Gate:** Verify that all output paths are quarantined under `local_outputs_do_not_commit/` and that the .gitignore gate is active.
9.  **W-01/W-02 Gates:** Confirm that the audit prompt strictly preserves W-01 (dirty tree) and W-02 (output debt) as future execution gates, with no physical file modifications in this phase.
10. **Adjective Sobriety:** Verify that no absolute or qualitative terms exist in the new files.
11. **No Execution:** Confirm that no execution was performed during the drafting of this prompt.

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
- Creating the audit report document `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M0_SYNTHETIC_EXECUTION_PROMPT_EXTERNAL_AUDIT_V1.md`.
- Staging, committing, and pushing ONLY the created audit report file to a new audit branch (`audit/m0-synthetic-execution-prompt-review-v1-20260517`).

---
*End of Prompt*
