# NEXT PROMPT — OWNER DECIDES MICRO-RUN PROTOCOL DESIGN AFTER GOVERNANCE AUDIT V1

## 0. Nature Of This Document
This is a governance-locked template. It does NOT authorize any code execution, micro-runs, dry-runs, backtests, validation unsealings, or parameter sweeps. It is designed to present three distinct, explicit choices to the owner regarding the future design-only phase.

---

## 1. Context
The post-extreme-audit governance hygiene patch has passed external read-only audit. All five warnings (W-01 dirty tree plan, W-02 output debt plan, W-03 missing BO01/MR02 rows, W-04 owner-less micro-run paths, W-05 TP01 lineage note) have been addressed for owner-decision purposes. W-01/W-02 are documented as remediation plans, not physically cleaned.

The strategy candidates **BO01** and **MR02** are currently at `IMPLEMENTED_TESTS_AUDITED_OWNER_PROTOCOL_DECISION_PENDING` status. No execution exists. The next step under the taxonomy is an owner decision on whether to commission a **design-only micro-run protocol** (which is a document, not a dynamic run).

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

## 3. The Three Decision Options For The Owner
You must present the owner with these exact three options and wait for an explicit written decision:

### OPTION A: Approve Commissioning of Design-Only Micro-Run Protocol
*   **What this authorizes:** Writing the protocol *design* document strictly as markdown under `06_GOVERNANCE_AND_COMPLIANCE/research_registry/` (re-routing the strategy statuses to `MICRO_RUN_PROTOCOL_DESIGN_PENDING`).
*   **What this DOES NOT authorize:** ANY execution, micro-runs, dry-runs, backtests, validation, or code changes.
*   **Target branch:** `research/microrun-protocol-design-v1-20260517`

### OPTION B: Request Further Governance Patching
*   **What this authorizes:** Refining the existing taxonomies, plans, or registry files before advancing to any design work.
*   **Target branch:** `governance/additional-patch-v1-20260517`

### OPTION C: Do Not Advance / Lock Laboratory
*   **What this authorizes:** Archiving the current branch state. The laboratory execution remains unauthorized under a frozen status.

---

## 4. Forbidden Actions
Under any option chosen:
*   **NO dynamic backtests or optimizations are authorized.**
*   **NO micro-runs or dry-runs are permitted.**
*   **NO unsealing of validation sets or holdout (2025/2026) exposure is allowed.**
*   **NO changes can be made to the core engine or official runner.**
*   **NO parallel writing agents are permitted in the laboratory.**
*   **NO use of production, demo, real, or FTMO accounts is allowed.**

---

## 5. Required Action
Stop immediately, present this prompt and the three choices to the owner, and wait for their explicit written selection.

---
*End of Prompt*
