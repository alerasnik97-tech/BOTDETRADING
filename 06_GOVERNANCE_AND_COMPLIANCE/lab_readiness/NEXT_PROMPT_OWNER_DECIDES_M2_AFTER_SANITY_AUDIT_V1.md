# NEXT PROMPT — OWNER DECIDES M2 AFTER SANITY AUDIT V1

This prompt is to be executed in **OWNER DECISION MODE**. No execution, data loading, or code changes are permitted. The owner must choose the next course of action following the successful corrective design audit.

---

## 1. Context and Current State
- **Audited Branch:** `research/next-train-only-protocol-bo01-mr02-v1-20260518`
- **Sanity Review Branch:** `audit/m2-design-audit-sane-review-v1-20260518`
- **Audit Sanity Status:** `M2_DESIGN_AUDIT_SANITY_READY_FOR_OWNER_DECISION`
- **State to Validate:** `M2_TRAIN_ONLY_PROTOCOL_DESIGN_VERIFIED`

---

## 2. Decision Matrix for the Owner
The owner must select exactly **one** of the following options:

### Option A: Draft and Execute M2 Conservative Protocol Prompt
Authorize the team to draft the exact prompt for the M2 Conservative Train-Only Structural Evaluation and prepare for execution.
- *Strict Bounds:* 3-month window (`2015-01-01` to `2015-03-31`), strictly no performance metrics (PnL, Sharpe, WR), strictly train-only, no validation/holdout, no 2025/2026.
- *Runner Requirement:* If no audited runner is found or provided, the execution must immediately abort as `BLOCKED_M2_RUNNER_NOT_AUDITED_OR_NOT_FOUND`.
- *Execution:* A separate, explicit owner activation gate phrase will be required to execute.

### Option B: Patch Remaining Warnings First
Instruct the team to address any residual warnings or administrative points from the design audit.

### Option C: Pause Operations
Freeze all quantitative research lab operations under safety status.

---

## 3. Strict Safety Declarations
Under all options, the following remain strictly **PROHIBITED**:
- **NO immediate M2 execution under this prompt.**
- **NO immediate backtesting or parameter sweeps.**
- **NO validation or holdout data access.**
- **NO 2025/2026 data loading.**
- **NO production or incubation staging operations.**
- **NO paper, live, or FTMO execution claims.**
