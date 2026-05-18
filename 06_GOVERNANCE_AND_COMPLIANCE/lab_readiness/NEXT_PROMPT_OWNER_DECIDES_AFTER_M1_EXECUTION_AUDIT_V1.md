# NEXT PROMPT — OWNER DECIDES AFTER M1 EXECUTION AUDIT V1

This prompt is to be executed in **OWNER DECISION MODE**. No execution, data loading, or code changes are permitted during this phase. The owner must choose the next course of action following the successful M1 Train-Only plumbing audit.

---

## 1. Context and Current State
- **Audited Branch:** `research/m1-train-only-bo01-mr02-v1-20260518`
- **Audit Verdict:** `M1_TRAIN_ONLY_MICRORUN_EXECUTION_AUDIT_PASS_WITH_WARNINGS` (Prstine plumbing, minor administrative warnings)
- **State to Validate:** `M1_TRAIN_ONLY_PLUMBING_VERIFIED`

---

## 2. Decision Matrix for the Owner
The owner must choose exactly **one** of the following options:

### Option A: Design Next Train-Only Protocol
Authorize the quantitative committee to design the protocol for the next research phase (e.g., a broader, train-only historical backtest on `BO01` and `MR02` utilizing the official runner on multiple years of train-only data).
- *Strict Bounds:* Train-only, no validation, no holdout, no 2025/2026, no optimization sweeps.

### Option B: Patch M1 Execution Warnings First
Instruct the team to address the minor administrative warnings recorded in the audit report (e.g., standardizing runner hashing to resolve CRLF newline variance, updating prompt commit placeholders, and refining clean/untracked directories).

### Option C: Pause Operations
Freeze all quantitative research lab operations under safety status.

---

## 3. Strict Safety Declarations
Under all options, the following remain strictly **PROHIBITED**:
- **NO immediate backtesting or parameter sweeps.**
- **NO validation or holdout data access.**
- **NO 2025/2026 data loading.**
- **NO production or incubation staging operations.**
- **NO paper, live, or FTMO execution claims.**
