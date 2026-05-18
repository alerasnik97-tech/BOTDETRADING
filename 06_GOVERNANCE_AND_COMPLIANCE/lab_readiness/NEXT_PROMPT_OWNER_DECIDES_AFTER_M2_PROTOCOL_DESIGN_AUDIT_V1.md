# NEXT PROMPT — OWNER DECIDES AFTER M2 PROTOCOL DESIGN AUDIT V1

This prompt is to be executed in **OWNER DECISION MODE**. No execution, data loading, or code changes are permitted during this phase. The owner must choose the next course of action following the successful M2 design audit.

---

## 1. Context and Current State
- **Audited Branch:** `research/next-train-only-protocol-bo01-mr02-v1-20260518`
- **Audit Verdict:** `NEXT_TRAIN_ONLY_PROTOCOL_DESIGN_AUDIT_PASS_WITH_WARNINGS` (Clean design, minor administrative warnings)
- **State to Validate:** `M2_TRAIN_ONLY_PROTOCOL_DESIGN_VERIFIED`

---

## 2. Decision Matrix for the Owner
The owner must choose exactly **one** of the following options:

### Option A: Authorize M2 Conservative Protocol Execution
Authorize the team to draft and execute the M2 Conservative Train-Only Structural Evaluation protocol.
- *Strict Bounds:* 3-month window (`2015-01-01` to `2015-03-31`), strictly no performance metrics (PnL, Sharpe, WR), strictly train-only, no validation/holdout, no 2025/2026.
- *Execution:* Needs a specific activation gate phrase signed by the owner.

### Option B: Patch M2 Design Warnings First
Instruct the team to address the minor administrative warnings recorded in the design audit report (e.g. resolve declared SHA abbreviation variance in reports, clean pre-existing dirty backlogs W-01/W-02).

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
