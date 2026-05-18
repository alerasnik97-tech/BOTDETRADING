# NEXT PROMPT — OWNER DECIDES EXECUTE M2 CONSERVATIVE AFTER DRAFT AUDIT V1

This prompt is to be executed in **OWNER DECISION MODE**. 
Under blocker penalty, the following are strictly **PROHIBITED** during this phase:
- NO executing Python scripts or commands.
- NO executing helper scripts (such as `safety_scan.py`).
- NO M2 execution.
- NO loading of market data.
- NO modifying strategy code, engine, runner, or test files.
- NO backtesting or formal training.
- NO validation or holdout partition access.
- NO 2025 or 2026 data loading.
- NO optimization sweeps, grid searches, or walk-forward parameters.

---

## 1. Context and Current State
- **Audited Branch:** `research/fix-m2-conservative-draft-lineage-v1-20260518`
- **Audit Branch:** `audit/m2-conservative-execution-prompt-draft-v1-20260518`
- **Audit Sanity Status:** `M2_CONSERVATIVE_EXECUTION_PROMPT_DRAFT_AUDIT_PASS_WITH_WARNINGS`
- **State to Validate:** `M2_CONSERVATIVE_EXECUTION_PROMPT_DRAFT_VERIFIED`

---

## 2. Decision Matrix for the Owner
The owner must select exactly **one** of the following options:

### Option A: Execute M2 Conservative Train-Only Structural Evaluation
Authorize the execution of M2 Conservative Train-Only Structural Evaluation by providing the exact activation gate phrase in the next turn:

“APRUEBO EJECUTAR M2 CONSERVATIVE TRAIN-ONLY STRUCTURAL EVALUATION BO01/MR02, SOLO MÉTRICAS ESTRUCTURALES, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026, SIN BACKTEST, SIN TRAIN FORMAL Y SIN OPTIMIZATION/SWEEP.”

- *Strict Bounds:* 3-month window (`2015-01-01` to `2015-03-31`), strictly no performance metrics (PnL, Sharpe, WR), strictly train-only, no validation/holdout, no 2025/2026.
- *Runner Requirement:* If no audited runner is found or provided, the execution must immediately abort as `BLOCKED_M2_RUNNER_NOT_AUDITED_OR_NOT_FOUND`.
- *Execution Limits:* This option ONLY authorizes executing the limited train-only structural evaluation under the designated gitignored local outputs. It does NOT authorize any backtest or train now.
- *Activation Phrase:* Future execution is strictly prohibited without the exact, explicit owner activation gate phrase.

### Option B: Patch Warnings First
Instruct the team to address the residual warning W-03 (language patch) from the draft audit.

### Option C: Pause Operations
Freeze all quantitative research lab operations under safety status.

---

## 3. Strict Safety Declarations
Under all options, the following remain strictly **PROHIBITED** during the current turn:
- **NO immediate M2 execution under this prompt.**
- **NO immediate backtesting or parameter sweeps.**
- **NO validation or holdout data access.**
- **NO 2025/2026 data loading.**
- **NO production or incubation staging operations.**
- **NO paper, live, or FTMO execution claims.**
