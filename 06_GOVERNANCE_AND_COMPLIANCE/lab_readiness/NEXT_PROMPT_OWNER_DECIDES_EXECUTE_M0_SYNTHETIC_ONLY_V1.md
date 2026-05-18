# NEXT PROMPT — OWNER DECIDES: EXECUTE M0 SYNTHETIC-ONLY V1

## 0. Nature Of This Document
This document is an owner-decision template. It executes NOTHING automatically.
It does NOT authorize any code execution, micro-run, dry-run, backtest, train,
validation, holdout, 2025/2026 access, or optimization/sweep. It exists only so
the owner can choose how to proceed after the external read-only audit of the
M0 synthetic-only execution prompt draft.

---

## 1. Audit Outcome Recap
- External audit status: `M0_SYNTHETIC_EXECUTION_PROMPT_AUDIT_PASS_WITH_WARNINGS`.
- Audit artifact: `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M0_SYNTHETIC_EXECUTION_PROMPT_EXTERNAL_AUDIT_V1.md`.
- No blockers. The cleanup commit is docs-only and lineage-correct; the future
  execution prompt is owner-gated, synthetic-only, and output-quarantined.
- Open minor warnings (recommended, not required): W-A (no explicit
  anti-self-citation / anti-paraphrase clause on the activation gate), W-B
  (Sub-Batch 1B / parallel writers absent from Forbidden Scope), W-C
  (daily_trade_count / active_position gate scenarios not enumerated), W-D
  (report mandatory declarations omit explicit no-2025/2026 and
  no-optimization/sweep), W-E (cosmetic branch-name wording).
- W-01 (dirty tree) and W-02 (output debt) remain intact as future execution
  gates and were not touched.

### 1A. Post-Audit Update
A documentation-only hardening patch addressing W-A through W-E has since been
applied on branch `research/m0-synthetic-execution-prompt-hardening-v1-20260518`.
That patch has NOT yet been externally audited. W-A and W-B are considered
material hardening warnings.

---

## 2. Owner Options

> Recommendation (maximum discipline): **Option B before Option A.** Because W-A
> (activation-gate anti-ambiguity) and W-B (expansion lock) are material, the
> safest process is to externally audit the hardening patch read-only first, and
> only then make the execution decision. Option A remains permitted under the
> existing audit status, but it is not the preferred path if the owner wants the
> strictest sequence. This document selects nothing on the owner's behalf.

### Option A — Approve execution of the already-audited prompt
The owner may, in a separate future prompt, authorize execution of
`NEXT_PROMPT_EXECUTE_M0_SYNTHETIC_MICRORUN_BO01_MR02_V1.md` as audited.
- This still requires the EXACT activation phrase to be provided autonomously by
  the owner in that future prompt:
  "APRUEBO EJECUTAR M0 SYNTHETIC-ONLY MICRORUN BO01/MR02, SIN DATOS REALES, SIN
  DATA VAULT, SIN BACKTEST, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026
  Y SIN OPTIMIZATION/SWEEP."
- The phrase appearing inside this document or the audited prompt does NOT count
  as activation. Selecting Option A here does NOT start execution.

### Option B — Externally audit the applied hardening patch first (recommended)
The docs-only patch addressing W-A through W-E has already been applied on
`research/m0-synthetic-execution-prompt-hardening-v1-20260518`. Under Option B,
the owner authorizes a separate read-only external audit of that patch (see
`NEXT_PROMPT_AUDIT_M0_SYNTHETIC_EXECUTION_PROMPT_HARDENING_V1.md`) before any
owner-use / execution decision. This is the preferred path for maximum
discipline; it still authorizes no execution.

### Option C — Do not advance
Hold at the current state. No execution, no patch, no further action.

---

## 3. Hard Constraints (apply to all options)
- No execution now.
- No real data.
- No data vault (`05_MARKET_DATA_VAULT`).
- No dry-run.
- No backtest.
- No train.
- No validation.
- No holdout.
- No 2025/2026.
- No optimization/sweep.
- No Sub-Batch 1B.
- No parallel writers.
- No production/demo/real/FTMO.
- W-01 and W-02 remain future execution gates until formally resolved.

---

## 4. What This Document Does Not Do
- It does not authorize execution.
- It does not select an option on the owner's behalf.
- It does not modify code, tests, or data.
- It does not unseal validation or holdout.
- It does not access the data vault or real data.

---
*End of Owner-Decision Prompt*
