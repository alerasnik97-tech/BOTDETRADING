# NEXT PROMPT — OWNER DECIDES: M1 TRAIN-ONLY PROTOCOL DESIGN V1

## 0. Nature Of This Document
Owner-decision template. It executes NOTHING. It does NOT authorize M1, any
execution, backtest, train, validation, holdout, 2025/2026, optimization/sweep,
Sub-Batch 1B, or parallel writers. It only lets the owner choose whether to
*design* (design-only, on paper) a future M1 train-only protocol.

---

## 1. Audit Outcome Recap
- M0 synthetic microrun execution audit:
  `M0_SYNTHETIC_MICRORUN_EXECUTION_AUDIT_PASS_WITH_WARNINGS`.
- Audit artifact:
  `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M0_SYNTHETIC_MICRORUN_EXECUTION_EXTERNAL_AUDIT_V1.md`.
- No blockers. Manifest SHA256 hashes independently verified (match), with a
  warning that `output_manifest.json` does not embed branch/commit metadata or
  self-hash. Commit docs-only. W-01/W-02 untouched. One disclosed in-loop
  remediation (a 2026 synthetic calendar label corrected to `2001-01-02`, zero
  data impact). No edge/performance asserted.
- M0 verified technical plumbing only (imports, fail-closed, session gate,
  daily_trade_count gate, active_position gate, negative control). It did NOT
  test edge, profitability, or readiness.

---

## 2. Owner Options

> This document selects nothing on the owner's behalf. It only presents choices.

### Option A — Approve design-only M1 train-only protocol
Authorize drafting (markdown design only — NO execution) of an M1 train-only
controlled micro-run protocol. M1 would be the first stage where *real* data is
contemplated, but strictly: train-only, NO validation, NO holdout, NO 2025/2026,
NO optimization/sweep. The protocol itself would still require:
its own external read-only audit, then a separate explicit autonomous owner
approval, then a separate external audit, before any M1 execution. Choosing A
here authorizes only the *design document*, nothing executable.

### Option B — Request a minor patch/remediation before M1 protocol design
If the owner wants any M0 artifact wording, the transparency NOTE, or the
standing W-01/W-02 gates addressed first, authorize a docs-only patch and a
read-only re-audit before the M1 design decision.

### Option C — Do not advance
Hold at the current state. No design, no execution, no further action.

---

## 3. Hard Constraints (apply to all options)
- No M1 execution now.
- No backtest.
- No train.
- No dry-run.
- No validation.
- No holdout.
- No 2025/2026.
- No optimization/sweep.
- No Sub-Batch 1B.
- No parallel writers.
- No production/demo/real/FTMO.
- W-01 and W-02 remain future execution gates until formally resolved.
- M1, if ever designed, stays gated behind: external audit → explicit
  autonomous owner approval → external audit → only then execution.

---

## 4. What This Document Does Not Do
- It does not authorize M1 or any execution.
- It does not select an option on the owner's behalf.
- It does not modify code, tests, or data.
- It does not touch the data vault, validation, holdout, or 2025/2026.
- It asserts no edge, performance, or profitability for BO01/MR02.

---
*End of Owner-Decision Prompt*
