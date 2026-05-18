# NEXT PROMPT — OWNER DECIDES SUB-BATCH 1A MICRO-RUN PROTOCOL DESIGN V1

## 0. Nature Of This Document

This is an owner-gated decision prompt. It does NOT authorize and does NOT
execute any micro-run. It exists only so the owner can decide whether to
commission a separate, independently audited micro-run protocol DESIGN for
Sub-Batch 1A (BO01/MR02). Designing a protocol is not running it.

## 1. Preconditions

- The Sub-Batch 1A blocker patch passed external read-only audit:
  `SUBBATCH_1A_BLOCKER_PATCH_EXTERNAL_AUDIT_V1.md`
  (status `AUDIT_PASS_SUBBATCH_1A_BLOCKER_PATCH_READY_FOR_OWNER_REVIEW`).
- Audited commit: `fdce9603f28e03ba24f92a64235f5a031e758a14`.
- Two non-blocking, pre-existing repository warnings (W-01 dirty tree,
  W-02 tracked output debt) are documented and are NOT in scope here.

## 2. Owner Decision Required

The owner must explicitly choose one of:

- A) Commission a separate micro-run PROTOCOL DESIGN prompt (design only,
  no execution, to be independently audited before anything runs).
- B) Request a minor documentation/wording follow-up first (e.g. the
  optional report-tally wording note I-01) and defer the protocol-design
  decision.
- C) Take no further action at this time.

No option in this prompt runs anything. Option A produces only a design
document to be audited; it does not start a micro-run.

## 3. Hard Constraints (apply to every option)

- owner decision required before any further step.
- no execution yet — design only.
- no micro-run execution.
- no dry-run execution.
- no formal backtest.
- no formal train.
- no validation.
- no holdout.
- no 2025/2026 data.
- no optimization.
- no sweep.
- no grid search.
- no walk-forward.
- no Sub-Batch 1B.
- no code, test, data, engine, runner, registry, factory, or
  `strategies/__init__.py` changes.
- no parallel writers — at most one writer at any time.

## 4. If The Owner Chooses Option A — Scope Of The Future DESIGN Prompt

The future design prompt (a separate document, to be written and then
independently audited before any execution) must, at minimum:

- state that it is design-only and authorizes no execution.
- restrict any later micro-run to synthetic or small, controlled,
  owner-approved data only — never holdout, never 2025/2026, never the
  market-data vault, never production data.
- define explicit pass/fail gates BEFORE any execution is proposed.
- require exactly one writer; forbid parallel writers.
- require a separate external read-only audit of the design itself
  before any execution prompt may be drafted.
- require a further, separate owner approval and a further, separate
  external audit before any micro-run is ever executed.
- preserve the BO01/MR02 fail-closed Asian-range completeness contract
  and causality (rows strictly before `i`) as invariants.
- forbid any edge, performance, profitability, champion, demo, real, or
  FTMO claims.

## 5. Explicitly Out Of Scope Here

- running the micro-run.
- writing the micro-run protocol itself (only the owner's go/no-go on
  whether to later commission that design is in scope).
- touching code, tests, data, engine, runner, registry, factory.
- resolving the pre-existing W-01/W-02 repository warnings.
- Sub-Batch 1B or any other batch.

## 6. Reminder

Passing the blocker-patch audit means the declared blockers were corrected
and covered by tests. It does not mean any strategy has edge, is profitable,
or is approved for any laboratory or live execution. Micro-run, dry-run,
backtest, formal train, validation, holdout, 2025/2026, optimization, and
sweep all remain unauthorized until each is separately designed, separately
owner-approved, and separately externally audited.
