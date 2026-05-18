# NEXT PROMPT — OWNER DECIDES MICRO-RUN PROTOCOL DESIGN AFTER EXTREME AUDIT V1

## 0. Nature Of This Document

This is an owner-gated decision prompt. It does NOT authorize and does NOT
execute anything. It exists only so the owner can decide how to proceed after
the extreme nightly end-to-end audit. Deciding is not designing; designing is
not running.

## 1. Preconditions

- Extreme end-to-end audit completed:
  `EXTREME_NIGHTLY_END_TO_END_AUDIT_V1.md`
  (status `EXTREME_AUDIT_PASS_WITH_WARNINGS_OWNER_DECISION_ALLOWED`).
- Head commit audited: `025ca8f787691760a356a897f1e43a9391f8e8ea`
  (parent `fdce9603f28e03ba24f92a64235f5a031e758a14`).
- No blockers. Seven warnings recorded: W-01 dirty tree, W-02 tracked output
  debt, W-03 registry missing BO01/MR02 rows, W-04 latent taxonomy/execution-
  plan owner-less micro-run path, W-05 TP-01 lineage traceability, W-06
  reactive anti-self-deception, W-07 language/process/count debt.
- BO01/MR02 code and tests are byte-identical to the already-audited
  `fdce9603`; 85 lightweight tests pass; no edge/performance claim exists.

## 2. Owner Decision Required

The owner must explicitly choose one of:

- A) Approve commissioning a separate, design-only micro-run protocol
  prompt (a document only, to be independently audited; it would NOT run
  a micro-run).
- B) Request a minor governance/hygiene patch FIRST, then revisit the
  design decision. Recommended scope of that patch: add BO01/MR02
  `PRE_REGISTERED` rows to `STRATEGY_RESEARCH_REGISTRY.md` per its
  Maintenance Protocol (W-03); reconcile `STRATEGY_STATUS_TAXONOMY.md` and
  `FIRST_BATCH_EXECUTION_PLAN_V1.md` to require an explicit owner approval
  + external audit gate before any micro-run preflight (W-04); document a
  remediation plan for W-01/W-02; reconcile the TP-01 registry lineage
  (W-05). Markdown/governance only, single writer, separately audited.
- C) Take no further action at this time.

No option here runs anything. Option A produces only a design document to
be audited. Option B produces only governance/markdown changes to be
audited. The auditor's recorded recommendation is **B before A**.

## 3. Hard Constraints (apply to every option)

- owner decision required before any further step.
- no execution of any kind.
- no micro-run.
- no dry-run.
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
- no parallel writers — at most one writer at any time.
- no production / demo / real / FTMO.
- no code, test, data, engine, runner, factory, or
  `strategies/__init__.py` changes (option B may touch only the named
  governance/registry markdown files, single-writer, separately audited).

## 4. If The Owner Chooses Option A — Scope Of The Future DESIGN Prompt

The future design prompt (separate document, independently audited before
any execution) must, at minimum:

- state it is design-only and authorizes no execution.
- explicitly distinguish a micro-run as a PLUMBING preflight (order/stop/
  fill/telemetry) that proves nothing about edge, and forbid any inference
  of edge, performance, or profitability from a micro-run.
- restrict any later micro-run to synthetic or small, controlled, owner-
  approved data only — never holdout, never 2025/2026, never the market-
  data vault, never production data.
- define explicit pass/fail gates BEFORE any execution is proposed.
- require exactly one writer; forbid parallel writers.
- require resolution (or explicit owner-accepted deferral with rationale)
  of W-01/W-02/W-03/W-04 before any micro-run execution.
- require a separate external read-only audit of the design itself, then a
  further separate owner approval, then a further separate external audit
  before any micro-run is ever executed.
- preserve the BO01/MR02 fail-closed Asian-range completeness contract and
  causality (rows strictly before `i`) as invariants.
- forbid any edge / performance / profitability / champion / demo / real /
  FTMO claims and any "perfect / guaranteed / 100% / indestructible"
  language.

## 5. Explicitly Out Of Scope Here

- running or designing the micro-run (only the owner's go/no-go is in scope).
- touching code, tests, data, engine, runner, factory.
- Sub-Batch 1B or any other batch.
- declaring any strategy to have edge, to be profitable, or to be
  production/demo/real/FTMO ready.

## 6. Reminder

A passing extreme audit means the chain is disciplined in outcome and the
next step is safe to *decide*, not that any strategy has edge or is approved
for any laboratory or live execution. Micro-run, dry-run, backtest, formal
train, validation, holdout, 2025/2026, optimization, and sweep all remain
unauthorized until each is separately designed, separately owner-approved,
and separately externally audited.
