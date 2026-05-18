# NEXT PROMPT — OWNER DECIDES AFTER BO01 PHASE A H02 FLOW HARDENING AUDIT V1

This document executes nothing. It does not load data, run code, generate the Phase A-0
script, run a backtest, run Phase A-0 or Phase A-1, run validation/holdout, touch
2025/2026, optimize, sweep, or authorize demo/real/FTMO. It only presents the owner's
decision options following
`BO01_PHASE_A_H02_FLOW_HARDENING_PATCH_EXTERNAL_AUDIT_V1.md`.

## Audit outcome (summary)

- Verdict: **BO01_PHASE_A_H02_FLOW_HARDENING_AUDIT_PASS_WITH_WARNINGS**.
- Blockers: 0. High: 1 (F-01). Medium: 1 (F-02). Low: 2 (F-03, F-04). Info: 3.
- H-02 is correctly converted into a mandatory three-gate technical control
  (Phase A-0 script generation, no data → dedicated read-only script audit → Phase A-1
  execution of the unmodified, SHA256-verified script). H-01 is correctly pre-registered
  as a mandatory pre-Phase-B blocker. Diff scope, no-subjective-acceptance, no-execution,
  and git/output security all pass.
- F-01 (HIGH): the Phase A prompt's Section 1 owner activation phrase was not re-scoped
  to the A-0/A-1 split; it still reads as a single "execute the backtest" authorization.
  The document operatively forbids direct execution (multiple explicit prohibitions plus
  the mandatory hash gate), so this is a latent gate-phrasing inconsistency, not an
  operative permission to execute — but it must be corrected before any activation
  phrase is issued.

## The owner chooses exactly ONE of the following

### Option B — Patch the warnings first (recommended)
Apply, in a normal research/governance branch (markdown-only, no code/data/Python), a
small follow-up micro-patch that:
- splits the Section 1 activation gate into (i) a Phase A-0 phrase that authorizes only
  script generation with no data, and (ii) a separate, later Phase A-1 phrase that
  authorizes execution only after the script audit passes and the SHA256 matches (F-01);
- labels sections 4-13 as Phase A-1 mechanics and cross-references Section 5 ⇄ 2-BIS so
  the data-proof is clearly the content the A-0 script implements and the script audit
  verifies (F-02);
- updates Section 3 base/branch for the A-0/A-1 flow (F-03);
- adds an explicit activation-gate ↔ split consistency check to the next-audit prompt
  (F-04).
Then a short read-only review of that micro-patch. This closes the HIGH warning before
any owner activation phrase exists.

### Option A — Owner decision to generate the Phase A-0 script draft
Authorize generating the Phase A-0 execution/data-proof script draft **without executing
it and without loading any data** (script written to the gitignored local outputs
folder; only the governance draft report + script-audit next-prompt are committable).
Recommended only AFTER Option B closes F-01, so the owner is not issuing or relying on an
ambiguous activation phrase.

### Option C — Pause
Take no further action now. The patch remains audited PASS_WITH_WARNINGS; nothing is
executed; H-01 remains pending pre-Phase-B.

### Option D — Schedule the H-01 data-prep causality audit before Phase B
H-01 (`ema_m15_200` / `atr14` causal construction) remains a pre-registered mandatory
blocker for Phase B and any edge/profitability interpretation. It does not block Phase A
plumbing. Schedule it (do not run it now) before any Phase B or window widening. This is
complementary to B/A/C, not a substitute.

## Recommended sequence

1. Option B (close F-01/F-02/F-03/F-04 via a markdown-only micro-patch + short review).
2. Option A (owner-gated Phase A-0 script generation, no data, no execution).
3. Dedicated read-only audit of the generated Phase A-0 script.
4. Only then, a separate later owner phrase for Phase A-1 (hash-verified execution).
5. Option D (H-01 data-prep causality audit) before any Phase B or window widening.

## Hard constraints carried forward (unchanged)

- No Phase A-0 script generation, no Phase A-1, no direct Phase A execution now.
- No data loading, no real CSV read, no backtest, no formal train.
- No validation, no holdout, no 2025/2026, no optimization/sweep.
- No demo/real/FTMO; no edge/profitability/rentabilidad claims.
- H-01 stays pending and blocks Phase B until its own dedicated read-only audit passes.
- No modification of the audited runner or strategy classes (preserves the W-01 pin).
- No `git add .`; explicit per-file staging only; no history-rewriting git operations.
- This document authorizes none of the above — the owner does, with an explicit phrase.
