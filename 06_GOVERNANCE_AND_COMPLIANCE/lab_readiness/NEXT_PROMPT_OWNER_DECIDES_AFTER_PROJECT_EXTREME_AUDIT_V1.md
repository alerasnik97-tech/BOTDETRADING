# NEXT PROMPT — OWNER DECIDES AFTER PROJECT EXTREME READONLY AUDIT V1

This document does not execute anything. It does not load data, run code, run a backtest,
run Phase A, run validation, run holdout, touch 2025/2026, optimize, sweep, or authorize
demo/real/FTMO. It only presents decision options for the owner following
`PROJECT_EXTREME_READONLY_AUDIT_V1.md`.

## Audit outcome (summary)

- Verdict: **PROJECT_EXTREME_AUDIT_PASS_WITH_WARNINGS**.
- Blockers: 0. High: 2. Medium: 10. Low: 10. Info: 9.
- The W-01 / W-02 / W-03 Phase A prompt warning patch was independently re-verified as
  correctly applied. A separate patch-specific audit is therefore not required.
- The two HIGH findings are about the data-preparation / data-loading surface that the
  audited runner does not cover, not about the runner, strategies, prompt, or chain.

> UPDATE (H-02 flow hardening patch, branch
> `research/bo01-phase-a-h02-flow-hardening-v1-20260518`): the prior recommendation that
> Phase A could proceed once H-02 was "formally accepted" by the owner is **superseded
> and is no longer valid**. H-02 is NOT closed by subjective owner acceptance. It is
> converted into an auditable technical control: an exact execution/data-proof script
> must first be generated (Phase A-0, no data loaded), then pass a dedicated read-only
> audit, and only then may Phase A-1 execute the unmodified, hash-verified script.
> H-01 is now formally pre-registered as a mandatory pre-Phase-B blocker.

## Strict policy on H-02 (superseding the prior version)

- H-02 must NOT be closed by simple owner acceptance.
- H-02 must be converted into an auditable technical control.
- The exact execution / data-proof script (the "loader") must EXIST and PASS a dedicated
  read-only audit BEFORE any real CSV is touched.
- No direct Phase A execution from the prior prompt is permitted. Phase A is now a
  three-gate flow: Phase A-0 (script generation, no data) → script audit → Phase A-1
  (execute the unmodified, hash-verified script only after the audit passes).

## The owner chooses exactly ONE of the following

### Option A — Close H-02 via Phase A-0 script generation + audit (recommended)
Generate the exact Phase A execution/data-proof script (Phase A-0, no data loaded, no
CSV read, no Python executed yet), then run a dedicated destructive read-only audit of
that script. This is how H-02 is closed — as a technical control, not an opinion.

### Option B — Execute Phase A-1 only after the script audit passes
Phase A-1 may run only after Option A's script audit passes, only on the unmodified
script, and only after Phase A-1 recomputes and matches the audited script SHA256
(`BLOCKED_SCRIPT_HASH_MISMATCH` otherwise), under the runner pinned at
`5bdb4bed1f829eb7e8bfe65dc30a6e2f49657d89`, and only with a later explicit owner phrase.

### Option C — Schedule the H-01 data-prep causality audit before Phase B
H-01 (`ema_m15_200` / `atr14` causal construction) is now a pre-registered mandatory
blocker for Phase B and for any edge/profitability interpretation. It does not block
Phase A plumbing. Schedule it (do not run it now) before any Phase B or window widening.

### Option D — Do NOT execute Phase A directly from the current/prior prompt
The earlier "owner accepts H-02 then runs Phase A" path is withdrawn. Any attempt to run
Phase A without Phase A-0 + script audit + Phase A-1 hash check is a blocker.

## Recommended sequence

1. Option A — generate the Phase A-0 script (no data) and produce its draft report +
   next audit prompt. 2. Audit that script (dedicated read-only audit). 3. Option B —
   only then, owner-gated Phase A-1 plumbing run on the hash-verified script. 4. Option C
   — before Phase B: the H-01 data-prep causality audit + parameter pre-registration check.
   Optional hygiene/language pass (M-03/M-04/M-05/M-06/M-07/M-08/M-10/L-01/L-02/L-09) may
   run in a normal research/governance branch at any point before Phase B.

## Hard constraints carried forward (unchanged)

- No validation, no holdout, no 2025/2026, no optimization/sweep, no demo/real/FTMO.
- No edge or profitability claims from Phase A (plumbing only).
- No modification of the audited runner or strategy classes (preserves the W-01 pin).
- No `git add .`; explicit per-file staging only; no history-rewriting git operations.
- This audit, and this document, authorize none of the above — the owner does.
