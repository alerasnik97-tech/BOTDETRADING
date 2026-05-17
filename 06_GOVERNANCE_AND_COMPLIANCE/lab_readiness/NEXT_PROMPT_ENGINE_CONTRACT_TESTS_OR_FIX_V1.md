# NEXT PROMPT — ENGINE/STRATEGY CONTRACT TESTS (TEST-ONLY, FAIL-SAFE)

> Created because `ENGINE_STRATEGY_CONTRACT_AUDIT_VEORB_V1.md` confirmed a real
> **systemic gap** (no universal no-lookahead test, no zero-activity sentinel,
> O(N²) signal pattern uncaught, undocumented `signal()` contract). This is a
> future prompt only. It is NOT authorization to act now.

## Why this is needed (evidence)

- Engine passes the **entire multi-year frame** to `signal(frame, i, params)`
  with no causal sandbox (`engine.py:725`); no engine-level lookahead guard.
- No timeout / no signal-density check; a strategy emitting 0 signals produces a
  clean gate-passing dossier indistinguishable from "regime obsolete"
  (VE-ORB's exact shape).
- Only no-lookahead test is TP01-specific
  (`tests/test_tp01_performance_equivalence.py:128`).
- VE-ORB `signal()` is O(N²) (per-call full-frame ATR recompute + O(i) OR scan).

## Objective of the future task

Add **lightweight, read-only / test-only** safeguards so the *next* strategy
batch cannot silently leak the future or silently die undetected. **Do NOT
"fix" or revive VE-ORB.** VE-ORB stays archived/rejected.

## Mandatory safety rules for the future task

- NO backtest. NO `formal_train_runner --execute`. NO validation. NO holdout.
  NO 2025/2026. NO optimization/sweep/walk-forward.
- NO modification of `engine.py`, `data_loader.py`, runner, `report.py`,
  `metric_reconciliation.py`, cost profiles, or any strategy **behavior**.
  (New *test* files only; engine/strategy code changes require a SEPARATE,
  explicitly-authorized prompt and only if a confirmed bug — none found here.)
- NO `git add .`, NO force push, NO main, NO destructive git.
- NO ZIP. NO touching production / CORE_PRODUCTION / data vault / holdout.
- New tests must use **tiny synthetic in-memory frames** (no market data IO).

## Scope of allowed work (tests only)

1. **Universal no-lookahead contract test:** for every registered strategy,
   on a small synthetic frame, assert that mutating bars at `> i` does not
   change `signal(frame, i, params)` (generalize the TP01 pattern; tolerate
   strategies that legitimately return None).
2. **Causality sandbox test:** assert strategies only read `frame.iloc[:i+1]`
   (e.g., feed a frame whose `> i` rows are NaN/poisoned and assert no error /
   no signal change).
3. **Zero-activity sentinel (test + optional telemetry):** a check that flags a
   profile producing ~0 signals over the period so it cannot be silently
   labeled "regime obsolete" without an explicit human note.
4. **Performance smoke test:** assert `signal()` per-call cost is ~O(1)/O(window)
   on a synthetic frame (catch O(N) per-call / O(N²) patterns) — *report only*,
   do not auto-fail legacy strategies; gate only new ones.
5. **Written `signal()` contract** doc + optional `typing.Protocol` (no behavior
   change): inputs, the "read only `≤ i`" rule, return schema.
6. **Pin effective timeframe**: a test asserting the runner/loader/manifest
   agree on the strategy timeframe (close the `target_timeframe="M1"` vs
   M5/M15 traceability gap) — assertion only, no pipeline change.

## Deliverables of the future task

- New test files under `03_RESEARCH_LAB/research_lab/tests/`.
- A short markdown note summarizing coverage added.
- Commit/push only those test files + the note (explicit `git add`, never `.`).

## Explicit non-goals

- Not reviving, optimizing or re-running VE-ORB.
- Not changing engine/strategy behavior.
- Not approving any strategy for validation/holdout/production.

## Trigger

Run this future task **before** any large new strategy batch / new family
pipeline. If no new batch is planned, this remains a backlog item; VE-ORB
archival does not depend on it.
