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

## The owner chooses exactly ONE of the following

### Option A — Patch blockers / high risks first
There are no blockers. "High risks" here are H-01 (precomputed-feature causality, a
pre-Phase-B concern) and H-02 (the optional, unaudited data-loader/data-proof code that
Phase A will exercise). Choosing A means: do a read-only audit of the data-loader/data-proof
code path and the data-prep causal alignment of `ema_m15_200`/`atr14` before anything else.
Best if the owner wants maximum assurance before any real-data contact.

### Option B — Audit the Phase A warning patch specifically
Not recommended as a standalone step: this extreme audit already re-verified W-01/W-02/W-03
two independent ways. Selecting B would duplicate completed work. Listed only for completeness.

### Option C — Execute Phase A later (recommended path), conditioned
Proceed toward Phase A **only** when ALL of the following hold:
1. The owner has formally accepted or remediated H-02 and the data-policy MEDIUMs
   (M-05 unprotected `data_candidates_2022_2025`/`data_free_2020`; M-06 fragile
   `*_DO_NOT_COMMIT*` ignore; M-09 protocol vs prompt SHA256 mismatch).
2. An H-01 data-prep causality audit is scheduled before Phase B (not necessarily before
   Phase A, since Phase A is plumbing-only and draws no edge conclusions).
3. The owner issues, as an autonomous declaration, the exact Phase A activation phrase
   defined in `BO01_PHASE_A_TRAIN_ONLY_REALDATA_BACKTEST_EXECUTION_PROMPT_DRAFT_V1.md`
   (reproduced below for reference only — quoting it here does NOT authorize execution):

   "AUTORIZO EJECUTAR PHASE A BO01 TRAIN-ONLY REAL-DATA BACKTEST, VENTANA 2015-01-05 A
   2015-01-09, SOLO TRAIN-ONLY, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026, SIN
   OPTIMIZATION/SWEEP, SIN DEMO/REAL/FTMO Y SIN EDGE CLAIMS."

4. Phase A runs on the unmodified runner pinned at `5bdb4bed1f829eb7e8bfe65dc30a6e2f49657d89`
   and on the verified Phase A prompt with no edits.

### Option D — Freeze and run the hygiene/language pass first
Freeze Phase A. First execute (in a normal research/governance branch, not this read-only
audit) the repo-hygiene + sober-language remediation: gitignore hardening (M-05/M-06),
data/output-in-git decision (M-03/M-04), provenance/anchor fixes (L-01/L-02), language
pass (M-07), stale worktree/branch pruning (M-10/L-09), and the Telegram remediation
consolidation (M-08). Then return to Option C.

## Recommended sequence

1. Now: owner records a decision on H-02 and the data-policy MEDIUMs (accept-with-rationale
   or remediate). 2. Optionally D for hygiene/language. 3. Then C: owner-gated Phase A
   plumbing run. 4. Before Phase B: H-01 data-prep causality audit + parameter
   pre-registration evidence check.

## Hard constraints carried forward (unchanged)

- No validation, no holdout, no 2025/2026, no optimization/sweep, no demo/real/FTMO.
- No edge or profitability claims from Phase A (plumbing only).
- No modification of the audited runner or strategy classes (preserves the W-01 pin).
- No `git add .`; explicit per-file staging only; no history-rewriting git operations.
- This audit, and this document, authorize none of the above — the owner does.
