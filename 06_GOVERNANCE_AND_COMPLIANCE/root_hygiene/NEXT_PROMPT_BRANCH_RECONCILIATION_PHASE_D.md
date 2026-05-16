# NEXT PROMPT — BRANCH RECONCILIATION (PHASE D → LAB)

**Priority: B1 — highest-order blocker. Resolve before B2/B3/B5 re-audit.**

## Problem (evidence from Phase E audit, 2026-05-16)
`governance/root-hygiene-20260516` is a narrow root-hygiene branch. It does
**not** contain the actual lab / engine / security state:

- `governance ↔ clean-sync-branch`: **57 commits exist only on
  `clean-sync-branch`** (which == `origin/clean-sync-branch`, fully pushed).
- Those 57 include lab-critical work: `[security] revoke and mask exposed
  telegram token`, `[v40/engine] lockdown definitivo del core`, `[governance]
  P0/P1 lab hardening`, `[governance] block V50B rerun (contamination)`,
  `[governance] data news vault integrity audit`, `[v49.8/r1] final freeze`,
  V39→V50B / R1 research evidence.
- `governance ↔ main`: 34/42 diverged (context only — **do not touch main**).
- F06 research branches (`research/f06-clean-train-only-rerun-20260515`,
  `research/f06-d5-behavior-neutral-telemetry-20260516`) are fully contained
  in governance (right=0) — no action needed for those.

Lab readiness **cannot** be assessed on governance because the engine/security
hardening and research freeze state are not present here.

## Objective
Produce a single integrated branch (or a clear owner decision on the source of
truth) that contains BOTH: (a) the Phase D root canonicalization, AND (b) the
`clean-sync-branch` lab/engine/security/governance state.

## Scope of the next prompt (audit + plan; no destructive ops)
1. Establish source-of-truth policy with the owner: is `clean-sync-branch` the
   trunk and governance a feature to land onto it, or vice-versa?
2. Compute a precise 3-way picture: `git merge-base`, per-path conflict
   forecast (`git merge-tree` / dry-run), and a categorized list of all 57
   commits (security / engine / governance / research-evidence / cloud).
3. Decide integration mechanism with the owner: merge, rebase, or curated
   cherry-pick of the canonicalization onto `clean-sync-branch`. **Do not
   execute** until the owner approves the mechanism in writing.
4. Special care: the Phase D `config.py` remap (`DEFAULT_NEWS_FILE` →
   `DEFAULT_NEWS_FILE_OBSOLETE`) will likely conflict with the
   `clean-sync-branch` config — plan the resolution so active runners are NOT
   left import-broken (ties into B2).
5. Re-run the Phase E integrity audit on the reconciled branch.

## Hard rules
- No `main`. No force push. No merge/rebase/cherry-pick **executed** without
  explicit owner approval of the mechanism. No history rewrite of shared
  branches. Investigate-before-overwrite.
- No backtest / strategy / F06-real / optimization / sweep / validation /
  holdout / 2025-2026 analysis. No data touched. No ZIP workflow.
- Preserve the `[security]` token-revocation commit through any integration —
  never drop it.
