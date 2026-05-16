# NEXT PROMPT — APPLY PHASE D RECONCILIATION (curated, non-destructive)

**Precondition:** owner has read `PHASE_D_BRANCH_RECONCILIATION_AUDIT.md` and
decided the source-of-truth policy. Do not start otherwise.

## Established facts (do not re-litigate)
- `governance/root-hygiene-20260516` is **canonical** and **already contains a
  correct, closed Phase D** (`8ee830e6` + audit docs). Phase D itself does
  **not** need re-application.
- `clean-sync-branch` is an **unrelated history** (no common ancestor, local
  AND origin). It holds lab/engine/security work (V39→V50B, R1 freeze,
  `[security] revoke telegram token`, `[v40/engine] core lockdown`) absent from
  governance. It has **no** Phase D/E artifacts.
- A blind merge is **impossible/forbidden** (`--allow-unrelated-histories`).

## Objective
Stand up a safe workspace and a **curated, owner-approved** plan to bring the
*needed* clean-sync-branch content onto the canonical governance line — without
merging unrelated histories and without losing any work.

## Steps (audit/setup + plan; no destructive ops)
1. From `governance/root-hygiene-20260516`, create
   `governance/phase-d-reconciliation-20260516`
   (`git switch -c governance/phase-d-reconciliation-20260516 governance/root-hygiene-20260516`).
   Do **not** branch from clean-sync-branch.
2. Produce a categorized inventory of the 57 clean-sync-branch-only commits:
   `security` / `engine` / `governance` / `research-evidence` / `cloud`.
   For each, decide: cherry-pick candidate, patch-extract candidate, or
   docs-only/skip. **Prioritize the `[security]` token-revocation commit.**
3. For each approved item, prefer **`git cherry-pick -x <sha>`** or
   `git format-patch`→`git apply` onto the new branch, one logical unit at a
   time, running safe tests between units. Resolve conflicts manually
   (investigate-before-overwrite); never blanket-overwrite.
4. **Never** `git merge clean-sync-branch` (unrelated histories). **Never**
   force-push, delete, or rebase shared branches. No `main`.
5. Safe verification after each batch:
   - `PYTHONPATH=03_RESEARCH_LAB python -c "import research_lab"`
   - F06 unittest discovery (expect 119/119)
   - (optionally) `research_lab/tests` to track B5 — classify, never fake PASS.
6. Push the new branch with a plain `git push -u origin
   governance/phase-d-reconciliation-20260516` (non-force). Open a PR
   **into governance**, not main; merge only on explicit owner approval.

## Hard rules
- No backtest / strategy / F06-real / optimization / sweep / validation /
  holdout / 2025-2026 analysis. No engine or trading-logic edits.
- No data regeneration; do not delete or "fix" `canonical_anchor_events.csv`
  (untracked/local-only) — its provenance audit is a separate owner task.
- No ZIP workflow. No new root files. No force push, no unrelated-history
  merge, no branch deletion. Lab stays NOT authorized until Phase E blockers
  clear on the canonical line.
