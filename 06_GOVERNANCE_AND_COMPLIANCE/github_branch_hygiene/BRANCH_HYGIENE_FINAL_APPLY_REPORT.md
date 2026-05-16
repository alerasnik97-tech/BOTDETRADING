# BRANCH HYGIENE FINAL APPLY REPORT

## 1. Status

BRANCH_HYGIENE_PARTIAL_OWNER_REVIEW_REQUIRED

Operational skeleton blocker: NO, if owner accepts the remaining protected branches as non-blocking historical/PR surfaces.

Residual branch-owner review: YES.

Reason: the canonical branch is clear and current, and the highest-risk contaminated branch was archived and deleted. However, several remote branches remain alive because they either have open PRs, divergent history, or unique historical research/governance evidence. No unsafe delete was forced.

## 2. Executive Summary

This pass audited GitHub branch hygiene before Priority A skeleton work. It did not touch main, did not merge, did not rebase, did not force push, did not modify strategy/engine/data code, and did not run backtests or strategy execution.

Confirmed canonical branch:

- `governance/claude-strategy-intake-audit-20260516`
- head: `519e3054c47dfa8ce1cbee4b3cbd2b19527517a4`
- subject: `docs: re-arbitrate final EURUSD queue post-Grok recovery audit`

Remote branches before cleanup: 21.
Remote branches after cleanup: 13.

Applied cleanup:

- 8 legacy tags created and pushed.
- 8 remote branches deleted after tag/PR safety checks.
- PR #4 closed because it was already explicitly marked `SUPERSEDED` by PR #5.
- No other PR was closed.

Conservative stop points:

- `governance/root-strict-final-pass-20260516` is not an ancestor of the canonical branch. It contains a small unique docs commit, so it was kept for owner review instead of being deleted.
- Divergent governance/research branches with unique evidence were kept.
- Local stale branches that required `git branch -D` were not deleted.
- Pre-existing unstaged research-intake working tree changes were detected before this phase and left untouched.

## 3. Canonical Branch

Selected:

- `governance/claude-strategy-intake-audit-20260516`

Expected head:

- `519e3054c47dfa8ce1cbee4b3cbd2b19527517a4`

Observed local head:

- `519e3054c47dfa8ce1cbee4b3cbd2b19527517a4`

Observed remote head:

- `519e3054c47dfa8ce1cbee4b3cbd2b19527517a4`

Recent canonical log:

- `519e3054 docs: re-arbitrate final EURUSD queue post-Grok recovery audit`
- `63643109 docs: arbitrate final EURUSD strategy implementation queue`
- `8dd92a67 AUDIT: Independent Claude Opus 4.7 audit of Gemini strategy intake`
- `51e69e72 RESEARCH: Institutional Strategy Backlog Intake (25 Hypotheses) - Workstream A Ready`
- `e462a914 docs: coordinate strategy intake and F06 adapter workstreams`

Required ancestry checks:

| Candidate branch | Ancestor of canonical | Decision |
|---|---:|---|
| `governance/smoke-incident-and-strategy-intake-prep-20260516` | YES | tagged and deleted |
| `governance/engine-base-preflight-fix-v3-20260516` | YES | tagged and deleted |
| `governance/root-strict-final-pass-20260516` | NO | owner review required |
| `clean-sync-branch` | YES | tagged and deleted |

Canonical decision:

- `CANONICAL_CURRENT = governance/claude-strategy-intake-audit-20260516`
- `clean-sync-branch` is no longer operational and was retired through tag plus delete.
- Root strictness is physically present in the current checkout, but the historical `root-strict-final-pass` branch head was not deleted because its exact commit is not in canonical ancestry.

## 4. Branch Count Before / After

Remote branch count excludes `origin/HEAD`.

| Metric | Count |
|---|---:|
| Remote branches before | 21 |
| Remote branches after | 13 |
| Remote branches deleted | 8 |
| Open PRs before | 5 |
| Open PRs after | 4 |
| PRs closed in this pass | 1 |

Final remote branches:

- `origin/agent/research-manipulante4-sweep-quality`
- `origin/governance/claude-strategy-intake-audit-20260516`
- `origin/governance/engine-base-preflight-fix-20260516`
- `origin/governance/parallel-research-lab-failure-triage-20260516`
- `origin/governance/root-strict-final-pass-20260516`
- `origin/main`
- `origin/research/eurusd-daytime-strategy-01`
- `origin/research/f06-clean-train-only-rerun-20260515`
- `origin/research/f06-d5-behavior-neutral-telemetry-20260516`
- `origin/research/f06-evidence-rebuild-foundation-v2-20260515`
- `origin/research/pre-claude-blocker-remediation-20260515`
- `origin/research/v50b-evidence-reconciliation-20260515`
- `origin/research/v50b-train-only-rerun-single-writer-20260515`

## 5. Branches Kept

| Branch | Classification | Reason |
|---|---|---|
| `main` | KEEP_MAIN_PROTECTED | Protected base. Not touched. |
| `governance/claude-strategy-intake-audit-20260516` | KEEP_CANONICAL_CURRENT | Current final arbitration branch at `519e3054`. |
| `research/f06-d5-behavior-neutral-telemetry-20260516` | PROTECT_UNTIL_PR_DECISION | Open PR #7. |
| `research/f06-clean-train-only-rerun-20260515` | PROTECT_UNTIL_PR_DECISION | Open PR #6. |
| `research/f06-evidence-rebuild-foundation-v2-20260515` | PROTECT_UNTIL_PR_DECISION | Open PR #5. |
| `research/pre-claude-blocker-remediation-20260515` | PROTECT_UNTIL_PR_DECISION | Open PR #3. |
| `governance/root-strict-final-pass-20260516` | OWNER_REVIEW_REQUIRED | Not ancestor of canonical; unique docs commit. |
| `governance/engine-base-preflight-fix-20260516` | OWNER_REVIEW_REQUIRED | Divergent branch; v2/v3 were archived/deleted but this exact head is not ancestor. |
| `governance/parallel-research-lab-failure-triage-20260516` | OWNER_REVIEW_REQUIRED | Divergent governance evidence; not deleted. |
| `research/v50b-evidence-reconciliation-20260515` | OWNER_REVIEW_REQUIRED | Historical V50B evidence branch; unique divergent history. |
| `research/v50b-train-only-rerun-single-writer-20260515` | OWNER_REVIEW_REQUIRED | Historical V50B rerun branch; unique divergent history. |
| `research/eurusd-daytime-strategy-01` | OWNER_REVIEW_REQUIRED | Historical research branch; local worktree marker present, not deleted. |
| `agent/research-manipulante4-sweep-quality` | OWNER_REVIEW_REQUIRED | Historical Manipulante branch with unique history. |

Local-only branches still present were not counted as remote GitHub branches. Branches requiring `git branch -D` were preserved.

## 6. Branches Tagged

All tags below were created as annotated tags and pushed to origin.

| Tag | Archived ref |
|---|---|
| `legacy/clean-sync-branch-retired-20260516` | `origin/clean-sync-branch` |
| `legacy/engine-base-preflight-fix-v2-20260516` | `origin/governance/engine-base-preflight-fix-v2-20260516` |
| `legacy/engine-base-preflight-fix-v3-20260516` | `origin/governance/engine-base-preflight-fix-v3-20260516` |
| `legacy/phase-d-reconciliation-20260516` | `origin/governance/phase-d-reconciliation-20260516` |
| `legacy/root-hygiene-20260516` | `origin/governance/root-hygiene-20260516` |
| `legacy/smoke-incident-intake-prep-20260516` | `origin/governance/smoke-incident-and-strategy-intake-prep-20260516` |
| `legacy/v50b-cost-hardening-clean-20260515` | `origin/research/v50b-cost-hardening-clean-20260515` |
| `legacy/f06-evidence-rebuild-foundation-v1-superseded-20260515` | `origin/research/f06-evidence-rebuild-foundation-20260515` |

## 7. Branches Deleted

Remote branches deleted with `git push origin --delete`:

- `clean-sync-branch`
- `governance/engine-base-preflight-fix-v2-20260516`
- `governance/engine-base-preflight-fix-v3-20260516`
- `governance/phase-d-reconciliation-20260516`
- `governance/root-hygiene-20260516`
- `governance/smoke-incident-and-strategy-intake-prep-20260516`
- `research/v50b-cost-hardening-clean-20260515`
- `research/f06-evidence-rebuild-foundation-20260515`

Local branches deleted with `git branch -d`:

- `clean-sync-branch`
- `governance/engine-base-preflight-fix-v2-20260516`
- `governance/engine-base-preflight-fix-v3-20260516`
- `governance/phase-d-reconciliation-20260516`
- `governance/root-hygiene-20260516`
- `governance/smoke-incident-and-strategy-intake-prep-20260516`
- `research/v50b-cost-hardening-clean-20260515`

Local branch not deleted because Git required `-D`:

- `research/f06-evidence-rebuild-foundation-20260515`

Local test branch not deleted because Git required `-D`:

- `research/push-test-20260515`

No `git branch -D` was used.

## 8. Branches Retired Do Not Use

- `clean-sync-branch`

Action:

- Archived as `legacy/clean-sync-branch-retired-20260516`.
- Deleted from origin.
- Deleted locally with `git branch -d`.

Decision:

- `RETIRE_DO_NOT_USE`
- Do not use as a base for skeleton implementation.

## 9. PR Actions

GitHub CLI was not available in this environment. PR audit and PR action used the GitHub connector.

| PR | State after | Head branch | Base branch | Action |
|---:|---|---|---|---|
| #7 | open | `research/f06-d5-behavior-neutral-telemetry-20260516` | `research/f06-clean-train-only-rerun-20260515` | Keep open active. |
| #6 | open | `research/f06-clean-train-only-rerun-20260515` | `research/f06-evidence-rebuild-foundation-v2-20260515` | Keep open active. |
| #5 | open | `research/f06-evidence-rebuild-foundation-v2-20260515` | `research/pre-claude-blocker-remediation-20260515` | Keep open active. |
| #4 | closed | `research/f06-evidence-rebuild-foundation-20260515` | `research/pre-claude-blocker-remediation-20260515` | Closed as superseded by #5 after tag archive. |
| #3 | open | `research/pre-claude-blocker-remediation-20260515` | `main` | Keep open active. |

Closed PR detail:

- PR #4 already stated `SUPERSEDED` by PR #5.
- Added PR comment documenting archive tag and closure basis.
- Closed without merge.

No other PR was closed.

## 10. Owner Review Remaining

Remote branches requiring owner review:

- `governance/root-strict-final-pass-20260516`
- `governance/engine-base-preflight-fix-20260516`
- `governance/parallel-research-lab-failure-triage-20260516`
- `research/v50b-evidence-reconciliation-20260515`
- `research/v50b-train-only-rerun-single-writer-20260515`
- `research/eurusd-daytime-strategy-01`
- `agent/research-manipulante4-sweep-quality`

Open PR branches requiring PR decision:

- `research/pre-claude-blocker-remediation-20260515`
- `research/f06-evidence-rebuild-foundation-v2-20260515`
- `research/f06-clean-train-only-rerun-20260515`
- `research/f06-d5-behavior-neutral-telemetry-20260516`

Local-only cleanup requiring owner review:

- `research/f06-evidence-rebuild-foundation-20260515` was archived/deleted remotely and PR #4 was closed, but local `git branch -d` refused because it is not fully merged.
- `research/push-test-20260515` also refused `git branch -d`.
- No forced local deletion was used.

## 11. Root Strictness Check

Root listing after cleanup:

- `.git`
- `.github`
- `.gitignore`
- `01_CORE_PRODUCTION`
- `02_INCUBATION_STAGING`
- `03_RESEARCH_LAB`
- `04_INFRASTRUCTURE_ENGINEERING`
- `05_MARKET_DATA_VAULT`
- `06_GOVERNANCE_AND_COMPLIANCE`
- `07_BACKUPS`
- `08_CLOUD_FREE_RUN_LAB`

Root strict result:

- strict_ok: YES
- allowed technical exception: `.github`
- forbidden ZIP in root: NO
- loose CSV in root: NO
- loose PY in root: NO
- loose parquet in root: NO
- scratch/temp/output root items: NO

## 12. Safety Verification

| Control | Result |
|---|---|
| main_touched | NO |
| force_push | NO |
| merge_run | NO |
| rebase_run | NO |
| code_modified | NO by this phase |
| data_modified | NO |
| backtest_run | NO |
| strategy_run | NO |
| f06_real_run | NO |
| validation_run | NO |
| holdout_used | NO |
| 2025_2026_used | NO |
| git_add_dot_used | NO |
| zip_created | NO |
| engine_modified | NO by this phase |
| strategy_modified | NO by this phase |

Safe import checks:

- `research_lab_import`: PASS, `research_lab OK`
- `strategy_registry`: PASS, `63`
- `engine_import`: PASS, `engine OK`

Pre-existing working tree note:

- `git status --short` was dirty before branch hygiene started.
- The dirty files were under `03_RESEARCH_LAB/strategy_research_intake/external_research_20260516/...`.
- They were not staged, modified, reverted, or committed by this branch hygiene pass.

## 13. Copy-Paste Summary for ChatGPT

BRANCH_HYGIENE_PARTIAL_OWNER_REVIEW_REQUIRED

Canonical branch is `governance/claude-strategy-intake-audit-20260516` at `519e3054c47dfa8ce1cbee4b3cbd2b19527517a4`.

Remote branches were reduced from 21 to 13. `clean-sync-branch` was archived as `legacy/clean-sync-branch-retired-20260516` and deleted from origin. Eight legacy tags were created and pushed. Eight remote branches were deleted. PR #4 was closed as already superseded by PR #5. PRs #3, #5, #6, and #7 remain open and protected.

No main touch, no force push, no merge, no rebase, no code/data/engine/strategy edits, no backtest, no strategy run, no F06 real, no validation, no holdout, no 2025/2026, no `git add .`, no ZIP.

Remaining owner review branches are preserved because they have open PRs, divergent history, or unique historical evidence. Skeleton work should use only the canonical branch and only the final Priority A list:

1. MR-01 Anchor Elastic
2. MR-02 VWAP Stretch Reversion
3. TP-01 London-NY Momentum Pullback, reformulated
4. VE-ORB Volatility Expansion

Do not include VE-01, SD-01, or ED-01 in skeleton implementation.
