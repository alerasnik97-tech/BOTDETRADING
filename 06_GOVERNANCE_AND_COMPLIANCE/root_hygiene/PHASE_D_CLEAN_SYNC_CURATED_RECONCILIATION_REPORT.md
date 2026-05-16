# PHASE D CLEAN-SYNC CURATED RECONCILIATION REPORT

**Date:** 2026-05-16
**Reconciliation branch:** `governance/phase-d-reconciliation-20260516`
(from `governance/root-hygiene-20260516` @ `e15df044`)
**clean-sync source:** `origin/clean-sync-branch` @ `786eb3c3`
**Mode:** curated, non-destructive. No merge, no `--allow-unrelated-histories`,
no force push, no main.

## 1. Status

**`CLEAN_SYNC_RECONCILIATION_PARTIAL_OWNER_REVIEW_REQUIRED`**
(5 low-risk governance docs ported; security-doc port blocked by the canonical
secret-hygiene `.gitignore`; the large majority deferred to owner review.)
Lab remains **NOT authorized**.

## 2. Executive Summary

- A reconciliation branch was created **from governance** (never from
  clean-sync). No history was merged.
- clean-sync-branch is an unrelated orphan history (57 commits, no common
  ancestor). Cherry-pick is structurally unsafe (wholesale tree divergence
  4,229 D / 1,855 A) → **no commit was cherry-picked**.
- Only **file-level docs-only ports** (Method C, `git show`) were used, for
  **secret-free governance/security policy markdown** in brand-new additive
  paths (zero conflict, zero overwrite).
- **Applied:** 5 governance source-of-truth policy docs (from `d65dac5f`).
- **Blocked (notable):** the security token-revocation docs (from `66768383`)
  cannot be committed — the canonical `.gitignore:64 *token*` (a deliberate
  secret-prevention rule) ignores the `token_revocation_cleanup_telegram/`
  path. **Circumventing a secrets-protection ignore is an owner decision and
  was NOT done.**
- Everything else (engine, research evidence, cloud, data/news-vault, root
  duplicates, merge) was classified and deferred — no unsafe action taken.

## 3. Source Branches

| | branch | head | history |
|---|---|---|---|
| base/canonical | `governance/root-hygiene-20260516` | `e15df044` | Phase A–E |
| working | `governance/phase-d-reconciliation-20260516` | `e15df044` (+1 docs commit) | from governance |
| donor (read-only) | `origin/clean-sync-branch` | `786eb3c3` | unrelated orphan, 57 commits, NO common ancestor |

## 4. Commit Inventory (57 clean-sync-only)

| Category | ~count | Examples | Action |
|---|---|---|---|
| RESEARCH_EVIDENCE (r1/v40–v50b runs, gauntlets, sweeps, probes) | ~30 | `406654a3`,`f288af86`,`e2d11d26`,`54436f0e` | SKIP_UNSAFE (result data; validation/holdout/2025-26 risk) |
| GOVERNANCE_DOCS | ~10 | `d65dac5f`,`cd7b97ba`,`ef90af18`,`280c161c`,`786eb3c3` | `d65dac5f`→5 docs **PORTED**; rest OWNER_REVIEW |
| ENGINE_CORE | 2 | `3f016741`,`b281229e` (core lockdown) | SKIP_UNSAFE (no engine w/o specific owner approval) |
| SECURITY_CRITICAL | 1 | `66768383` (telegram token revoke/mask) | docs port **BLOCKED by `*token*` .gitignore** → OWNER_REVIEW |
| LAB_HARDENING | 2 | `d12f52f6` (P0/P1), `39ea4abc` | OWNER_REVIEW (non-invasive only post-review) |
| DATA_NEWS_VAULT | 2 | `cd7b97ba`,`946a750b` | SKIP_UNSAFE (data completeness/provenance) |
| CLOUD_RUNNER | 3 | `ad9069ec`,`7d074bd0`,`8a7b04e8` (kaggle) | SKIP/docs-only deferred |
| ROOT_HYGIENE_DUPLICATE / LEGACY | 2 | `cc7eed4a` (clean start), `e082c40d` (zip restore) | SKIP (gov already resolved root; `e082c40d`=ZIP, forbidden) |
| MERGE | 1 | `6d547537` | SKIP (merge commit) |

## 5. File Inventory (security + governance focus)

`66768383` adds, under `06_GOVERNANCE_AND_COMPLIANCE/security/token_revocation_cleanup_telegram/`:
5 `.md` (policy/decision/confirmation/lockdown/scan-plan), 3 `.csv`, 1 `.txt`
(+ 2 `M` audit md against clean-sync's own tree — not portable).
`d65dac5f` adds 5 clean policy `.md` under
`06_GOVERNANCE_AND_COMPLIANCE/github_source_of_truth_policy/` amid heavy
data/backup/log/secret-inventory noise (not portable as a commit).

Destination paths confirmed **absent + untracked** on governance → ports are
purely additive.

## 6. Candidate Selection

Secret scan of 10 candidate docs: 9 clean; `PRE_COMMIT_SECRET_SCAN_PLAN.md`
matched only a **documented detection regex** (false positive) → gated.
Selected for port: the 9 clean docs. Outcome split by `.gitignore` reality
(see §7/§8).

## 7. Units Applied

| source_commit | method | files | reason | risk | result |
|---|---|---|---|---|---|
| `d65dac5f` | Method C `git show` (docs-only, file-level) | 5 × `06_GOVERNANCE_AND_COMPLIANCE/github_source_of_truth_policy/*.md` (ANTIGRAVITY_OPERATING_RULES_UPDATE, CHATGPT_GITHUB_HANDOFF_POLICY, POST_ZIP_GITHUB_SOURCE_OF_TRUTH_POLICY, ROOT_HYGIENE_POLICY, ZIP_DEPRECATION_POLICY) | preserve valuable governance source-of-truth policy; additive new path; secret-free | LOW | **APPLIED & staged (md only, no secrets, no conflict)** |

## 8. Units Skipped / Blocked

| source | reason | future gate |
|---|---|---|
| `66768383` security `.md` ×5 | canonical `.gitignore:64 *token*` ignores path — cannot commit; circumventing a secrets-protection ignore is an owner call | OWNER_REVIEW: rename path away from `*token*` OR add a scoped, audited gitignore exception |
| `66768383` `.csv`/`.txt` audit artifacts ×4 | binary/secret-risk (token exposure/masking audits) | OWNER_REVIEW + secret inspection |
| `PRE_COMMIT_SECRET_SCAN_PLAN.md` | contains detection regex (false-positive); ultra-conservative gate | OWNER_REVIEW (trivially clearable) |
| ENGINE_CORE `3f016741`,`b281229e` | engine change w/o specific owner approval forbidden | OWNER_REVIEW + engine test gate |
| RESEARCH_EVIDENCE ~30 | research result data; validation/holdout/2025-26 prohibitions | SKIP (not lab-relevant to port) |
| DATA_NEWS_VAULT, CLOUD_RUNNER, ROOT dup, MERGE | data/provenance/duplicate/merge | SKIP / OWNER_REVIEW |
| ~remaining GOVERNANCE_DOCS commits | reference clean-sync's own (unrelated) state | OWNER_REVIEW per-doc |

No conflicts were resolved blindly. No `cherry-pick --abort` needed (no
cherry-pick attempted).

## 9. Tests

- `PYTHONPATH=03_RESEARCH_LAB python -c "import research_lab"` → **OK**
- F06 pipeline unittest → **Ran 119 — OK** (exit 0)
- Broader `research_lab/tests`: **`BROADER_RESEARCH_LAB_KNOWN_BLOCKERS_STILL_STAND`**
  (Phase E: 105/149, active `light_runner` ImportError — not re-run here, not
  faked PASS).

## 10. Security / Data Safety

- `git diff --cached` secret regex (telegram/api/private-key/aws/slack/gh):
  **no matches** (initial heuristic flag was a shell exit-code artifact;
  exact-match re-scan empty).
- Staged content: **md only** — no `.zip/.csv/.parquet/.db/.env/.key/.pem`,
  no data, no quarantine, no caches.
- `secrets_detected=NO`, `data_staged=NO`, `zips_staged=NO`.
- The `*token*` gitignore blocking the security docs is **defense-in-depth
  working as designed** — reported, not bypassed.

## 11. Remaining Phase E Blockers (lab still NOT authorized)

- B2 IMPORT: `light_runner` et al. import removed `DEFAULT_NEWS_FILE`.
- B3 PATH: active refs to relocated root `scripts/`.
- B4 DATA: `forex_factory_cache.csv`, `news_eurusd_v2_utc.csv`, hi-precision
  dukascopy bundle missing.
- B5 TEST: `research_lab/tests` 105/149.
- B6 (new): security token docs blocked by `*token*` gitignore — owner policy.
- B7 (open): 50+ clean-sync commits (engine/research/cloud) need owner review;
  `canonical_anchor_events.csv` provenance unverified.

## 12. Next Step

1. Owner reviews `NEXT_PROMPT_DATA_COMPLETENESS_RECOVERY.md` (B4).
2. Owner decides the `*token*` security-docs gitignore policy (B6) and the
   per-category fate of the deferred clean-sync commits (B7).
3. Resolve B2/B3/B5 on the canonical line.
4. Only then run `NEXT_PROMPT_CLAUDE_FINAL_PRE_LAB_AUDIT.md` (final gate).

## 13. Copy-Paste Summary for ChatGPT

```
PHASE D CLEAN-SYNC CURATED RECONCILIATION
STATUS: PARTIAL_OWNER_REVIEW_REQUIRED. Lab NOT authorized.

- New branch governance/phase-d-reconciliation-20260516 from governance
  (NOT from clean-sync). NO merge, NO unrelated-history merge, NO force.
- clean-sync = unrelated orphan (57 commits, no common ancestor).
  No cherry-pick (structurally unsafe). Only docs-only file ports.
- APPLIED: 5 governance source-of-truth policy .md (from d65dac5f),
  additive, secret-free, F06 still 119/119.
- BLOCKED: security token-revocation docs (66768383) — canonical
  .gitignore *token* ignores the path; NOT bypassed (owner decision).
  Engine/research-evidence/cloud/data commits deferred (owner review).
- Safety: no secrets/data/zip staged (md only). No backtest/strategy/
  F06-real/validation/holdout. No main. No force push.
- Phase E blockers B2/B3/B4/B5 still stand + B6 token-gitignore policy
  + B7 50+ clean-sync commits owner review + anchor csv provenance.

NEXT: owner decisions (data recovery, *token* policy, commit triage),
fix B2/B3/B5, THEN Claude final pre-lab audit. Do not declare lab ready.
```
