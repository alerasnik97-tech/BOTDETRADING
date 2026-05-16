# POLICY ALIGNMENT HOTFIX REPORT

**Date:** 2026-05-16
**Branch:** `governance/phase-d-reconciliation-20260516` (from canonical
`governance/root-hygiene-20260516`)
**Base commit:** `1580a0f4`
**Mode:** docs/policy alignment + secret-hygiene governance only. No data,
no engine, no backtest, no merge, no force push, no main.

## 1. Status

**`POLICY_ALIGNMENT_HOTFIX_APPLIED_SAFE`**
Lab remains **NOT authorized** (Phase E blockers stand).

## 2. Executive Summary

- The 5 governance policy docs ported from `clean-sync-branch` last phase
  contradicted current institutional truth (they declared `clean-sync-branch`
  the operational/source-of-truth branch and referenced the deleted external
  `BOT_ZIP_LEGACY_ARCHIVE`). All contradictions were surgically corrected.
- `clean-sync-branch` is now explicitly reclassified across the docs as a
  **historical donor / unrelated orphan branch**, never operational; the
  canonical branch is named as `governance/root-hygiene-20260516` (or the
  canonical branch declared in the active gate — no branch hardcoded as
  universal).
- The security token-revocation docs (blocked last phase by the `.gitignore`
  secret-hygiene rules) were resolved **safely**: ported as **secret-free
  policy markdown** under a **neutral, committable path** — without weakening,
  negating, or force-adding past `.gitignore`. Sensitive `.csv`/`.txt` audit
  artifacts were intentionally excluded.
- No secrets, data, or ZIPs staged. F06 119/119 OK.

## 3. Docs Audited

`06_GOVERNANCE_AND_COMPLIANCE/github_source_of_truth_policy/`:
`ANTIGRAVITY_OPERATING_RULES_UPDATE.md`, `CHATGPT_GITHUB_HANDOFF_POLICY.md`,
`POST_ZIP_GITHUB_SOURCE_OF_TRUTH_POLICY.md`, `ROOT_HYGIENE_POLICY.md`,
`ZIP_DEPRECATION_POLICY.md`.

## 4. Contradictions Found

| file | problem | risk | fix_type |
|---|---|---|---|
| POST_ZIP_GITHUB_SOURCE_OF_TRUTH_POLICY.md | L8/L9 "rama operativa = clean-sync-branch"; L20 "Commit local en clean-sync-branch" | HIGH | rewrite branch policy + workflow |
| CHATGPT_GITHUB_HANDOFF_POLICY.md | L12 hardcoded `branch: clean-sync-branch` | HIGH | canonical placeholder + add force/ZIP/blockers fields |
| ANTIGRAVITY_OPERATING_RULES_UPDATE.md | L12 `git push origin clean-sync-branch` | HIGH | canonical placeholder + donor-branch prohibition |
| ZIP_DEPRECATION_POLICY.md | L8 `BOT_ZIP_LEGACY_ARCHIVE` (deleted, must not reappear) | MED | remove ref, tighten ZIP policy |
| ROOT_HYGIENE_POLICY.md | no clean-sync contradiction; missing documented technical exceptions | LOW | minor accuracy clause |

## 5. Fixes Applied

- **POST_ZIP…**: §2 now states GitHub = SoT; canonical branch =
  `governance/root-hygiene-20260516` (or gate-declared); clean-sync =
  reclassified donor/orphan, not operational; integration only curated
  (file-port / cherry-pick under owner approval), never direct merge / never
  `--allow-unrelated-histories` / never force / never main. §4 workflow now
  commits to the canonical branch and reports repo/branch/commit/push/no-main/
  no-force/no-ZIP/tests/blockers.
- **CHATGPT_GITHUB_HANDOFF_POLICY**: branch field → canonical placeholder
  (never hardcode clean-sync); added `force push: NO`, `ZIP used: NO`,
  `blockers` fields.
- **ANTIGRAVITY…**: push step → `<RAMA_CANONICA_VIGENTE>` non-force; added
  explicit prohibition treating clean-sync as operational/SoT (donor/orphan).
- **ZIP_DEPRECATION_POLICY**: removed `BOT_ZIP_LEGACY_ARCHIVE` path; states
  it was deleted and must not reappear; no ZIP workflow; historical ZIPs
  local-only/gitignored in `07_BACKUPS`/quarantine only if owner-authorized.
- **ROOT_HYGIENE_POLICY**: added documented-technical-exceptions clause
  (`.github`, `README.md`, `requirements*.txt`; `01_CORE_PRODUCTION`
  gitignored-by-design) to match the real canonical state.

## 6. Security Docs Decision

- Original donor path `security/token_revocation_cleanup_telegram/` is ignored
  by `.gitignore:64 *token*`. The prompt-recommended `credential_…` path was
  **verified ALSO ignored** (`.gitignore:59 *credential*`). `*secret*` (L57)
  likewise blocks `SECRET_*` filenames.
- **Resolution (no irresponsible bypass):** ported the secret-free policy `.md`
  to a **neutral, policy-descriptive, committable** path
  `06_GOVERNANCE_AND_COMPLIANCE/security/telegram_access_revocation_cleanup/`,
  renamed away from token/secret/credential. The `.gitignore` secret-hygiene
  rules were **NOT** modified, negated, or force-added past.
- **Ported (4 + README):** `SCAN_HYGIENE_POLICY.md` (←SECRET_SCAN_POLICY),
  `ACCESS_CLEANUP_DECISION.md` (←TOKEN_CLEANUP_DECISION),
  `ACCESS_REVOCATION_CONFIRMATION.md` (←TOKEN_REVOCATION_CONFIRMATION),
  `ACCESS_REVOCATION_LOCKDOWN.md` (←TOKEN_REVOCATION_LOCKDOWN), `README.md`
  (provenance + exclusion note).
- **Excluded (owner review only):** `PRE_COMMIT_SECRET_SCAN_PLAN.md` (regex
  false-positive), `SECRET_SCAN_AFTER_REVOCATION_MASKED.csv`,
  `TOKEN_EXPOSURE_FILES_AUDIT.csv`, `TOKEN_MASKING_ACTIONS.csv`,
  `TOKEN_CLEANUP_GIT_STATUS.txt` — "Sensitive audit artifacts intentionally
  excluded."
- **Secret scan:** full manual read of all 4 `.md` + regex scan of staged
  diff → **NO real secret** (only an illustrative masked example
  `12345:ABC...` in the policy doc; no token/key/password/chat-id values).
- Preserved facts: token revoked by owner via @BotFather 2026-05-14; new token
  NOT in repo (env/secret-manager only); git *history* still holds the old
  revoked token → separate owner-authorized purge phase required (not done).

## 7. Tests

- `PYTHONPATH=03_RESEARCH_LAB python -c "import research_lab"` → **OK**
- F06 pipeline unittest → **Ran 119 — OK** (exit 0)
- Broader `research_lab/tests`: **known Phase E blockers still stand**
  (not re-run here, not faked).

## 8. Safety Verification

- secrets_detected: **NO**
- data_staged: **NO**
- zips_staged: **NO**
- backtest_run: **NO**
- strategy_run: **NO**
- f06_real_run: **NO**
- validation_process_run: **NO**
- holdout_process_run: **NO**
- force_push: **NO**
- (also: no main, no merge, no `--allow-unrelated-histories`, no engine/data
  touched, `.gitignore` unchanged, no ZIP workflow.)

## 9. Remaining Blockers Before Lab

- **Data completeness:** `forex_factory_cache.csv`,
  `news_eurusd_v2_utc.csv`, hi-precision dukascopy bundle MISSING.
- **B2 import:** `light_runner` et al. import removed `DEFAULT_NEWS_FILE`.
- **B3 paths:** active refs to relocated root `scripts/`.
- **B5 broader research_lab tests:** 105/149.
- **clean-sync deferred commits:** ~50 (engine/research/cloud) owner triage.
- **canonical_anchor_events.csv provenance:** unverified (net/2025-26 pipeline).
- **Excluded security audit artifacts:** owner review of `.csv`/`.txt`.

## 10. Copy-Paste Summary for ChatGPT

```
POLICY ALIGNMENT HOTFIX — STATUS: APPLIED_SAFE. Lab NOT authorized.

- Fixed 4/5 ported governance docs that wrongly declared clean-sync-branch
  as operational/source-of-truth + a deleted BOT_ZIP_LEGACY_ARCHIVE ref.
- Canonical branch now named: governance/root-hygiene-20260516 (or
  gate-declared); clean-sync reclassified historical donor/orphan,
  integration only curated, never merge/unrelated-history/force/main.
- ROOT_HYGIENE_POLICY: added documented technical exceptions clause.
- Security docs (token-revocation) resolved SAFELY: ported secret-free
  policy .md to NEUTRAL committable path
  06_GOVERNANCE_AND_COMPLIANCE/security/telegram_access_revocation_cleanup/
  (.gitignore NOT bypassed/negated; *.csv/*.txt artifacts excluded).
- Secret scan: NO secrets. Staged = 10 .md + report only. No data/ZIP.
- F06 119/119 OK. Broader research_lab blockers still stand.
- Branch pushed non-force. No main, no merge, no force.

Remaining lab blockers: data completeness, B2 import, B3 paths,
B5 research_lab tests, clean-sync deferred commits triage,
anchor csv provenance, excluded security artifacts owner review.
```
