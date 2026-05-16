# PHASE E FINAL PRE-LAB INTEGRITY AUDIT

**Audit date:** 2026-05-16
**Repo:** `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`
**Branch:** `governance/root-hygiene-20260516`
**HEAD at audit:** `e8bf8411fb049e851bc28fdb2ba07ec27dc84de4`
**Auditor mode:** read-only integrity audit (no lab, no backtest, no strategy,
no F06-real, no optimization, no sweep, no validation, no holdout, no 2025/2026).

## 1. Status

**`NOT_READY_BRANCH_RECONCILIATION_BLOCKERS`** (primary)

Concurrent secondary blockers also TRUE:
`NOT_READY_PATH_OR_IMPORT_BLOCKERS`, `NOT_READY_DATA_COMPLETENESS_BLOCKERS`,
`NOT_READY_TEST_REGRESSION_BLOCKERS`.

**Lab green light: NO. This is an explicit RED. No false green light issued.**

## 2. Executive Summary

Root hygiene (the Phase D objective) is genuinely excellent and the F06
pipeline regression suite is genuinely green (119/119). However the repository
is **not ready for lab pre-check** for four independent, evidence-backed
reasons:

1. **Branch reconciliation (meta-blocker).** `governance/root-hygiene-20260516`
   is a *narrow hygiene branch*. `clean-sync-branch` (fully synced with origin)
   contains **57 commits absent from governance**, including security
   remediation, engine lockdown, governance/lab hardening, and the entire
   V39→V50B / R1-freeze research state. Lab readiness **cannot** be assessed on
   a branch that does not contain the actual lab/engine/security state.
2. **Active import breakage.** Phase D's `config.py` remap renamed
   `DEFAULT_NEWS_FILE` → `DEFAULT_NEWS_FILE_OBSOLETE` but did **not** update
   active runners. `import research_lab.light_runner` (a primary entrypoint)
   raises `ImportError`. The prior Phase D "SUCCESS" only ran the F06 subset and
   therefore masked this regression.
3. **Data completeness.** `forex_factory_cache.csv` and `news_eurusd_v2_utc.csv`
   are missing everywhere; the high-precision M1 dukascopy bundle is also
   absent.
4. **Test regression.** `research_lab/tests`: 105/149 pass, **44 fail** across
   IMPORT_BROKEN / PATH_BROKEN / DATA_MISSING / TEST_BROKEN classes.

**Readiness score: 59 / 100** (threshold for lab pre-check ≥ 90).

## 3. Root Strictness Audit

`STRICT_ROOT_CONFIRMED = YES` (with documented exceptions).

Canonical (8) + `.gitignore` all present. `01_CORE_PRODUCTION` exists on disk;
contents gitignored-by-design (production engine/configs/live logs) — not in
`git ls-tree HEAD`, acceptable.

Technical exceptions (documented, not violations): `.git`, `.github`,
`README.md`, `requirements.txt`, `requirements-vps-optional.txt`.

**Root violations: 0.** No loose `.py`/`.csv`/outputs/ZIP in root.

## 4. Git / Branch Integrity

| Comparison | only-in-governance | only-in-OTHER | Verdict |
|---|---|---|---|
| governance ↔ `clean-sync-branch` | 122 | **57** | **RECONCILIATION REQUIRED** |
| governance ↔ `research/f06-clean-train-only-rerun-20260515` | 12 | 0 | Contained — OK |
| governance ↔ `research/f06-d5-behavior-neutral-telemetry-20260516` | 11 | 0 | Contained — OK |
| governance ↔ `main` | 34 | 42 | Diverged — OUT OF SCOPE (no-touch) |

`clean-sync-branch` == `origin/clean-sync-branch` (0/0, fully pushed — this is
real shared work, not local scratch).

Representative unreflected commits on `clean-sync-branch` (not in governance):
- `66768383 [security] revoke and mask exposed telegram token`
- `3f016741 [v40/engine] lockdown definitivo del core v7 v6`
- `b281229e [v40/engine] endurecer bloqueo core sin bypass permanente`
- `d12f52f6 [governance] P0/P1 lab hardening during V50B limited run`
- `786eb3c3 [governance] block V50B rerun due to multi-run_id contamination`
- `cd7b97ba [governance] data news vault integrity audit for v49.7`
- `2322f021 [v49.8/r1] final freeze and postmortem`
- (+ ~50 more across V39–V50B, R1 freeze, cloud/kaggle evidence)

**branch_reconciliation_needed = YES.**

## 5. Python Import / Package Audit

- `import research_lab` → **OK** only with `PYTHONPATH=03_RESEARCH_LAB`
  (fails from bare root — non-canonical resolution, tracked separately).
- `import research_lab.light_runner` → **ImportError: cannot import name
  'DEFAULT_NEWS_FILE' from 'research_lab.config'** (PROOF captured).
- `research_lab/tests` discovery: **Ran 149, FAILED (failures=3, errors=41)** →
  **105 pass / 44 fail**.

Root-cause classification (no failure masked as PASS):

| Class | Evidence | Examples |
|---|---|---|
| `IMPORT_BROKEN_BLOCKER` | `config.py` defines `DEFAULT_NEWS_FILE_OBSOLETE`, not `DEFAULT_NEWS_FILE`; active modules still import old name | `light_runner.py:19`, `news_rebuild.py:20`, `audit_project.py:14`, `audit_level3.py:17`, `tests/test_integration_real_project.py:8`, `tests/test_level3_precision.py:10` |
| `PATH_BROKEN_BLOCKER` | references to **old root `scripts/`** relocated by Phase D `8ee830e6` | `tests/test_h6_paper_shadow_runner.py:11` (`parents[3]/scripts/h6_paper_shadow_runner.py`), `eurusd_ltf_objective_entry_replacement_ecb_autopilot.py:60` (`PROJECT_ROOT/scripts/build_chatgpt_bundle.py`) |
| `DATA_MISSING_BLOCKER` | `FileNotFoundError` | `forex_factory_cache.csv`; high-precision M1 dukascopy bundle (`05_MARKET_DATA_VAULT\legacy_data\data_precision\dukascopy`) |
| `TEST_BROKEN_BLOCKER` | engine/test interface drift + missing test dep | 29× `AttributeError: type object 'LongSignalStrategy' has no attribute 'generate_signal'` (`engine.py:1030`); `ModuleNotFoundError: No module named 'pytest'` (`test_e2e_canonical_flow`) |

## 6. F06 Pipeline Regression

`python -m unittest discover -s 03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/pipelines/f06_evidence_rebuild/tests -p "test_*.py"`
→ **Ran 119 tests — OK** (exit 0, 3.8s). **No F06 regression.** Genuinely green.

## 7. Data Completeness Audit

| File | Expected | Found | Required by | Regenerable? | Blocks lab? |
|---|---|---|---|---|---|
| `canonical_anchor_events.csv` | `05_MARKET_DATA_VAULT/data/official_anchors/out/` | **YES** (68,915 B, 2026-05-16) | anchor pipeline | n/a (present) | No |
| `forex_factory_cache.csv` | `05_MARKET_DATA_VAULT/data/` | **MISSING (everywhere)** | USDJPY news fortress, JPY datasets, news_phase3 | Requires scrape OR owner file (NOT done here) | **YES** |
| `news_eurusd_v2_utc.csv` | `05_MARKET_DATA_VAULT/data/` | **MISSING (everywhere)** | `build_am_grade_news_dataset.py:23`, config `DEFAULT_NEWS_V2_UTC_FILE` | Requires rebuild pipeline (may depend on FF cache) OR owner file | **YES** |
| high-precision M1 dukascopy bundle | `05_MARKET_DATA_VAULT/legacy_data/data_precision/dukascopy` | **MISSING** | high-precision data loader | Requires dukascopy acquisition (NOT done here) | **YES (high-precision scope)** |

No scrape / download / internet / dataset generation performed — existence
checks only.

## 8. Path Reference Audit

- **Active broken refs (non-test):** `light_runner.py`, `news_rebuild.py`,
  `audit_project.py`, `audit_level3.py` (import removed `DEFAULT_NEWS_FILE`);
  `eurusd_ltf_objective_entry_replacement_ecb_autopilot.py:60` and
  `audit_news_engine.py:9` (old root `scripts/` / `data/` paths).
  → **`BLOCKED_PATH_REFERENCES_REMAIN = YES`**
- **Config refs:** `config.py` uses `*_OBSOLETE` names + vault paths (correct,
  but two referenced datasets are physically missing — see §7).
- **Test refs:** broken `DEFAULT_NEWS_FILE` / old `scripts/` imports (see §5).
- **Docs/backup refs:** `config_BACKUP_*`, archived/quarantined — inert.
- `news_phase3_mass_validate.py` still writes to `PROJECT_ROOT(=repo root)
  /reports/...` (latent root re-pollution if run) — carried from Phase D audit,
  tracked in `NEXT_PROMPT_REPORTS_PATH_MIGRATION.md`.

## 9. ZIP / Root Output Policy

- Active ZIPs: **NONE.** No `.zip` in root; no ZIP workflow active.
- Inert ZIPs (gitignored, archival only): 1 in
  `06_GOVERNANCE_AND_COMPLIANCE/quarantine/_LOCAL_QUARANTINE_DO_NOT_COMMIT/`,
  3 in `07_BACKUPS/` (incl. neutralized `.zipbak`). Acceptable.
- Root non-dir files: only `.gitignore`, `README.md`, `requirements.txt`,
  `requirements-vps-optional.txt`. No loose outputs.

## 10. Readiness Score

| Dimension | Max | Score |
|---|---|---|
| Root hygiene | 20 | 19 |
| Git / branch integrity | 15 | 4 |
| Python imports / tests | 20 | 5 |
| F06 pipeline | 15 | 15 |
| Data completeness | 20 | 6 |
| No forbidden actions | 10 | 10 |
| **TOTAL** | **100** | **59** |

Gate (ALL required): score ≥ 90 ✗ · no critical blockers ✗ · no missing
required files ✗ · no path broken refs ✗ · safe tests PASS ✗ (research_lab).
**RESULT: NOT READY.**

## 11. Blockers

| # | Class | Blocker | Fix owner |
|---|---|---|---|
| B1 | BRANCH | `clean-sync-branch` has 57 lab/security/engine/governance commits not in governance branch | `NEXT_PROMPT_BRANCH_RECONCILIATION_PHASE_D.md` |
| B2 | IMPORT | `DEFAULT_NEWS_FILE` removed from `config.py`; active runners (`light_runner` etc.) ImportError | Path/Import remediation (post-reconciliation) |
| B3 | PATH | active refs to relocated root `scripts/` | Path/Import remediation |
| B4 | DATA | `forex_factory_cache.csv`, `news_eurusd_v2_utc.csv`, hi-precision dukascopy bundle missing | `NEXT_PROMPT_DATA_RECOVERY_MISSING_NEWS_FILES.md` |
| B5 | TEST | 44/149 research_lab tests fail (incl. `pytest` missing, engine `generate_signal` drift) | Test-harness remediation (post-reconciliation) |

Non-blockers (genuinely green): root hygiene, F06 pipeline (119/119),
ZIP/root-output policy.

## 12. What Is Still Forbidden

No lab. No backtest. No strategy run. No F06-real. No optimization. No sweep.
No validation process. No holdout process. No 2025/2026 analysis. No data
delete. No scrape/download. No ZIP workflow. No new root files. No `main`.
No force push. No merge. Investigate-before-overwrite remains in force.

## 13. Next Step

1. **`NEXT_PROMPT_BRANCH_RECONCILIATION_PHASE_D.md`** — resolve B1 first; this
   gates everything (the lab/engine/security state lives on `clean-sync-branch`).
2. **`NEXT_PROMPT_DATA_RECOVERY_MISSING_NEWS_FILES.md`** — owner action for B4.
3. Re-run THIS Phase E audit on the reconciled branch; only then evaluate B2/B3/B5.
4. `READY_FOR_LAB_PRECHECK` may **only** be declared when score ≥ 90 with zero
   critical blockers — not before.

## 14. Copy-Paste Summary for ChatGPT

```
PHASE E PRE-LAB AUDIT — RESULT: NOT READY (score 59/100)
Branch audited: governance/root-hygiene-20260516 @ e8bf8411 (clean, pushed)

GREEN:
- Root strictly canonical (8 folders + .gitignore + documented exceptions, 0 violations)
- F06 pipeline tests 119/119 OK (re-verified)
- No active ZIP, no loose root outputs
- Zero forbidden actions performed

RED / BLOCKERS:
- B1 BRANCH: clean-sync-branch has 57 commits NOT in governance
  (security telegram-token revoke, v40 engine lockdown, governance P0/P1
  lab hardening, V50B/R1-freeze state). Governance is a narrow hygiene
  branch and does NOT contain the real lab/engine/security state.
- B2 IMPORT: Phase D renamed DEFAULT_NEWS_FILE -> *_OBSOLETE in config.py
  but did not update active runners. `import research_lab.light_runner`
  -> ImportError. Prior Phase D "SUCCESS" only ran F06 subset, masking this.
- B3 PATH: active refs to relocated root scripts/ (autopilot, h6 runner test).
- B4 DATA: forex_factory_cache.csv MISSING, news_eurusd_v2_utc.csv MISSING,
  high-precision M1 dukascopy bundle MISSING. canonical_anchor_events.csv OK.
- B5 TEST: research_lab/tests 105/149 pass, 44 fail
  (IMPORT_BROKEN + PATH_BROKEN + DATA_MISSING + TEST_BROKEN incl. pytest
  not installed and engine/test `generate_signal` interface drift).

DO NOT declare lab ready. Next:
1) Branch reconciliation (NEXT_PROMPT_BRANCH_RECONCILIATION_PHASE_D.md)
2) Data recovery, owner (NEXT_PROMPT_DATA_RECOVERY_MISSING_NEWS_FILES.md)
3) Re-run Phase E audit on reconciled branch; then fix B2/B3/B5.
```
