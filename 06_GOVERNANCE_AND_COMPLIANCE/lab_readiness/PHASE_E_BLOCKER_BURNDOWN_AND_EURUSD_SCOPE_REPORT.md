# PHASE E BLOCKER BURNDOWN AND EURUSD SCOPE REPORT

Date: 2026-05-16
Branch: governance/phase-d-reconciliation-20260516
Base commit: 62901434c4e439e87f1892fb6c9bdd82111c2948
Mode: audit plus compatibility/test-architecture cleanup only.

## 1. Status

`PHASE_E_BURNDOWN_PARTIAL_OWNER_REVIEW_REQUIRED`

EURUSD core lab is blocked. The missing news/Jpy/high-precision assets are not the direct core blocker, but EURUSD prepared OHLCV is missing at the configured loader path. In addition, broader research_lab tests are still non-green after honest cleanup, so no lab authorization is possible.

## 2. Executive Summary

- Branch/head/process precheck passed: correct branch, expected commit, origin aligned, no active python research process.
- The four known missing/non-certified assets were segmented:
  - `forex_factory_cache.csv`: soft blocker for Jpy/news rebuild, not EURUSD core.
  - `news_eurusd_v2_utc.csv`: soft/legacy blocker for legacy news rebuild, not EURUSD core.
  - hi-precision dukascopy M1 bundle: soft blocker for high-precision module, not prepared-OHLCV core.
  - `canonical_anchor_events.csv`: present with 169 lines and SHA256 `1E7EB737163941A41C37ECEC824BF8588EABB7D3C5AA69DCE2B75C756D1AB556`, but provenance remains governance-unverified.
- A direct EURUSD core blocker was discovered: `DEFAULT_DATA_DIRS` points to empty prepared folders under `05_MARKET_DATA_VAULT/eurusd_data/*/prepared`.
- Raw EURUSD tick material exists under `05_MARKET_DATA_VAULT/BOT_MARKET_DATA/tick/EURUSD` (3548 files, 9,210,754,686 bytes, 137 monthly parquet files plus pilot), but `research_lab.data_loader.load_prepared_ohlcv` does not consume it.
- Compatibility fixes were applied without touching engine/trading/strategy logic.
- Broader research_lab tests improved from import/path/data errors to a clearer state, but remain failing: 164 run, 16 failures, 9 errors, 13 skipped.
- One ignored legacy ZIP inside the repo was moved outside the project to `C:\Users\alera\Desktop\Bot\BOT_ZIP_LEGACY_ARCHIVE\` and documented in `PHASE_E_ZIP_LEGACY_MOVE_MANIFEST.md`.

## 3. EURUSD Core Dependency Map

| component | required_for_eurusd_lab | depends_on_forex_factory_cache | depends_on_news_eurusd_v2_utc | depends_on_dukascopy | depends_on_canonical_anchor_events | depends_on_relocated_scripts | risk | decision |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `research_lab` package | YES | NO | NO | NO | NO | NO | import drift was present | FIXED |
| `research_lab.light_runner` | YES/entrypoint | NO | default constant only | NO | NO | NO | old `DEFAULT_NEWS_FILE` import | FIXED |
| `research_lab.audit_level3` | NO/core optional audit | YES via legacy defaults | YES via legacy defaults | NO | YES/summary | NO | old config symbols | FIXED_IMPORT_ONLY |
| `data_loader.load_prepared_ohlcv` | YES | NO | NO | NO | NO | NO | prepared CSVs missing | HARD_BLOCKER |
| EURUSD prepared OHLCV | YES | NO | NO | NO | NO | NO | absent in configured dirs | HARD_BLOCKER |
| Raw EURUSD tick vault | possible source only | NO | NO | NO | NO | NO | present but not loader-wired | OWNER_DECISION_REQUIRED |
| `engine.run_backtest` | YES later | NO | NO | optional via news mask only | NO | NO | imports, but tests still fail | MUST_TRIAGE_BEFORE_LAB |
| `STRATEGY_REGISTRY` | YES later | NO | NO | NO | NO | NO | imports 63 strategies | PASS |
| `EngineConfig` cost model | YES | NO | NO | NO | NO | NO | import OK | PASS |
| NY session helpers/splitters | YES later | NO | NO | NO | NO | NO | code present, not executed as lab | PASS_IMPORT_ONLY |
| EURUSD canonical news fortress_v3 | OPTIONAL | NO | NO | NO | anchor-derived | NO | present but source approval is governance-scoped | OPTIONAL_DISABLED_UNTIL_APPROVED |
| USDJPY news fortress builder | NO | YES | NO | NO | NO | NO | raw FF cache missing | SOFT_BLOCKER |
| high-precision loader | NO for prepared core | NO | NO | YES | NO | NO | dukascopy bundle missing | SOFT_BLOCKER |
| legacy ECB autopilot | NO | NO | YES/optional | YES | NO | YES | old ZIP/root script refs | LEGACY_BLOCKER |
| F06 evidence pipeline | adjacent regression suite | NO | NO | NO | NO | pipeline-local scripts only | 119/119 pass | PASS |

## 4. Missing Data Classification

| asset | current_status | required_for_eurusd_core | required_for_news_filter_lab | required_for_jpy_lab | required_for_high_precision_lab | can_be_deferred | lab_decision | safe_next_step |
|---|---|---:|---:|---:|---:|---:|---|---|
| `05_MARKET_DATA_VAULT/data/forex_factory_cache.csv` | MISSING | NO | YES for raw rebuild | YES | NO | YES | SOFT_BLOCKER_OPTIONAL_MODULE | keep Jpy/news rebuild blocked or owner-supply real file |
| `05_MARKET_DATA_VAULT/data/news_eurusd_v2_utc.csv` | MISSING | NO | YES for legacy AM rebuild | NO | NO | YES | SOFT_BLOCKER_OPTIONAL_MODULE / LEGACY_BLOCKER | use only if owner wants news rebuild; do not fake |
| hi-precision M1 dukascopy bundle | MISSING: `legacy_data/data_precision*/dukascopy` empty | NO for prepared core | NO | NO | YES | YES | SOFT_BLOCKER_OPTIONAL_MODULE | keep high-precision mode blocked until real bundle supplied |
| `canonical_anchor_events.csv` | PRESENT, 169 lines, SHA256 recorded, provenance unverified | NO if news disabled | YES for anchor-governed news | NO | NO | YES | GOVERNANCE_BLOCKER | Claude/owner provenance ruling before news authority |
| EURUSD prepared OHLCV in `DEFAULT_DATA_DIRS` | MISSING: configured prepared dirs empty | YES | NO | NO | NO | NO | HARD_BLOCKER_FOR_EURUSD_LAB | owner decision: populate prepared CSVs from approved source or approve loader bridge |

## 5. Import Drift Fixes

Files changed:

- `03_RESEARCH_LAB/research_lab/config.py`
  - restored `DEFAULT_NEWS_FILE = DEFAULT_NEWS_FILE_OBSOLETE`
  - restored `DEFAULT_RAW_NEWS_FILE = DEFAULT_RAW_NEWS_FILE_OBSOLETE`
  - added `DEFAULT_NEWS_SUMMARY_FILE` derived from the obsolete default news path

Evidence:

- `python -c "import research_lab"` -> `research_lab OK`
- `python -c "import research_lab.light_runner"` -> `light_runner OK`
- `python -c "import research_lab.audit_level3"` -> `audit_level3 OK`

No data file was created. The aliases restore import compatibility only; missing legacy news files remain missing and are skipped or blocked honestly in tests.

## 6. Path Drift Fixes

Files changed:

- `03_RESEARCH_LAB/research_lab/tests/test_h6_paper_shadow_runner.py`
  - repointed the legacy H6 test import from removed root `scripts/` to `04_INFRASTRUCTURE_ENGINEERING/legacy_scripts/root_scripts_archive/h6_paper_shadow_runner.py`.
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/tests/remediation/test_integrity.py`
  - inserted the local `BOT_V2_DAYTIME_LAB` path so `scripts.utils.integrity` resolves to the package-local `scripts/` folder, not removed root scripts.

Not fixed deliberately:

- `03_RESEARCH_LAB/research_lab/eurusd_ltf_objective_entry_replacement_ecb_autopilot.py`
  - still references old `scripts/build_chatgpt_bundle.py` and related ZIP-era targets.
  - classified as LEGACY_BLOCKER because repointing it would revive deprecated ZIP workflow and is not required for EURUSD core lab.

## 7. Broader Research Lab Test Inventory

Initial inventory before cleanup:

- `python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_*.py"`
- Result: 149 tests, 3 failures, 41 errors.

Post-cleanup inventory:

- Compact runner result: 164 tests, 16 failures, 9 errors, 13 skipped.
- Final full discover is recorded in the safe-test section below.

Classification after cleanup:

| class | evidence | required_for_eurusd_lab | decision |
|---|---|---:|---|
| MISSING_DATA_REQUIRED | skips for absent hi-precision bundle, legacy/default news files, old audit CSV, USDJPY FF cache | NO for prepared core | honest SKIPPED_MISSING_REQUIRED_DATA |
| IMPORT_DRIFT | `DEFAULT_NEWS_*` imports, `pytest` import, `audit_level3` import | could block tooling | fixed |
| PATH_DRIFT | H6 root script test, BOT_V2 integrity test | NO for core | fixed where safe |
| LEGACY_MODULE_NOT_FOR_EURUSD_LAB | ECB autopilot root scripts/ZIP refs, USDJPY news builder, old news validation expectations | NO | documented/deferred |
| TEST_CONTRACT_BROKEN | test stubs exposing `signal` while engine calls `generate_signal`; stale LS-SR/SB mock columns | could hide failures | fixed; real assertions now visible |
| REAL_REGRESSION_OR_ENGINE_CONTRACT_DRIFT | remaining engine/level2/high-precision assertions after stubs execute | YES, because engine is core | MUST_TRIAGE_BEFORE_LAB |

Remaining non-green groups:

- Engine/stop-entry/level2/level3 synthetic behavior assertions: 21 combined failures/errors after stubs now execute.
- AM news builder expectations: 2 failures tied to anchor/news family coverage and ECB press conference derivation.
- `NewsConfig().enabled` legacy expectation: 1 failure because current config has news enabled by default.

## 8. Tests Fixed / Tests Remaining

Fixed or made honest:

- `research_lab`, `light_runner`, `audit_level3` imports.
- `pytest` optional dependency no longer prevents unittest discovery.
- H6 legacy test path no longer points at removed root `scripts/`.
- BOT_V2 integrity test resolves package-local `scripts/utils/integrity.py`.
- hi-precision, USDJPY raw news, and legacy default news tests now emit explicit `SKIPPED_MISSING_REQUIRED_DATA` instead of failing as missing files.
- test strategy stubs now expose `generate_signal`, revealing actual engine contract assertions instead of masking them as AttributeError.
- stale LS-SR and AM silver bullet mock columns updated to current loader/strategy contract.

Remaining:

- Broader research_lab suite is not green.
- Remaining engine/level2/high-precision failures are not fixed here because that would require engine behavior decisions or expected-result rewrites.
- Remaining AM/news failures are not fixed because they are data/provenance/expectation issues, not compatibility-only fixes.

## 9. EURUSD Core Pre-Lab Readiness

| item | verdict |
|---|---|
| root strictness | PASS_WITH_DOCUMENTED_EXCEPTIONS and ZIP moved out |
| branch canonical | PASS |
| no ZIP workflow | PASS_AFTER_MOVE |
| DATA_ROOT points to vault | PASS |
| EURUSD required train data exists | FAIL_HARD_BLOCKER |
| EURUSD data path excludes 2025/2026 | NOT_EVALUATED; no lab data loaded |
| strategy registry importable | PASS |
| engine importable | PASS |
| cost model config exists | PASS |
| output dir policy exists | PASS |
| optional modules isolated | PASS |
| holdout untouched | PASS |
| validation untouched | PASS |
| Claude final audit required | PASS |

## 10. Remaining Blockers

| id | blocker | class | blocks core lab |
|---|---|---|---:|
| CORE-DATA-1 | EURUSD prepared OHLCV absent from configured `DEFAULT_DATA_DIRS` | HARD_BLOCKER_FOR_EURUSD_LAB | YES |
| CORE-TEST-1 | engine/level2/high-precision synthetic tests still fail after test stubs execute | MUST_FIX_BEFORE_ANY_LAB / OWNER_TRIAGE | YES until triaged |
| GOV-ANCHOR-1 | `canonical_anchor_events.csv` provenance unverified | GOVERNANCE_BLOCKER | NO if news disabled |
| SOFT-NEWS-1 | `forex_factory_cache.csv` missing | SOFT_BLOCKER_OPTIONAL_MODULE | NO |
| SOFT-NEWS-2 | `news_eurusd_v2_utc.csv` missing | SOFT/LEGACY_BLOCKER | NO |
| SOFT-PREC-1 | hi-precision M1 dukascopy bundle missing | SOFT_BLOCKER_OPTIONAL_MODULE | NO |
| LEGACY-PATH-1 | ECB autopilot root scripts/ZIP references | LEGACY_BLOCKER | NO |
| GOV-GIT-1 | deferred clean-sync commits need owner review | GOVERNANCE_BLOCKER | NO for this branch audit |

## 11. What Is Still Forbidden

- No backtest.
- No strategy run.
- No F06 real run.
- No optimization.
- No sweep.
- No validation process.
- No holdout process.
- No 2025/2026 analysis.
- No data download.
- No data regeneration.
- No synthetic data.
- No engine/trading/strategy logic modification.
- No push to main.
- No force push.
- No ZIP workflow.

## 12. Copy-Paste Summary for ChatGPT

```text
PHASE E BURNDOWN + EURUSD SCOPE STATUS: PHASE_E_BURNDOWN_PARTIAL_OWNER_REVIEW_REQUIRED

Lab is NOT authorized.

Known missing assets classified:
- forex_factory_cache.csv: SOFT_BLOCKER for Jpy/news rebuild, not EURUSD core.
- news_eurusd_v2_utc.csv: SOFT/LEGACY news blocker, not EURUSD core.
- hi-precision dukascopy M1 bundle: SOFT high-precision blocker, not prepared-OHLCV core.
- canonical_anchor_events.csv: present, 169 lines, SHA256 1E7EB737163941A41C37ECEC824BF8588EABB7D3C5AA69DCE2B75C756D1AB556, but provenance unverified = GOVERNANCE_BLOCKER.

Discovered hard core blocker:
- EURUSD prepared OHLCV is absent at configured DEFAULT_DATA_DIRS.
- Raw EURUSD tick vault exists (3548 files, 9,210,754,686 bytes, 137 monthly parquets plus pilot) but research_lab.data_loader.load_prepared_ohlcv does not consume it.
- Owner must choose: approved prepared CSV materialization OR approved loader bridge. No silent repointing.

Fixes applied:
- config aliases: DEFAULT_NEWS_FILE, DEFAULT_RAW_NEWS_FILE, DEFAULT_NEWS_SUMMARY_FILE.
- research_lab/light_runner/audit_level3 imports OK.
- safe test path fixes and explicit SKIPPED_MISSING_REQUIRED_DATA guards.
- test stubs now use generate_signal; stale mock columns fixed.
- F06 pipeline remains 119/119 OK.
- legacy ZIP moved outside repo to BOT_ZIP_LEGACY_ARCHIVE with manifest.

Remaining:
- broader research_lab suite still non-green: 164 run, 16 failures, 9 errors, 13 skipped in compact post-cleanup inventory.
- core engine/level2/high-precision synthetic assertions still need owner/Claude triage before any lab.

Next prompt: NEXT_PROMPT_FIX_EURUSD_CORE_BLOCKERS.md
```
