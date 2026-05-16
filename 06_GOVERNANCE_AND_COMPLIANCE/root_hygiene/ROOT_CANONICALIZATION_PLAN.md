# ROOT CANONICALIZATION PLAN

## 1. Status

ROOT_CANONICALIZATION_PLAN_READY_FOR_OWNER_DECISION

## 2. Executive Summary

This is a plan-only governance artifact. No tracked files were moved, deleted, or removed from Git. No `git mv`, `git rm`, backtest, strategy run, F06 run, optimization, sweep, validation, holdout, 2025, 2026, raw data, tick data, or parquet workflow was executed.

The root hygiene audit showed that the visual disorder is structural: 112 tracked root items are outside the official root allowlist. Because they are tracked, making the root visually canonical requires an owner-approved repository canonicalization phase, not local cleanup.

Recommended path: OPTION C - Hybrid institutional. Keep technical root exceptions temporarily, move/archive legacy material by category, remove the tracked ZIP from Git while preserving a local ignored copy, and defer data-like and strategy-authority moves until owner decisions are explicit.

## 3. Current Root Problem

Current tracked root inventory from `git ls-files`:

| metric | value |
|---|---:|
| tracked root items | 115 |
| tracked items in official allowlist | 3 |
| tracked non-canonical root items | 112 |
| missing canonical folder on disk | 1 (`08_CLOUD_FREE_RUN_LAB`) |
| tracked root ZIPs | 1 (`000_PARA_CHATGPT.zip`) |

Primary problem:

1. Many legacy scripts, reports, labs, checkpoints, strategy folders, data-like folders, and ZIP workflow artifacts are tracked directly in root.
2. `000_PARA_CHATGPT.zip` is tracked even though current policy says GitHub is the source of truth and ZIPs must not live in the repo root.
3. Some root items are high-risk to move because imports, docs, workflows, or operator procedures reference their current root paths.

## 4. Official Root Allowlist

Owner-desired root:

| item | status |
|---|---|
| `01_CORE_PRODUCTION` | present on disk |
| `02_INCUBATION_STAGING` | present on disk |
| `03_RESEARCH_LAB` | present and tracked |
| `04_INFRASTRUCTURE_ENGINEERING` | present on disk |
| `05_MARKET_DATA_VAULT` | present on disk; ignored/local data surface |
| `06_GOVERNANCE_AND_COMPLIANCE` | present and tracked |
| `07_BACKUPS` | present on disk; ignored/local backup surface |
| `08_CLOUD_FREE_RUN_LAB` | missing |
| `.gitignore` | present and tracked |
| `.git` | technical hidden Git directory; not touched |

Potential technical exceptions for owner decision:

| item | recommendation |
|---|---|
| `.github` | keep as root exception if GitHub Actions remain active |
| `README.md` | keep as root exception in hybrid mode |
| `requirements.txt` | keep as root exception in hybrid mode |
| `requirements-vps-optional.txt` | keep as root exception in hybrid mode or move to infra with install docs update |
| `research_lab` | keep temporarily as root exception until import migration is separately approved |

## 5. Tracked Non-Canonical Inventory Summary

Move map file:

`06_GOVERNANCE_AND_COMPLIANCE/root_hygiene/ROOT_CANONICALIZATION_MOVE_MAP.csv`

Summary:

| dimension | count |
|---|---:|
| total move-map rows | 115 |
| low risk | 46 |
| medium risk | 59 |
| high risk | 10 |
| owner approval required | 112 |
| keep root exception | 8 |
| move to canonical folder | 85 |
| archive to `07_BACKUPS` | 12 |
| remove from Git and keep local quarantine | 1 |
| needs owner decision before action | 9 |

Category summary:

| category | count |
|---|---:|
| governance docs or audit material | 29 |
| infrastructure or legacy scripts | 29 |
| research legacy or checkpoints | 15 |
| infrastructure ops or local support | 10 |
| legacy ZIP workflow | 9 |
| strategy sources | 5 |
| market data-like | 4 |
| legacy archive | 3 |
| technical root metadata | 3 |
| official/root policy/current package/github/zip artifact/operator groups | remaining rows |

## 6. Root Exceptions Recommendation

Recommended temporary root exceptions under OPTION C:

| item | reason | risk if moved now |
|---|---|---|
| `.github` | GitHub Actions normally require this location | workflow breakage |
| `README.md` | common repository entrypoint | documentation/tooling expectations |
| `requirements.txt` | package/dependency tooling commonly expects root path | install breakage |
| `requirements-vps-optional.txt` | related dependency metadata | install/docs drift |
| `research_lab` | many imports/docs reference root package | import breakage |

These exceptions should be documented explicitly. They are not a waiver for adding more root clutter.

## 7. Options A/B/C

| option | root after cleanup | pros | cons | risk | estimated tracked items affected | tests needed | recommended |
|---|---|---|---|---|---:|---|---|
| OPTION A - Minimal visual cleanup | canonical folders plus many technical/legacy root remnants | lowest disruption; can remove ZIP workflow noise first | root remains visually busy; does not satisfy strict owner target | low | 20-40 | root allowlist audit; ZIP absence check; rg path refs | no |
| OPTION B - Strict 8-folder canonical root | only eight official folders plus `.gitignore` and hidden `.git` | cleanest visual result; strongest policy | high chance of path/import/workflow breakage; requires broad docs/import updates | medium/high | 112 | full path reference migration; import tests; workflow tests; safe unit tests | no for immediate application |
| OPTION C - Hybrid institutional | official folders plus `.github`, `README.md`, requirements files, and `research_lab` temporary exception | materially cleans root while preserving critical compatibility | not perfectly strict; requires exception register | moderate | about 107 | path/reference audit; import checks; targeted unit tests; root allowlist check | yes |

## 8. Recommended Option

Recommend OPTION C - Hybrid institutional.

Rationale:

1. It respects the owner's visual cleanup goal while avoiding unnecessary breakage.
2. It keeps technical exceptions that are likely to be assumed by GitHub, package tools, and Python imports.
3. It removes the active root clutter sources: ZIP workflow artifacts, old reports, legacy scripts, checkpoint folders, and scattered governance files.
4. It defers high-risk data and strategy-authority moves until owner decisions are explicit.

## 9. Proposed Move Map Summary

Action summary from the move map:

| action | count | meaning |
|---|---:|---|
| `KEEP_ROOT_EXCEPTION` | 8 | official root item or temporary technical exception |
| `MOVE_TO_CANONICAL_FOLDER` | 85 | move to one of the eight canonical folders after approval |
| `ARCHIVE_TO_07_BACKUPS` | 12 | preserve historical/legacy material under backups |
| `REMOVE_FROM_GIT_KEEP_LOCAL_QUARANTINE` | 1 | remove tracked ZIP from Git while preserving an ignored local copy |
| `NEEDS_OWNER_DECISION` | 9 | do not move until the owner classifies authority/data status |

Destination policy:

| class | destination |
|---|---|
| research/checkpoints/labs | `03_RESEARCH_LAB/legacy_root_research/` |
| legacy/current strategy sources | `03_RESEARCH_LAB/legacy_strategy_sources/` unless owner marks production |
| infrastructure/scripts/ops | `04_INFRASTRUCTURE_ENGINEERING/legacy_scripts/` or `legacy_ops/` |
| data-like folders | `05_MARKET_DATA_VAULT/legacy_data/` only after data-owner approval |
| governance/docs/audits/status | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/` |
| ZIP workflow/historical archives | `07_BACKUPS/legacy_zip_workflow/` or local quarantine for Git removal |

## 10. High-Risk Items

| item | risk | recommended handling |
|---|---|---|
| `research_lab` | import/path breakage | keep root exception until dedicated import migration |
| `DATA MANUAL` | data mutation risk | separate data audit before move |
| `data_usdjpy_2016_2019` | data mutation/path risk | separate data audit before move |
| `data_usdjpy_2016_2021` | data mutation/path risk | separate data audit before move |
| `data_usdjpy_2022_2025` | data mutation/2025 sensitivity | separate data audit before move |
| `MANIPULANTE` | possible authority/production strategy surface | owner classification required |
| `ROCKI_AM` | strategy lineage risk | owner classification required |
| `ESTRATEGIAS` | strategy source risk | owner classification required |
| `STRATEGIES` | strategy source risk | owner classification required |
| `LAB_STRATEGIES` | strategy source risk | owner classification required |

## 11. Items Requiring Owner Decision

Owner must decide:

1. Which cleanup option to apply: A, B, or C.
2. Whether `.github` remains a root exception.
3. Whether `README.md` and requirements files remain root exceptions.
4. Whether `research_lab` remains a root exception until import migration.
5. What to do with `000_PARA_CHATGPT.zip`: remove from Git and preserve local quarantine, archive under `07_BACKUPS`, or keep temporarily.
6. Whether data-like root folders may move to `05_MARKET_DATA_VAULT/legacy_data/`.
7. Whether strategy folders move to `03_RESEARCH_LAB/legacy_strategy_sources/` or one or more belong in `01_CORE_PRODUCTION/`.
8. Whether the full move map is approved for a future implementation phase.

## 12. Tests Required Before/After Applying

Before applying:

1. `git status --short`
2. `git branch --show-current`
3. `git rev-parse HEAD`
4. `git ls-files | ForEach-Object { ($_ -split "/")[0] } | Sort-Object -Unique`
5. Restricted `rg` reference audit excluding `_LOCAL_QUARANTINE_DO_NOT_COMMIT`, `.git`, `05_MARKET_DATA_VAULT`, `07_BACKUPS`, ZIPs, parquet, CSV, sqlite/db, and jsonl.

After applying:

1. Root allowlist check.
2. `git status --short`.
3. `git diff --name-status`.
4. Path-reference audit for every moved root item.
5. Import checks for `research_lab` if it is moved; otherwise assert it remains an approved exception.
6. GitHub workflow path audit if `.github` or workflow targets change.
7. Safe unit tests only. No backtest, no strategy run, no F06 real, no validation, no holdout, no 2025/2026 data touch.

Reference-risk audit already observed:

| surface | references found | risk conclusion |
|---|---:|---|
| `.github` / workflow paths | 35 | keep root exception unless workflow migration is planned |
| `research_lab` | 778 | high import/docs risk |
| `BOT_V2_DAYTIME_LAB` | 6262 | high documentation/path migration volume |
| `MANIPULANTE` | 1797 | owner authority decision required |
| ZIP workflow | 694 | remove/archive with path doc updates |
| root entrypoints | 78 | wrapper/path compatibility needed |
| phase scripts | 813 | legacy script migration requires rg cleanup |
| data roots | 69 | data-owner approval required |

## 13. What Not To Move Yet

Do not move yet:

1. `research_lab`, unless a dedicated import migration is approved.
2. `MANIPULANTE`, `ROCKI_AM`, `ESTRATEGIAS`, `STRATEGIES`, `LAB_STRATEGIES`, until owner classifies strategy authority.
3. `DATA MANUAL` and `data_usdjpy_*`, until a separate data audit approves the move.
4. `.github`, unless workflows are updated and tested.
5. `README.md` and requirements files, unless owner chooses strict mode and install/docs tooling is updated.
6. `000_PARA_CHATGPT.zip`, until owner chooses remove-from-Git, archive, or temporary keep.

## 14. Next Step

Owner should fill `ROOT_CANONICALIZATION_OWNER_DECISION_TEMPLATE.md`. Only after explicit approval should a future implementation prompt apply the move map using controlled `git mv` / `git rm`, path-reference updates, safe tests, and a new implementation report.
