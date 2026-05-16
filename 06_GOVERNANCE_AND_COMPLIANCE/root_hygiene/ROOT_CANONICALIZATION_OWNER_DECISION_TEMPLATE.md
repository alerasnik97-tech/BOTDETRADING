# ROOT CANONICALIZATION OWNER DECISION TEMPLATE

## 1. Purpose

This template records the owner decision required before any tracked root files are moved, removed, archived, or preserved as exceptions.

No implementation is authorized by this document alone.

## 2. Current Status

| field | value |
|---|---|
| current status | `ROOT_CANONICALIZATION_PLAN_READY_FOR_OWNER_DECISION` |
| tracked root items | 115 |
| tracked non-canonical root items | 112 |
| missing canonical folder | `08_CLOUD_FREE_RUN_LAB` |
| move map | `ROOT_CANONICALIZATION_MOVE_MAP.csv` |
| recommended option | `OPTION C - Hybrid institutional` |

## 3. Owner Decisions Required

### D1 - Canonicalization Option

Choose one:

- [ ] OPTION A - Minimal visual cleanup
- [ ] OPTION B - Strict 8-folder canonical root
- [ ] OPTION C - Hybrid institutional

Owner decision:

`D1_SELECTED_OPTION = `

### D2 - `.github`

Choose one:

- [ ] Keep `.github` as root technical exception.
- [ ] Move or redesign GitHub workflow structure in strict mode.

Owner decision:

`D2_GITHUB_ROOT_EXCEPTION = YES/NO`

### D3 - README and Requirements

Choose one:

- [ ] Keep `README.md`, `requirements.txt`, and `requirements-vps-optional.txt` as root technical exceptions.
- [ ] Move docs/requirements into canonical folders and update all tooling/docs references.

Owner decision:

`D3_README_REQUIREMENTS_ROOT_EXCEPTION = YES/NO`

### D4 - `research_lab`

Choose one:

- [ ] Keep `research_lab` as root technical exception until a dedicated import migration.
- [ ] Move `research_lab` now and update imports/path references in the same implementation phase.

Owner decision:

`D4_RESEARCH_LAB_ROOT_EXCEPTION_UNTIL_IMPORT_MIGRATION = YES/NO`

### D5 - ZIPs

Choose one for `000_PARA_CHATGPT.zip` and related legacy ZIP workflow artifacts:

- [ ] Remove `000_PARA_CHATGPT.zip` from Git and preserve a local ignored quarantine copy.
- [ ] Move tracked ZIP material into `07_BACKUPS/legacy_zip_workflow/`.
- [ ] Keep ZIP material temporarily in root.

Owner decision:

`D5_ZIP_POLICY = REMOVE_FROM_GIT_KEEP_LOCAL / ARCHIVE_TO_07_BACKUPS / KEEP_TEMPORARILY`

### D6 - Data-Like Root Folders

Applies to:

- `DATA MANUAL`
- `data_usdjpy_2016_2019`
- `data_usdjpy_2016_2021`
- `data_usdjpy_2022_2025`

Choose one:

- [ ] Move to `05_MARKET_DATA_VAULT/legacy_data/` after data inventory and manifest.
- [ ] Leave in root until a separate data audit.

Owner decision:

`D6_DATA_LIKE_FOLDERS = MOVE_AFTER_DATA_AUDIT / LEAVE_UNTIL_SEPARATE_DATA_AUDIT`

### D7 - Legacy Strategy Folders

Applies to:

- `MANIPULANTE`
- `ROCKI_AM`
- `ESTRATEGIAS`
- `STRATEGIES`
- `LAB_STRATEGIES`

Choose one:

- [ ] Move to `03_RESEARCH_LAB/legacy_strategy_sources/`.
- [ ] Move selected certified/authority surfaces to `01_CORE_PRODUCTION/`.
- [ ] Leave in root until a separate strategy authority audit.

Owner decision:

`D7_LEGACY_STRATEGY_FOLDERS = MOVE_TO_03 / SELECTIVE_PRODUCTION_SPLIT / LEAVE_UNTIL_STRATEGY_AUDIT`

### D8 - Approval To Apply Move Map

Choose one:

- [ ] Approve future implementation of the move map under the selected decisions.
- [ ] Do not apply yet; revise plan first.

Owner decision:

`D8_APPROVE_APPLY_MOVE_MAP = YES/NO`

## 4. Explicit Non-Authorization

This owner decision does not authorize:

- backtest execution,
- strategy execution,
- F06 execution,
- optimization,
- sweep,
- validation data access,
- holdout access,
- 2025/2026 data access,
- raw/tick/parquet mutation,
- force push,
- merge to main,
- deleting tracked files outside the approved move map.

## 5. Signature Block

Owner approval status:

`OWNER_ROOT_CANONICALIZATION_DECISION_STATUS = PENDING / APPROVED / REJECTED / NEEDS_REVISION`

Owner notes:

```

```
