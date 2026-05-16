# ROOT CANONICALIZATION OWNER DECISION TEMPLATE

## 1. Purpose

This template records the owner decision required before any tracked root files are moved, removed, archived, or preserved as exceptions.

No implementation is authorized by this document alone.

## 2. Current Status

| field | value |
|---|---|
| current status | `ROOT_CANONICALIZATION_OWNER_DECISION_APPROVED_WITH_LIMITS` |
| tracked root items | 115 |
| tracked non-canonical root items | 112 |
| missing canonical folder | `08_CLOUD_FREE_RUN_LAB` |
| move map | `ROOT_CANONICALIZATION_MOVE_MAP.csv` |
| recommended option | `OPTION C - Hybrid institutional` |

Formal owner decision:

`APPROVED_OPTION = OPTION_C_HYBRID_INSTITUTIONAL`

Decision rationale:

The owner wants to clean the repository root aggressively while preserving technical exceptions needed to avoid breaking imports, GitHub workflows, tooling, and current dependency workflows.

## 3. Owner Decisions Required

### D1 - Canonicalization Option

Choose one:

- [ ] OPTION A - Minimal visual cleanup
- [ ] OPTION B - Strict 8-folder canonical root
- [x] OPTION C - Hybrid institutional

Owner decision:

`D1_SELECTED_OPTION = OPTION_C_HYBRID_INSTITUTIONAL`

### D2 - `.github`

Choose one:

- [x] Keep `.github` as root technical exception.
- [ ] Move or redesign GitHub workflow structure in strict mode.

Owner decision:

`D2_GITHUB_ROOT_EXCEPTION = YES`

Reason:

`.github` remains in the repository root as a normal technical exception for professional repositories when it contains workflows, actions, or GitHub configuration.

### D3 - README and Requirements

Choose one:

- [x] Keep `README.md`, `requirements.txt`, and `requirements-vps-optional.txt` as root technical exceptions.
- [ ] Move docs/requirements into canonical folders and update all tooling/docs references.

Owner decision:

`D3_README_REQUIREMENTS_ROOT_EXCEPTION = YES`

Scope:

- `README.md`: keep root exception.
- `requirements.txt`: keep root exception.
- `requirements-vps-optional.txt`: keep root exception.

Reason:

These are standard root files in professional repositories. Do not move them in the current canonicalization path.

### D4 - `research_lab`

Choose one:

- [x] Keep `research_lab` as root technical exception until a dedicated import migration.
- [ ] Move `research_lab` now and update imports/path references in the same implementation phase.

Owner decision:

`D4_RESEARCH_LAB_ROOT_EXCEPTION_UNTIL_IMPORT_MIGRATION = YES_TEMPORARY`

Reason:

`research_lab` remains a temporary root exception to avoid breaking imports until a dedicated import migration is scoped and approved.

### D5 - ZIPs

Choose one for `000_PARA_CHATGPT.zip` and related legacy ZIP workflow artifacts:

- [x] Remove `000_PARA_CHATGPT.zip` from Git and preserve a local ignored quarantine copy.
- [ ] Move tracked ZIP material into `07_BACKUPS/legacy_zip_workflow/`.
- [ ] Keep ZIP material temporarily in root.

Owner decision:

`D5_ZIP_POLICY = REMOVE_FROM_GIT_KEEP_LOCAL`

Approved action:

`D5_APPROVED_ACTION = REMOVE_FROM_GIT_KEEP_LOCAL_QUARANTINE`

Required local preservation path:

`_LOCAL_QUARANTINE_DO_NOT_COMMIT/root_zip_legacy/000_PARA_CHATGPT.zip`

Additional owner constraints:

- Do not use ZIP as the primary workflow.
- Do not permanently delete the local ZIP copy in the canonicalization phase.
- Do not upload the ZIP again.

### D6 - Data-Like Root Folders

Applies to:

- `DATA MANUAL`
- `data_usdjpy_2016_2019`
- `data_usdjpy_2016_2021`
- `data_usdjpy_2022_2025`

Choose one:

- [ ] Move to `05_MARKET_DATA_VAULT/legacy_data/` after data inventory and manifest.
- [x] Leave in root until a separate data audit.

Owner decision:

`D6_DATA_LIKE_FOLDERS = LEAVE_UNTIL_SEPARATE_DATA_AUDIT`

Blocked until separate data audit:

- `DATA MANUAL`
- `data_usdjpy_*`
- any CSV/parquet/data folder

Reason:

Data-like folders require a separate data audit, inventory, hashes, and explicit owner decision before any move.

### D7 - Legacy Strategy Folders

Applies to:

- `MANIPULANTE`
- `ROCKI_AM`
- `ESTRATEGIAS`
- `STRATEGIES`
- `LAB_STRATEGIES`

Choose one:

- [x] Move to `03_RESEARCH_LAB/legacy_strategy_sources/`.
- [ ] Move selected certified/authority surfaces to `01_CORE_PRODUCTION/`.
- [ ] Leave in root until a separate strategy authority audit.

Owner decision:

`D7_LEGACY_STRATEGY_FOLDERS = MOVE_TO_03`

Approved action:

`D7_APPROVED_ACTION = APPROVE_MOVE_TO_03_RESEARCH_LAB_LEGACY_STRATEGY_SOURCES`

Implementation constraint:

This approves the destination policy only for legacy strategy folders and equivalent strategy folders listed in the move map. It does not authorize strategy execution, backtest execution, production changes, or any trading logic change. If any selected legacy strategy folder is classified high-risk in the move map, implementation remains blocked until a separate high-risk phase is explicitly approved.

### D8 - Approval To Apply Move Map

Choose one:

- [x] Approve future implementation of the move map under the selected decisions.
- [ ] Do not apply yet; revise plan first.

Owner decision:

`D8_APPROVE_APPLY_MOVE_MAP = YES_LOW_MEDIUM_RISK_ONLY`

Formal approval:

`D8_APPROVED = YES_CONTROLLED_FUTURE_PHASE_ONLY`

Application boundary:

- Approved now: low-risk and medium-risk move map rows only.
- Medium-risk rows require path/import checks before and after moving.
- Blocked now: high-risk move map rows.
- High-risk rows require a separate future phase and explicit owner approval before implementation.
- Data-like folders remain blocked.
- `research_lab` migration remains blocked.
- validation, holdout, 2025, and 2026 remain out of scope.

## 4. High-Risk Block

`HIGH_RISK_CANONICALIZATION_STATUS = BLOCKED_FOR_SEPARATE_PHASE`

High-risk examples currently blocked from implementation:

- `research_lab`
- `DATA MANUAL`
- `data_usdjpy_*`
- `MANIPULANTE`
- `ROCKI_AM`
- `ESTRATEGIAS`
- `STRATEGIES`
- `LAB_STRATEGIES`

No high-risk move is authorized by this decision.

## 5. Explicit Non-Authorization

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
- deleting tracked files outside the approved move map,
- applying any high-risk move map row in the low/medium implementation phase.

## 6. Signature Block

Owner approval status:

`OWNER_ROOT_CANONICALIZATION_DECISION_STATUS = APPROVED_WITH_HIGH_RISK_BLOCK`

Owner notes:

```
Owner approves OPTION C - Hybrid Institutional.
Owner approves future application only for low/medium risk move map rows.
High-risk rows remain blocked for a separate scoped phase.
No move map application is authorized in this documentation-only update.
```
