# ROOT CANONICALIZATION OWNER DECISION APPROVED

## 1. Status

OWNER_ROOT_CANONICALIZATION_DECISION_APPROVED

This document records owner approval only. It does not apply the move map, move files, delete files, run `git mv`, run `git rm`, execute strategies, execute backtests, touch validation, touch holdout, touch 2025/2026, or mutate raw/data files.

## 2. Approved Option

OPTION C - Hybrid Institutional

Rationale:

The owner approves a strong root cleanup while preserving technical exceptions required to avoid breaking imports, GitHub workflows, tooling, and current dependency workflows.

## 3. Approved Root Exceptions

The following root exceptions are approved:

- `.github`
- `README.md`
- `requirements.txt`
- `requirements-vps-optional.txt`
- `research_lab` TEMPORARY

`research_lab` remains a temporary exception because direct imports/pathing may depend on the root package location. It must not be moved until a dedicated import/path migration is approved.

## 4. ZIP Decision

Approved future action for `000_PARA_CHATGPT.zip`:

- remove from Git in a future apply phase,
- preserve a local copy in ignored quarantine,
- target local preservation path: `_LOCAL_QUARANTINE_DO_NOT_COMMIT/root_zip_legacy/`,
- no ZIP workflow going forward,
- do not permanently delete the local ZIP copy,
- do not upload the ZIP again.

## 5. Data Folder Decision

Data-like folders remain blocked until a separate data audit.

Blocked examples:

- `DATA MANUAL`
- `data_usdjpy_2016_2019`
- `data_usdjpy_2016_2021`
- `data_usdjpy_2022_2025`
- any folder or file containing CSV/parquet/data material requiring inventory

Required before any data movement:

- data inventory,
- size and hash manifest,
- sensitivity classification,
- explicit owner approval.

## 6. Strategy Legacy Decision

Legacy strategy folders are approved for future move to:

`03_RESEARCH_LAB/legacy_strategy_sources/`

Applies to:

- `MANIPULANTE`
- `ESTRATEGIAS`
- `STRATEGIES`
- `LAB_STRATEGIES`
- `ROCKI_AM`
- equivalent legacy strategy folders listed in the move map

Conditions:

- no strategy execution,
- no backtest execution,
- no production changes,
- no trading logic changes,
- only move items present in the move map,
- high-risk rows remain blocked until separate approval.

## 7. Apply Scope

Approved for a future controlled apply phase:

- low-risk move map rows,
- medium-risk move map rows after path/import checks.

Blocked:

- high-risk rows,
- data-like folders,
- `research_lab` migration,
- any unplanned item,
- any item not present in the move map,
- any item whose risk changes during preflight.

The future apply phase must use `NEXT_PROMPT_APPLY_ROOT_CANONICALIZATION.md` and must create its own implementation report.

## 8. Safety Boundaries

The approval does not authorize:

- backtest,
- strategy run,
- F06 run,
- optimization,
- sweep,
- validation access,
- holdout access,
- 2025/2026 access,
- raw data mutation,
- CSV/parquet/data mutation,
- force push,
- merge,
- touching main,
- deleting tracked files outside the approved future scope.

## 9. Next Step

Use:

`06_GOVERNANCE_AND_COMPLIANCE/root_hygiene/NEXT_PROMPT_APPLY_ROOT_CANONICALIZATION.md`

in a separate controlled phase. That phase may apply only the approved low/medium-risk subset and must keep high-risk rows blocked unless a new explicit owner approval is provided.
