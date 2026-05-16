# NEXT PROMPT - APPLY ROOT CANONICALIZATION

Act as a senior institutional repo hygiene engineer, Git safety specialist, path compatibility auditor, and controlled migration operator.

## Objective

Apply the owner-approved root canonicalization move map for:

`C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`

Use:

- `06_GOVERNANCE_AND_COMPLIANCE/root_hygiene/ROOT_CANONICALIZATION_PLAN.md`
- `06_GOVERNANCE_AND_COMPLIANCE/root_hygiene/ROOT_CANONICALIZATION_MOVE_MAP.csv`
- `06_GOVERNANCE_AND_COMPLIANCE/root_hygiene/ROOT_CANONICALIZATION_OWNER_DECISION_TEMPLATE.md`

## Absolute Preconditions

Before making changes:

1. Read the owner decision template.
2. Confirm `OWNER_ROOT_CANONICALIZATION_DECISION_STATUS = APPROVED`.
3. Confirm D1-D8 are explicit and non-ambiguous.
4. If any required owner decision is missing, stop with:

`ROOT_CANONICALIZATION_BLOCKED_OWNER_DECISION_REQUIRED`

## Absolute Rules

NO backtest.
NO strategy run.
NO F06 run.
NO optimization.
NO sweep.
NO validation data.
NO holdout.
NO 2025.
NO 2026.
NO raw/tick/parquet mutation.
NO force push.
NO merge.
NO touch main.
NO destructive cleanup.
NO `git clean -fdx`.
NO unplanned move outside the approved move map.

If a proposed action is not present in the approved move map, stop and document it.

## Precheck

Run:

```powershell
pwd
git status --short
git branch --show-current
git rev-parse HEAD
git remote -v
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, StartTime, CPU, WorkingSet
Get-WmiObject Win32_Process -Filter "name='python.exe'" | Select-Object ProcessId, CommandLine | Format-List
```

If active backtest/sweep/optimization is detected, abort with:

`BLOCKED_ACTIVE_RESEARCH_PROCESS_DETECTED`

## Implementation Scope

Allowed only after owner approval:

- `git mv` approved tracked paths to approved canonical destinations.
- `git rm --cached` approved ZIP artifact if owner chose remove-from-Git-keep-local.
- Create missing `08_CLOUD_FREE_RUN_LAB` if approved.
- Update path references only when required by approved moves.
- Update docs/indexes that point to old root paths.
- Add compatibility wrappers only if explicitly approved.

Forbidden:

- changing trading logic,
- changing strategy logic,
- changing backtest engine behavior,
- touching raw/tick/parquet data,
- touching validation/holdout/2025/2026,
- deleting local data without manifest and owner approval.

## Required Process

1. Load and validate `ROOT_CANONICALIZATION_MOVE_MAP.csv`.
2. Filter rows by owner-approved decisions.
3. For each row:
   - verify path exists,
   - verify tracked state,
   - verify action matches owner decision,
   - verify destination parent is canonical,
   - apply only approved action.
4. After each batch, run:

```powershell
git status --short
```

5. Run restricted reference audit:

```powershell
rg -n "<old_path_or_item>" . --hidden --glob "!.git/**" --glob "!_LOCAL_QUARANTINE_DO_NOT_COMMIT/**" --glob "!05_MARKET_DATA_VAULT/**" --glob "!07_BACKUPS/**" --glob "!**/*.zip" --glob "!**/*.parquet" --glob "!**/*.csv" --glob "!**/*.jsonl"
```

6. Update references only when safe and within scope.

## Required Tests

Run safe tests only:

- root allowlist check,
- import/path checks,
- GitHub workflow path check if `.github` or workflow targets changed,
- unit tests that do not execute strategies or real backtests,
- no validation/holdout/2025/2026 tests.

If tests would execute backtests, strategies, F06, validation, holdout, 2025, 2026, or raw data, do not run them.

## Required Report

Create:

`06_GOVERNANCE_AND_COMPLIANCE/root_hygiene/ROOT_CANONICALIZATION_APPLY_REPORT.md`

Include:

1. status,
2. owner decisions applied,
3. files moved,
4. files removed from Git but preserved locally,
5. files archived,
6. root before/after,
7. path/import reference audit,
8. tests run,
9. safety verification,
10. remaining exceptions,
11. final decision.

## GitHub Update

Stage only planned files and report.

Commit:

```powershell
git commit -m "chore: canonicalize tracked project root"
git push origin <branch>
```

No force push. No merge. Do not touch main.

## Final Response Format

Respond:

1. STATUS:
2. OWNER_DECISIONS:
3. FILES_MOVED:
4. FILES_REMOVED_FROM_GIT:
5. ROOT_AFTER:
6. TESTS:
7. SAFETY:
8. GITHUB:
9. DECISION:
10. NEXT_STEP:
