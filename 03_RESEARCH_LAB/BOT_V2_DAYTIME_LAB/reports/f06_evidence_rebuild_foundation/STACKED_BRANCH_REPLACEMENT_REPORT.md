# STACKED BRANCH REPLACEMENT REPORT
**Status**: COMPLETED
**Fecha**: 2026-05-15

## 1. Problem
PR #4 (`research/f06-evidence-rebuild-foundation-20260515`, head `e62da979`) diverged from
PR #3 (`research/pre-claude-blocker-remediation-20260515`, head `73c4ea63`) after PR #3
received a governance fix. No force-push was allowed.

## 2. Solution: Clean Replacement Branch

| Item | Value |
|:---|:---|
| Old PR4 branch | `research/f06-evidence-rebuild-foundation-20260515` |
| New branch | `research/f06-evidence-rebuild-foundation-v2-20260515` |
| Base (PR3 HEAD) | `73c4ea63c58647cbfd7c6e35e2db2f640d5a0725` |
| Force push used | NO |
| Merge unrelated histories | NO |

## 3. Cherry-Picked Commits
| Order | SHA | Message |
|:---|:---|:---|
| 1 | `1d04e696` | research: add F06 evidence rebuild foundation guards |
| 2 | `398162a8` | research: refresh dry-run manifest post-tracking |
| 3 | `56090960` | test: harden F06 evidence rebuild foundation guards |
| 4 | `e62da979` | test: close PR4 foundation warning guards |

Cherry-pick result: **CLEAN** (0 conflicts).

## 4. Fixture Reproducibility Fix
**Problem**: All 8 fixture `output_*/` directories had a `results/` sub-folder. The root
`.gitignore` contains `results/`, so these folders were untracked and would be missing
on a clean clone, breaking test reproducibility.

**Fix applied (Option A)**: Renamed all `results/` → `ranking/` in fixtures. Updated:
- 8 `MANIFEST_*.json` files (output_hashes keys)
- 8 `HASHES_*.txt` files
- `MANIFEST_good.json`: recalculated `script_sha256`, `config_sha256`, and all
  `output_hashes` (non-circular bootstrap order)
- Created `tests/__init__.py` to make tests/ importable by Python 3.14

## 5. New Test Added
`tests/test_fixture_artifacts_tracked.py` — guards:
- No `results/` sub-folder exists inside fixtures
- All artifacts declared in MANIFEST `output_hashes` exist on disk
- All artifacts are tracked by git (`git ls-files`)
- No manifest `output_hashes` key references `/results/`

## 6. Test Results
**82/82 PASS** (0 failures, 0 errors)

## 7. Validator Results
| Validator | Expected | Actual |
|:---|:---|:---|
| validate_config | PASS | ✅ PASS |
| dry_run | DRY_RUN_SCHEMA_VALIDATED | ✅ DRY_RUN_SCHEMA_VALIDATED |
| output_good | READY_FOR_CLAUDE_AUDIT | ✅ READY_FOR_CLAUDE_AUDIT |
| bad_multi_runid | BLOCKED_GUARD_FAILED | ✅ BLOCKED_GUARD_FAILED |
| bad_validation_columns | BLOCKED_GUARD_FAILED | ✅ BLOCKED_GUARD_FAILED |
| bad_2025 | BLOCKED_GUARD_FAILED | ✅ BLOCKED_GUARD_FAILED |
| bad_hash | BLOCKED_GUARD_FAILED | ✅ BLOCKED_GUARD_FAILED |
| bad_sample_size | BLOCKED_GUARD_FAILED | ✅ BLOCKED_GUARD_FAILED |
| bad_cost | BLOCKED_GUARD_FAILED | ✅ BLOCKED_GUARD_FAILED |
| bad_quarantined_path | BLOCKED_GUARD_FAILED | ✅ BLOCKED_GUARD_FAILED |

## 8. Safety Verification
- **strategy_run**: NO
- **backtest_run**: NO
- **validation_touched**: NO
- **holdout_touched**: NO
- **2025_2026_touched**: NO
- **raw_data_mutated**: NO
- **old_quarantined_outputs_used**: NO

## 9. PR #4 Status
PR #4 (`research/f06-evidence-rebuild-foundation-20260515`) is **SUPERSEDED** by this
branch. It must NOT be merged. See PR body for cross-reference.

## 10. Decision
**READY_FOR_CLAUDE_AUDIT** (F06 pipeline scaffold only — no strategy certified).
