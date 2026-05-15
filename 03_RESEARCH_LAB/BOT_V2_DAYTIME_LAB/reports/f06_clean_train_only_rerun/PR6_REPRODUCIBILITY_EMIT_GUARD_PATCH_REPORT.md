# PR6 REPRODUCIBILITY + EMIT GUARD PATCH REPORT

## 1. Status

PR6_REPRO_EMIT_PATCH_READY_FOR_CLAUDE_REAUDIT

## 2. Claude Blockers Addressed

| blocker | fix | test | status |
|---|---|---|---|
| `dry_run --emit` accepted forbidden tokens | `cmd_dry_run` now validates raw emit, basename, and resolved emit path before output dir reservation | `test_dry_run_rejects_emit_with_quarantined_token` | FIXED |
| `dry_run --emit V50B_RERUN_TRADES.csv` could write a legacy-looking artifact | same emit guard blocks legacy V50B filenames before write | `test_dry_run_rejects_emit_legacy_v50b_name` | FIXED |
| fixture `script_sha256` was tied to physical working-tree bytes | synthetic fixture uses `__CURRENT_SCRIPT_SHA256__`, allowed only under `fixtures/`; real outputs still require physical SHA256 | `test_fixture_current_script_sha_marker_allowed_only_in_fixtures`, `test_current_script_sha_marker_forbidden_outside_fixtures`, `test_output_good_passes_after_script_change_without_manual_hash_edit`, `test_real_manifest_requires_physical_script_sha256` | FIXED |
| stale reports could confuse audit evidence | historical reports now carry superseded banners or current command-surface text | `rg` stale phrase check | FIXED |

## 3. Emit Guard Fix

`dry_run` now computes and validates the final `emit` path before calling
`reserve_output_dir_atomic()`.

The guard checks:

- raw `--emit` argument,
- raw basename,
- resolved absolute emit path,
- resolved basename,
- containment under the dry-run output directory.

If any forbidden token is found, status is `BLOCKED_FORBIDDEN_OUTPUT_PATH` and no
artifact is written. In clean verification, both bad emit cases also left the
requested output directories absent.

## 4. Script SHA Reproducibility Fix

The synthetic good fixture now uses:

`"script_sha256": "__CURRENT_SCRIPT_SHA256__"`

This marker is allowed only when the manifest file itself lives under
`pipelines/f06_evidence_rebuild/fixtures/`.

Real outputs remain strict:

- direct manifest validation rejects the marker,
- output validation rejects the marker outside fixtures,
- real manifests must carry the physical script SHA256.

## 5. ResourceWarning / File Handle Cleanup

Reviewed touched code and tests for file-handle handling. New tests use context
managers and `TemporaryDirectory()` for synthetic copies. No extra file-handle
leak was introduced.

## 6. Stale Report Cleanup

Historical reports were updated to avoid presenting old counts or old command
surfaces as current authority:

- `PHASE3_RUNNER_IMPLEMENTATION_REPORT.md`
- `PHASE3_RUNNER_NOT_IMPLEMENTED.md`
- `PR6_RUNNER_HARDENING_PATCH_REPORT.md`

Current authority is this report plus the source and tests in the PR branch.

## 7. Tests

- total: 118
- passed: 118
- failed: 0
- command: `python -m unittest discover -s 03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/pipelines/f06_evidence_rebuild/tests -p "test_*.py"`
- environment: clean detached worktree from the PR #6 patch branch

## 8. Safe Checks

| check | result |
|---|---|
| `validate_config` | PASS |
| `dry_run` good path | DRY_RUN_SCHEMA_VALIDATED |
| `dry_run --emit QUARANTINED_DO_NOT_USE.json` | BLOCKED_FORBIDDEN_OUTPUT_PATH |
| `dry_run --emit V50B_RERUN_TRADES.csv` | BLOCKED_FORBIDDEN_OUTPUT_PATH |
| bad emit artifact existence | absent |
| bad emit output dir existence | absent |
| `preflight_phase3` | PREFLIGHT_PHASE3_PASS |
| `prepare_phase3_run` | PHASE3_RUN_PREPARED |
| `run_phase3` without confirmation | BLOCKED_MISSING_EXPLICIT_REAL_RUN_CONFIRMATION |
| `run_phase3` with confirmation and no adapter | NOT_IMPLEMENTED_FAIL_CLOSED |

## 9. Safety Verification

- adapter_implemented: NO
- real_f06_run: NO
- backtest_run: NO
- validation_touched: NO
- holdout_touched: NO
- 2025_touched: NO
- 2026_touched: NO
- raw_data_mutated: NO
- old_outputs_used: NO
- zip_used_as_primary_delivery: NO

## 10. Decision

READY_FOR_CLAUDE_REAUDIT

This patch does not authorize adapter implementation, F06 execution, validation,
holdout, 2025/2026 access, demo, FTMO, or real trading.

## 11. Copy-Paste Summary for ChatGPT

STATUS: PR6_REPRO_EMIT_PATCH_READY_FOR_CLAUDE_REAUDIT
EMIT_GUARD: FIXED
SCRIPT_SHA_REPRODUCIBILITY: FIXED_FOR_FIXTURES_ONLY
REAL_OUTPUT_HASHING: STILL_REQUIRES_PHYSICAL_SHA256
STALE_REPORTS: CLEANED_OR_MARKED_HISTORICAL
TESTS: 118/118 PASS
SAFE_CHECKS: PASS_EXPECTED_FAIL_CLOSED_GATES
ADAPTER_IMPLEMENTED: NO
REAL_F06_RUN: NO
BACKTEST_RUN: NO
VALIDATION_TOUCHED: NO
HOLDOUT_TOUCHED: NO
2025_TOUCHED: NO
2026_TOUCHED: NO
F06_CERTIFIED: NO
NEXT_STEP: CLAUDE_REAUDIT
