# LOCAL CLEANUP AND 119 TEST REPRODUCTION REPORT

Generated: 2026-05-15
Branch: research/f06-clean-train-only-rerun-20260515
Pre-cleanup-commit HEAD: 94ca831bfafa510993d96319045283ae13667141
Mode: surgical READ-MOSTLY remediation. No adapter, no real F06, no backtest, no validation/holdout/2025/2026.

## 1. Status
LOCAL_CLEANUP_COMPLETE_119_PASS

## 2. Executive Summary
The only blocker for SAFE_ENGINE_ADAPTER_IMPLEMENTATION_GATE was a single untracked, pre-existing,
foundation-era nested project tree polluting the audited workspace and tripping the precondition of
`test_dry_run_does_not_create_nested_project_tree_under_pipeline` (forced 118/119 instead of 119/119).
The tree contained exactly one 1586-byte dry-run manifest (no real data, no secrets, no tracked file),
produced by an earlier pre-hardening commit (8ea0b9e5). It was forensically documented, confirmed safe,
and surgically removed (only that untracked path). The full suite now reproduces 119/119 PASS in the
audited workspace, all four minimal safe checks pass, and the hardened runner does NOT re-create the
nested tree. Adapter, real F06, backtest, validation, holdout, and 2025/2026 remain fully blocked.

## 3. Cleanup Target
- path: 03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/pipelines/f06_evidence_rebuild/03_RESEARCH_LAB/
- existed_before: YES (Test-Path = True)
- tracked_files_inside: 0 (`git ls-files` empty; re-verified immediately before deletion)
- deleted: YES (Remove-Item -Recurse -Force, scoped to exactly this one untracked path)
- exists_after: NO (Test-Path = False)
- re_polluted_after_safe_checks: NO (path ABSENT after BLOQUE 5)

## 4. Safety Verification Before Delete
- raw_data_found: NO
- parquet_found: NO
- zip_found: NO
- secrets_found: NO (.key/.pem/.env scan empty)
- tracked_files_found: NO
- heavy_files_found: NO (single file, 1586 bytes; total tree 1586 bytes)
- content: 1 dry-run manifest (input_dataset_path=DRY_RUN_NO_INPUT, row_count_input=0, trade_count=0,
  validation_evaluated=false, holdout_touched=false, allow_2025=false, allow_2026=false,
  status=DRY_RUN_SCHEMA_VALIDATED) — fully reproducible byproduct, not valuable output
- safe_to_delete: YES (SAFE_TO_DELETE_UNTRACKED_POLLUTION; see LOCAL_UNTRACKED_POLLUTION_AUDIT.md)

## 5. Test Reproduction
- command: `python -m unittest discover -s 03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/pipelines/f06_evidence_rebuild/tests -p "test_*.py"`
- total: 119
- passed: 119
- failed: 0
- runtime: ~2.8s
- result_line: `Ran 119 tests in 2.809s` / `OK`

## 6. Safe Checks
- validate_config: PASS (exit 0) — "config invariants satisfied (TRAIN_ONLY, no 2025/2026, single family F06, cost components present, sample floor ok)"
- dry_run_good: DRY_RUN_SCHEMA_VALIDATED (exit 0) — manifest written under allowed reports subtree, NOT nested under pipeline
- dry_run_bad_emit_traversal: BLOCKED_FORBIDDEN_OUTPUT_PATH (exit 2) — "emit path escapes dry_run output_dir: ../MANIFEST.json"; output dir never created (fail-closed)
- run_phase3_without_confirmation: BLOCKED_MISSING_EXPLICIT_REAL_RUN_CONFIRMATION (exit 2)
- run_phase3_with_confirmation: NOT EXECUTED (intentionally out of scope; adapter still NOT_IMPLEMENTED_FAIL_CLOSED)

## 7. Git Status
- tracked_changes: NONE
- tracked_deletions: NONE (no tracked file removed)
- untracked_pollution_target_remaining: NONE (removed)
- other_preexisting_untracked_left_untouched: cost_hardening_v50b_train_only_20260515_1020/COST_HARDENING_BEST_CONFIGS.csv; prepared_claude_audit/; prepared_run/; v50b_limited_real_gauntlet_rerun_sw/trades/ (all intentionally NOT touched, NOT committed)
- safe_check_outputs_created (NOT committed): reports/f06_clean_train_only_rerun/dry_run_after_local_cleanup/
- files_to_commit (allowlist only):
  - 03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/f06_clean_train_only_rerun/LOCAL_UNTRACKED_POLLUTION_AUDIT.md
  - 03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/f06_clean_train_only_rerun/LOCAL_CLEANUP_AND_119_REPRO_REPORT.md

## 8. Safety Verification
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

## 9. Decision
READY_FOR_SAFE_ENGINE_ADAPTER_IMPLEMENTATION_GATE
(Local reproducibility blocker cleared. Implementation NOT authorized here — next gate is a Claude short
confirmation; until then adapter remains design-only and all real execution stays blocked.)

## 10. Copy-Paste Summary for ChatGPT
PR6 local pollution remediation COMPLETE.
- Untracked foundation-era nested tree `pipelines/f06_evidence_rebuild/03_RESEARCH_LAB/` (1 dry-run
  manifest, 1586 bytes, no tracked files, no data, no secrets) forensically documented and surgically
  removed. No tracked file deleted; other preexisting untracked artifacts untouched.
- Suite reproduces 119/119 PASS in the audited workspace (was 118/119).
- Safe checks: validate_config=PASS, dry_run_good=DRY_RUN_SCHEMA_VALIDATED,
  dry_run_bad_emit_traversal=BLOCKED_FORBIDDEN_OUTPUT_PATH,
  run_phase3_without_confirmation=BLOCKED_MISSING_EXPLICIT_REAL_RUN_CONFIRMATION.
- Adapter NOT implemented; real F06/backtest/validation/holdout/2025/2026 NOT touched; PR #6 stays
  open/draft, no merge, no ready conversion.
- Decision: READY_FOR_SAFE_ENGINE_ADAPTER_IMPLEMENTATION_GATE. Next step: Claude short confirmation,
  then adapter DESIGN-ONLY. Do NOT start adapter implementation or real F06 yet.
