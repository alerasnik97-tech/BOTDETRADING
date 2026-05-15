# LOCAL UNTRACKED POLLUTION AUDIT

Generated: 2026-05-15 (READ-ONLY forensic snapshot, written BEFORE any deletion)
Branch: research/f06-clean-train-only-rerun-20260515
HEAD: 94ca831bfafa510993d96319045283ae13667141
Scope: surgical cleanup of one untracked nested project tree blocking 119/119 reproduction.

## 1. Status Before Cleanup
- folder_exists: TRUE
- target_path: 03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/pipelines/f06_evidence_rebuild/03_RESEARCH_LAB/
- git_tracked_files_inside: 0  (`git ls-files -- <path>` returned empty)
- git_status_classification: "?? " (untracked, whole directory)
- untracked_files_count: 1
- nested_dirs_count: 3 (BOT_V2_DAYTIME_LAB / reports / f06_evidence_rebuild_foundation)
- total_size_bytes: 1586
- sensitive_or_heavy_files_found: NONE (danger scan for .parquet/.zip/.pkl/.h5/.hdf5/.sqlite/.db/.key/.pem/.env returned empty)
- decision: SAFE_TO_DELETE_UNTRACKED_POLLUTION

## 2. Contents Summary
| Relative path (under target) | Type | Size (bytes) | LastWriteTime |
|---|---|---|---|
| \BOT_V2_DAYTIME_LAB | DIR | - | 2026-05-15 12:40:34 |
| \BOT_V2_DAYTIME_LAB\reports | DIR | - | 2026-05-15 12:40:34 |
| \BOT_V2_DAYTIME_LAB\reports\f06_evidence_rebuild_foundation | DIR | - | 2026-05-15 12:40:34 |
| \BOT_V2_DAYTIME_LAB\reports\f06_evidence_rebuild_foundation\DRYRUN_MANIFEST.json | FILE | 1586 | 2026-05-15 13:56:12 |

Only file content (DRYRUN_MANIFEST.json) key fields:
- run_id: RBf9d4471a
- generated_at: 2026-05-15T16:56:12Z
- git_commit_sha: 8ea0b9e5882724c0abe40d42403099d5db877942 (earlier pre-hardening commit on this branch)
- input_dataset_path: DRY_RUN_NO_INPUT
- input_dataset_sha256_or_reference: DRY_RUN_NO_INPUT
- row_count_input: 0 / trade_count: 0 / rejected_count: 0
- train_only: true / validation_evaluated: false / holdout_touched: false / allow_2025: false / allow_2026: false
- safety_flags: test_touched=false, validation_touched=false, holdout_touched=false, raw_data_mutated=false, sweep_run=false, optimization_run=false
- status: DRY_RUN_SCHEMA_VALIDATED

## 3. Risk Assessment
- trackable code risk: NONE — no git-tracked file inside (`git ls-files` empty); deletion cannot remove tracked content.
- raw data risk: NONE — input_dataset_path = DRY_RUN_NO_INPUT, row_count_input = 0; no tick/parquet/csv data present.
- secrets risk: NONE — no .key/.pem/.env or credential files; sole file is a JSON dry-run manifest.
- heavy/binary risk: NONE — single 1586-byte JSON; total tree = 1586 bytes.
- evidence contamination risk: NONE for real evidence (no real outputs); the artifact itself is a fully reproducible dry-run byproduct. However its PRESENCE contaminates the audited workspace and trips the precondition of `test_dry_run_does_not_create_nested_project_tree_under_pipeline`, forcing 118/119 instead of 119/119.
- provenance: foundation-era / pre-final-hardening stale artifact (path references `f06_evidence_rebuild_foundation`; produced by commit 8ea0b9e5 before the path-hardening commits 1b43b031 / 94ca831b). NOT part of PR #6 diff.
- deletion safety: SAFE — only this untracked path will be removed; no tracked file, no raw data, no secrets, no valuable output.

## 4. Cleanup Decision
SAFE_TO_DELETE_UNTRACKED_POLLUTION
