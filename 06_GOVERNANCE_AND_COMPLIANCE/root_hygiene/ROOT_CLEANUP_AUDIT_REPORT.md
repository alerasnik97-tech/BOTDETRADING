# ROOT CLEANUP AUDIT REPORT

## 1. Status

ROOT_CLEANUP_PARTIAL_OWNER_REVIEW_REQUIRED

## 2. Executive Summary

The local project root was audited under a non-destructive repo hygiene scope. No backtest, strategy, F06 runner, optimizer, validation, holdout, 2025, 2026, raw, tick, or parquet data workflow was executed.

The disorder is not mainly untracked local trash. The root contains many non-canonical items that are already tracked by Git, including `000_PARA_CHATGPT.zip`. Those files cannot be moved or deleted safely in this scope without a separate repository restructuring/change-control decision.

Six untracked generated output paths were moved into an ignored local quarantine folder. No tracked file was deleted or moved.

## 3. Initial Root Inventory

Precheck context:

| field | value |
|---|---|
| repo_root | `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo` |
| initial_branch | `research/f06-d5-behavior-neutral-telemetry-20260516` |
| hygiene_branch | `governance/root-hygiene-20260516` |
| initial_head | `ef89fddc52f14657fb2fa15712f2fdd767331b98` |
| python_processes_detected | `NO` |

Root classification summary before cleanup:

| classification | count | action |
|---|---:|---|
| KEEP_REQUIRED canonical or required root items present | 9 | keep |
| Missing canonical root item | 1 | owner review: `08_CLOUD_FREE_RUN_LAB` absent |
| NEEDS_OWNER_REVIEW tracked non-canonical root items | 112 | keep in place; separate structural cleanup required |
| MOVE_TO_LOCAL_QUARANTINE root items | 0 | none |
| SAFE_DELETE root cache items | 0 | none |
| Nested untracked generated output paths | 6 | moved to local quarantine |

Tracked non-canonical root items observed and left untouched:

`.github`, `.mplconfig`, `.tmp.driveupload`, `00_READ_THIS_FIRST.md`, `000_PARA_CHATGPT.zip`, `01_CURRENT_PROJECT_STATUS.json`, `01_CURRENT_PROJECT_STATUS.md`, `02_STRATEGY_AUTHORITY_MAP.json`, `02_STRATEGY_AUTHORITY_MAP.md`, `03_OBSOLETE_AND_SUPERSEDED_INDEX.json`, `03_OBSOLETE_AND_SUPERSEDED_INDEX.md`, `ABRIR_MANIPULANTE_AQUI.txt`, `AUDITORIA_EJECUCION_FINAL.md`, `audits`, `BLS_ACCESS_EVIDENCE_REPORT.md`, `bls_debug_fetch.py`, `bls_html_samples`, `BLS_HYBRID_ACCESS_COMPLETE_REPORT.md`, `bootstrap.ini`, `BOT_V2_DAYTIME_LAB`, `BOT_V2_profile_bootstrap_hack.py`, `bypass_zip_output.json`, `CANONICAL_EXECUTION_CONTRACT.md`, `CHANGELOG.md`, `CLOUD_WORKFLOW.md`, `COMPARABILITY_2020_2025_NOTE.md`, `CPI_PPI_INSTRUCTIONS.md`, `CPI_PPI_manual_fill_template.json`, `CPI_PPI_SOURCE_AUDIT.md`, `DATA MANUAL`, `data_usdjpy_2016_2019`, `data_usdjpy_2016_2021`, `data_usdjpy_2022_2025`, `debug_zip_identity_output.json`, `desktop_bypass_output.json`, `docs`, `ecb_stage2_checkpoints`, `ESTRATEGIAS`, `ESTRUCTURA_DEL_PROYECTO.md`, `external_scbi_research_harness`, `force_visible_zip_output.json`, `git_operations.py`, `htf_ny_window_scbi_stage2_checkpoints`, `INFRASTRUCTURE_STATUS_FINAL.md`, `institutional_research_candidate_lab`, `inventory_check.py`, `LAB_STRATEGIES`, `LEER_PARA_SUBIR_ZIP.txt`, `legacy`, `legacy_archive_2026`, `MANIPULANTE`, `manual_trade_chartpacks`, `micro_pilot_gate`, `micro_pilot_protocol`, `monitoring`, `mt5_demo_executor_lab`, `mt5_deployment_audit`, `news_impact_analysis.py`, `news_impact_analysis_v2.py`, `next_hypothesis_discovery_checkpoints`, `OOS_REJECTION_PROTOCOL.md`, `ops_external`, `phase34_core_docs_audit.py`, `phase34_git_push.py`, `phase34_manipulante_shadow_sync.py`, `phase34_manipulante_validation.py`, `phase34_path_audit.py`, `phase34_preflight.py`, `phase34_python_path_audit.py`, `phase34_report_generator.py`, `phase34_update_master_docs.py`, `phase35_config_audit.py`, `phase35_mt5_safety.py`, `phase35_preflight.py`, `phase35_python_audit.py`, `phase35_repo_zip_audit.py`, `phase35_safety_gates.py`, `phase35_signal_sync.py`, `phase35_structure_audit.py`, `phase35_time_audit.py`, `phase35_update_master_docs.py`, `phase37e_run_mql5_calendar_script.py`, `preflight_check.py`, `PROJECT_ZONES_AND_BRANCHING_RULES.md`, `README.md`, `real_htf_filter_ab_checkpoints`, `real_readiness_gate`, `reports`, `requirements.txt`, `requirements-vps-optional.txt`, `research_lab`, `research_scripts`, `results_REHEARSAL`, `ROCKI_AM`, `run_canonical.py`, `scbi_2020_2025_durability_checkpoints`, `scbi_full_campaign_checkpoints`, `scbi_global_validation_checkpoints`, `scratch`, `scripts`, `shadow_line_lab`, `STRATEGIES`, `STRATEGY_PROMOTION_POLICY.md`, `SUBIR_ESTE_ZIP_A_CHATGPT.txt`, `tests_external`, `untracked_files.txt`, `validation_check.py`, `VPS_READINESS`, `VSE_BASELINE_REPORT.md`, `zip_builder.py`, `ZIP_CONTENTS_MANIFEST.md`, `ZIP_VALIDADO_SUBIR_ESTE.txt`.

## 4. What Caused the Disorder

Evidence indicates two separate causes:

1. The visible root clutter is mostly tracked repository content, not local accidental files. This includes root reports, scripts, historical workflow files, legacy folders, USDJPY data folders, and `000_PARA_CHATGPT.zip`.
2. The working tree was also dirty from generated, untracked research outputs under `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/...`. These were local output artifacts and were safe to quarantine because Git reported them as untracked.

The root cannot be made visually canonical in one safe cleanup pass without a broader tracked-file restructuring plan.

## 5. Actions Taken

| path | action | reason | tracked | ignored | size |
|---|---|---|---|---|---:|
| `.gitignore` | updated | added `_LOCAL_QUARANTINE_DO_NOT_COMMIT/` so quarantine cannot be committed accidentally | YES | NO | n/a |
| `_LOCAL_QUARANTINE_DO_NOT_COMMIT/root_cleanup_20260516_015032/` | created | local safety quarantine for untracked artifacts | NO | YES | n/a |
| `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/cost_hardening_v50b_train_only_20260515_1020/COST_HARDENING_BEST_CONFIGS.csv` | moved to quarantine | untracked generated output | NO | NO | 375 |
| `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/f06_clean_train_only_rerun/dry_run_after_local_cleanup/` | moved to quarantine | untracked dry-run output | NO | NO | 1586 |
| `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/f06_clean_train_only_rerun/dry_run_final_confirmation/` | moved to quarantine | untracked dry-run output | NO | NO | 1586 |
| `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/f06_clean_train_only_rerun/prepared_claude_audit/` | moved to quarantine | untracked prepared artifact output | NO | NO | 244 |
| `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/f06_clean_train_only_rerun/prepared_run/` | moved to quarantine | untracked prepared artifact output | NO | NO | 235 |
| `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/v50b_limited_real_gauntlet_rerun_sw/trades/` | moved to quarantine | untracked generated trades output; preserved, not deleted | NO | NO | 22255391 |

## 6. Items Moved To Quarantine

Quarantine root:

`_LOCAL_QUARANTINE_DO_NOT_COMMIT/root_cleanup_20260516_015032/`

Moved items:

| original path | quarantine destination |
|---|---|
| `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/cost_hardening_v50b_train_only_20260515_1020/COST_HARDENING_BEST_CONFIGS.csv` | `_LOCAL_QUARANTINE_DO_NOT_COMMIT/root_cleanup_20260516_015032/03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/cost_hardening_v50b_train_only_20260515_1020/COST_HARDENING_BEST_CONFIGS.csv` |
| `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/f06_clean_train_only_rerun/dry_run_after_local_cleanup/` | `_LOCAL_QUARANTINE_DO_NOT_COMMIT/root_cleanup_20260516_015032/03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/f06_clean_train_only_rerun/dry_run_after_local_cleanup/` |
| `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/f06_clean_train_only_rerun/dry_run_final_confirmation/` | `_LOCAL_QUARANTINE_DO_NOT_COMMIT/root_cleanup_20260516_015032/03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/f06_clean_train_only_rerun/dry_run_final_confirmation/` |
| `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/f06_clean_train_only_rerun/prepared_claude_audit/` | `_LOCAL_QUARANTINE_DO_NOT_COMMIT/root_cleanup_20260516_015032/03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/f06_clean_train_only_rerun/prepared_claude_audit/` |
| `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/f06_clean_train_only_rerun/prepared_run/` | `_LOCAL_QUARANTINE_DO_NOT_COMMIT/root_cleanup_20260516_015032/03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/f06_clean_train_only_rerun/prepared_run/` |
| `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/v50b_limited_real_gauntlet_rerun_sw/trades/` | `_LOCAL_QUARANTINE_DO_NOT_COMMIT/root_cleanup_20260516_015032/03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/v50b_limited_real_gauntlet_rerun_sw/trades/` |

## 7. Items Deleted

None.

No cache deletion was performed because there were no root-level cache folders requiring removal. Ignored nested caches remain untouched.

## 8. Items Requiring Owner Review

Owner review is required for any future visual root cleanup of tracked non-canonical root files and folders. The largest governance issue is that `000_PARA_CHATGPT.zip` is tracked even though the current project policy prohibits ZIP workflow and `.gitignore` now blocks ZIPs. This cannot be fixed by local deletion because it is versioned content and would require an explicit tracked-file removal/restructure commit.

Additional owner-review items:

| item | issue | safe next step |
|---|---|---|
| `000_PARA_CHATGPT.zip` | tracked root ZIP conflicts with current no-ZIP policy | separate tracked-removal decision |
| root `phase34_*`, `phase35_*`, `news_impact_*`, `zip_builder.py`, `validation_check.py`, `preflight_check.py` | tracked scripts live in root instead of infrastructure/governance folders | separate move map with tests/import checks |
| root legacy folders such as `MANIPULANTE`, `BOT_V2_DAYTIME_LAB`, `reports`, `scratch`, `legacy*`, checkpoint folders | tracked or historically significant root folders outside canonical root policy | separate repository restructure plan |
| `data_usdjpy_*` and `DATA MANUAL` | tracked data-like root folders; may contain important historical content | do not move without owner/data inventory |
| missing `08_CLOUD_FREE_RUN_LAB` | canonical folder absent | create in separate root-canonicalization commit if still desired |

## 9. Final Root Inventory

Final root summary:

| classification | count | notes |
|---|---:|---|
| Total root items after cleanup | 122 | includes ignored local quarantine folder |
| Canonical/required/local quarantine root items | 10 | includes `.git`, `.gitignore`, canonical folders present, and ignored quarantine |
| Tracked non-canonical root items remaining | 112 | not moved; owner review required |
| Untracked non-canonical root items remaining | 0 | no remaining root move candidates |
| Deleted items | 0 | no deletion performed |

`git status --short` after quarantine showed only the expected `.gitignore` modification before report creation. `git status --short --ignored` shows the ignored quarantine and existing ignored local/cache/data-vault surfaces.

## 10. Safety Verification

| check | result |
|---|---|
| git_tracked_files_deleted | NO |
| git_tracked_files_moved | NO |
| raw_data_touched | NO |
| validation_touched | NO |
| holdout_touched | NO |
| 2025_touched | NO |
| 2026_touched | NO |
| backtest_run | NO |
| strategy_run | NO |
| force_push | NO |
| main_touched | NO |
| `.git` touched | NO |

## 11. Recommended Permanent Fixes

1. Approve a separate root-canonicalization plan for tracked root files. This should be a real repo restructure, not a local cleanup.
2. Decide whether tracked ZIPs, especially `000_PARA_CHATGPT.zip`, should be removed from the repository in a dedicated commit.
3. Create a root allowlist enforcement check that permits only canonical project folders, `.gitignore`, `.github` if required, and explicitly approved root docs.
4. Require all dry-run and agent outputs to write under dated report subfolders or ignored local quarantine, never directly into the root.
5. Add a lightweight root hygiene CI/preflight that fails when new non-allowlisted root files appear.

## 12. Copy-Paste Summary for ChatGPT

ROOT_CLEANUP_PARTIAL_OWNER_REVIEW_REQUIRED. The repo root was audited without executing trading, strategy, F06, validation, holdout, 2025, 2026, or raw data workflows. Six untracked generated outputs were moved to ignored local quarantine under `_LOCAL_QUARANTINE_DO_NOT_COMMIT/root_cleanup_20260516_015032/`. No tracked files were deleted or moved. The visible root remains non-canonical because 112 non-canonical root items are tracked by Git, including `000_PARA_CHATGPT.zip`; this requires a separate owner-approved tracked restructure/removal plan.
