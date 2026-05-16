# ROOT CANONICALIZATION APPLY REPORT

## 1. Status

ROOT_CANONICALIZATION_PARTIAL_OWNER_REVIEW_REQUIRED

Low/medium approved root canonicalization was applied conservatively. The root is materially cleaner, but it is not fully canonical because high-risk, data-like, research_lab, validation/2025/2026-sensitive, and code/workflow-reference rows remain blocked.

## 2. Executive Summary

Applied Option C - Hybrid Institutional under owner-approved constraints.

No high-risk move was applied. No data-like folder was moved. `research_lab` was not moved. No validation, holdout, 2025/2026, raw data, strategy, backtest, F06 real run, optimization, or sweep was executed.

The tracked root ZIP `000_PARA_CHATGPT.zip` was preserved locally under ignored quarantine, hash-verified, removed from Git, and removed from the root working copy.

## 3. Owner Decisions Applied

| decision | applied |
|---|---|
| Option C - Hybrid Institutional | YES |
| low/medium only | YES |
| high-risk blocked | YES |
| data-like folders blocked | YES |
| `research_lab` blocked | YES |
| ZIP policy applied | YES |

## 4. Root Before

| metric | value |
|---|---:|
| total root items | 122 |
| canonical folders present | 7 |
| root exceptions present | 7 |
| tracked noncanonical roots after Option C exceptions | 107 |
| ZIP present in root | YES |
| data-like root items present | 4 |

## 5. Files Moved

Moved using `git mv` only for approved move-map rows. Counts below are top-level move-map rows, not individual tracked paths.

| source | destination | risk | action | git command | status |
|---|---|---|---|---|---|
| `.mplconfig` | `04_INFRASTRUCTURE_ENGINEERING/legacy_ops/.mplconfig` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `00_READ_THIS_FIRST.md` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/00_READ_THIS_FIRST.md` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `01_CURRENT_PROJECT_STATUS.json` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/01_CURRENT_PROJECT_STATUS.json` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `01_CURRENT_PROJECT_STATUS.md` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/01_CURRENT_PROJECT_STATUS.md` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `02_STRATEGY_AUTHORITY_MAP.json` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/02_STRATEGY_AUTHORITY_MAP.json` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `02_STRATEGY_AUTHORITY_MAP.md` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/02_STRATEGY_AUTHORITY_MAP.md` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `03_OBSOLETE_AND_SUPERSEDED_INDEX.json` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/03_OBSOLETE_AND_SUPERSEDED_INDEX.json` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `03_OBSOLETE_AND_SUPERSEDED_INDEX.md` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/03_OBSOLETE_AND_SUPERSEDED_INDEX.md` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `ABRIR_MANIPULANTE_AQUI.txt` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/ABRIR_MANIPULANTE_AQUI.txt` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `AUDITORIA_EJECUCION_FINAL.md` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/AUDITORIA_EJECUCION_FINAL.md` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `BLS_ACCESS_EVIDENCE_REPORT.md` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/BLS_ACCESS_EVIDENCE_REPORT.md` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `bls_debug_fetch.py` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/bls_debug_fetch.py` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `BLS_HYBRID_ACCESS_COMPLETE_REPORT.md` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/BLS_HYBRID_ACCESS_COMPLETE_REPORT.md` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `CANONICAL_EXECUTION_CONTRACT.md` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/CANONICAL_EXECUTION_CONTRACT.md` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `CHANGELOG.md` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/CHANGELOG.md` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `CLOUD_WORKFLOW.md` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/CLOUD_WORKFLOW.md` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `CPI_PPI_INSTRUCTIONS.md` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/CPI_PPI_INSTRUCTIONS.md` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `CPI_PPI_manual_fill_template.json` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/CPI_PPI_manual_fill_template.json` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `CPI_PPI_SOURCE_AUDIT.md` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/CPI_PPI_SOURCE_AUDIT.md` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `docs` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/docs` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `ESTRUCTURA_DEL_PROYECTO.md` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/ESTRUCTURA_DEL_PROYECTO.md` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `INFRASTRUCTURE_STATUS_FINAL.md` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/INFRASTRUCTURE_STATUS_FINAL.md` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `OOS_REJECTION_PROTOCOL.md` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/OOS_REJECTION_PROTOCOL.md` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `PROJECT_ZONES_AND_BRANCHING_RULES.md` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/PROJECT_ZONES_AND_BRANCHING_RULES.md` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `STRATEGY_PROMOTION_POLICY.md` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/STRATEGY_PROMOTION_POLICY.md` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `VSE_BASELINE_REPORT.md` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/VSE_BASELINE_REPORT.md` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `untracked_files.txt` | `06_GOVERNANCE_AND_COMPLIANCE/legacy_root_docs/untracked_files.txt` | LOW | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `htf_ny_window_scbi_stage2_checkpoints` | `03_RESEARCH_LAB/legacy_root_research/htf_ny_window_scbi_stage2_checkpoints` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `micro_pilot_gate` | `03_RESEARCH_LAB/legacy_root_research/micro_pilot_gate` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `next_hypothesis_discovery_checkpoints` | `03_RESEARCH_LAB/legacy_root_research/next_hypothesis_discovery_checkpoints` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `real_htf_filter_ab_checkpoints` | `03_RESEARCH_LAB/legacy_root_research/real_htf_filter_ab_checkpoints` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `research_scripts` | `03_RESEARCH_LAB/legacy_root_research/research_scripts` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `scbi_full_campaign_checkpoints` | `03_RESEARCH_LAB/legacy_root_research/scbi_full_campaign_checkpoints` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `mt5_deployment_audit` | `04_INFRASTRUCTURE_ENGINEERING/legacy_ops/mt5_deployment_audit` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `ops_external` | `04_INFRASTRUCTURE_ENGINEERING/legacy_ops/ops_external` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `real_readiness_gate` | `04_INFRASTRUCTURE_ENGINEERING/legacy_ops/real_readiness_gate` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `news_impact_analysis.py` | `04_INFRASTRUCTURE_ENGINEERING/legacy_scripts/news_impact_analysis.py` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `phase35_config_audit.py` | `04_INFRASTRUCTURE_ENGINEERING/legacy_scripts/phase35_config_audit.py` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `phase35_mt5_safety.py` | `04_INFRASTRUCTURE_ENGINEERING/legacy_scripts/phase35_mt5_safety.py` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `phase35_preflight.py` | `04_INFRASTRUCTURE_ENGINEERING/legacy_scripts/phase35_preflight.py` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `phase35_python_audit.py` | `04_INFRASTRUCTURE_ENGINEERING/legacy_scripts/phase35_python_audit.py` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `phase35_repo_zip_audit.py` | `04_INFRASTRUCTURE_ENGINEERING/legacy_scripts/phase35_repo_zip_audit.py` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `phase35_safety_gates.py` | `04_INFRASTRUCTURE_ENGINEERING/legacy_scripts/phase35_safety_gates.py` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `phase35_signal_sync.py` | `04_INFRASTRUCTURE_ENGINEERING/legacy_scripts/phase35_signal_sync.py` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `phase35_time_audit.py` | `04_INFRASTRUCTURE_ENGINEERING/legacy_scripts/phase35_time_audit.py` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `phase35_update_master_docs.py` | `04_INFRASTRUCTURE_ENGINEERING/legacy_scripts/phase35_update_master_docs.py` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `run_canonical.py` | `04_INFRASTRUCTURE_ENGINEERING/legacy_scripts/run_canonical.py` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `tests_external` | `04_INFRASTRUCTURE_ENGINEERING/legacy_tests/tests_external` | MEDIUM | MOVE_TO_CANONICAL_FOLDER | `git mv` | MOVED |
| `legacy` | `07_BACKUPS/legacy_archives/legacy` | MEDIUM | ARCHIVE_TO_07_BACKUPS | `git mv` | MOVED |
| `bypass_zip_output.json` | `07_BACKUPS/legacy_zip_workflow/bypass_zip_output.json` | LOW | ARCHIVE_TO_07_BACKUPS | `git mv` | MOVED |
| `debug_zip_identity_output.json` | `07_BACKUPS/legacy_zip_workflow/debug_zip_identity_output.json` | LOW | ARCHIVE_TO_07_BACKUPS | `git mv` | MOVED |
| `desktop_bypass_output.json` | `07_BACKUPS/legacy_zip_workflow/desktop_bypass_output.json` | LOW | ARCHIVE_TO_07_BACKUPS | `git mv` | MOVED |
| `force_visible_zip_output.json` | `07_BACKUPS/legacy_zip_workflow/force_visible_zip_output.json` | LOW | ARCHIVE_TO_07_BACKUPS | `git mv` | MOVED |
| `LEER_PARA_SUBIR_ZIP.txt` | `07_BACKUPS/legacy_zip_workflow/LEER_PARA_SUBIR_ZIP.txt` | LOW | ARCHIVE_TO_07_BACKUPS | `git mv` | MOVED |
| `SUBIR_ESTE_ZIP_A_CHATGPT.txt` | `07_BACKUPS/legacy_zip_workflow/SUBIR_ESTE_ZIP_A_CHATGPT.txt` | LOW | ARCHIVE_TO_07_BACKUPS | `git mv` | MOVED |
| `ZIP_CONTENTS_MANIFEST.md` | `07_BACKUPS/legacy_zip_workflow/ZIP_CONTENTS_MANIFEST.md` | LOW | ARCHIVE_TO_07_BACKUPS | `git mv` | MOVED |
| `ZIP_VALIDADO_SUBIR_ESTE.txt` | `07_BACKUPS/legacy_zip_workflow/ZIP_VALIDADO_SUBIR_ESTE.txt` | LOW | ARCHIVE_TO_07_BACKUPS | `git mv` | MOVED |
| `zip_builder.py` | `07_BACKUPS/legacy_zip_workflow/zip_builder.py` | LOW | ARCHIVE_TO_07_BACKUPS | `git mv` | MOVED |

Tracked path impact:

| metric | count |
|---|---:|
| moved top-level move-map rows | 58 |
| renamed tracked paths | 115 |
| low-risk rows moved | 35 |
| medium-risk rows moved | 23 |

## 6. Files Removed From Git But Preserved Locally

| file | original hash | quarantine hash | local quarantine path | root copy removed | Git status |
|---|---|---|---|---|---|
| `000_PARA_CHATGPT.zip` | `F37FD660206ACE6B881C0EEE99FE13DB90F0369BDD4E13292ECC6ED5788D6C47` | `F37FD660206ACE6B881C0EEE99FE13DB90F0369BDD4E13292ECC6ED5788D6C47` | `_LOCAL_QUARANTINE_DO_NOT_COMMIT/root_zip_legacy/000_PARA_CHATGPT.zip` | YES | removed from Git |

The quarantine copy is ignored by Git through `_LOCAL_QUARANTINE_DO_NOT_COMMIT/`.

## 7. Files / Rows Skipped

Skipped rows:

| reason | count | future phase needed |
|---|---:|---|
| HIGH_RISK_BLOCKED | 10 | high-risk phase |
| CONTAINS_SENSITIVE_DATA_OR_FORBIDDEN_YEAR_PATHS | 18 | data/scope audit |
| MEDIUM_CODE_OR_WORKFLOW_REFERENCES_BLOCKED | 17 | path/import/workflow migration |
| VALIDATION_HOLDOUT_2025_2026_SCOPE_BLOCK | 6 | explicit scope approval |
| NOT_A_MOVE_ARCHIVE_ROW | 6 | no action required or separate policy |

Known skipped examples:

| item | reason | future phase |
|---|---|---|
| `research_lab` | root exception / high import risk | import migration |
| `DATA MANUAL` and `data_usdjpy_*` | data-like folders blocked | data audit |
| `MANIPULANTE`, `ROCKI_AM`, `ESTRATEGIAS`, `STRATEGIES`, `LAB_STRATEGIES` | high-risk strategy surfaces | high-risk strategy authority phase |
| `BOT_V2_DAYTIME_LAB` | contains sensitive/forbidden scoped paths and workflow refs | dedicated path migration |
| `reports`, `audits`, `scratch`, `scripts` | contain sensitive/forbidden scoped paths or code refs | separate review |
| `validation_check.py`, `phase34_manipulante_validation.py` | validation scope string | explicit validation-safe phase |
| `COMPARABILITY_2020_2025_NOTE.md`, `scbi_2020_2025_durability_checkpoints`, `legacy_archive_2026` | 2025/2026 scope strings | explicit year-scope approval |

## 8. Path / Import Reference Audit

Restricted reference audit was run excluding `.git`, `_LOCAL_QUARANTINE_DO_NOT_COMMIT`, `05_MARKET_DATA_VAULT`, `07_BACKUPS`, ZIPs, parquet, CSV, and JSONL.

| metric | result |
|---|---:|
| moved top-level items audited | 58 |
| no/blocker-free moved items | 28 |
| review references remaining | 30 |

The remaining references are treated as non-blocking for this phase because:

1. medium items with code/workflow references were blocked before move,
2. `research_lab` import smoke passed,
3. F06 pipeline unit tests passed,
4. no strategy/backtest/runtime path was executed.

No broad import rewrite was performed.

## 9. Root After

| metric | value |
|---|---:|
| total root items | 64 |
| allowed root items present | 16 |
| remaining unexpected root items | 48 |
| tracked noncanonical roots remaining | 48 |
| ZIP present in root | NO |
| ZIP quarantine copy exists | YES |

Allowed root items present:

`.git`, `.gitignore`, `.github`, `_LOCAL_QUARANTINE_DO_NOT_COMMIT`, `01_CORE_PRODUCTION`, `02_INCUBATION_STAGING`, `03_RESEARCH_LAB`, `04_INFRASTRUCTURE_ENGINEERING`, `05_MARKET_DATA_VAULT`, `06_GOVERNANCE_AND_COMPLIANCE`, `07_BACKUPS`, `08_CLOUD_FREE_RUN_LAB`, `README.md`, `requirements.txt`, `requirements-vps-optional.txt`, `research_lab`.

Remaining unexpected root items are intentionally blocked/review items, including data-like folders, high-risk strategy folders, validation/2025/2026-sensitive names, and code/workflow-reference blocked items.

## 10. Tests / Checks

| command | result | notes |
|---|---|---|
| `git status --short` | PASS | expected staged moves/deletion plus new report/.gitkeep before commit |
| `python -c "import research_lab; print('research_lab import OK')"` | PASS | `research_lab import OK` |
| `python -m unittest discover -s "03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\pipelines\f06_evidence_rebuild\tests" -p "test_*.py"` | PASS | 119 tests ran, 119 passed |
| ZIP SHA256 match | PASS | original and quarantine hashes matched |
| root allowlist check | PARTIAL | 48 blocked/review items remain |

## 11. Safety Verification

| check | result |
|---|---|
| tracked_files_deleted_without_approval | NO |
| tracked_files_moved_outside_move_map | NO |
| raw_data_touched | NO |
| validation_touched | NO |
| holdout_touched | NO |
| 2025_touched | NO |
| 2026_touched | NO |
| backtest_run | NO |
| strategy_run | NO |
| force_push | NO |
| main_touched | NO |

## 12. Remaining Work

1. High-risk phase for strategy authority/root migration decisions.
2. Separate data audit before moving `DATA MANUAL`, `data_usdjpy_*`, or any data-like folder.
3. Dedicated `research_lab` import/path migration if strict root cleanup is later desired.
4. Dedicated path/workflow migration for medium rows blocked by code or GitHub workflow references.
5. Optional final strict-root pass after the above gates.

## 13. Copy-Paste Summary for ChatGPT

ROOT_CANONICALIZATION_PARTIAL_OWNER_REVIEW_REQUIRED. Option C low/medium root canonicalization was applied conservatively. 58 approved move-map rows were moved with `git mv`, representing 115 tracked path renames. `000_PARA_CHATGPT.zip` was hash-preserved in ignored local quarantine and removed from Git/root. `08_CLOUD_FREE_RUN_LAB/.gitkeep` was created. High-risk rows, data-like folders, `research_lab`, validation/holdout/2025/2026-sensitive rows, and medium rows with code/workflow references remain blocked. `research_lab` import smoke passed and F06 pipeline unit tests passed 119/119. No backtest, strategy, F06 real run, validation, holdout, 2025/2026, or raw data mutation occurred.
