# PR6 RUNNER HARDENING PATCH REPORT

## 1. Status
PR6_RUNNER_HARDENED_READY_FOR_CLAUDE_REAUDIT

## 2. Claude Blockers Addressed
| blocker | fix | test | status |
|---|---|---|---|
| `validate_config` accepted `symbol: USDJPY` | `_config_invariants` now requires `symbol == EURUSD` | `test_validate_config_rejects_usdjpy` | FIXED |
| `validate_config` accepted one exact month | exact month list must equal the five Phase 3 train months in order | `test_validate_config_rejects_one_exact_month_only` | FIXED |
| `validate_config` accepted missing `session` / `risk` | both mappings and exact values are required | `test_validate_config_rejects_missing_session`, `test_validate_config_rejects_missing_risk` | FIXED |
| `validate_config` accepted `output_dir_must_not_exist: false` | all `output_rules` booleans must be true | `test_validate_config_rejects_output_dir_must_not_exist_false` | FIXED |
| `prepare_phase3_run` reused output dir | output dir is checked absent and reserved with `os.mkdir` | `test_prepare_phase3_run_rejects_existing_output_dir` | FIXED |
| `prepare_phase3_run` wrote to quarantined path | config/output paths are checked for forbidden tokens before writes | `test_prepare_phase3_run_rejects_quarantined_path` | FIXED |
| `dry_run` was cwd-dependent | output dirs resolve from the pipeline root/repo root and must live under the Phase 3 report subtree | `test_dry_run_does_not_create_nested_project_tree_under_pipeline` | FIXED |
| engine report referenced missing `src/v7_engine` / `src/v6_utils` | report now states those paths are not present and identifies `research_lab/engine.py` as visible engine surface | report diff + grep check | FIXED |

## 3. Config Validation Hardening
`_config_invariants` now fails closed on missing or different values for:
- `phase == F06_PHASE3_CLEAN_TRAIN_ONLY_RERUN`
- `mode == TRAIN_ONLY`
- `symbol == EURUSD`
- `families == ["F06"]`
- `session.timezone/start/end == America/New_York / 07:00 / 17:00`
- `risk.max_trades_per_day == 3`
- `data_scope` flags all false
- exact train months in fixed order: `2020-03`, `2021-08`, `2022-05`, `2023-01`, `2024-04`
- all input/output rule booleans
- required spread/slippage/commission cost flags
- sample floors: family `>=100`, monthly reporting `>=10`

No silent defaults are accepted for these Phase 3 controls.

## 4. Path Handling Hardening
Added path helpers:
- `resolve_repo_root()`
- `resolve_output_dir(path)`
- `assert_output_under_allowed_reports(output_dir)`
- `assert_no_forbidden_path_tokens(path)`
- `reserve_output_dir_atomic(output_dir)`

Outputs are only allowed under:
`03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/f06_clean_train_only_rerun/`

Forbidden tokens remain blocked:
`QUARANTINED`, `DO_NOT_USE`, `v50b_limited_real_gauntlet_rerun_sw`, `V50B_RERUN_TRADES.csv`, `V50B_RERUN_MASTER_RANKING.csv`.

## 5. prepare_phase3_run Hardening
`prepare_phase3_run` now:
- validates strict config invariants,
- validates config/output paths,
- rejects existing output dirs,
- rejects paths outside the allowed Phase 3 report subtree,
- reserves output dirs atomically with `os.mkdir`,
- creates only `PRE_RUN_MANIFEST_DRAFT.json`, `COMMANDS_PLANNED.md`, and `SAFETY_PRECHECK.md`.

It still does not create trades, ranking, raw reads, backtests, or strategy outputs.

## 6. dry_run Hardening
`dry_run` now:
- resolves output paths independent of process cwd,
- rejects existing output dirs with `BLOCKED_OUTPUT_DIR_EXISTS`,
- rejects forbidden paths with `BLOCKED_FORBIDDEN_OUTPUT_PATH`,
- writes `DRYRUN_MANIFEST.json` inside the requested allowed output directory,
- does not create `pipelines/f06_evidence_rebuild/03_RESEARCH_LAB`.

## 7. Engine Interface Report Correction
`PHASE3_ENGINE_INTERFACE_AUDIT.md` now records:
- `src/v7_engine/`: not present in this checkout,
- `src/v6_utils/`: not present in this checkout,
- visible engine surface: `research_lab/engine.py`,
- visible callable surface: `research_lab.engine.run_backtest(...)`,
- next step after hardening: engine inventory / adapter design only.

## 8. Tests
- total: 112
- passed: 112
- failed: 0
- command: `python -m unittest discover -s 03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/pipelines/f06_evidence_rebuild/tests -p "test_*.py"`

## 9. Safe Checks
| check | result |
|---|---|
| validate_config | PASS |
| dry_run | DRY_RUN_SCHEMA_VALIDATED |
| preflight_phase3 | PREFLIGHT_PHASE3_PASS |
| prepare_phase3_run | PHASE3_RUN_PREPARED |
| run_phase3 without confirmation | BLOCKED_MISSING_EXPLICIT_REAL_RUN_CONFIRMATION |

## 10. Safety Verification
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

## 11. Decision
READY_FOR_CLAUDE_REAUDIT

This patch does not certify F06 and does not authorize adapter implementation. It only closes the runner hardening blockers raised by Claude so PR #6 can be re-audited.

## 12. Copy-Paste Summary for ChatGPT
STATUS: PR6_RUNNER_HARDENED_READY_FOR_CLAUDE_REAUDIT
TESTS: 112/112 PASS
SAFE_CHECKS: validate_config PASS; dry_run DRY_RUN_SCHEMA_VALIDATED; preflight_phase3 PREFLIGHT_PHASE3_PASS; prepare_phase3_run PHASE3_RUN_PREPARED; run_phase3 without confirmation BLOCKED_MISSING_EXPLICIT_REAL_RUN_CONFIRMATION
ADAPTER_IMPLEMENTED: NO
REAL_F06_RUN: NO
BACKTEST_RUN: NO
VALIDATION_TOUCHED: NO
HOLDOUT_TOUCHED: NO
2025_TOUCHED: NO
2026_TOUCHED: NO
F06_CERTIFIED: NO
NEXT_STEP: Claude re-audit of PR #6 runner hardening patch.
