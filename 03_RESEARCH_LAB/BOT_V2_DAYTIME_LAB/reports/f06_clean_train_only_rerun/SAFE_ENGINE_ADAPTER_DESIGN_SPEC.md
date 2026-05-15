# SAFE ENGINE ADAPTER DESIGN SPEC

Generated: 2026-05-15 — DESIGN ONLY. No executable adapter code is created by this document.
PR #6 head: d988c03ad7a60c0080a767baf740161346b1222c
Engine target: `research_lab/engine.py::run_backtest` (see SAFE_ENGINE_ADAPTER_ENGINE_INVENTORY.md).

## 1. Objective
Specify exactly how a future `phase3_f06_engine_adapter` will connect the PR #6 fail-closed runner (`f06_rebuild_pipeline.py run_phase3`) to the existing `research_lab` engine, **fail-closed**, train-only, F06-only, with full Phase 3 output-contract production and a mandatory post-run validator — without leakage and without improvisation at implementation time.

## 2. Non-Goals
- NO validation, NO holdout, NO WFA/OOS.
- NO 2025/2026 data.
- NO V50B/V50C, NO F08/F12, NO non-F06 family.
- NO optimization, NO sweep, NO parameter search.
- NO F06 certification, NO demo/FTMO/live.
- This spec does NOT authorize a real F06 run; implementing the adapter and running F06 are separate, independently gated steps.

## 3. Adapter Location (proposed, not created)
`03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/pipelines/f06_evidence_rebuild/adapters/phase3_f06_engine_adapter.py`
- New `adapters/` package, sibling to `scripts/`. It MUST NOT be created until the implementation gate (Section 12) is cleared.
- The adapter is the ONLY module allowed to import from `research_lab`. The runner never imports the engine directly.

## 4. Adapter Responsibilities
The adapter MUST:
- accept ONLY an already-strict-validated config object handed by the runner (re-run `_config_invariants`; refuse if not identical),
- refuse arbitrary configs (hash the config; record `config_sha256`; reject unknown keys),
- load ONLY the 5 exact train months (`2020-03, 2021-08, 2022-05, 2023-01, 2024-04`) for EURUSD,
- block 2025/2026 (and any non-train month) BEFORE building the engine `frame` (pre-load temporal guard),
- select EXACTLY the audited F06 strategy module(s); refuse any other family/strategy,
- inject and assert session = America/New_York 07:00–17:00 (via `params`/`SessionConfig`/`session_cutoff`),
- enforce and assert `max_trades_per_day == 3` (engine does not enforce this),
- pin cost model and assert all three components applied: real spread component, slippage component, round-turn commission (record per-trade `entry_spread_pips`, `entry_slippage_pips`, `entry_commission_usd`),
- pin determinism: fixed `DEFAULT_SEED`, pinned `execution_mode`/`cost_profile`/`intrabar_policy`, recorded in manifest,
- produce the full Phase 3 output contract (Section 9),
- generate `MANIFEST.json` + `HASHES.txt` (Section 8),
- execute the post-run validator and ABORT if it fails,
- never interpret/emit metrics if the validator fails (fail-closed; results institutionally void).

## 5. Adapter Forbidden Behavior
The adapter MUST refuse (fail-closed, non-zero exit, explicit STATUS) on:
- old/quarantined outputs or any path token: `QUARANTINED`, `DO_NOT_USE`, `v50b_limited_real_gauntlet_rerun_sw`, `V50B_RERUN_TRADES.csv`, `V50B_RERUN_MASTER_RANKING.csv`,
- any import of `research_lab/validation.py` or `research_lab/wfa.py` or any OOS/holdout/WFA call,
- validation columns in `RANKING.csv` (e.g. `N_val`, `PF_val`, `val_pass`),
- any 2025/2026 date anywhere in `TRADES.csv`/`RANKING.csv`,
- output dir already existing; more than one `run_id`,
- missing cost report / missing hashes / missing manifest,
- direct mutation of `research_lab` core, engine native results dir, or raw data,
- `*_BACKUP_*` or `reports/canonical_*` engine copies as the engine source.

## 6. Proposed Call Flow (NON-EXECUTABLE pseudocode)
```
run_phase3(config, output_dir, --confirm-real-run TOKEN)
  -> assert confirmation token exact                  # else BLOCKED_MISSING_EXPLICIT_REAL_RUN_CONFIRMATION
  -> _config_invariants(config) strict                # else BLOCKED_GUARD_FAILED
  -> check_phase3_output_path(output_dir)             # allowed subtree, must-not-exist
  -> reserve_output_dir_atomic(output_dir)            # os.mkdir, atomic
  -> run_id = new single run id
  -> adapter.preflight(config):
       assert symbol==EURUSD, families==["F06"], months==5 train months
       assert no 2025/2026, validation_enabled==false, holdout_enabled==false
       assert cost components required, session==NY 07:00-17:00, max_trades_per_day==3
  -> frame = adapter.load_train_only(config)          # 5 months EURUSD; 2025/2026 blocked pre-load
  -> engine_config = adapter.build_engine_config(config)   # pinned cost/session/seed/mode
  -> result = research_lab.engine.run_backtest(F06_module, frame, params, engine_config, ...)
  -> artifacts = adapter.build_contract(result, config, run_id)   # into TEMP dir only
  -> adapter.validate_output_contract(artifacts)      # structural+semantic; else abort, no publish
  -> adapter.atomic_publish(temp_dir -> final_dir)    # all-or-nothing
  -> validate_rebuild_outputs(final_dir)              # MUST print READY_FOR_CLAUDE_AUDIT
       if not READY_FOR_CLAUDE_AUDIT: STATUS=BLOCKED_GUARD_FAILED; do not interpret
  -> write PHASE3_CLEAN_F06_RERUN_REPORT.md           # final report
  -> STATUS: PHASE3_F06_TRAIN_ONLY_COMPLETE_PENDING_CLAUDE_AUDIT
```
On ANY failed step: print explicit `STATUS: BLOCKED_*`, exit 2, leave NO partial published artifacts.

## 7. Atomic Output Strategy
- temp dir: `<final_dir>.__tmp_<run_id>/` (same allowed subtree, must-not-exist, reserved with `os.mkdir`).
- final dir: `reports/f06_clean_train_only_rerun/run_<RUN_ID>/`.
- no partial publish: artifacts are fully built + contract-validated in temp, then a single atomic rename to final.
- cleanup rules: on any failure, remove temp dir; never leave a half-written final dir; final dir is created only by the atomic rename.
- failure behavior: fail-closed, exit 2, explicit STATUS; the run is institutionally void.

## 8. Manifest Strategy
`MANIFEST.json` mandatory fields:
- `run_id` (single), `generated_at`, `git_branch`, `git_commit_sha`,
- `config_path`, `config_sha256`, `script_sha256` (runner), `adapter_sha256`, `engine_sha256` (`research_lab/engine.py`), `engine_config_sha256`,
- `input_dataset_path`, `input_dataset_sha256_or_reference`, `input_is_quarantined_path: false`,
- `symbol: EURUSD`, `families: ["F06"]`, `exact_months` (the 5),
- `train_only: true`, `validation_evaluated: false`, `holdout_touched: false`, `allow_2025: false`, `allow_2026: false`,
- `row_count_input`, `trade_count`, `rejected_count`,
- `cost_model` (spread/slippage/round-turn asserted), `sample_size_floor`,
- `safety_flags` (test/validation/holdout/raw_data_mutated/sweep/optimization all false),
- `output_hashes` (every artifact → sha256, matching `HASHES.txt` and disk),
- `seed`, `execution_mode_used`, `cost_profile_used`, `intrabar_policy_used`, `status`.

## 9. Output Contract Mapping (per PHASE3_OUTPUT_CONTRACT.md)
| artifact | source |
|---|---|
| `MANIFEST.json` | adapter, Section 8 |
| `CONFIG_USED.yaml` | exact copy of the validated Phase 3 config |
| `COMMANDS_RUN.md` | the exact runner command(s) executed |
| `ENVIRONMENT_SUMMARY.json` | python/pkg versions, git, host (no secrets) |
| `ledger/TRADES.csv` | `BacktestResult.trades` (no 2025/2026; F06 only) |
| `ranking/RANKING.csv` | adapter-computed train-only ranking (NO validation columns) |
| `cost/COST_REPORT.json` | derived from per-`Position` spread/slippage/commission; asserts all three |
| `hashes/HASHES.txt` | sha256 of every produced artifact |
| `SAFETY_VERIFICATION.md` | explicit NO for adapter-bypass/validation/holdout/2025/2026/raw mutation |
| `PHASE3_CLEAN_F06_RERUN_REPORT.md` | final human/agent report; only written if validator passed |

Sample-size gate: `>= 100` trades for the F06 family; monthly reporting floor `>= 10`. Below floor → fail-closed.

## 10. Failure Modes
| failure | detection | action | status |
|---|---|---|---|
| confirmation token missing/wrong | runner arg check | abort pre-engine | BLOCKED_MISSING_EXPLICIT_REAL_RUN_CONFIRMATION |
| config not strictly valid | `_config_invariants` | abort pre-engine | BLOCKED_GUARD_FAILED |
| output path forbidden/exists | path guards | abort pre-engine | BLOCKED_FORBIDDEN_OUTPUT_PATH / BLOCKED_OUTPUT_DIR_EXISTS |
| 2025/2026 in data | pre-load temporal guard | abort pre-engine | BLOCKED_TEST_LEAKAGE_RISK |
| non-F06 family / non-EURUSD | preflight assert | abort pre-engine | BLOCKED_SCOPE_ESCALATION_DETECTED |
| validation/holdout import attempted | static import allowlist | abort | BLOCKED_GUARD_FAILED |
| cost component missing | COST_REPORT assert | abort, no publish | BLOCKED_GUARD_FAILED |
| sample size below floor | post-run count | abort, no publish | BLOCKED_GUARD_FAILED |
| contract/structure invalid | contract validator | abort, no publish | BLOCKED_GUARD_FAILED |
| hash mismatch | hash recompute | abort, no publish | BLOCKED_GUARD_FAILED |
| validator != READY_FOR_CLAUDE_AUDIT | mandatory validator | do not interpret | BLOCKED_GUARD_FAILED |
| engine raises | try/except around run_backtest | abort, cleanup temp | BLOCKED_GUARD_FAILED |

## 11. Required Tests for Implementation (future)
- adapter blocks 2025; blocks 2026 (pre-load).
- adapter blocks validation; blocks holdout; refuses importing `validation.py`/`wfa.py`.
- adapter blocks non-F06 family; blocks non-EURUSD.
- adapter blocks old/quarantined outputs and forbidden tokens.
- adapter blocks output dir already existing; enforces single run_id.
- adapter writes NO partial artifacts on any failure (temp-only, atomic publish).
- adapter requires cost report with all three components.
- adapter requires hashes that match disk + manifest.
- adapter calls the mandatory validator; refuses to interpret on validator failure.
- adapter pins seed/mode and records them; reproducible manifest.
- adapter does not write to engine native results dir.

## 12. Implementation Gate
Before ANY adapter code is written, ALL must hold:
1. PR #6 still: open, draft, no merge, 119/119 reproduced, safe checks green.
2. A dedicated read-only F06 engine-discovery pass maps: the exact F06 strategy module(s), the canonical train-data loader contract, and the exact `EngineConfig` fields for Phase 3.
3. This spec + risk register reviewed and approved by a Claude design audit.
4. Explicit written authorization to implement the adapter (implementation ≠ real F06 run; real run is a further separate gate).
