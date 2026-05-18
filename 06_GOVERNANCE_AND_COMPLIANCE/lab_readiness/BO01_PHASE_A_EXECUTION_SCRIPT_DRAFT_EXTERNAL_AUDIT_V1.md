# BO01 PHASE A EXECUTION SCRIPT DRAFT EXTERNAL AUDIT V1

## 1. Audit Status

AUDIT_BLOCKED_MANIFEST_INCONSISTENT

## 2. Executive Verdict

The Phase A-0 script draft remains blocked. The local script exists, is gitignored, and
its physical SHA256 matches the declared script hash, but the local manifest provenance
does not match the commit under audit. This prevents a clean chain of custody before any
future Phase A-1 decision.

No Python was executed. The script was not executed or imported. No data was loaded. No
CSV was read. No backtest, train, validation, holdout, 2025/2026 access, optimization,
sweep, demo, real, or FTMO workflow was run.

This audit does not evaluate edge, profitability, or strategy readiness.

## 3. Scope Audited

- branch: `audit/bo01-phase-a0-execution-script-draft-v1-20260518`
- audited research branch: `research/bo01-phase-a0-execution-script-draft-v1-20260518`
- audited commit: `e65a7bf39ff6b73eff13a04a83e40d7be79e52ec`
- base branch: `audit/bo01-phase-a-h02-warning-micro-patch-v1-20260518`
- base commit: `1c55ecbd42250c6d041471c9ce2d6b399e9d6966`
- local script path: `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_phase_a_execution_script_drafts/BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_20260518_191749/PHASE_A_EXECUTION_SCRIPT_DRAFT.py`
- local manifest path: `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_phase_a_execution_script_drafts/BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_20260518_191749/SCRIPT_DRAFT_MANIFEST.json`
- local report path: `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_phase_a_execution_script_drafts/BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_20260518_191749/SCRIPT_DRAFT_REPORT.md`
- script hash observed: `a75e014936b78fa9aefe96c3e1e290208e30228d01a3b688ce7c3088b9ce7c07`
- expected script hash: `a75e014936b78fa9aefe96c3e1e290208e30228d01a3b688ce7c3088b9ce7c07`
- no Python: YES
- no script execution: YES
- no data loading: YES
- no CSV read: YES
- no backtest: YES

## 4. Safety Verification

- code modified by audit: NO
- tests modified: NO
- data modified: NO
- Python executed: NO
- script executed: NO
- data loaded: NO
- CSV read: NO
- backtest: NO
- formal train: NO
- validation: NO
- holdout: NO
- 2025/2026: NO
- optimization/sweep: NO
- git add dot: NO
- reset/rebase/clean/stash: NO
- force push: NO

## 5. Diff Scope Audit

PASS_DIFF_SCOPE_GOVERNANCE_DOCS_ONLY.

`git diff --name-status audit/bo01-phase-a-h02-warning-micro-patch-v1-20260518..HEAD`
shows only:

1. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_PHASE_A_EXECUTION_SCRIPT_DRAFT_REPORT_V1.md`
2. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_BO01_PHASE_A_EXECUTION_SCRIPT_DRAFT_V1.md`

No tracked Python code, tests, data, data vault, runner, strategy, engine, data_loader,
local outputs, notebooks, ZIPs, or root-level unauthorized files were introduced by the
audited branch.

## 6. Local Artifacts Audit

PASS_LOCAL_ARTIFACTS_PRESENT_WITH_BLOCKING_PROVENANCE_FINDING.

Observed local artifacts:

- `PHASE_A_EXECUTION_SCRIPT_DRAFT.py`
- `SCRIPT_DRAFT_MANIFEST.json`
- `SCRIPT_DRAFT_REPORT.md`

`git check-ignore -v` confirms the script path is ignored by `.gitignore`. `git ls-files`
did not show the script, manifest, local report, or `bo01_phase_a_execution_script_drafts`
as tracked files.

## 7. Hash / Manifest Audit

BLOCKED.

Script SHA256:

- observed: `a75e014936b78fa9aefe96c3e1e290208e30228d01a3b688ce7c3088b9ce7c07`
- expected: `a75e014936b78fa9aefe96c3e1e290208e30228d01a3b688ce7c3088b9ce7c07`
- result: hash matches.

Manifest provenance:

- expected `generated_from_commit`: `e65a7bf39ff6b73eff13a04a83e40d7be79e52ec`
- observed `generated_from_commit`: `1c55ecbd42250c6d041471c9ce2d6b399e9d6966`
- result: manifest is inconsistent with the audited commit.

Other manifest safety flags were coherent:

- `script_executed`: false
- `data_loaded`: false
- `csv_read`: false
- `backtest_run`: false
- `validation_used`: false
- `holdout_used`: false
- `optimization_sweep`: false
- `phase_a1_authorized`: false
- `required_next_step`: `external_readonly_script_audit`

## 8. Script Header / Main Guard Audit

PASS_SCRIPT_HEADER_AND_MAIN_GUARD.

The script contains the expected header constants:

- `SCRIPT_ID = "BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_V1"`
- `PURPOSE = "Future Phase A-1 BO01 train-only real-data data-proof/backtest execution"`
- `GENERATED_IN_PHASE = "PHASE_A0_SCRIPT_DRAFT_ONLY"`
- `DO_NOT_EXECUTE_IN_PHASE_A0 = True`
- `EXPECTED_RUNNER_AUDIT_COMMIT = "5bdb4bed1f829eb7e8bfe65dc30a6e2f49657d89"`
- `EXPECTED_BASE_AUDIT_COMMIT = "1c55ecbd42250c6d041471c9ce2d6b399e9d6966"`

The script has `if __name__ == "__main__": main()`. Static review found no CSV read,
hash calculation, data loading, backtest, or output writing at top level outside
definitions/constants.

## 9. Phase A-1 Activation / Hash Gate Audit

PASS_PHASE_A1_ACTIVATION_AND_HASH_GATE.

The script requires:

- exact future Phase A-1 owner phrase through `BO01_PHASE_A1_OWNER_AUTHORIZATION`;
- script audit status `PASS`;
- audited script SHA256 through `BO01_PHASE_A1_AUDITED_SCRIPT_SHA256`;
- recomputation of the script hash before execution;
- abort with `BLOCKED_SCRIPT_HASH_MISMATCH` if hashes differ;
- authorized execution mode `phase_a1_train_only_plumbing`.

No old single direct Phase A phrase was accepted by the reviewed guard.

## 10. Path Authorization Audit

PASS_PATH_AUTHORIZATION.

The script contains fixed relative paths for:

- M5: `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared/EURUSD_M5.csv`
- M15: `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared/EURUSD_M15.csv`
- runner: `03_RESEARCH_LAB/research_lab/runners/bo01_backtest_runner.py`
- BO01 strategy: `03_RESEARCH_LAB/research_lab/strategies/BO01Strategy.py`
- Phase A-1 output root: `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_first_train_only_realdata_backtest`

`assert_authorized_path` rejects wildcards, exact-path mismatches, and external paths.
`load_csv_strict` accepts only the authorized M5/M15 paths.

## 11. Data Proof Logic Audit

PASS_DATA_PROOF_LOGIC.

Static review found:

- strict CSV loader exists and is only called from future `main()`;
- UTC timestamp normalization and null timestamp aborts;
- `DatetimeIndex` requirement;
- OHLC required columns;
- empty dataset abort;
- monotonic increasing index check;
- duplicate timestamp abort;
- critical NaN abort;
- full-index 2025 and 2026 guards;
- validation/holdout/unknown partition guards;
- selected window restricted to `2015-01-05 00:00:00+00:00` through `2015-01-09 23:59:59+00:00`;
- selected-window empty/leak aborts;
- M5 and optional M15 cadence checks with gap reporting.

## 12. Runner / Strategy Gate Audit

PASS_RUNNER_STRATEGY_GATE.

The script verifies:

- `RUNNER_ID = BO01_BACKTEST_RUNNER_SYNTHETIC_V1`
- `ENTRY_POLICY = ENTRY_NEXT_CANDLE_OPEN`
- `SAME_BAR_POLICY = STOP_FIRST`
- `MAX_TRADES_PER_DAY = 1`
- `MAX_ACTIVE_POSITIONS = 1`
- runner entrypoint `run_bo01_backtest_on_frame`
- BO01 strategy name and required entrypoints

No MR02 path, runner fallback, or alternate strategy selector was found.

## 13. Execution Policy Audit

PASS_EXECUTION_POLICY.

Future execution is restricted to:

- BO01 strategy path;
- M5 base data path;
- optional M15 path;
- runner contract with `ENTRY_NEXT_CANDLE_OPEN` and `STOP_FIRST`;
- max 1 trade per day;
- max 1 active position;
- fixed execution mode;
- no optimization/sweep/grid search/walk-forward/parameter search modes.

## 14. Cost / Metrics / Output Policy Audit

PASS_COST_METRICS_OUTPUT_POLICY.

The script defines fixed cost profiles:

- base: spread 1.2, slippage 0.2, commission 7.0, max spread guard 3.0
- conservative: spread 1.62, slippage 0.5, commission 7.0, max spread guard 3.0
- stress: spread 3.0, slippage 1.0, commission 7.0, max spread guard 4.0

The future output root is gitignored via a `git check-ignore` guard. Required future
files are listed in the script:

- `BO01_TRAIN_ONLY_REALDATA_BACKTEST_REPORT.md`
- `output_manifest.json`
- `command_log.txt`
- `data_access_log.txt`
- `diagnostic_counts.json`
- `trades_structural.csv`
- `equity_R.csv`
- `monthly_summary.csv`
- `cost_profile_summary.csv`

## 15. Security / Destructive Command Audit

PASS_SECURITY_STATIC_REVIEW_WITH_NOTES.

No literal secrets, passwords, API keys, credential values, broker execution calls,
Telegram sends, web calls, destructive delete commands, `os.system`, `eval`, `pickle`,
`requests`, `curl`, `wget`, or `Invoke-WebRequest` were found in the script.

Notes:

- `subprocess.run` appears only for `git check-ignore`.
- `exec_module` appears only as fixed-path module loading for the future runner/strategy
  gate.

These notes do not override the manifest blocker.

## 16. No Execution Proof Audit

PASS_NO_EXECUTION_PROOF.

Coherent evidence:

- manifest: `script_executed=false`, `data_loaded=false`, `csv_read=false`,
  `backtest_run=false`, `validation_used=false`, `holdout_used=false`,
  `optimization_sweep=false`, `phase_a1_authorized=false`;
- local report: script not executed, no Python, no data, no CSV, no backtest;
- governance report: script not executed, no Python, no data, no CSV, no backtest;
- status/diff: no generated backtest outputs in tracked diff.

## 17. Static Safety Scan

PASS_STATIC_SAFETY_SCAN_WITH_MANIFEST_BLOCKER_REPORTED_SEPARATELY.

The broad static scan over script, manifest, local report, governance report, and next
audit prompt produced 181 textual hits. Classification:

- NEGATIVE_DECLARATION_OK: validation, holdout, 2025/2026, optimization/sweep,
  demo/real/FTMO, edge/profitability, and forbidden execution terms used as guards or
  prohibitions.
- GOVERNANCE_TERM_OK: branch, report, audit prompt, and read-only scope references.
- FUTURE_SCRIPT_TERM_OK: future Phase A-1 script logic, backtest plumbing, output files,
  and cost profile terms.
- SCRIPT_REQUIRED_LOGIC_OK: `pd.read_csv` inside future strict loader, fixed-path module
  loading, fail-closed errors, and future output names.
- SECURITY_SCAN_OK: `subprocess.run` limited to `git check-ignore`; `exec_module`
  limited to fixed runner/strategy paths.
- LANGUAGE_WARNING: 0
- BLOCKER: 1, manifest commit provenance mismatch documented in section 7.

## 18. Git / Output Security Audit

PASS_GIT_OUTPUT_SECURITY_FOR_AUDITED_DIFF.

The audited diff introduces only two governance markdowns. The local script, manifest,
and local report are gitignored and not tracked.

A broad repository-level `git ls-files` scan returns pre-existing tracked data/data-vault
and legacy output matches outside this audited diff. Those are not introduced by this
branch and were not read as market data in this audit.

## 19. Findings Table

| id | severity | category | finding | evidence | implication | required_action |
| --- | --- | --- | --- | --- | --- | --- |
| F-01 | BLOCKER | Manifest provenance | Manifest `generated_from_commit` does not match the audited commit. | Expected `e65a7bf39ff6b73eff13a04a83e40d7be79e52ec`; observed `1c55ecbd42250c6d041471c9ce2d6b399e9d6966` in `SCRIPT_DRAFT_MANIFEST.json`. | Chain of custody is inconsistent; Phase A-1 owner decision cannot rely on the manifest as the audited commit record. | Generate a new Phase A-0 script-draft package with coherent manifest provenance, then re-audit read-only. |

## 20. Decision

AUDIT_BLOCKED_MANIFEST_INCONSISTENT

The script is not apt for Phase A-1 owner decision in this state. Blockers: 1. Warnings:
0.

This audit did not execute the script, load data, read CSV, prove edge, prove
profitability, authorize validation/holdout/2025/2026, authorize optimization/sweep, or
authorize demo/real/FTMO.

## 21. Allowed Next Step

B) Patch script draft warnings/blockers by generating a new Phase A-0 script draft.

The correction must not manually patch the audited local script package. It must generate
a new Phase A-0 script-draft package with a coherent manifest and then undergo a new
external read-only audit.

## 22. Forbidden Next Steps

- no direct Phase A execution outside audited script;
- no script modification before execution;
- no execution if hash mismatch;
- no validation;
- no holdout;
- no 2025/2026;
- no optimization/sweep;
- no demo/real/FTMO;
- no edge/profitability claims.
