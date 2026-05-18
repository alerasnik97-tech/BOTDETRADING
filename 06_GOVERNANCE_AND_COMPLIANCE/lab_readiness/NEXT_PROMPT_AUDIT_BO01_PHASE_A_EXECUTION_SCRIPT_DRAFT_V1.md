# NEXT PROMPT - AUDIT BO01 PHASE A EXECUTION SCRIPT DRAFT V1

Act as an institutional destructive read-only auditor, Senior Python Reviewer, Data
Leakage Auditor, Backtesting Execution Auditor, Risk Governance Officer, and Git Safety
Officer for the Trading BOT project.

## Objective

Audit the Phase A-0 generated BO01 execution/data-proof script draft in read-only mode.
Do not execute the script. Do not execute Python. Do not load market data. Do not read
real CSV files. Do not run backtest, train, validation, holdout, 2025/2026,
optimization, sweep, demo, real, or FTMO workflows.

## Activation Gate

The exact owner phrase must appear as a standalone declaration:

"AUTORIZO AUDITORIA EXTERNA READ-ONLY DEL SCRIPT DRAFT PHASE A-0 BO01 TRAIN-ONLY REAL-DATA, SIN EJECUTAR PYTHON, SIN EJECUTAR EL SCRIPT, SIN CARGAR DATOS DE MERCADO, SIN LEER CSV REAL, SIN BACKTEST CON REAL DATA, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP."

If it is missing exactly, abort with:

`BLOCKED_MISSING_EXPLICIT_OWNER_APPROVAL`

## Context

Repo local:

`C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`

Branch to audit:

`research/bo01-phase-a0-execution-script-draft-v1-20260518`

Base:

`audit/bo01-phase-a-h02-warning-micro-patch-v1-20260518`

Expected governance diff:

1. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_PHASE_A_EXECUTION_SCRIPT_DRAFT_REPORT_V1.md`
2. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_BO01_PHASE_A_EXECUTION_SCRIPT_DRAFT_V1.md`

Expected local gitignored artifacts:

1. `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_phase_a_execution_script_drafts/BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_20260518_191749/PHASE_A_EXECUTION_SCRIPT_DRAFT.py`
2. `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_phase_a_execution_script_drafts/BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_20260518_191749/SCRIPT_DRAFT_MANIFEST.json`
3. `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_phase_a_execution_script_drafts/BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_20260518_191749/SCRIPT_DRAFT_REPORT.md`

Expected script SHA256:

`a75e014936b78fa9aefe96c3e1e290208e30228d01a3b688ce7c3088b9ce7c07`

## Required Checks

1. Precheck Git:
   - active branch is `research/bo01-phase-a0-execution-script-draft-*`;
   - no staged changes;
   - no worktree drift beyond documented pre-existing untracked files;
   - no Python process from research/backtest/train/optimization.

2. Diff Scope:
   - governance diff contains only the two expected markdowns;
   - local script artifacts are gitignored and not staged;
   - no code/test/data/runner/strategy/data-vault/output ZIP/notebook/root files are
     committed.

3. Local Artifact Check:
   - script exists;
   - manifest exists;
   - local report exists;
   - script SHA256 matches manifest and governance report;
   - manifest says script_executed=false, data_loaded=false, csv_read=false,
     backtest_run=false, validation_used=false, holdout_used=false,
     optimization_sweep=false, phase_a1_authorized=false.

4. Script Content Safety:
   - script header constants match expected IDs and commits;
   - script has `if __name__ == "__main__": main()`;
   - script fails closed;
   - script requires exact future Phase A-1 owner phrase;
   - script requires external script audit status;
   - script verifies audited SHA256 before execution;
   - Phase A-1 stays blocked until script audit passes.

5. Path Authorization:
   - M5 path is exactly authorized;
   - M15 path is exactly authorized;
   - no wildcard path;
   - no external paths;
   - output root must be gitignored.

6. Data Proof Logic:
   - strict CSV loader exists but was not executed;
   - timestamp normalization requires UTC proof;
   - full-index scan blocks 2025 and 2026;
   - validation/holdout/unknown partition guards exist;
   - OHLC checks exist;
   - duplicate timestamp check exists;
   - critical NaN check exists;
   - selected-window slicing is restricted to 2015-01-05 through 2015-01-09;
   - cadence checks allow normal market gaps but fail incompatible cadence.

7. Runner and Strategy Gates:
   - runner path is fixed to `03_RESEARCH_LAB/research_lab/runners/bo01_backtest_runner.py`;
   - runner ID must be `BO01_BACKTEST_RUNNER_SYNTHETIC_V1`;
   - entry policy must be `ENTRY_NEXT_CANDLE_OPEN`;
   - same-bar policy must be `STOP_FIRST`;
   - max trades per day = 1;
   - max active positions = 1;
   - BO01 strategy path is fixed and importability is verified;
   - MR02 is not used.

8. Output and Cost Policy:
   - future Phase A-1 outputs are restricted to gitignored local output root;
   - required future output files are enumerated;
   - base, conservative, and stress cost profiles are fixed;
   - no optimization/sweep/grid search/walk-forward/parameter search.

9. Security Scan:
   - no sensitive values;
   - no external execution connectors;
   - no destructive file commands;
   - no destructive Git commands;
   - no unsupported paths;
   - no demo/real/FTMO logic.

10. Static Safety Scan:
   - use `rg` on script, manifest, local report, governance report, and next audit prompt;
   - classify hits as `NEGATIVE_DECLARATION_OK`, `GOVERNANCE_TERM_OK`,
     `FUTURE_SCRIPT_TERM_OK`, `SCRIPT_REQUIRED_LOGIC_OK`, `SECURITY_SCAN_OK`, or
     `BLOCKER`.

## Decision Values

- `BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_AUDIT_PASS_READY_FOR_PHASE_A1_OWNER_DECISION`
- `BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_AUDIT_PASS_WITH_WARNINGS`
- `AUDIT_BLOCKED_SCRIPT_HASH_MISMATCH`
- `AUDIT_BLOCKED_SCRIPT_EXECUTED_OR_DATA_LOADED`
- `AUDIT_BLOCKED_DIFF_SCOPE`
- `AUDIT_BLOCKED_SCRIPT_SAFETY`
- `AUDIT_BLOCKED_STATIC_SAFETY_SCAN`

If the audit passes, create the next owner decision prompt. It must not authorize direct
execution unless the owner later provides the audited Phase A-1 phrase and the script hash
is verified.
