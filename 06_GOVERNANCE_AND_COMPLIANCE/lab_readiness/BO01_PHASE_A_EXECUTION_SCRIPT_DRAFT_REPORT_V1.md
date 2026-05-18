# BO01 PHASE A EXECUTION SCRIPT DRAFT REPORT V1

## 1. Status

BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_READY_FOR_EXTERNAL_AUDIT

## 2. Scope

- script draft generated;
- script not executed;
- no Python executed;
- no data loaded;
- no CSV read;
- no backtest;
- no formal train;
- no validation;
- no holdout;
- no 2025/2026;
- no optimization/sweep;
- no code/test/data/runner changes.

## 3. Local Artifacts

- run_id: `BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_20260518_191749`
- output_root: `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_phase_a_execution_script_drafts/BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_20260518_191749/`
- script_path: `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_phase_a_execution_script_drafts/BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_20260518_191749/PHASE_A_EXECUTION_SCRIPT_DRAFT.py`
- script_sha256: `a75e014936b78fa9aefe96c3e1e290208e30228d01a3b688ce7c3088b9ce7c07`
- manifest: `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_phase_a_execution_script_drafts/BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_20260518_191749/SCRIPT_DRAFT_MANIFEST.json`
- local_report: `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_phase_a_execution_script_drafts/BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_20260518_191749/SCRIPT_DRAFT_REPORT.md`

The local output root is gitignored and is not committed.

## 4. Script Content Summary

- path gates: exact authorized path checks for M5, M15, runner, and BO01 strategy;
- SHA256 gates: chunked hash function and pre-execution script hash verification;
- data proof gates: strict CSV load function, UTC timestamp normalization, OHLC checks,
  monotonic index check, duplicate timestamp check, critical NaN check, partition guard,
  selected-window slicing, and cadence checks;
- forbidden-date guard: full-index scan blocks 2025 and 2026 before selected-window use;
- validation/holdout guard: partition columns are checked and forbidden values fail closed;
- runner gate: verifies runner ID, entry policy, same-bar policy, max trades per day, max
  active positions, and entrypoint;
- strategy gate: verifies BO01 importability and required strategy entrypoints;
- cost profiles: fixed base, conservative, and stress profiles;
- output policy: future Phase A-1 outputs restricted to gitignored local output root;
- abort conditions: missing authorization, hash mismatch, unauthorized paths, forbidden
  years/partitions, bad timestamps, bad columns, runner mismatch, strategy mismatch,
  unsupported execution mode, and non-gitignored output root;
- final handoff: future Phase A-1 prints a JSON handoff after successful plumbing.

## 5. Phase A-1 Not Authorized

Phase A-1 is not authorized by this report.

Required before any Phase A-1 execution:

- external read-only audit of the script draft;
- audited script SHA256 recorded;
- exact later Phase A-1 owner activation phrase;
- pre-execution hash verification against the audited SHA256;
- confirmation that the script was not modified after audit.

## 6. Decision

Ready for external read-only audit of the Phase A-0 script draft.

## 7. Allowed Next Step

External read-only audit of generated script draft.

## 8. Forbidden Next Steps

- no script execution;
- no data loading;
- no CSV read;
- no backtest;
- no validation;
- no holdout;
- no 2025/2026;
- no optimization/sweep;
- no demo/real/FTMO.
