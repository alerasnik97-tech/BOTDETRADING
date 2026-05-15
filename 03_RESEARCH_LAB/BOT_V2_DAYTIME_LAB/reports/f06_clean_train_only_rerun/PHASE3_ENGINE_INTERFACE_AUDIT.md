# PHASE 3 ENGINE INTERFACE AUDIT

## 1. Context
- **Objective**: Audit existing engine interfaces (`src/v7_engine` and `src/v6_utils`) to determine if a safe, reusable Phase 3 clean train-only runner adapter can be built without touching the core files.
- **Constraints**: No modifying `src/v7_engine` or `src/v6_utils`. No using quarantined or legacy V50B runners as the source of truth. The interface must support the rigid restrictions defined in `F06_PHASE3_CLEAN_TRAIN_ONLY.yaml`.

## 2. Interface Audit Results
The existing engine in `src/v7_engine/` and `src/v6_utils/` contains robust logic but has historically been wrapped by scripts that were either overly permissible or relied on hardcoded directories that have since been quarantined.
- **Data Loading**: Can be accomplished via standard utility functions assuming paths are passed explicitly.
- **Execution**: The core execution loop (e.g. `execution.py`) accepts configuration parameters. However, ensuring it strictly adheres to `TRAIN_ONLY` without `validation_enabled` or `holdout_enabled` requires the runner to construct a failsafe dictionary of arguments before passing them.
- **Cost Model**: The cost model logic expects explicit parameters for spread, slippage, and commission. The Phase 3 runner must calculate these and apply them to gross metrics before outputting to `COST_REPORT.json` and `TRADES.csv`.

## 3. Safe vs Forbidden Routes
- **Safe Route**: Create a pure adapter (e.g., `run_phase3`) inside the `pipelines/f06_evidence_rebuild/scripts/` directory that imports functions from `src/v7_engine` but handles all directory isolation and `fail-closed` validations locally.
- **Forbidden Route**: Attempting to reuse the old `v50b_limited_real_runner.py` or modifying the core `engine.py` to bake in Phase 3 rules.

## 4. Decision
We will build the `run_phase3` command strictly inside `f06_rebuild_pipeline.py`. Because we cannot execute actual backtesting right now, `run_phase3` will be implemented to require an explicit confirmation flag (`--confirm-real-run PHASE3_F06_TRAIN_ONLY_APPROVED`). If this flag is provided, it will abort with `NOT_IMPLEMENTED_FAIL_CLOSED` until the actual engine linkage is finalized in a later audit, but it will structurally hold the lock and the required validation sequences.
