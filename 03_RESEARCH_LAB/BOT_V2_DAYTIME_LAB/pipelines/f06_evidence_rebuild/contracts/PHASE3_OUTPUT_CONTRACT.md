# PHASE 3 OUTPUT CONTRACT

## 1. Overview
This document defines the exact contract that MUST be satisfied by the output directory of any `Phase 3 Clean F06 Train-Only Rerun`. If an output directory fails to meet these exact structural and semantic requirements, the CLI validator (`validate_rebuild_outputs.py`) will return `BLOCKED_GUARD_FAILED` and the results will be institutionally void.

## 2. Directory Structure
```text
reports/f06_clean_train_only_rerun/run_<RUN_ID>/
├── MANIFEST.json
├── CONFIG_USED.yaml
├── COMMANDS_RUN.md
├── ENVIRONMENT_SUMMARY.json
├── SAFETY_VERIFICATION.md
├── PHASE3_CLEAN_F06_RERUN_REPORT.md
├── ledger/
│   └── TRADES.csv
├── ranking/
│   └── RANKING.csv
├── cost/
│   └── COST_REPORT.json
└── hashes/
    └── HASHES.txt
```

## 3. Strict Rules
- **Run ID**: All data within the output directory must correspond to exactly ONE `run_id`.
- **Validation Columns**: `RANKING.csv` MUST NOT contain any validation columns (e.g., `N_val`, `PF_val`, `val_pass`).
- **Dates**: No dates corresponding to 2025 or 2026 are permitted anywhere in `TRADES.csv` or `RANKING.csv`.
- **Source Paths**: No inputs may originate from directories containing `QUARANTINED` or `v50b_limited`.
- **Hashes**: All artifacts generated must have their `SHA-256` hash recorded in `HASHES.txt` and inside `MANIFEST.json['output_hashes']`. The hashes must perfectly match the disk files.
- **Cost Model**: The `COST_REPORT.json` must assert that `spread`, `slippage`, and `commission_round_turn_usd` were applied.
- **Sample Size**: The output must contain `>= 100` trades per family.
- **Mandatory Validator**: Before results can be read or interpreted by any human or agent, `validate_rebuild_outputs.py` must be executed. It must output `READY_FOR_CLAUDE_AUDIT`.
