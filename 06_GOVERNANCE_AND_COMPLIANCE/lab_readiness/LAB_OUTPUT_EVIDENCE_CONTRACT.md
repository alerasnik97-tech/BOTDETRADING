# LAB OUTPUT EVIDENCE CONTRACT

## 1. Objective
Ensure that all research activities within the EURUSD Train-Only Laboratory generate high-integrity, auditable, and non-leaky evidence.

## 2. Mandatory Identification
Every research run must include:
- `run_id`: Unique identifier (e.g., `EURUSD_TRAIN_SMOKE_20260516_001`).
- `timestamp`: Execution start time in UTC.
- `codebase_commit`: SHA of the active branch.
- `branch_name`: Name of the research branch.

## 3. Storage Protocols
### Allowed Output Directories
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/` (Primary for strategy runs).
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/` (Only for authorization/audit reports).

### Forbidden Directories
- Repository Root (`/`).
- `07_BACKUPS/`.
- `01_CORE_PRODUCTION/`.
- `02_INCUBATION_STAGING/`.
- Any legacy or quarantine folder.

## 4. Evidence Package Content
A valid evidence package must contain:
1. `manifest.json`: List of all generated files with SHA256 hashes.
2. `config_audit.json`: Snapshot of `EngineConfig` and `NewsConfig` used.
3. `data_manifest_hash`: Verification of the dataset used (must match `prepared_train_2015_2024`).
4. `trades.csv`: Full trade log (empty if no trades).
5. `results_summary.md`: Performance metrics (PF, WR, etc.).

## 5. Security Declarations
Each report must explicitly state:
- `train_only_scope`: TRUE.
- `no_holdout_access`: TRUE.
- `no_2025_2026_data_used`: TRUE.
- `news_filter_status`: DISABLED (or ENABLED with provenance check).

## 6. Fail-Closed Validation
Any run that fails the internal `lab_preflight` must immediately terminate and not generate research output.
