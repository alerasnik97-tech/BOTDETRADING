# NEXT PROMPT — AUDIT M2 TRAIN-ONLY STRUCTURAL RETRY EXECUTION V1

This prompt is to be executed in **READ-ONLY AUDIT MODE**.
Under blocker penalty, the following are strictly **PROHIBITED** during this audit:
- NO executing Python scripts with real market data.
- NO loading of market data.
- NO modifying strategy code, engine, runner, or test files.
- NO backtesting or formal training.
- NO validation or holdout partition access.
- NO 2025 or 2026 data loading.
- NO optimization sweeps or parameters search.
- NO git add .
- NO force push, reset --hard, rebase, clean, stash.

---

## 1. Audit Objective
Verify the execution of the M2 Train-Only Structural Retry for strategies BO01 and MR02.

---

## 2. Verification Checklist

### 2.1 File Scope and Whitelist Verification
- Confirm that the base branch is `audit/m2-structural-runner-warning-patch-v1-20260518` at commit `f01bdf77f8daa35905bdc831dbf93b51f93e38e9`.
- Confirm that the execution branch is `research/m2-conservative-structural-retry-bo01-mr02-v1-20260518` (active branch).
- Verify the physical local HEAD SHA is descendant of commit `f01bdf77f8daa35905bdc831dbf93b51f93e38e9`.
- Verify that ONLY the whitelisted 2 files are modified/created in git status:
  - `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M2_TRAIN_ONLY_STRUCTURAL_RETRY_EXECUTION_REPORT_V1.md`
  - `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_M2_TRAIN_ONLY_STRUCTURAL_RETRY_EXECUTION_V1.md`
- Verify that the local output root contains exactly the 7 gitignored files:
  - `M2_TRAIN_ONLY_STRUCTURAL_REPORT.md`
  - `output_manifest.json`
  - `command_log.txt`
  - `data_access_log.txt`
  - `diagnostic_counts.json`
  - `signal_structure_summary.json`
  - `m2_retry_executor.py`

### 2.2 Data Scope Verification
- Confirm M5 dataset used is `prepared_train_2015_2024` from path `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared/EURUSD_M5.csv`.
- Confirm M15 dataset used is `prepared_train_2015_2024` from path `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared/EURUSD_M15.csv`.
- Verify that no 2025/2026 data was loaded or processed.
- Verify that no validation or holdout data was accessed.

### 2.3 Structural Counts Audit
- Verify that BO01 generated exactly `638` valid signals over `17,999` candles (17,361 nones).
- Verify that MR02 generated exactly `5` valid signals over `17,999` candles (17,994 nones).
- Verify that exception_count and fail_closed_count are both exactly `0` for both strategies.
- Verify that temporal distribution of signals is strictly limited between `07:00` and `10:00` UTC/GMT (designated session hours).
- Confirm that no performance metrics (PnL, Sharpe, winrate, drawdown, Profit Factor, Sortino) are calculated or written to any local output file.

### 2.4 Safety Verification
- Confirm git status shows no tracked/staged modifications outside the 2 whitelisted governance markdown files.
- Confirm W-01/W-02 remain untouched.
- Confirm no forbidden outputs (`trades.csv`, `equity_curve.csv`, `pnl.csv`, or ZIP files) are created.

---

## 3. Allowed Methods
The auditor is permitted to use **ONLY** read-only text commands:
- Git inspection commands (`git status`, `git branch`, `git log`, `git diff`).
- Reading markdown files using file viewers.
- Reading local json manifests and reports in the output root.
- Running native PowerShell static search checks.

---

## 4. Final Audit Decision
The auditor must report a final safety status:
- **STATUS = PASS:** If all structural counts match, no performance metrics are computed, data scopes are strictly respected, git status is clean, and the 2 governance markdown files are successfully created and committed.
- **STATUS = BLOCKER:** If any python script execution is allowed during audit, any performance metrics are computed, or any files are mutated.
