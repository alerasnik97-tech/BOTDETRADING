# NEXT PROMPT — EXECUTE M1 TRAIN-ONLY BO01/MR02 V1

## 0. Activation Gate
This prompt requires the exact autonomous activation phrase from the owner to proceed.
If the owner does not declare this exact phrase as a new, top-level, separate statement:
**ABORT** with: `BLOCKED_MISSING_EXPLICIT_OWNER_APPROVAL`

The required exact approval phrase is:
`“APRUEBO EJECUTAR M1 TRAIN-ONLY BO01/MR02, SOLO M1A METADATA PREFLIGHT Y M1B TINY CONTROLLED EXECUTION, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026, SIN BACKTEST, SIN TRAIN FORMAL Y SIN OPTIMIZATION/SWEEP.”`

> [!IMPORTANT]
> - The phrase must be a new, autonomous, and top-level statement.
> - Paraphrasing, short answers (e.g., "ok", "run it", "go"), or quotes do not count.
> - Any ambiguity must trigger `BLOCKED_AMBIGUOUS_OWNER_APPROVAL`.

---

## 1. Nature Of This Prompt
This prompt authorizes the controlled execution of:
- **M1A:** Metadata and data availability preflight only (no strategy execution, no price calculations, no performance metrics).
- **M1B:** A tiny, train-only controlled execution slice to verify the real-data plumbing of strategies BO01 and MR02.

> [!WARNING]
> This phase does NOT search for an edge, does NOT validate strategies, does NOT run standard backtests, and does NOT generate performance metrics (PnL, Win Rate, Expectancy, Sharpe, etc.). Its only purpose is structural plumbing verification.

---

## 2. Allowed Scope

### M1A — Metadata Preflight
- Verify data source file path and existence.
- Verify file identity and read-only access.
- Calculate and verify the SHA-256 hash of the data source file if available.
- Read and verify the exact row count of the dataset.
- Read the minimum and maximum timestamps of the dataset.
- Read and verify the column names list.
- Read and verify the train-only partition label inside the dataset.
- **Strictly Prohibited:** Strategy instantiation, strategy execution, signal calculations, or price statistics.

### M1B — Tiny Controlled Execution
- Load a tiny, contiguous, pre-declared train-only date slice.
- **Allowed range:** Bounded within 2015-01-01 and 2024-12-31.
- **Candidates:** Strategies BO01 and MR02 only.
- **Asset:** EURUSD only.
- Import strategy candidates (`BO01Strategy.py` and `MR02Strategy.py`).
- Call `default_params` to verify parameter initialization.
- Call the `signal` path on the selected slice.
- Verify exact timezone, GMT cadence, and Asian range window alignment (00:00-06:30 GMT).
- Verify strategy behavior with missing or duplicate timestamps.
- Verify fail-closed implementation when required columns are absent.
- Bounded signal call count.
- Verify output containment inside local ignored folders.
- Create execution logs and dynamic manifests.

---

## 3. Prohibited Scope
The following actions are strictly and absolutely prohibited:
- **NO Validation partition access.**
- **NO Holdout partition access.**
- **NO 2025 data access.**
- **NO 2026 data access.**
- **NO Backtesting.**
- **NO Formal training runner.**
- **NO Optimization or parameter sweeps.**
- **NO Grid search or walk-forward verification.**
- **NO Performance metrics calculation** (strictly no Profit Factor, Win Rate, Drawdown, Sharpe Ratio, Expectancy, or PnL).
- **NO Creation of `trades.csv` or `equity_curve.csv`.**
- **NO Sub-Batch 1B or other candidates (e.g., MR03, LS01, LS02).**
- **NO Portfolio expansion or parallel execution writers.**
- **NO Code modifications** (no changes to `BO01Strategy.py`, `MR02Strategy.py`, tests, engine, runner, data loader, or factory).
- **NO Data Vault writes or mutations.**
- **NO ZIP creation.**
- **NO Output committing to GitHub** (local outputs must remain 100% ignored).

---

## 4. Pre-Execution Safety Checks
Before executing any script, the executing agent must verify:
1. **Branch check:** The current branch must be `research/m1-train-only-bo01-mr02-v1-YYYYMMDD` (replaces YYYYMMDD with the current local date).
2. **Worktree check:** No staged files exist. Worktree stability snapshots A and B match under a 60-second observation window.
3. **Quarantine check:** Pre-existing dirty backlog (W-01) under `03_RESEARCH_LAB/strategy_research_intake/external_research_20260516/` must be stable or quarantined. Any drift aborts execution.
4. **Output debt check:** Pre-existing output debt (W-02) remains untouched and uncleaned.
5. **Process check:** No active Python research processes are running.
6. **Ignores check:** The target output root must be verified as gitignored.
7. **Vault check:** Read-only locks on `05_MARKET_DATA_VAULT/` must be active.
8. **Range check:** The declared date range must be validated to have no 2025/2026 timestamps.
9. **Single writer check:** No parallel agent is writing or executing.

---

## 5. Branch Strategy
- **Branch:** `research/m1-train-only-bo01-mr02-v1-YYYYMMDD` (created from `research/draft-m1-train-only-execution-prompt-v1-20260518` at commit `BRANCH_HEAD`).
- **Push policy:** Force-push to this branch is strictly prohibited.
- **Main lock:** No direct push, merge, or rebase to `main` is allowed.

---

## 6. Data Policy
- The market data source must be read-only.
- **Target source ID:** `EURUSD_PREPARED_TRAIN_2015_2024_M5`
- The executing agent must verify the actual canonical filename in the system and document its size and location.
- **Safety range limits:**
  - `min_timestamp >= 2015-01-01 00:00:00`
  - `max_timestamp <= 2024-12-31 23:59:59`
- Any observed timestamp from 2025 or 2026 will immediately abort with: `BLOCKED_FORBIDDEN_DATE_RANGE`

---

## 7. M1A Metadata Preflight Execution
The agent must execute a read-only metadata verification script to print:
- Complete file/path location of the dataset.
- Dataset size in bytes and SHA-256 hash.
- Total row count.
- Exact minimum and maximum timestamps.
- Complete list of column names.
- Explicit validation that the data belongs strictly to the train partition.
- **Check:** No strategy is instantiated, and no price statistics or signals are computed.

---

## 8. M1B Tiny Controlled Execution Setup
The agent must pre-declare the exact tiny slice before executing the call path:
- **Pre-declared slice:** (e.g., `2015-01-05 00:00:00` to `2015-01-07 23:59:59`, or a similar 3-day window containing a London session).
- **Max length:** Bounded to the minimum required bars for parameter warm-up and one Asian range evaluation.
- **Command template:**
  `DRAFT_DO_NOT_RUN - TEMPLATE ONLY: python -m research_lab.runners.m1_controlled_runner --slice-start "2015-01-05" --slice-end "2015-01-07"`
- **Required checks during execution:**
  - Import BO01 and MR02 classes.
  - Call default parameters and verify schema compatibility.
  - Call `signal` on the pre-declared slice.
  - Confirm M5 timezone-aware GMT candle cadence.
  - Verify that if missing features occur, the strategies fail closed.
  - Verify that ATR calculations do not include future lookahead.
- **Allowed outputs:**
  - Structural check counts only (bars evaluated, signals generated, missing bars handled).
  - Absolutely no trades or equity curve output files.

---

## 9. Hardened Manifest Schema
The execution must output a detailed manifest containing:
- `run_id` (UUID format).
- `phase` (`M1_TRAIN_ONLY_PLUMBING_VERIFICATION`).
- `strategy_ids` (`[BO01, MR02]`).
- `branch` and `commit_sha`.
- `parent_commit_sha`.
- `repo_status_before` and `repo_status_after`.
- `python_version`.
- `timestamp_start_utc` and `timestamp_end_utc`.
- `data_policy_version` (`V1_TRAIN_ONLY`).
- `data_source_id`.
- `data_range_declared` and `data_range_observed`.
- `selected_slice_declared_before_run`.
- `validation_used: false` (strictly false).
- `holdout_used: false` (strictly false).
- `used_2025_2026: false` (strictly false).
- `optimization_sweep: false` (strictly false).
- `backtest_run: false` (strictly false).
- `formal_train_run: false` (strictly false).
- `output_root` (`local_outputs_do_not_commit/...`).
- `created_files` (list of created markdown/json/txt files with individual SHA-256 hashes).
- `manifest_sha256_external`.
- `command_log_sha256`.
- `report_sha256`.
- `data_access_log_sha256`.
- `no_secrets_detected: true`.
- `no_forbidden_outputs_detected: true`.

---

## 10. Output Policy
- **Target output root:**
  `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/m1_train_only_bo01_mr02/<RUN_ID>/`
- The executing agent must verify this path is 100% ignored.
- **Allowed output files:**
  - `M1_TRAIN_ONLY_MICRORUN_REPORT.md` (local copy of structural results).
  - `output_manifest.json` (hardened manifest).
  - `command_log.txt` (exact command history).
  - `data_access_log.txt` (features read and feature contract checks).
- **Prohibited:** Any `trades.csv`, `equity_curve.csv`, ZIPs, or root files. None of the local output files can be staged or committed.

---

## 11. Post-Execution Safety Verification
Following the execution, the agent must run:
- `git status --short` to verify no source code, tests, or market data files were modified.
- Verify that pre-existing backlog W-01 and W-02 are unaffected.
- Verify that created files are only in the allowed output root and are completely unstaged.
- Verify that no secrets or credentials are exposed in command logs.

---

## 12. Governance Records
The executing agent may only stage and commit the following two governance documents:
1. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M1_TRAIN_ONLY_MICRORUN_EXECUTION_REPORT_V1.md`
2. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_M1_TRAIN_ONLY_MICRORUN_EXECUTION_V1.md`

No other files are allowed in git staging.

---

## 13. Final Response Format
The executing agent must respond in the following exact format:
1. STATUS:
2. BRANCH:
3. SAFETY:
4. M1A_METADATA_PREFLIGHT:
5. M1B_TINY_EXECUTION:
6. DATA_POLICY:
7. OUTPUTS:
8. MANIFEST:
9. POSTRUN_SCAN:
10. DECISION:
11. ALLOWED_NEXT_STEP:
12. FORBIDDEN_NEXT_STEPS:
13. ARTIFACTS:
14. GITHUB:

---

## 14. Success vs. Performance Metrics
A successful execution of M1 only means that:
- The data contract is verified and correct.
- Strategies can load and parse M5 candles GMT timezone-aware.
- ATR and EMA calculations handle monotonic indexes without leaks.
- Strategies fail closed when required features are missing.

A successful M1 execution **does not prove edge**, does not prove profitability, does not prove strategy robustness, and does not authorize validation or paper trading.
