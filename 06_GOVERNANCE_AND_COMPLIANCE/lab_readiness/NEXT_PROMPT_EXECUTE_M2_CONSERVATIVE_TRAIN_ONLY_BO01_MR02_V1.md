# NEXT PROMPT — EXECUTE M2 CONSERVATIVE TRAIN-ONLY STRUCTURAL EVALUATION BO01/MR02 V1

## 0. Activation Gate
This prompt must be executed only when the owner provides the exact future activation phrase as an autonomous declaration:

“APRUEBO EJECUTAR M2 CONSERVATIVE TRAIN-ONLY STRUCTURAL EVALUATION BO01/MR02, SOLO MÉTRICAS ESTRUCTURALES, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026, SIN BACKTEST, SIN TRAIN FORMAL Y SIN OPTIMIZATION/SWEEP.”

- **Rules:**
  - This phrase must appear as a standalone declaration from the owner.
  - Paraphrases, logs, citations, or examples do not qualify.
  - Standard approvals like "ok", "go", "dale", "proceed" do not qualify.
  - Any ambiguity or missing declaration must result in: `BLOCKED_AMBIGUOUS_OWNER_APPROVAL`.

---

## 1. Nature and Objective
The M2 Conservative execution is a:
- Train-only evaluation.
- Structural-only count and check.
- `BO01Strategy` and `MR02Strategy` only.
- `EURUSD` M5 data only.
- **Explicitly excludes:** performance metrics, economic inference, edge inference, profitability inference, and FTMO/demo/real suitability claims.

---

## 2. Data Policy
- **Allowed Dataset:** `EURUSD_PREPARED_TRAIN_2015_2024_M5`
- **Allowed Window:** `2015-01-01 00:00:00 UTC` to `2015-03-31 23:59:59 UTC`.
  - *Note:* If the dataset starts later, use the actual first available timestamp, but do not extend past `2015-03-31`.
- **Hard Blocks:**
  - Any 2025 timestamp.
  - Any 2026 timestamp.
  - Validation partition.
  - Holdout partition.
  - Unknown partition.
  - Non-EURUSD data.
  - Non-M5 cadence data.
  - Any result-based window change.

---

## 3. Runner Policy
- **Before Execution:**
  - The team must verify that an audited runner exists in the repository.
  - If no audited runner is found or exists:
    `BLOCKED_M2_RUNNER_NOT_AUDITED_OR_NOT_FOUND`
- **Repository Constraints:**
  - NO runner creation.
  - NO runner modification.
  - NO engine/data_loader/factory changes.
  - NO `BO01Strategy.py` or `MR02Strategy.py` code changes.
  - NO test file changes.
- **Temporary Script Limits:**
  - Allowed only if created inside the gitignored local output root.
  - Must not be committed to GitHub.
  - Must not modify repository source code, tests, or market data.
  - Must not write outside the designated local output root.
  - Must not compute performance metrics.
  - Must not access validation, holdout, 2025, or 2026 partitions.

---

## 4. Allowed Structural Metrics
The execution is permitted to record and report **only** the following metrics:
- `row_count` (total candles loaded in scope).
- `signal_call_count` (total attempts to generate signals).
- `valid_signal_count` (signals matching structural rules).
- `none_count` (cases with no signal).
- `exception_count` (handled exceptions).
- `days_with_signal` (days displaying at least one signal).
- `max signals per day`.
- `signals by hour` / `signals by month` (distribution check).
- `missing column failure count` (robustness check).
- `fail-closed counts` / `contract-valid counts`.
- `timestamp anomalies` / `cadence anomalies`.
- `forbidden date count` / `validation/holdout access count` (security scans).
- `output file count` / `hash checks` (manifest verification).

---

## 5. Forbidden Metrics
The execution is strictly prohibited from calculating, displaying, or logging:
- PnL (Profit and Loss).
- PF / Profit Factor.
- Winrate.
- Drawdown (Max or relative).
- Sharpe Ratio / Sortino Ratio.
- Expectancy.
- Equity curve data.
- R multiples.
- Average winner / Average loser / Average profit per trade.
- Trade list for profitability.
- Optimization score or rankings.
- Any profitability claims.

---

## 6. Output Policy
- **Output Root:** `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/m2_train_only_structural_bo01_mr02/<RUN_ID>/`
- **Allowed Local Outputs (Must be Gitignored):**
  - `M2_TRAIN_ONLY_STRUCTURAL_REPORT.md`
  - `output_manifest.json`
  - `command_log.txt`
  - `data_access_log.txt`
  - `diagnostic_counts.json`
  - `signal_structure_summary.json`
  - Temporary script (if used for execution).
- **Forbidden Outputs:**
  - `trades.csv` / `equity_curve.csv` / `pnl.csv`
  - Performance reports.
  - Optimization sweeps or search results.
  - ZIP archives.
  - Screenshots.
  - Files in repository root or data vault.
  - Staged or committed local outputs.

---

## 7. Execution Scope
For each strategy (`BO01Strategy` and `MR02Strategy`):
1. Import the class.
2. Initialize with default parameters.
3. Perform signal calls across the pre-declared 3-month window.
4. Count structural outcomes only.
5. Verify fail-closed checks.
- **Rule:** Parameter changes by results are prohibited.

---

## 8. Abort Conditions
The team must immediately abort operations if:
- Currently on the `main` branch.
- Any active, unknown python process is running on the system.
- Staged files exist before running.
- The worktree is unstable.
- `W-01` or `W-02` strategies contain modifications.
- Designated output root is not gitignored.
- Dataset source is not found.
- Train-only status is not proven.
- Any forbidden date is detected in the input scope.
- Validation or holdout partitions are accessed.
- Code, test, or runner modifications are required to run.
- Any performance metric is requested or calculated.
- Any local output file becomes staged.
- Any forbidden output is generated.

---

## 9. Governance Report
Upon completion, the execution is permitted to commit only:
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M2_TRAIN_ONLY_STRUCTURAL_EXECUTION_REPORT_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_M2_TRAIN_ONLY_STRUCTURAL_EXECUTION_V1.md`
NO local outputs under `03_RESEARCH_LAB/` may be committed.

---

## 10. Final Response Format
The final response of the future turn must strictly follow this structure:

1. **STATUS:** [COMPLETED / BLOCKED / RED / INCONCLUSIVE]
2. **BRANCH:** [Branch Name, parent, head]
3. **SAFETY:** [Pre-run scans, python checks]
4. **DATA_POLICY:** [Verification of prepared data, no holdout]
5. **STRUCTURAL_WINDOW:** [Start/end dates, verification]
6. **BO01_STRUCTURAL_COUNTS:** [Total signals, exceptions, fail-closed cases]
7. **MR02_STRUCTURAL_COUNTS:** [Total signals, exceptions, fail-closed cases]
8. **OUTPUTS:** [List of created local files and their gitignore status]
9. **MANIFEST:** [JSON manifest containing hashes of all generated files]
10. **POSTRUN_SCAN:** [Validation that no forbidden files/metrics exist]
11. **DECISION:** [Next phase recommendation]
12. **ALLOWED_NEXT_STEP:** [Designated audit next step]
13. **FORBIDDEN_NEXT_STEPS:** [Prohibited operations]
14. **ARTIFACTS:** [Paths to created governance reports]
15. **GITHUB:** [Branch, commit SHA, push verification]

---

## 11. Reminder
M2 completion does not mean edge.
M2 completion does not mean profitability.
M2 completion does not mean ready for backtest.
M2 completion only means a train-only structural evaluation completed under governance.
