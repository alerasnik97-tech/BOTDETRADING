# NEXT PROMPT — EXECUTE M0 SYNTHETIC-ONLY MICRORUN BO01/MR02 V1

## 0. Activation Gate
The owner must provide the exact following phrase in the prompt to authorize execution:
“APRUEBO EJECUTAR M0 SYNTHETIC-ONLY MICRORUN BO01/MR02, SIN DATOS REALES, SIN DATA VAULT, SIN BACKTEST, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”

If this exact phrase is not provided:
ABORT IMMEDIATELY with:
`BLOCKED_MISSING_EXPLICIT_OWNER_APPROVAL`

---

## 1. Nature
This prompt executes ONLY the M0 synthetic-only plumbing verification for strategy candidates BO01 and MR02.
- No real market data.
- No data vault.
- No validation.
- No holdout.
- No 2025/2026.
- No backtest.
- No train.
- No optimization/sweep.
- No edge inference.
- No performance inference.

---

## 2. Allowed Execution Scope
The execution in this phase is strictly restricted to:
- importing strategy candidates (`BO01Strategy`, `MR02Strategy`);
- calling their `default_params` functions;
- building minimal, temporary, in-memory M5 bar fixtures (using `pandas.DataFrame`);
- calling their `signal(frame, i, params)` functions on the constructed fixtures;
- verifying `fail-closed` behavior on incomplete or malformed bar index sets;
- verifying lack of signals outside entry session hours;
- writing a simple status markdown report and an output manifest;
- saving all execution outputs strictly inside the approved `local_outputs_do_not_commit` subdirectory.

---

## 3. Forbidden Scope
This execution strictly prohibits:
- reading or writing any real market data from files or vaults;
- importing from or writing to `05_MARKET_DATA_VAULT`;
- accessing validation or holdout data;
- accessing 2025/2026 data;
- running backtests, training, or runner scripts (such as `formal_train_runner`);
- using `--execute` or dynamic execution options;
- performing parameter optimization, sweeps, grids, or walk-forward sweeps;
- modifying any Python source files in the workspace;
- modifying unit or targeted test files;
- writing files to the repository root directory;
- committing or staging any output files (including `trades.csv` or `equity_curve.csv`);
- generating screenshot files;
- creating binary archives or ZIP files;
- uploading outputs or data to GitHub.

---

## 4. Prechecks Future Execution
Before performing any execution checks, you must:
1. Run `git status` to ensure a clean index.
2. Confirm the active git branch is not `main`.
3. Confirm W-01 (dirty tree) is quarantined or clean.
4. Confirm W-02 (output debt) remains untouched.
5. Verify `.gitignore` ignores `local_outputs_do_not_commit/`.
6. Confirm no other python execution processes exist.

---

## 5. Branch Future Execution
If authorized, create a new branch strictly named:
`research/m0-synthetic-microrun-bo01-mr02-v1-YYYYMMDD`
*(Where YYYYMMDD represents the current execution date. Do not push to main or force push).*

---

## 6. Synthetic Fixture Policy
You must define and construct small, in-memory Pandas DataFrames representing synthetic bar fixtures to feed the strategy `signal` function:
- Timeframe: M5 timezone-aware (UTC/GMT index).
- Size: Minimal bars required (e.g., 81 bars to satisfy WARMUP_BARS = 80).
- Columns required for BO01: `"open", "high", "low", "close", "volume", "ema_m15_200"`.
- Columns required for MR02: `"open", "high", "low", "close"`.
- Scenarios to test:
  1. **BO01 Valid Breakout case:** Prices trigger a long/short breakout outside the Asian range width within entry hours.
  2. **BO01 Malformed case:** DataFrame is missing columns or timezone info; signal function must return `None` (fail-closed).
  3. **MR02 Valid Reversion case:** Price spikes out of the Asian range bound and returns, with engulfing bar pattern confirmed.
  4. **MR02 Malformed case:** DataFrame lacks timezone or required columns; signal function must return `None` (fail-closed).
  5. **Session hour verification:** Bars feed data outside entry hours; signal must return `None`.

---

## 7. Output Policy Future Execution
All outputs must be written exclusively to:
`03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/m0_synthetic_microrun_bo01_mr02/<RUN_ID>/`

You must create only:
1. `M0_SYNTHETIC_MICRORUN_REPORT.md` (summarizing the synthetic tests executed, passes, and fails).
2. `output_manifest.json` (listing exact file names, sizes, and SHA256 hashes).
3. `command_log.txt` (recording commands executed).

No other files are allowed. If the output folder is not ignored in `.gitignore`, the execution must immediately abort.

---

## 8. Future Execution Script Policy
To execute this, you must write an inline temporary Python script or a temporary script located strictly inside `local_outputs_do_not_commit/`.
- No new runner or strategy files may be added to the repository's source directories.
- The script must load the strategies, create the in-memory fixtures, run signal calculations, collect checks, write the report, and terminate.
- The script must never access the disk to read price data.

---

## 9. Future Safety Scan
Immediately post-run, execute a scan to confirm:
- `git status --short` shows no staged or untracked changes outside the approved output directory.
- No forbidden files (`trades.csv`, `equity_curve.csv`, ZIPs) exist in the repository tree.
- No files were added to `05_MARKET_DATA_VAULT` or repository root.

---

## 10. Future Report
The output `M0_SYNTHETIC_MICRORUN_REPORT.md` must clearly state:
- Branch and commit SHA of execution.
- List of commands ran.
- Explicit declaration that no real price data or data vaults were touched.
- Confirmation that no backtest, train, validation, or holdout was exposed.
- Passes/fails of all synthetic signal tests.
- Target verdict: `M0_SYNTHETIC_MICRORUN_COMPLETED_READY_FOR_EXTERNAL_AUDIT_ONLY`.

---

## 11. Future Git Policy
- Do NOT stage or commit any outputs under `local_outputs_do_not_commit/`.
- Committing the final `M0_SYNTHETIC_MICRORUN_REPORT.md` is permitted only if separately authorized.

---

## 12. Future Final Response Format
The final response of that execution phase must strictly present:
1. STATUS:
2. BRANCH:
3. SAFETY:
4. SYNTHETIC_CHECKS:
5. OUTPUTS:
6. GIT_STATUS:
7. DECISION:
8. ALLOWED_NEXT_STEP:
9. FORBIDDEN_NEXT_STEPS:

---

## 13. Critical Reminder
Success of this M0 synthetic plumbing test means only that strategy candidate signal loops and fail-closed pathways are technically verified. It does NOT imply that the strategy has an edge, is profitable, or is approved for backtesting, train, validation, holdout, or live/demo/FTMO accounts.
