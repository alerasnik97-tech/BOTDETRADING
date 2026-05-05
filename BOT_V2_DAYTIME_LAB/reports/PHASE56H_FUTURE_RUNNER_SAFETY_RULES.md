# PHASE 56H - FUTURE RUNNER SAFETY RULES

To ensure the integrity of the forensic audit and prevent accidental data loss or corruption, the following rules are **MANDATORY** for all future automated runs:

## 1. Operational Prohibitions
- **NO `taskkill`:** Prohibited unless explicitly approved by a human operator. If a process hangs, investigate logs before termination.
- **NO `rmdir` / `del` / `Remove-Item`:** No directory or file deletion is allowed within the workspace or data folders. 
- **NO `git reset` / `git clean`:** Source control must be handled manually or via approved CI scripts.

## 2. Data Integrity
- **Canonical Parquet Only:** All replays must use the pre-validated Parquet files in `BOT_MARKET_DATA`.
- **Atomic Checkpointing:** The orchestrator must save progress to the global checkpoint JSON **immediately** after each month is processed.
- **Deduplication:** The runner must check for existing entries in the checkpoint before starting a month to avoid redundant processing.

## 3. Environment Isolation
- **No Live/Demo Connectivity:** The replay engine must remain 100% disconnected from MT5 and brokers.
- **No Strategy Mutation:** TP, BE, BF, and exit windows are hard-locked. Any change requires a NEW phase designation.

## 4. Failure Handling
- **Data Repair Required:** If a checkpoint file is found to be corrupted, stop execution and flag as `PHASE56H_REQUIRES_DATA_REPAIR`.
- **Safety Repair Required:** If core strategy files are missing or modified, flag as `PHASE56H_REQUIRES_SAFETY_REPAIR`.
