# M1 TRAIN-ONLY MICRORUN EXECUTION REPORT V1

## 1. Run Identification
- **RUN_ID:** `M1_TRAIN_ONLY_BO01_MR02_20260518_112700`
- **Phase:** `M1_TRAIN_ONLY_PLUMBING_VERIFICATION`
- **Execution Date:** `2026-05-18`
- **Execution Branch:** `research/m1-train-only-bo01-mr02-v1-20260518`
- **Base Commit:** `a59557d11aace57326183f3b35e3beb7ca7def46`
- **Pushed Branch:** `research/m1-train-only-bo01-mr02-v1-20260518`
- **Parent Commit:** `0905406c84ebc1cb5e47488b00abd4eb86c984ca`

---

## 2. Executive Summary
The M1 train-only microrun was executed successfully in two subfases: M1A (metadata/data availability preflight) and M1B (tiny controlled execution slice). This is a purely structural verification designed to prove plumbing compatibility with real train-only data. 

No backtesting was performed, no parameters were optimized, and no strategy performance or edge is asserted. The laboratory remains non-operative, and all validation/holdout partitions remain strictly sealed.

---

## 3. M1A — Metadata Preflight Result
**`STATUS = PASS`**
- **Data Source ID:** `EURUSD_PREPARED_TRAIN_2015_2024_M5`
- **Dataset Path:** `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared/EURUSD_M5.csv`
- **Size:** `60,488,231 bytes`
- **SHA-256 Hash:** `386ab589d14e52236581201b03aa7d8e6c5d2c9771bc59eea00d34abc1afa625`
- **Row Count:** `729,382`
- **Min Timestamp (UTC):** `2015-01-01 22:05:00+00:00`
- **Max Timestamp (UTC):** `2024-12-31 22:00:00+00:00`
- **Columns Present:** `['timestamp', 'open', 'high', 'low', 'close', 'volume']`
- **Timezone/Cadence:** UTC, 5-minute bars.
- **Train-Only Status:** **Proven.** Verified that no 2025/2026 dates are present in the dataset.

---

## 4. M1B — Tiny Controlled Execution Result
**`STATUS = PASS`**
- **Selected Slice:** `2015-01-05 00:00:00+00:00 to 2015-01-07 23:55:00+00:00`
- **Slice Duration:** 3 calendar days.
- **Slice Row Count:** `864` M5 candles.
- **Warmup Bars:** `80` candles.

### Strategy BO01 (Breakout Candidate)
- **Import Status:** `PASS`
- **Candles Evaluated:** `864`
- **None Returns:** `850`
- **Valid Signals Generated:** `14`
- **Exceptions Count:** `0`
- **Fail-Closed Gate Status:** `PASS` *(Verified that removing required `ema_m15_200` column results in None signals returned without exception)*

### Strategy MR02 (Mean Reversion Candidate)
- **Import Status:** `PASS`
- **Candles Evaluated:** `864`
- **None Returns:** `864`
- **Valid Signals Generated:** `0`
- **Exceptions Count:** `0`
- **Fail-Closed Gate Status:** `PASS` *(Verified that removing required `close` column results in None signals returned without exception)*

---

## 5. Safety Declarations
- **No Validation Partition Access:** Checked and confirmed.
- **No Holdout Partition Access:** Checked and confirmed.
- **No 2025/2026 dates accessed:** Checked and confirmed.
- **No Backtesting:** No PnL, Win Rate, Expectancy, Drawdown, or Sharpe was computed.
- **No Parameter Sweeps/Optimization:** No grids, parameter search, or walk-forward verification occurred.
- **No Code Mutated:** Code files `BO01Strategy.py` and `MR02Strategy.py`, unit tests, data loader, and registry remain completely clean.
- **W-01 and W-02 Backlogs:** Untouched. Pre-existing backlog under `03_RESEARCH_LAB/strategy_research_intake/...` is strictly quarantined.
- **Output Policy:** All generated execution reports, manifestations, command logs, and access logs are contained strictly within the gitignored output root:  
  `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/m1_train_only_bo01_mr02/M1_TRAIN_ONLY_BO01_MR02_20260518_112700/`

---

## 6. Decision
**`M1_TRAIN_ONLY_PLUMBING_COMPLETED_READY_FOR_EXTERNAL_AUDIT`**  
The execution successfully verified the real-data plumbing of strategies BO01 and MR02. The results are fully logged and structured under gitignored folders, ready for the external read-only audit.
