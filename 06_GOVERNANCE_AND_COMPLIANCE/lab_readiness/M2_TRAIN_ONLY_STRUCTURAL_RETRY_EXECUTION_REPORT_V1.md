# M2 CONSERVATIVE TRAIN-ONLY STRUCTURAL RETRY EXECUTION REPORT V1

## 1. Status

**`M2_CONSERVATIVE_TRAIN_ONLY_STRUCTURAL_RETRY_EXECUTION_READY_FOR_EXTERNAL_AUDIT`**

All M2 Conservative structural retry counts were computed successfully using the warning-patched, audited runner `M2_STRUCTURAL_RUNNER_BO01_MR02_V1`.

---

## 2. Scope & Boundaries

- **Candidate Strategies**: `BO01Strategy` and `MR02Strategy` (skeleton candidates).
- **Execution Level**: Structural signal counting only.
- **Real Market Data Used**: `prepared_train_2015_2024` from the `05_MARKET_DATA_VAULT`.
- **Declared M2 Window**: `2015-01-01 00:00:00 UTC` to `2015-03-31 23:59:59 UTC`.
- **Observed/Processed Window**: `2015-01-01 22:05:00 UTC` (first real tick) to `2015-03-31 23:55:00 UTC`.
- **Validation Split Access**: Locked (no validation partition rows parsed or used).
- **Holdout Split Access**: Locked (no holdout partition rows parsed or used).
- **Future Dates (2025/2026)**: Sealed (fully guarded; zero 2025 or 2026 rows loaded).
- **Backtesting & PnL**: Banned (no profit calculations, no equity curves, no trade lists).
- **Optimization / Sweep**: Banned (no parameter ranges evaluated, default parameters only).

---

## 3. Executive Summary of Counts

The structural runner executed row-by-row over `17,999` candles (5-minute bars) spanning January 1 to March 31, 2015.

| Metric | Strategy BO01 | Strategy MR02 | Verdict |
|---|---|---|---|
| **row_count** | 17,999 | 17,999 | Matching slice length |
| **signal_call_count** | 17,999 | 17,999 | Processed all candles |
| **valid_signal_count** | 638 | 5 | Signals passed contract checks |
| **contract_valid_count**| 638 | 5 | Programmatic check matches |
| **none_count** | 17,361 | 17,994 | Zero signal generated |
| **exception_count** | 0 | 0 | 100% execution stability |
| **fail_closed_count** | 0 | 0 | Zero structural errors |
| **days_with_signal** | 41 | 3 | Calendar signal presence |
| **max_signals_per_day** | 32 | 3 | Daily volume bound |
| **fail_closed_gate_test**| PASS | PASS | Validated by missing column tests |

### Temporal Distribution

- **BO01 (London Breakout)**:
  - Hours (UTC): 07:00 (101 signals), 08:00 (207 signals), 09:00 (300 signals), 10:00 (30 signals). Zero signals at any other hour.
  - Months: January (152), February (206), March (280).
- **MR02 (VWAP Stretch Reversion)**:
  - Hours (UTC): 07:00 (2 signals), 08:00 (2 signals), 09:00 (1 signal). Zero signals at any other hour.
  - Months: February (5).

All signals are located strictly within the designated session hours (07:00–10:00 GMT), validating the timezone-aware engine alignment.

---

## 4. Run Metadata

- **RUN_ID**: `M2_CONSERVATIVE_STRUCTURAL_RETRY_BO01_MR02_20260518_165500`
- **Output Root**: `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/m2_train_only_structural_bo01_mr02/M2_CONSERVATIVE_STRUCTURAL_RETRY_BO01_MR02_20260518_165500/`
- **Runner Module**: `research_lab.runners.m2_structural_runner`
- **Runner ID**: `M2_STRUCTURAL_RUNNER_BO01_MR02_V1`
- **Active Branch**: `research/m2-conservative-structural-retry-bo01-mr02-v1-20260518`
- **Commit SHA**: `f01bdf77f8daa35905bdc831dbf93b51f93e38e9`
- **Parent Commit SHA**: `62851977c223889d16107e74393076d41d88b315`

---

## 5. Output Manifest File Integrity

The local outputs are stored inside the gitignored output root and verified by sha256 hashes recorded inside `output_manifest.json`:

| File | Size (Bytes) | Hash (SHA256) |
|---|---|---|
| `M2_TRAIN_ONLY_STRUCTURAL_REPORT.md` | 3,463 | `64d3cd2db0e1ec0755a6d5162ad9b921316e632860b73c4d7ecbb207a90f2305` |
| `command_log.txt` | 27 | `cd21dd44a86846174a806950290ca83df1492c3008fe51c6c06a3861214ab6d3` |
| `data_access_log.txt` | 984 | `0da29db429d20c57c433364f9bfcd688755088ebc22ebf4c54ad3d940e7293a3` |
| `diagnostic_counts.json` | 2,194 | `7a493a388b1cc94966a3ca0c497491b5c216cdfb1c5ee8beacdf069cd5b8d960` |
| `signal_structure_summary.json` | 1,736 | `091001859d042637955f242aa6858e999be3273cc4b6fcc8c027419e48f07421` |
| `m2_retry_executor.py` | 13,603 | `a39031c9a62ee7156976ce7336fccf3653139369ce5eb270a6c6e7687884d642` |
| `output_manifest.json` | 2,706 | `5a7f9a2ee64e83c27e85c1bf435df2a92683fe7ee4d8c7c9441a1827b521255e` |

---

## 6. Safety Verification Summary

| Verification | Status | Evidence / Notes |
|---|---|---|
| **git_status** | CLEAN | No core files mutated or tracked |
| **dirty_trees** | UNTOUCHED | Pre-existing dirty trees W-01/W-02 remain untouched |
| **forbidden_dates** | PASS | Programmatic date check confirms no 2025/2026 rows |
| **forbidden_splits** | PASS | Programmatic split checks confirm no validation or holdout data |
| **performance_leak** | PASS | Zero active performance computation (no PnL, winrate, drawdown) |
| **forbidden_files** | PASS | No `trades.csv`, `equity_curve.csv`, or ZIP files created |
| **runner_correctness** | PASS | Using warning-patched runner; counts perfectly semantic |

---

## 7. Decision & Next Step

The M2 Conservative Train-Only Structural Retry was executed flawlessly under complete laboratory command discipline:

1. **Successful Execution**: The fully patched runner successfully parsed `17,999` candles in UTC and generated semantic structural counts for both strategies.
2. **Fail-Closed Verified**: Programmatic missing columns fail-closed tests verified that strategies return `None` immediately under missing columns context.
3. **Registry Match**: Reconciles the pre-registered skeletons state in the quant laboratory registry.

**Ready for the next external read-only audit of the execution.**
