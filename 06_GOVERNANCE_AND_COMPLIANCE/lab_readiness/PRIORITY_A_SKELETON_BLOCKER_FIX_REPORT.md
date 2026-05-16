# PRIORITY A SKELETON BLOCKER FIX REPORT

## 1. Status

PRIORITY_A_SKELETON_BLOCKERS_FIXED_READY_FOR_REPEAT_AUDIT

## 2. Executive Summary

This phase fixed only the blockers identified by the external Priority A skeleton audit. No backtest, strategy run, optimization, sweep, validation, holdout, event feed, high precision, F06 adapter, engine edit, data edit, force push, ZIP workflow, or root-file creation was performed.

Primary result:

- VE-ORB now fails closed when the 07:00-08:00 NY opening range is incomplete.
- The exact synthetic failure case from the audit now returns `None`.
- Priority A skeleton tests were expanded from 11 to 16 tests.
- Safe engine/preflight/stop-entry tests remain green.
- Static safety scan on the four Priority A skeletons and test file found no blocked tokens.

One additional fail-closed bug was found by the new tests and fixed surgically:

- MR-02 VWAP stats now reject NaN/non-finite `close` or `volume` values in the prior VWAP window.

## 3. Original Audit Blocker

External audit:

- branch: `audit/eurusd-priority-a-skeletons-code-audit-20260516`
- commit: `5baf89095db0265e52a273a38e31bb79d8e78f3e`
- status: `PRIORITY_A_SKELETONS_BLOCKED_CONTRACT_FAILURE`

Blocker:

- `ve_orb_volatility_expansion` accepted any non-empty 07:00-08:00 OR window.
- With only one row inside the OR, it could emit a long breakout signal after 08:00.
- Expected behavior was fail-closed `None`.

Pre-fix reproduction:

- `expected=None`
- actual: long signal dict
- `or_rows=1`

Post-fix verification:

- `expected=None`
- `actual=None`
- `or_rows=1`

## 4. VE-ORB Fix

Changed file:

- `03_RESEARCH_LAB/research_lab/strategies/ve_orb_volatility_expansion.py`

Fix summary:

- Added `min_or_coverage_pct = 0.90`.
- Added `allow_inferred_timeframe = True`.
- Added optional `min_or_bars = None`.
- Added `_infer_cadence_minutes()`.
- Added `_or_window_is_complete()`.
- `_opening_range()` now rejects incomplete, duplicate, non-finite, or unverifiable OR windows.

Cadence inference:

- Uses positive timestamp deltas from prior rows only.
- Uses median delta in minutes.
- Accepts only conservative cadence values: `1, 2, 3, 5, 10, 15`.
- Returns fail-closed if cadence cannot be inferred.

Expected bars:

- `expected_bars = floor((or_end - or_start) / cadence_minutes)`.
- If `min_or_bars` is provided, expected bars is at least that value.
- For M1: expected about `60`.
- For M5: expected about `12`.

Minimum coverage:

- `required_bars = ceil(expected_bars * min_or_coverage_pct)`.
- With default `0.90`, M1 requires at least `54` OR bars and M5 requires at least `11`.

Edge coverage:

- First OR timestamp must be no later than `07:00 + cadence`.
- Last OR timestamp must be no earlier than `08:00 - cadence`.
- OR timestamps must live inside `[07:00, 08:00)`.
- Duplicate timestamps fail closed.
- NaNs in OR `high`, `low`, or `close` fail closed.

No-lookahead:

- OR construction still uses `frame.iloc[:i]`.
- No post-08:00 rows are used to build OR high/low.
- Breakout rules were not changed, only blocked when OR completeness cannot be verified.

## 5. Tests Added

Updated file:

- `03_RESEARCH_LAB/research_lab/tests/test_priority_a_skeletons.py`

Added/expanded coverage:

- `test_ve_orb_fails_closed_with_incomplete_opening_range`
- `test_ve_orb_allows_complete_opening_range`
- `test_ve_orb_short_signal`
- `test_tp01_short_signal`
- `test_mr02_nan_fail_closed`
- `test_tp01_nan_fail_closed`
- expanded file-access patch coverage across all four skeletons

Existing coverage preserved:

- imports;
- insufficient data fail-closed;
- registry keys;
- MR-01 long/short;
- MR-02 long/short;
- TP-01 long and lateral no-signal;
- VE-ORB pre-OR no-signal;
- source blocked-token assertions.

## 6. Test Results

Priority A skeleton tests:

- Command: `python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_priority_a_skeletons.py" -v`
- Result: `Ran 16 tests in 0.218s - OK`

Safe imports:

- `import research_lab`: `research_lab OK`
- `STRATEGY_REGISTRY`: `67`; all four Priority A keys present
- `import research_lab.engine`: `engine OK`

Optional safe tests:

- `test_engine.py`: `Ran 17 tests - OK`
- `test_lab_preflight*.py`: `Ran 6 tests - OK`
- `test_engine_stop_entry.py`: `Ran 3 tests - OK`

Not executed:

- no backtest;
- no strategy run;
- no smoke run;
- no optimization;
- no sweep;
- no F06 real;
- no validation;
- no holdout.

## 7. Static Safety Scan

Scope:

- `03_RESEARCH_LAB/research_lab/strategies/mr01_anchor_elastic.py`
- `03_RESEARCH_LAB/research_lab/strategies/mr02_vwap_stretch_reversion.py`
- `03_RESEARCH_LAB/research_lab/strategies/tp01_london_ny_momentum_pullback.py`
- `03_RESEARCH_LAB/research_lab/strategies/ve_orb_volatility_expansion.py`
- `03_RESEARCH_LAB/research_lab/tests/test_priority_a_skeletons.py`

Pattern:

- `read_csv|to_csv|read_parquet|to_parquet|open\(|Path\(|os\.path|subprocess|requests|http|forex_factory|news|high_precision|level2|holdout|sealed_holdout|2025|2026|rv5|rv15|p30|0.08|zip|000_PARA_CHATGPT`

Result:

- no matches.

Classification:

- file I/O: NO
- event/feed dependency: NO
- high precision dependency: NO
- holdout: NO
- 2025/2026: NO
- ghost params: NO
- ZIP workflow: NO

## 8. Files Changed

Code:

- `03_RESEARCH_LAB/research_lab/strategies/ve_orb_volatility_expansion.py`
- `03_RESEARCH_LAB/research_lab/strategies/mr02_vwap_stretch_reversion.py`
- `03_RESEARCH_LAB/research_lab/tests/test_priority_a_skeletons.py`

Governance:

- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/PRIORITY_A_SKELETON_BLOCKER_FIX_REPORT.md`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_REPEAT_PRIORITY_A_SKELETON_EXTERNAL_AUDIT.md`

Not changed:

- `engine.py`
- `data_loader.py`
- data vault
- root files
- strategy registry
- MR-01
- TP-01 strategy code

## 9. Remaining Risks

- This fix closes the known pre-backtest contract blocker, but it does not prove strategy edge.
- No performance metrics were calculated.
- No train/validation/holdout data was touched.
- VE-ORB cadence inference is conservative and may reject irregular data until an external audit confirms the behavior is acceptable.
- Repeat external audit is still required before any micro-backtest.

## 10. Safety Verification

- backtest_run: NO
- strategy_run: NO
- optimization_run: NO
- sweep_run: NO
- validation_run: NO
- holdout_used: NO
- 2025_2026_used: NO
- news_used: NO
- high_precision_used: NO
- engine_modified: NO
- data_modified: NO
- force_push: NO
- git_add_dot_used: NO

## 11. Copy-Paste Summary for ChatGPT

PRIORITY_A_SKELETON_BLOCKERS_FIXED_READY_FOR_REPEAT_AUDIT

Fixed the external audit blocker from commit `5baf89095db0265e52a273a38e31bb79d8e78f3e`: VE-ORB now fails closed when the 07:00-08:00 opening range is incomplete. The original synthetic failure case with one OR row now returns `None`.

Added cadence inference and OR completeness checks:

- accepted cadence: 1, 2, 3, 5, 10, 15 minutes;
- expected bars based on 60-minute OR window;
- required coverage default 90%;
- first/last OR timestamp edge checks;
- duplicate timestamp and NaN rejection.

Expanded tests:

- incomplete OR fail-closed;
- complete OR valid long;
- VE-ORB short;
- TP-01 short;
- MR-02 NaN fail-closed;
- TP-01 NaN fail-closed;
- file-access patch expanded across all four skeletons.

Test results:

- Priority A skeleton tests: 16 OK
- `research_lab` import: OK
- registry: 67 and four keys present
- engine import: OK
- `test_engine.py`: 17 OK
- `test_lab_preflight*.py`: 6 OK
- `test_engine_stop_entry.py`: 3 OK

Static scan on relevant files: no file I/O, no event/feed dependency, no high precision, no holdout, no 2025/2026, no ghost params, no ZIP.

No backtest, strategy run, optimization, sweep, validation, holdout, engine edit, data edit, force push, `git add .`, or ZIP workflow. Repeat external audit before any micro-backtest.
