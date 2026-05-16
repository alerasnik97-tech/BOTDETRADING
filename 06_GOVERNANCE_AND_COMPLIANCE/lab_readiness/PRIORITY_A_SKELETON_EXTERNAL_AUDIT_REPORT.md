# PRIORITY A SKELETON EXTERNAL AUDIT REPORT

## 1. Status

PRIORITY_A_SKELETONS_BLOCKED_CONTRACT_FAILURE

## 2. Executive Summary

External pre-backtest audit was performed on branch `audit/eurusd-priority-a-skeletons-code-audit-20260516`, auditing implementation commit `087faddc37b53fedb25840b67cc4431acc5dcf5b`.

Verdict: do not run a train-only micro-backtest yet.

The implementation scope is mostly clean:

- only the four final Priority A skeletons were added;
- registry wiring is controlled;
- `engine.py` was not modified;
- `data_loader.py` was not modified;
- no data, output, ZIP, parquet, CSV, cache, or root file was committed by the skeleton implementation;
- tests and safe imports pass;
- no direct file I/O, external event/feed dependency, high precision dependency, holdout dependency, or 2025/2026 dependency was found in the four new skeletons.

However, VE-ORB has a pre-backtest blocker: it can emit a breakout signal even when the 07:00-08:00 opening range is incomplete. A direct synthetic unit call showed a signal with only one 07:00 opening-range row. That violates the skeleton fail-closed contract for opening-range completeness and must be fixed before any backtest.

## 3. Files Audited

Strategy files:

- `03_RESEARCH_LAB/research_lab/strategies/mr01_anchor_elastic.py`
- `03_RESEARCH_LAB/research_lab/strategies/mr02_vwap_stretch_reversion.py`
- `03_RESEARCH_LAB/research_lab/strategies/tp01_london_ny_momentum_pullback.py`
- `03_RESEARCH_LAB/research_lab/strategies/ve_orb_volatility_expansion.py`

Registry:

- `03_RESEARCH_LAB/research_lab/strategies/__init__.py`

Tests:

- `03_RESEARCH_LAB/research_lab/tests/test_priority_a_skeletons.py`

Implementation report:

- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/PRIORITY_A_SKELETON_IMPLEMENTATION_REPORT.md`

Commit surface:

- `git show --name-only --pretty="" 087faddc37b53fedb25840b67cc4431acc5dcf5b` returned exactly the expected eight files.
- No `engine.py`, `data_loader.py`, data file, output file, ZIP, parquet, CSV, cache, or root file was in the implementation commit.

## 4. Strategy Contract Audit

| Strategy | contract_ok | registry_ok | import_side_effects | file_io | external_dependency | signal_output_ok | stop_target_ok | decision |
|---|---|---|---|---|---|---|---|---|
| `mr01_anchor_elastic` | YES | YES | NO | NO | NO | YES | YES | PASS_CONTRACT |
| `mr02_vwap_stretch_reversion` | YES | YES | NO | NO | NO | YES | YES | PASS_CONTRACT |
| `tp01_london_ny_momentum_pullback` | YES | YES | NO | NO | NO | YES | YES | PASS_CONTRACT |
| `ve_orb_volatility_expansion` | NO | YES | NO | NO | NO | YES | YES | BLOCKED_CONTRACT_FAILURE |

Contract evidence:

- All four modules define `NAME`, `DEFAULT_PARAMS`, `default_params()`, `parameter_space()`, `parameter_grid()`, and `signal(frame, i, params)`.
- The registry imports the four modules and adds four keys at `__init__.py` lines 67-70 and 138-141.
- The engine supports legacy `signal()` when `generate_signal()` is absent at `engine.py` lines 703-707.
- The engine validates signal risk contract at `engine.py` lines 232-312 and schedules entry processing after signal generation at line 720.
- The four skeletons return `None` or an engine-compatible dict containing `direction`, hard stop, target, and metadata.

Blocking contract issue:

- `ve_orb_volatility_expansion._opening_range()` builds the OR from `frame.iloc[:i]` and returns any non-empty OR window as valid at lines 70-84.
- It does not require complete 07:00-08:00 coverage or a minimum expected bar count.
- A synthetic signal call with only one row in the 07:00-08:00 window returned a long signal after 08:00.

Ad hoc evidence, no backtest:

- `rows_0700_0800= 1`
- `signal= {'signal': 1, 'direction': 'long', 'stop_mode': 'price', 'stop_price': 1.0997, 'target_mode': 'rr', 'target_rr': 2.0, 'break_even_at_r': None, 'trailing_atr': False, 'session_name': 'all_day'}`

## 5. No-Lookahead Audit

| Strategy | No-lookahead verdict | Evidence |
|---|---|---|
| MR-01 | PASS_NO_LOOKAHEAD | `_session_slice()` uses `frame.iloc[:i]`; anchored stats exclude bar `i`; ADX fallback uses `i-1`; EMA fallback uses `close.iloc[:i]`. |
| MR-02 | PASS_NO_LOOKAHEAD | `_previous_session()` uses `frame.iloc[:i]`; VWAP stats exclude bar `i`; current bar is used only as closed-bar re-entry confirmation. |
| TP-01 | PASS_NO_LOOKAHEAD | ATR percentile window uses `atr_values.iloc[i - lookback : i]`; EMA uses `close_series.iloc[:i]`; current bar is used as closed-bar pullback/continuation confirmation. |
| VE-ORB | PASS_NO_LOOKAHEAD | `_opening_range()` uses `frame.iloc[:i]`; signals before 08:00 are blocked; ATR percentile window uses `atr_values.iloc[i - lookback : i]`. Contract still blocked because OR completeness is not enforced. |

Current-bar OHLC usage:

- The engine generates signals from completed bar `i`, stores `signal_index = i`, and processes entry on `i + 1`.
- Under that engine contract, using close/high/low of bar `i` for signal confirmation and stop construction is acceptable for these skeletons.

No future-row usage was found in the four new skeletons.

## 6. Fidelity to Final Arbitration

Priority A final list:

- MR-01: implemented.
- MR-02: implemented.
- TP-01 reformulated: implemented.
- VE-ORB: implemented but contract-blocked pending OR completeness fix.

Explicit exclusions:

- VE-01: not implemented.
- SD-01: not implemented.
- ED-01: not implemented.

Scan result:

- New skeleton files: no `VE-01`, `RV Shock`, `rv5`, `rv15`, `p30`, `SD-01`, `Europe Extreme`, `0.08`, `ED-01`, `Post-News`, `news`, `forex_factory`, `high_precision`, `level2`, `holdout`, `2025`, or `2026` hits.
- Full `strategies/` directory scan produced pre-existing `post_news` hits in older strategy files and `__init__.py`; these are outside the four new skeletons and were not introduced by commit `087faddc`.

Classification:

- New Priority A skeleton fidelity: PASS.
- Existing non-Priority-A directory noise: documented, not a blocker for this commit.

## 7. Unit Test Quality

Test file:

- `03_RESEARCH_LAB/research_lab/tests/test_priority_a_skeletons.py`

Positive coverage:

- imports all four strategies;
- verifies insufficient data fail-closed;
- patches file access for one signal path;
- source-scans the four modules for prohibited dependencies;
- verifies registry contains the four new approved keys;
- verifies MR-01 long/short synthetic extremes;
- verifies MR-02 long/short synthetic band re-entry;
- verifies TP-01 momentum pullback vs lateral range;
- verifies VE-ORB no signal during OR and signal after OR;
- verifies NaN fail-closed on MR-01 and VE-ORB;
- verifies excluded-family token absence in new module source.

Coverage gaps:

- No test proves VE-ORB fails closed when the 07:00-08:00 OR is incomplete.
- No TP-01 short-side synthetic signal test.
- No VE-ORB short-side synthetic signal test.
- File I/O patch is applied only around one MR-01 call, while source scan covers all modules.
- NaN critical-input testing covers MR-01 and VE-ORB only, not all four skeletons.

Decision:

- Tests are meaningful, not decorative.
- Tests are insufficient for approval because they missed the VE-ORB incomplete-opening-range contract failure.

## 8. Test Results

Executed safe tests only.

Priority A skeleton tests:

- Command: `python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_priority_a_skeletons.py" -v`
- Result: `Ran 11 tests in 0.131s - OK`

Safe imports:

- `import research_lab`: `research_lab OK`
- `STRATEGY_REGISTRY`: `67` and all four new keys present
- `import research_lab.engine`: `engine OK`

Optional safe engine tests:

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
- no holdout;
- no 2025/2026.

## 9. Static Safety Scan

Full command:

- `rg -n "read_csv|to_csv|read_parquet|to_parquet|open\(|Path\(|os\.path|subprocess|requests|http|forex_factory|news|high_precision|level2|holdout|sealed_holdout|2025|2026|rv5|rv15|p30|0.08|zip|000_PARA_CHATGPT" 03_RESEARCH_LAB/research_lab/strategies 03_RESEARCH_LAB/research_lab/tests/test_priority_a_skeletons.py`

Full-directory result:

- Hits exist in older strategies and backup registry for `post_news` and false-positive `dict(zip(...))`.
- These hits are pre-existing and not part of the four new skeletons.

New-file scoped command:

- same pattern applied only to the four new strategy files and `test_priority_a_skeletons.py`.

New-file result:

- no matches.

Safety classification for new skeletons:

- file_io: NO
- news: NO
- high_precision: NO
- holdout: NO
- 2025_2026: NO
- ghost_params: NO

## 10. Root / Git Safety

Root listing remains strict:

- `.git`
- `.github`
- `.gitignore`
- `01_CORE_PRODUCTION`
- `02_INCUBATION_STAGING`
- `03_RESEARCH_LAB`
- `04_INFRASTRUCTURE_ENGINEERING`
- `05_MARKET_DATA_VAULT`
- `06_GOVERNANCE_AND_COMPLIANCE`
- `07_BACKUPS`
- `08_CLOUD_FREE_RUN_LAB`

No root ZIP, loose CSV, loose PY, loose parquet, scratch, temp, or output file was observed.

Git status before audit report creation contained pre-existing unstaged research-intake changes under:

- `03_RESEARCH_LAB/strategy_research_intake/external_research_20260516/...`

Those files were present before this audit branch, were not touched by this audit, and must not be staged in this audit commit.

No code/data file was modified by the audit.

## 11. Required Fixes

### P0 - VE-ORB must fail closed on incomplete opening range

File:

- `03_RESEARCH_LAB/research_lab/strategies/ve_orb_volatility_expansion.py`

Problem:

- `_opening_range()` accepts any non-empty 07:00-08:00 window.
- It can emit a signal with only one OR bar.

Required correction:

- Add an explicit OR completeness guard.
- For M1/M5 ambiguity, use a conservative parameter such as `min_or_bars` or infer expected cadence from index deltas and require enough distinct bars.
- If the expected OR coverage cannot be verified, return `None`.

Required test:

- Add a synthetic unit test proving VE-ORB returns `None` when the 07:00-08:00 window is incomplete.

### P1 - Expand side coverage in tests

Required tests:

- TP-01 short-side momentum pullback signal.
- VE-ORB short-side breakout signal.

### P1 - Expand fail-closed tests

Required tests:

- NaN critical-input fail-closed for MR-02 and TP-01.
- Optional: patch file access around all four signal calls, not only MR-01.

No performance run is allowed until P0 is fixed and covered.

## 12. Decision

BLOCKED.

The code is clean enough in scope and no-lookahead posture, but not approved for train-only micro-backtest because VE-ORB violates fail-closed OR completeness. The correct next step is a surgical fix PR/commit for the blocker and test gaps, followed by a repeat external audit.

## 13. Safety Verification

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

## 14. Copy-Paste Summary for ChatGPT

PRIORITY_A_SKELETONS_BLOCKED_CONTRACT_FAILURE

External audit of commit `087faddc37b53fedb25840b67cc4431acc5dcf5b` found no engine/data/output/ZIP contamination and no critical lookahead in the four new skeletons. Tests and safe imports pass. Registry contains the four approved keys and no excluded strategy was implemented.

Blocker: `ve_orb_volatility_expansion` can emit a breakout signal with an incomplete 07:00-08:00 opening range. Synthetic evidence: with only one OR row, `signal()` returned a long signal. This violates fail-closed contract and blocks any train-only micro-backtest.

Required next step: fix VE-ORB OR completeness guard and add unit tests for incomplete OR, TP-01 short, VE-ORB short, and broader NaN/file-access fail-closed coverage. Repeat external audit before any backtest.
