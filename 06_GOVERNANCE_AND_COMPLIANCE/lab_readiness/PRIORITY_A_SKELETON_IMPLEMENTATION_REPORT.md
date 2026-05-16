# PRIORITY A SKELETON IMPLEMENTATION REPORT

## 1. Status

PRIORITY_A_SKELETONS_READY_FOR_AUDIT

## 2. Executive Summary

Implemented only the four final Priority A signal skeletons approved by the post-Grok arbitration:

- MR-01 Anchor Elastic
- MR-02 VWAP Stretch Reversion
- TP-01 London-NY Momentum Pullback, reformulated
- VE-ORB Volatility Expansion

This phase was limited to skeleton signal logic, registry wiring, synthetic unit tests, and governance documentation. No backtest, strategy run, optimization, sweep, validation, holdout, engine edit, data edit, main touch, force push, or ZIP workflow was performed.

The implementation follows the existing `research_lab.strategies` contract:

- each strategy is a module;
- each module exposes `NAME`, `WARMUP_BARS`, `default_params()`, `parameter_grid()`, and `signal(frame, i, params)`;
- `STRATEGY_REGISTRY` maps `NAME` to module;
- `signal()` returns either `None` or a dict compatible with `engine.validate_signal_risk_contract`.

## 3. Strategies Implemented

### MR-01

File:

- `03_RESEARCH_LAB/research_lab/strategies/mr01_anchor_elastic.py`

Contract:

- anchored intraday APM/VWAP from prior same-session bars;
- causal residual standard deviation;
- long after downside excursion and re-entry;
- short after upside excursion and re-entry;
- trend softness via ADX when present, otherwise causal EMA slope fallback;
- price stop from excursion extreme plus ATR buffer;
- price target capped by anchor / max R logic.

### MR-02

File:

- `03_RESEARCH_LAB/research_lab/strategies/mr02_vwap_stretch_reversion.py`

Contract:

- anchored VWAP from prior same-session bars;
- causal VWAP residual band at 2.25 standard deviations;
- long/short only on objective re-entry inside stretched band;
- price stop from excursion extreme plus ATR buffer;
- target at VWAP center by default.

### TP-01

File:

- `03_RESEARCH_LAB/research_lab/strategies/tp01_london_ny_momentum_pullback.py`

Contract:

- 08:00-12:00 NY signal window;
- prior five-bar momentum filter;
- causal ATR percentile filter with 200 prior ATR observations;
- EMA20 pullback and continuation confirmation;
- price stop from pullback extreme plus ATR buffer;
- fixed 2R target contract.

### VE-ORB

File:

- `03_RESEARCH_LAB/research_lab/strategies/ve_orb_volatility_expansion.py`

Contract:

- opening range built from 07:00-08:00 NY bars only;
- signals only after opening range is complete;
- causal ATR percentile expansion filter;
- opening-range width guard in ATR units;
- breakout close above OR high / below OR low;
- stop at opposite OR side;
- fixed 2R target contract.

## 4. Explicit Exclusions

Not implemented:

- VE-01 RV Shock Break
- SD-01 Europe Extreme Failure
- ED-01 Post-News Stabilization

Reason:

- VE-01 remains REVIEW because of unsupported ghost parameters.
- SD-01 remains REJECTED because of high correlation risk with Manipulante.
- ED-01 remains DEFERRED until event-data certification exists.

## 5. Signal Contract

Existing engine-compatible output:

- `signal`: `1` for long, `-1` for short
- `direction`: `long` or `short`
- `stop_mode`: `price`
- `stop_price`: finite hard stop
- `target_mode`: `price` or `rr`
- `target_price` or `target_rr`
- `break_even_at_r`: `None`
- `trailing_atr`: `False`
- `session_name`: `all_day`

Fail-closed output:

- `None`

No strategy reads files, calls data loaders, reads external feeds, or mutates global state during import or signal generation.

## 6. No-Lookahead Controls

Controls implemented:

- anchored VWAP/APM uses only rows before index `i`;
- residual standard deviation uses only prior session rows;
- MR re-entry uses the current closed bar only as confirmation;
- TP momentum uses prior bars and current closed bar confirmation;
- TP ATR percentile uses prior ATR observations and compares to current closed-bar ATR;
- VE opening range uses only 07:00-08:00 bars and refuses signals before 08:00;
- VE ATR percentile uses prior ATR observations and current closed-bar ATR;
- no future row is referenced in any skeleton;
- all missing/NaN critical inputs return `None`.

## 7. Unit Tests

New test file:

- `03_RESEARCH_LAB/research_lab/tests/test_priority_a_skeletons.py`

Executed:

- `python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_priority_a_skeletons.py"`
- Result: `Ran 11 tests in 0.135s - OK`

Coverage:

- imports of all four strategies;
- signal value constrained to `-1`, `0`, `1`;
- fail-closed with insufficient data;
- file-access guard during signal call;
- source scan for blocked external dependencies;
- registry contains the four approved keys;
- MR-01 long/short synthetic extremes;
- MR-02 long/short synthetic band re-entry;
- TP-01 momentum pullback vs lateral range;
- VE-ORB after OR completion and no signal during OR build;
- NaN critical inputs fail closed;
- excluded-family token absence.

Additional safe tests executed:

- `test_engine.py`: `Ran 17 tests - OK`
- `test_lab_preflight*.py`: `Ran 6 tests - OK`
- `test_engine_stop_entry.py`: `Ran 3 tests - OK`

Safe imports executed:

- `import research_lab`: `research_lab OK`
- `STRATEGY_REGISTRY`: total `67`; new keys present
- `import research_lab.engine`: `engine OK`

## 8. Static Safety Scan

Command scope: the four new strategy files and the new Priority A test file.

Pattern:

- `2025|2026|holdout|sealed_holdout|forex_factory|news|high_precision|level2|rv5|rv15|p30|0.08|clean-sync|zip|read_csv|to_csv|open\(|Path\(`

Result:

- no matches in new files.

Interpretation:

- forbidden date tokens: NO
- holdout tokens: NO
- external event/feed dependency tokens: NO
- high precision token: NO
- level2 token: NO
- ghost VE-01 params: NO
- clean-sync token: NO
- ZIP token: NO
- explicit file I/O tokens: NO

## 9. Registry Changes

Updated:

- `03_RESEARCH_LAB/research_lab/strategies/__init__.py`

Keys added:

- `mr01_anchor_elastic`
- `mr02_vwap_stretch_reversion`
- `tp01_london_ny_momentum_pullback`
- `ve_orb_volatility_expansion`

No existing strategy key was removed or renamed.

Total strategies after update:

- `67`

## 10. Risks / Owner Review

Remaining audit items before any backtest:

- external no-lookahead review should inspect indicator windows and signal-time assumptions;
- MR-01 and MR-02 are intentionally related mean-reversion families and should be monitored for duplication/correlation before any progression decision;
- TP-01 and VE-ORB use synthetic contract tests only in this phase, not performance validation;
- parameter values are initial conservative skeleton constants, not optimized values;
- no claim of edge, PF, winrate, drawdown, or readiness for demo/funding/real is made.

## 11. Safety Verification

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

## 12. Copy-Paste Summary for ChatGPT

PRIORITY_A_SKELETONS_READY_FOR_AUDIT

Implemented only four final Priority A skeletons on `research/eurusd-priority-a-skeletons-20260516`:

- `mr01_anchor_elastic`
- `mr02_vwap_stretch_reversion`
- `tp01_london_ny_momentum_pullback`
- `ve_orb_volatility_expansion`

Updated `STRATEGY_REGISTRY`; total strategies now `67`.

Tests passed:

- Priority A skeleton tests: 11 OK
- `research_lab` import: OK
- registry import: OK, four new keys present
- engine import: OK
- optional safe tests: `test_engine.py` 17 OK, `test_lab_preflight*.py` 6 OK, `test_engine_stop_entry.py` 3 OK

Static scan of new files found no blocked tokens, no ghost VE-01 params, no external event/feed dependency, no high precision dependency, and no explicit file I/O.

No backtest, strategy run, optimization, sweep, validation, holdout, engine edit, data edit, main touch, force push, `git add .`, or ZIP workflow.

Next step: external audit of the four skeletons before any backtest is authorized.
