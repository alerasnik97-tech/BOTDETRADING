# SUBBATCH 1A IMPLEMENTATION REPORT V1

## 1. Status

SKELETONS_AND_TESTS_COMPLETED_READY_FOR_EXTERNAL_AUDIT

## 2. Executive Summary

Sub-Batch 1A now has minimal BO01 and MR02 strategy skeletons plus targeted unit/contract tests. The work is limited to skeleton code, synthetic tests, this implementation report, and a future read-only audit prompt. No performance claim is made.

## 3. Scope

- strategies: BO01, MR02 only.
- skeletons only.
- unit/contract tests only.
- no micro-run.
- no dry-run.
- no backtest.
- no validation.
- no holdout.
- no 2025/2026.

## 4. Files Created/Modified

- `03_RESEARCH_LAB/research_lab/strategies/BO01Strategy.py`
- `03_RESEARCH_LAB/research_lab/strategies/MR02Strategy.py`
- `03_RESEARCH_LAB/research_lab/tests/test_strategy_contract_bo01.py`
- `03_RESEARCH_LAB/research_lab/tests/test_strategy_tz_bo01.py`
- `03_RESEARCH_LAB/research_lab/tests/test_strategy_contract_mr02.py`
- `03_RESEARCH_LAB/research_lab/tests/test_strategy_tz_mr02.py`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/SUBBATCH_1A_IMPLEMENTATION_REPORT_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_SUBBATCH_1A_TESTS_AUDIT_V1.md`

## 5. Specs Read

- `06_GOVERNANCE_AND_COMPLIANCE/research_registry/first_batch_specs/BO01_IMPLEMENTATION_SPEC_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/research_registry/first_batch_specs/MR02_IMPLEMENTATION_SPEC_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/research_registry/first_batch_specs/FIRST_BATCH_TEST_PLAN_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/research_registry/first_batch_specs/FIRST_BATCH_SUBBATCH_DECISION_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/FINAL_LANGUAGE_PATCH_BEFORE_OWNER_DECISION_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_OWNER_APPROVES_SUBBATCH_1A_SKELETONS_TESTS_V2.md`

## 6. Implementation Notes BO01

BO01 follows the repo module contract: `ID`, `FAMILY_ID`, `NAME`, `WARMUP_BARS`, `EXPLICIT_TIMEFRAME`, `DEFAULT_PARAMS`, `default_params()`, `parameter_space()`, `parameter_grid()`, and `signal(frame, i, params)`.

The signal uses only rows up to the current index. It checks a GMT Asian range from 00:00 to 06:30, an entry window from 07:00 to 10:00, causal ATR14, causal EMA20, required `ema_m15_200`, minimum Asian range width, one-trade state controls, midpoint stop, and fixed 2R target mode.

The module is not added to `strategies/__init__.py` because registry changes are outside the approved whitelist.

## 7. Implementation Notes MR02

MR02 follows the same repo module contract as BO01.

The signal uses only rows up to the current index. It checks a GMT Asian range from 00:00 to 06:30, an entry window from 07:00 to 11:00, causal ATR14, prior breach of up to three bars, strict re-entry inside the Asian range, objective engulfing, maximum Asian range width, one-trade state controls, fakeout swing stop plus 2 pips, and fixed 1.5R target mode.

The module is not added to `strategies/__init__.py` because registry changes are outside the approved whitelist.

## 8. Tests Created

- `test_strategy_contract_bo01.py`: import/module contract, signal return contract, no file access during signal, small-frame fail-closed, future poisoning invariance, warmup gate, missing columns, tz-naive index, NaNs, daily trade count, active position, and forbidden source tokens.
- `test_strategy_tz_bo01.py`: GMT 07:00 eligibility, March DST fixture, November DST fixture, before-window rejection, after-window rejection, and no objective breakout rejection.
- `test_strategy_contract_mr02.py`: import/module contract, signal return contract, no file access during signal, small-frame fail-closed, future poisoning invariance, warmup gate, missing columns, tz-naive index, NaNs, daily trade count, active position, and forbidden source tokens.
- `test_strategy_tz_mr02.py`: GMT 07:10 eligibility, March DST fixture, November DST fixture, before-window rejection, after-window rejection, and no objective breach/failure rejection.

## 9. Tests Run

- `$env:PYTHONPATH="03_RESEARCH_LAB"; python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_strategy_contract_bo01.py" -v` -> 7 tests, OK.
- `$env:PYTHONPATH="03_RESEARCH_LAB"; python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_strategy_tz_bo01.py" -v` -> 6 tests, OK.
- `$env:PYTHONPATH="03_RESEARCH_LAB"; python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_strategy_contract_mr02.py" -v` -> 7 tests, OK.
- `$env:PYTHONPATH="03_RESEARCH_LAB"; python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_strategy_tz_mr02.py" -v` -> 6 tests, OK.
- `$env:PYTHONPATH="03_RESEARCH_LAB"; python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_engine_strategy_contract.py" -v` -> 7 tests, OK.
- `$env:PYTHONPATH="03_RESEARCH_LAB"; python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_engine_time_contract.py" -v` -> 5 tests, OK.
- `$env:PYTHONPATH="03_RESEARCH_LAB"; python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_strategy_activity_gates.py" -v` -> 6 tests, OK.

Result: all targeted tests passed.

## 10. Safety Scan

- blockers: 0.
- allowed hits: 30.
- notes: all hits are negative restriction declarations or governance prompt terms; no hits were found in the BO01/MR02 skeleton files or the four new test files.

## 11. Forbidden Actions Confirmation

- no backtest;
- no micro-run;
- no dry-run;
- no validation;
- no holdout;
- no 2025/2026;
- no optimization/sweep;
- no data modified;
- no engine modified;
- no runner modified;
- no strategy outside BO01/MR02 modified;
- no git add dot.

## 12. Decision

Sub-Batch 1A skeletons/tests are ready for external read-only audit.

## 13. Allowed Next Step

External read-only audit of BO01/MR02 skeletons and tests.

## 14. Forbidden Next Steps

- no micro-run;
- no dry-run;
- no backtest;
- no train;
- no validation;
- no holdout;
- no 2025/2026;
- no optimization/sweep;
- no Sub-Batch 1B;
- no parallel writers.
