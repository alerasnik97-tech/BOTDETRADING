# SUBBATCH 1A BLOCKER PATCH REPORT V1

## 1. Status

SUBBATCH_1A_BLOCKERS_PATCHED_READY_FOR_EXTERNAL_AUDIT

## 2. Executive Summary

This patch addresses only the external audit blockers for BO01/MR02 Asian range completeness and targeted unit/contract coverage. It does not evaluate strategy performance, edge, or profitability. It does not authorize any laboratory execution.

## 3. Scope

- patch BO01/MR02 only.
- patch tests BO01/MR02 only.
- no micro-run.
- no dry-run.
- no backtest.
- no validation.
- no holdout.
- no 2025/2026.
- no optimization/sweep.

## 4. Audit Blockers Addressed

- F-001 BO01 Asian range completeness.
- F-002 MR02 Asian range completeness.
- F-003 BO01 test quality.
- F-004 MR02 test quality.
- F-005 directional coverage.
- F-006 MR02 breach third prior bar.

## 5. Files Modified

- `03_RESEARCH_LAB/research_lab/strategies/BO01Strategy.py`
- `03_RESEARCH_LAB/research_lab/strategies/MR02Strategy.py`
- `03_RESEARCH_LAB/research_lab/tests/test_strategy_contract_bo01.py`
- `03_RESEARCH_LAB/research_lab/tests/test_strategy_tz_bo01.py`
- `03_RESEARCH_LAB/research_lab/tests/test_strategy_contract_mr02.py`
- `03_RESEARCH_LAB/research_lab/tests/test_strategy_tz_mr02.py`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/SUBBATCH_1A_BLOCKER_PATCH_REPORT_V1.md`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_SUBBATCH_1A_BLOCKER_PATCH_V1.md`

## 6. BO01 Patch Notes

BO01 now builds the Asian range only from exact GMT timestamps from `00:00` through `06:30` inclusive on the current UTC trade date. The check requires strict M5 grid membership, one row per expected timestamp, no duplicate expected timestamp, no off-grid timestamp inside the range, and the `00:00` and `06:30` endpoints. The range calculation still uses only rows before `i`.

No BO01 entry logic, ATR logic, EMA logic, stop logic, target logic, parameters, registry, factory, or runner contract was changed.

## 7. MR02 Patch Notes

MR02 now applies the same fail-closed Asian range completeness contract: exact GMT timestamps, strict M5 cadence, unique expected timestamps, mandatory endpoints, rejection of missing bars, rejection of duplicate timestamps, rejection of off-grid timestamps, and no use of rows after `i`.

No MR02 fakeout logic, engulfing logic, ATR logic, stop buffer, target logic, parameters, registry, factory, or runner contract was changed.

## 8. Tests Added/Updated

- BO01 missing `06:30` endpoint fails closed.
- BO01 duplicate Asian timestamp replacing a missing bar fails closed.
- BO01 wrong Asian cadence fails closed.
- BO01 short-side eligible signal contract.
- BO01 timezone test for missing `06:30` endpoint during valid entry time.
- MR02 missing `06:30` endpoint fails closed.
- MR02 duplicate Asian timestamp replacing a missing bar fails closed.
- MR02 wrong Asian cadence fails closed.
- MR02 long-side eligible fakeout signal contract.
- MR02 breach on the third prior bar remains eligible when all objective conditions hold.
- MR02 timezone test for missing `06:30` endpoint during valid entry time.

## 9. Tests Run

- `$env:PYTHONPATH="03_RESEARCH_LAB"; python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_strategy_contract_bo01.py" -v`
  - result: 11 tests, OK.
- `$env:PYTHONPATH="03_RESEARCH_LAB"; python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_strategy_tz_bo01.py" -v`
  - result: 7 tests, OK.
- `$env:PYTHONPATH="03_RESEARCH_LAB"; python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_strategy_contract_mr02.py" -v`
  - result: 12 tests, OK.
- `$env:PYTHONPATH="03_RESEARCH_LAB"; python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_strategy_tz_mr02.py" -v`
  - result: 7 tests, OK.
- `$env:PYTHONPATH="03_RESEARCH_LAB"; python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_engine_strategy_contract.py" -v`
  - result: 7 tests, OK.
- `$env:PYTHONPATH="03_RESEARCH_LAB"; python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_engine_time_contract.py" -v`
  - result: 5 tests, OK.
- `$env:PYTHONPATH="03_RESEARCH_LAB"; python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_strategy_activity_gates.py" -v`
  - result: 6 tests, OK.

Summary: all targeted tests passed.

## 10. Safety Scan

- blockers: 0.
- allowed hits: 29 lines, all negative restriction declarations or required future-audit governance wording in this report and the next audit prompt.
- notes: no blockers were identified in BO01Strategy.py, MR02Strategy.py, or the four BO01/MR02 test files.

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
- no registry modified;
- no strategies/__init__.py modified;
- no Sub-Batch 1B;
- no git add dot.

## 12. Decision

Sub-Batch 1A blocker patch is ready for external read-only audit.

## 13. Allowed Next Step

External read-only audit of Sub-Batch 1A blocker patch.

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
