# NEXT PROMPT PATCH SUBBATCH 1A BLOCKER V1

Act as a single-writer institutional quant implementation agent.

## Activation

Use this prompt only if the owner explicitly approves patching the Sub-Batch 1A audit blockers from `SUBBATCH_1A_SKELETONS_TESTS_EXTERNAL_AUDIT_V1.md`.

## Objective

Patch only the BO01/MR02 Asian range completeness blockers and the missing targeted tests.

## Allowed Files

- `03_RESEARCH_LAB/research_lab/strategies/BO01Strategy.py`
- `03_RESEARCH_LAB/research_lab/strategies/MR02Strategy.py`
- `03_RESEARCH_LAB/research_lab/tests/test_strategy_contract_bo01.py`
- `03_RESEARCH_LAB/research_lab/tests/test_strategy_tz_bo01.py`
- `03_RESEARCH_LAB/research_lab/tests/test_strategy_contract_mr02.py`
- `03_RESEARCH_LAB/research_lab/tests/test_strategy_tz_mr02.py`
- one markdown patch report under `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/`

## Required Patch

1. Add a shared local completeness pattern in each skeleton:
   - require timezone-aware index;
   - require unique timestamps inside the Asian range;
   - require strict M5 cadence for the Asian range;
   - require exact GMT timestamps from `00:00` through `06:30` inclusive;
   - fail closed if any expected timestamp is missing or duplicated;
   - do not use rows after `i`.

2. Add tests:
   - BO01 fails closed when the `06:30` Asian endpoint is missing;
   - BO01 fails closed when a duplicate Asian timestamp replaces a missing bar;
   - BO01 fails closed on wrong cadence;
   - BO01 short-side eligible signal;
   - MR02 fails closed for the same missing endpoint, duplicate timestamp, and wrong cadence cases;
   - MR02 long-side eligible fakeout signal;
   - MR02 breach occurring on the third prior bar.

## Prohibited

- no engine changes;
- no runner changes;
- no registry or `strategies/__init__.py` changes;
- no data changes;
- no micro-run;
- no dry-run;
- no backtest;
- no formal train;
- no validation;
- no holdout;
- no 2025/2026;
- no optimization/sweep;
- no Sub-Batch 1B;
- no `git add .`.

## Tests Allowed

Run only the four BO01/MR02 unit/contract test files and the same three lightweight contract test files used by the external audit.

## Expected Result

The patch is complete only if the new fail-closed tests would fail on the current audited implementation and pass after the patch. The final decision must remain limited to readiness for another external read-only audit, not execution authorization.
