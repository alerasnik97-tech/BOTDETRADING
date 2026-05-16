# NEXT PROMPT AUDIT PRIORITY A SKELETONS BEFORE BACKTEST

Act as institutional external auditor before any backtest.

Audit only the Priority A skeleton implementation branch:

- `research/eurusd-priority-a-skeletons-20260516`

Audit these files:

- `03_RESEARCH_LAB/research_lab/strategies/mr01_anchor_elastic.py`
- `03_RESEARCH_LAB/research_lab/strategies/mr02_vwap_stretch_reversion.py`
- `03_RESEARCH_LAB/research_lab/strategies/tp01_london_ny_momentum_pullback.py`
- `03_RESEARCH_LAB/research_lab/strategies/ve_orb_volatility_expansion.py`
- `03_RESEARCH_LAB/research_lab/strategies/__init__.py`
- `03_RESEARCH_LAB/research_lab/tests/test_priority_a_skeletons.py`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/PRIORITY_A_SKELETON_IMPLEMENTATION_REPORT.md`

Scope:

1. Verify the current strategy contract without modifying engine code.
2. Confirm no lookahead in indicator windows and signal confirmation.
3. Confirm all skeletons are OHLCV-only and fail-closed.
4. Confirm no external event/feed dependency and no high precision dependency.
5. Confirm no file I/O, data-loader call, output creation, or import side effect.
6. Confirm registry adds only the four approved keys.
7. Confirm VE-01, SD-01, ED-01, and Priority B strategies were not implemented.
8. Confirm unit tests are synthetic and do not touch real historical datasets.

Forbidden during audit:

- no backtest
- no strategy run
- no optimization
- no sweep
- no validation
- no holdout
- no event feed activation
- no high precision activation
- no F06 real
- no F06 adapter
- no engine changes
- no data mutation
- no main touch
- no force push
- no ZIP workflow

Required output:

- PASS / BLOCKED / OWNER_REVIEW_REQUIRED
- file-cited findings
- exact reason for any blocker
- explicit authorization or refusal for a future small train-only backtest gate

Do not approve performance work unless the skeleton contract is clean.
