# NEXT PROMPT FIX PRIORITY A SKELETON BLOCKERS

Act as Lead Quant Developer and unit-test engineer.

Goal:

Fix only the external-audit blockers in the Priority A skeleton implementation.

Branch to start from:

- `audit/eurusd-priority-a-skeletons-code-audit-20260516`

Primary blocker:

- `ve_orb_volatility_expansion` can emit a signal with an incomplete 07:00-08:00 opening range.

Allowed code files:

- `03_RESEARCH_LAB/research_lab/strategies/ve_orb_volatility_expansion.py`
- `03_RESEARCH_LAB/research_lab/tests/test_priority_a_skeletons.py`
- governance report/prompt files under `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/`

Forbidden:

- no backtest
- no strategy run
- no optimization
- no sweep
- no validation
- no holdout
- no 2025/2026
- no event feed activation
- no high precision activation
- no F06 real
- no F06 adapter
- no engine changes
- no data mutation
- no main touch
- no force push
- no ZIP workflow
- no root files
- no `git add .`

Required fixes:

1. Add a fail-closed OR completeness guard to `ve_orb_volatility_expansion`.
2. Add a unit test proving VE-ORB returns `None` when the 07:00-08:00 window is incomplete.
3. Add TP-01 short-side synthetic signal coverage.
4. Add VE-ORB short-side synthetic signal coverage.
5. Add MR-02 and TP-01 NaN critical-input fail-closed coverage.
6. Optionally extend file-access patch coverage to all four skeletons.

Acceptance tests:

- `python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_priority_a_skeletons.py" -v`
- `python -c "from research_lab.strategies import STRATEGY_REGISTRY; print(len(STRATEGY_REGISTRY)); print([k for k in STRATEGY_REGISTRY.keys() if k in ['mr01_anchor_elastic','mr02_vwap_stretch_reversion','tp01_london_ny_momentum_pullback','ve_orb_volatility_expansion']])"`
- `python -c "import research_lab.engine; print('engine OK')"`

Static safety scan:

- Run the blocked-token scan on the four Priority A skeletons and `test_priority_a_skeletons.py`.
- There must be no file I/O, event/feed dependency, high precision dependency, holdout/2025/2026 dependency, or VE-01 ghost params.

Deliverable:

- Surgical code/test fix.
- Governance note summarizing the blocker closure.
- Commit and push to a fix branch.

Do not run any backtest after the fix. The next phase must be a repeat external audit.
