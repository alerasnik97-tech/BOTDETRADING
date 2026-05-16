# NEXT PROMPT REPEAT PRIORITY A SKELETON EXTERNAL AUDIT

Act as institutional external auditor before any backtest.

Audit branch:

- `fix/eurusd-priority-a-skeletons-blockers-20260516`

Audit purpose:

- Verify closure of the external-audit blocker in VE-ORB.
- Verify expanded unit tests.
- Decide whether the skeletons are ready for a first train-only micro-backtest prompt.

Files to audit:

- `03_RESEARCH_LAB/research_lab/strategies/ve_orb_volatility_expansion.py`
- `03_RESEARCH_LAB/research_lab/strategies/mr02_vwap_stretch_reversion.py`
- `03_RESEARCH_LAB/research_lab/tests/test_priority_a_skeletons.py`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/PRIORITY_A_SKELETON_BLOCKER_FIX_REPORT.md`

Required checks:

1. Confirm VE-ORB fails closed when 07:00-08:00 OR is incomplete.
2. Confirm VE-ORB still emits valid long and short signals with complete OR.
3. Confirm OR completeness guard is causal and uses no post-08:00 bars for OR construction.
4. Confirm cadence inference is conservative and fails closed if unverifiable.
5. Confirm MR-02 NaN/volume fail-closed fix is narrow and does not change strategic logic.
6. Confirm TP-01 short test is meaningful and no TP-01 strategy code changed.
7. Confirm no engine/data/root/output changes.
8. Confirm no file I/O, event/feed dependency, high precision, holdout, 2025/2026, ghost params, or ZIP workflow.

Allowed commands:

- `python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_priority_a_skeletons.py" -v`
- `python -c "import research_lab; print('research_lab OK')"`
- `python -c "from research_lab.strategies import STRATEGY_REGISTRY; print(len(STRATEGY_REGISTRY)); print([k for k in STRATEGY_REGISTRY.keys() if k in ['mr01_anchor_elastic','mr02_vwap_stretch_reversion','tp01_london_ny_momentum_pullback','ve_orb_volatility_expansion']])"`
- `python -c "import research_lab.engine; print('engine OK')"`
- `python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_engine.py"`
- `python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_lab_preflight*.py"`
- `python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_engine_stop_entry.py"`
- static scans over the relevant files

Forbidden:

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
- no root files

If audit passes:

- create `NEXT_PROMPT_PRIORITY_A_TRAIN_ONLY_MICRO_BACKTEST.md`;
- authorize only a first micro train-only run;
- one strategy at a time;
- no optimization;
- no sweep;
- no holdout;
- no 2025/2026;
- no ranking or promotion decision.

If audit fails:

- create a new fix prompt with exact blockers;
- do not authorize any backtest.
