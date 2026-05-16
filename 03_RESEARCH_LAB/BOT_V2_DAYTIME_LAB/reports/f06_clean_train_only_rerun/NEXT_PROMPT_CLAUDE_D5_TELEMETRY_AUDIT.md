# NEXT PROMPT - CLAUDE D5 TELEMETRY AUDIT

Act as an institutional quantitative governance auditor and senior behavior-neutral engine reviewer.

Audit only the scoped D5 telemetry change request branch:

`research/f06-d5-behavior-neutral-telemetry-20260516`

Expected base:

`research/f06-clean-train-only-rerun-20260515`

## Scope

Audit:

- code diff in `research_lab/engine.py`;
- synthetic tests in `research_lab/tests/test_d5_core_telemetry.py`;
- report `D5_TELEMETRY_CHANGE_REQUEST_REPORT.md`;
- updated `LAB_READINESS_GATE_CHECKLIST.md`;
- no real data access;
- no F06 real run;
- no validation/holdout/2025/2026;
- whether this is ready for the safe adapter implementation gate.

## Hard Prohibitions

Do not run F06 real.
Do not run real backtests.
Do not run strategy sweeps or optimization.
Do not touch validation.
Do not touch holdout.
Do not touch 2025.
Do not touch 2026.
Do not implement adapter.
Do not modify core.
Do not mark PR ready.
Do not merge.
Do not certify F06.
Do not claim lab green light.

If any command attempts real backtest, real strategy execution, raw/tick/parquet data, validation, holdout, 2025, or 2026, abort with:

`SCOPE_ESCALATION_BLOCKED`

## Required Audit Questions

1. Is the engine diff additive only?
2. Did any signal, entry, fill, stop, target, break-even, trailing, exit, cost, risk sizing, session, or max-trades logic change?
3. Are existing fields preserved and unchanged?
4. Are `pnl_usd` and `pnl_r` behavior-neutral?
5. Is `sl_pips` safely sourced from `initial_risk_distance / pip_size`?
6. Is `gross_r` correctly not faked when no explicit pre-cost PnL source exists?
7. Are cost R fields correctly unavailable rather than fabricated?
8. Are raw cost fields useful for future adapter audit?
9. Do the synthetic tests cover behavior-neutrality sufficiently for this narrow gate?
10. Did the F06 pipeline guard suite remain green?
11. Was any real market data, raw/tick/parquet, validation, holdout, 2025, or 2026 touched?
12. Can the project proceed to the next gate: safe adapter implementation with mocks/tests only?

## Expected Evidence

Review at minimum:

- `git diff research/f06-clean-train-only-rerun-20260515...research/f06-d5-behavior-neutral-telemetry-20260516 -- research_lab/engine.py`
- `git diff research/f06-clean-train-only-rerun-20260515...research/f06-d5-behavior-neutral-telemetry-20260516 -- research_lab/tests/test_d5_core_telemetry.py`
- test output:
  - `python -m unittest research_lab.tests.test_d5_core_telemetry`
  - `python -m unittest discover -s "03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/pipelines/f06_evidence_rebuild/tests" -p "test_*.py"`

## Required Output

Return:

1. `CLAUDE_D5_TELEMETRY_AUDIT_PASS`
2. `CLAUDE_D5_TELEMETRY_AUDIT_PASS_WITH_RESERVATIONS`
3. `CLAUDE_D5_TELEMETRY_AUDIT_FAIL`

Include:

- findings by severity;
- exact file/line references;
- whether behavior-neutrality is proven enough for this gate;
- whether `gross_r` unavailable handling is acceptable;
- whether safe adapter implementation may proceed after audit;
- remaining blockers before lab green light.

Final decision must be one clear state. Do not certify F06. Do not authorize lab green light.
