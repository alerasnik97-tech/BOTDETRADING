# NEXT PROMPT - D5 TELEMETRY CHANGE REQUEST

Use this prompt only after `OWNER_FINAL_APPROVAL_D3_D5.md` exists and states `OWNER_D3_D5_APPROVED`.

Purpose: implement the minimum additive, behavior-neutral telemetry needed to resolve D5 output-contract mapping. This is a core-touching task, but only within this future prompt's explicit scope. It still does not authorize adapter implementation, F06 real run, backtest, validation, holdout, 2025, 2026, certification, or lab green light.

## 1. Scope

Allowed:

- inspect `research_lab/engine.py`;
- add only telemetry fields needed for ledger mapping;
- add tests proving behavior neutrality;
- add/update documentation for the telemetry change;
- run targeted tests for telemetry behavior;
- run broader tests only if required by the changed surface.

Forbidden:

- adapter implementation;
- F06 real execution;
- backtest execution;
- strategy execution;
- optimization;
- sweep;
- validation;
- holdout;
- 2025;
- 2026;
- cost model logic changes;
- signal logic changes;
- fill logic changes;
- exit logic changes;
- parameter changes;
- schema weakening;
- PR ready conversion;
- merge;
- push to main;
- force push.

If any command attempts engine/backtest/strategy execution, abort with:

`SCOPE_ESCALATION_BLOCKED`

If any edit changes strategy behavior, cost behavior, fill behavior, or dates, abort with:

`CORE_BEHAVIOR_CHANGE_BLOCKED`

## 2. Required Telemetry

Add only fields required to map future engine trades into the F06 ledger contract:

- `initial_risk_distance`
- `sl` or equivalent stop telemetry
- `risk_usd`
- `entry_spread_pips`
- `entry_slippage_pips`
- `exit_slippage_pips` if available at close
- `entry_commission_usd`
- `exit_commission_usd`
- pre-cost or cost-component source sufficient to derive `gross_r`

If a field cannot be emitted without changing behavior, stop and document `TELEMETRY_BLOCKED_REQUIRES_DESIGN_DECISION`.

## 3. Required Tests

Tests must prove:

1. Same synthetic path before/after telemetry has unchanged `signal_time`, `entry_time`, `exit_time`, `entry_price`, `exit_price`, `exit_reason`, `pnl_usd`, and `pnl_r`.
2. Same number of trades before/after telemetry.
3. Telemetry fields are present and numeric where required.
4. Missing telemetry would fail future ledger mapping.
5. No validation, holdout, 2025, or 2026 is accessed.

Use synthetic fixtures/mocks where possible. Do not run F06 real.

## 4. Documentation Output

Create or update:

- `D5_TELEMETRY_CHANGE_REQUEST_REPORT.md`
- `NEXT_PROMPT_AFTER_TELEMETRY_SAFE_ADAPTER.md` only if telemetry tests and audit status are clear

Report:

- exact fields added;
- behavior-neutral evidence;
- tests run;
- tests not run and why;
- remaining blocker status;
- whether adapter prompt may proceed.

## 5. Final State Required

Acceptable final states:

- `D5_TELEMETRY_IMPLEMENTED_AWAITING_CLAUDE_AUDIT`
- `D5_TELEMETRY_BLOCKED_REQUIRES_OWNER_DECISION`
- `D5_TELEMETRY_REJECTED_BEHAVIOR_CHANGE_RISK`

Do not claim lab readiness.
