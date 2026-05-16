# NEXT PROMPT - AFTER TELEMETRY SAFE ADAPTER

Use this prompt only after the D5 telemetry change request has been implemented, tested, and Claude-audited with PASS.

This prompt is prepared in advance but is not active authorization. If telemetry audit has not passed, stop with:

`ADAPTER_BLOCKED_UNTIL_D5_TELEMETRY_AUDIT_PASS`

## 1. Scope

Allowed after telemetry audit PASS:

- implement the safe F06 adapter/wrapper with mocks and tests;
- map approved D3 config into the engine call boundary;
- map approved D4 cost policy into future manifest/cost report structures;
- map engine telemetry into future ledger fields;
- add fail-closed guards for missing telemetry or schema mismatch;
- update documentation.

Forbidden:

- F06 real execution;
- backtest execution;
- strategy execution against real data;
- optimization;
- sweep;
- validation;
- holdout;
- 2025;
- 2026;
- changing D3 params;
- changing D4 cost policy;
- changing D5 telemetry semantics;
- core modification;
- schema weakening;
- PR ready conversion;
- merge;
- push to main;
- force push;
- certification;
- lab green light.

If any command attempts engine/backtest/strategy execution, abort with:

`SCOPE_ESCALATION_BLOCKED`

## 2. Required Adapter Behavior

The adapter must:

- bridge `signal` to the engine-expected `generate_signal` behavior without changing strategy logic;
- inject exactly `config_id=F06_PHASE3_CANONICAL_001`;
- use the owner-approved D3 params and parameter hash;
- use the owner-approved D4 cost policy;
- map `pnl_r` to `net_r`;
- derive `sl_pips` only from emitted telemetry, not from fragile frame re-joins;
- derive `gross_r` only from emitted pre-cost/cost telemetry;
- fail closed if telemetry is missing or inconsistent;
- emit only mock/synthetic outputs in this phase.

## 3. Required Tests

Tests must cover:

1. D3 config hash and literals are unchanged.
2. D4 scenarios are embedded exactly.
3. `signal`/`generate_signal` bridging does not change signal payload semantics.
4. Ledger mapping rejects missing `gross_r`/`sl_pips` telemetry.
5. No validation columns are emitted.
6. No validation, holdout, 2025, or 2026 paths are read.
7. No real run occurs.

## 4. Documentation Output

Create/update:

- `SAFE_ENGINE_ADAPTER_IMPLEMENTATION_REPORT.md`
- `SAFE_ENGINE_ADAPTER_TEST_EVIDENCE.md`
- a next prompt for exactly one train-only micro-run only if adapter audit PASS

## 5. Final State Required

Acceptable final states:

- `SAFE_ADAPTER_IMPLEMENTED_AWAITING_CLAUDE_AUDIT`
- `SAFE_ADAPTER_BLOCKED`
- `SAFE_ADAPTER_REJECTED_SCOPE_RISK`

Do not claim lab readiness. Do not execute F06 real.
