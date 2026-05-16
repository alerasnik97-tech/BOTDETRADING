# NEXT PROMPT - OWNER APPROVAL D3-D5

Use this prompt only to approve or reject the D3-D5 governance package. This prompt does not authorize adapter implementation, F06 execution, backtest execution, validation, holdout, 2025, 2026, certification, or production/paper/demo use.

Authoritative documents:

- `OWNER_DECISION_D3_D5_COMPLETION_REPORT.md`
- `LAB_READINESS_GATE_CHECKLIST.md`

## 1. Approval Target

Approve or reject the following package:

- D3 canonical config proposal: `F06_PHASE3_CANONICAL_001`
- D4 explicit cost policy: base, conservative, stress
- D5 selected institutional resolution: Option A, future minimal additive behavior-neutral core telemetry

## 2. D3 To Approve

```json
{
  "config_id": "F06_PHASE3_CANONICAL_001",
  "strategy_name": "keltner_volatility_expansion_simple",
  "params": {
    "atr_mult_keltner": 2.0,
    "break_even_at_r": 1.0,
    "ema_base": 30,
    "ema_filter": 200,
    "expansion_atr_min": 1.2,
    "session_name": "research_08_1630",
    "stop_atr": 1.5,
    "target_rr": 1.5,
    "use_h1_context": false
  },
  "parameter_hash_preimage": "{\"atr_mult_keltner\":2.0,\"break_even_at_r\":1.0,\"ema_base\":30,\"ema_filter\":200,\"expansion_atr_min\":1.2,\"session_name\":\"research_08_1630\",\"stop_atr\":1.5,\"target_rr\":1.5,\"use_h1_context\":false}",
  "parameter_hash_sha256_proposed": "779946fcd23c73003baa9d9810bf52f1edbe50eef30aca43976a83086725ff65",
  "no_post_result_change": true
}
```

Owner response required:

- APPROVE_D3_AS_PROPOSED
- REJECT_D3_AND_EXPLAIN_LITERAL_CHANGES

## 3. D4 To Approve

| scenario | spread_pips | slippage_pips | commission_round_turn_usd | max_spread_entry_guard |
| :--- | ---: | ---: | ---: | :--- |
| base | `1.2` | `0.2` | `7.0` | `3.0` pips |
| conservative | `1.62` | `0.5` | `7.0` | `3.0` pips |
| stress | `3.0` | `0.8` | `7.0` | `3.0` pips |

Policy:

- Use explicit per-scenario spread values.
- Do not use `cost_profile="auto"` as the frozen decision.
- Conservative slippage floor is 0.5 pips.
- Commission is round-turn USD per lot.
- Stress scenario may interact with the 3.0-pip entry guard.

Owner response required:

- APPROVE_D4_AS_PROPOSED
- REJECT_D4_AND_EXPLAIN_LITERAL_CHANGES

## 4. D5 To Approve

Selected option:

`Option A - minimal additive behavior-neutral core telemetry`

Meaning:

- Future change request may append telemetry fields to engine trade output.
- It must not change strategy logic, fills, PnL, exits, costs, ranking, dates, or data scope.
- It must include tests proving behavior neutrality.
- It does not authorize implementation in this approval prompt.

Required future telemetry:

- `sl`
- `initial_risk_distance`
- `entry_spread_pips`
- `entry_slippage_pips`
- `entry_commission_usd`
- `exit_commission_usd`

Owner response required:

- APPROVE_D5_OPTION_A_FUTURE_TELEMETRY_CHANGE_REQUEST
- REJECT_D5_AND_CHOOSE_B_OR_C_WITH_REASON

## 5. Required Owner Countersign

The owner must state:

`I understand that approval of D3-D5 does not authorize adapter implementation, F06 real run, backtest, validation, holdout, 2025, 2026, certification, production, paper, demo, or funded trading.`

## 6. Next Agent Instructions After Approval

If and only if D3, D4, and D5 are approved exactly:

1. Update the owner decision docs to `D3_PINNED`, `D4_PINNED`, and `D5_RESOLVED_OPTION_A`.
2. Draft a design-only implementation plan for adapter mocks/tests and the separate Option A telemetry change request.
3. Do not run F06.
4. Do not run a backtest.
5. Do not touch validation, holdout, 2025, or 2026.
6. Do not modify core until the Option A change request is explicitly in scope.
7. Commit/push docs only unless the owner explicitly opens the next implementation scope.

If any approval is missing or modified:

1. Keep status `OWNER_D3_D5_PARTIAL_NEEDS_INPUT`.
2. Do not implement anything.
3. Record the exact unresolved literal or decision.
