# OWNER DECISION D3-D5 COMPLETION REPORT

## 1. Status

OWNER_D3_D5_READY_FOR_OWNER_APPROVAL

This is a governance decision package only. It does not authorize adapter implementation, F06 execution, backtest execution, validation, holdout, 2025, 2026, certification, or production/paper/demo use.

Precheck evidence:

| Check | Result |
| :--- | :--- |
| repo | `alerasnik97-tech/bottrading` |
| branch | `research/f06-clean-train-only-rerun-20260515` |
| local HEAD | `f68d957f1527f9028a45a073b42705420bfeafd2` |
| origin branch HEAD | `f68d957f1527f9028a45a073b42705420bfeafd2` |
| python backtest/sweep processes | none detected |
| tracked working tree changes before this package | none detected |

Local reservation: ignored/untracked historical artifacts were physically present before this task, including root-level ZIP/TXT clutter and previous untracked report folders. They were not used, moved, staged, or committed by this package.

## 2. Executive Summary

D1 is validated with warnings and D2 is validated. D3, D4, and D5 are now closed as explicit proposals suitable for owner approval.

D3 replaces the unsafe seeded `default_params()` draw with one literal, pre-registered canonical configuration proposal for `F06_PHASE3_CANONICAL_001`. The proposal is based on domain logic, simplicity, risk control, and compatibility with the F06 train-only session contract, not on historical performance.

D4 defines three numeric cost scenarios: base, conservative, and stress. The policy uses explicit spread values rather than `cost_profile="auto"` because a frozen decision package should be auditable from literals.

D5 selects the institutional option: minimal additive, behavior-neutral core telemetry in a future change request. This is slower than a passive adapter reconstruction, but it is the cleanest way to satisfy the ledger contract without weakening `gross_r` and `sl_pips`.

## 3. D3 Canonical Config Proposal

D3_CANONICAL_CONFIG_PROPOSAL:

| Field | Value |
| :--- | :--- |
| config_id | `F06_PHASE3_CANONICAL_001` |
| strategy_name | `keltner_volatility_expansion_simple` |
| status | `D3_PROPOSED_OWNER_APPROVAL_REQUIRED` |
| parameter_hash_required | YES |
| no_post_result_change | YES |
| owner_approval_required | YES |

Exact params:

| Param | Proposed value | Rationale |
| :--- | :--- | :--- |
| `ema_base` | `30` | Slower base EMA reduces noise sensitivity versus `20`; conservative for breakout confirmation. |
| `atr_mult_keltner` | `2.0` | Wider Keltner channel requires a more material expansion; less aggressive than `1.5`. |
| `ema_filter` | `200` | Long trend filter is simpler and less reactive than `100`; appropriate for first institutional train-only test. |
| `expansion_atr_min` | `1.2` | Middle value; avoids accepting every ATR-normal candle at `1.0` and avoids over-thinning with `1.5`. |
| `stop_atr` | `1.5` | Wider stop gives the breakout room; lower position size follows from fixed risk. |
| `target_rr` | `1.5` | Middle reward/risk; avoids shallow `1.2` and aggressive `2.0`. |
| `session_name` | `research_08_1630` | Subset of the F06 institutional NY `07:00`-`17:00` contract; avoids the late 16:30-17:00 rollover edge. |
| `use_h1_context` | `false` | Current strategy module does not reference this value; false avoids adding hidden H1 dependency before adapter design. |
| `break_even_at_r` | `1.0` | Risk-first exit management; once +1R is reached, the future engine behavior may protect capital without changing entry logic. |

parameter_hash_preimage:

```json
{"atr_mult_keltner":2.0,"break_even_at_r":1.0,"ema_base":30,"ema_filter":200,"expansion_atr_min":1.2,"session_name":"research_08_1630","stop_atr":1.5,"target_rr":1.5,"use_h1_context":false}
```

parameter_hash_sha256_proposed:

```text
779946fcd23c73003baa9d9810bf52f1edbe50eef30aca43976a83086725ff65
```

Important D3 limits:

- This hash is over the proposed literal preimage only.
- It is not derived from a backtest, seed draw, validation metric, or old output.
- If the owner changes any literal, the preimage and hash must be recomputed before any result is read.
- The seeded `default_params()` random draw remains prohibited as an official default.

## 4. D4 Cost Policy Proposal

D4_COST_POLICY_PROPOSAL:

| Field | Value |
| :--- | :--- |
| status | `D4_PROPOSED_OWNER_APPROVAL_REQUIRED` |
| spread policy | explicit per-scenario `spread_pips`; do not rely on `cost_profile="auto"` for the frozen decision |
| commission policy | round-turn USD per lot |
| owner_approval_required | YES |

Cost scenarios:

| scenario | spread_policy | spread_pips_or_multiplier | slippage_pips | commission_roundturn_usd | max_spread_entry_guard | purpose | acceptable_for_phase3? |
| :--- | :--- | :--- | ---: | ---: | :--- | :--- | :--- |
| base | explicit fixed spread | `1.2` pips | `0.2` | `7.0` | `3.0` pips | Reasonable non-zero baseline grounded in EURUSD code defaults. | YES |
| conservative | explicit fixed spread | `1.62` pips (`1.2 * 1.35`) | `0.5` | `7.0` | `3.0` pips | Owner-required conservative slippage floor plus stressed spread. | YES |
| stress | explicit fixed spread at entry guard cap | `3.0` pips | `0.8` | `7.0` | `3.0` pips | Severe but bounded friction; tests fragility and spread-guard interaction. | YES_FOR_STRESS_ONLY |

Manifest fields required:

- `cost_policy_version`
- `cost_scenarios`
- `spread_policy`
- `spread_pips`
- `slippage_pips`
- `commission_roundturn_usd`
- `max_spread_entry_guard`
- `cost_profile_used`
- `execution_mode_used`
- `components_applied.spread_component`
- `components_applied.slippage_component`
- `components_applied.round_turn_commission`
- `input_ledger_run_id`
- `input_is_quarantined_path`

Cost report fields required:

- `scenarios[].name`
- `scenarios[].spread_pips`
- `scenarios[].slippage_pips`
- `scenarios[].commission_round_turn_usd`
- `components_applied.spread_component=true`
- `components_applied.slippage_component=true`
- `components_applied.round_turn_commission=true`
- `input_ledger_run_id`
- `input_is_quarantined_path=false`

Operational rule: base can be reported, conservative must be the main governance decision scenario, and stress must be shown as fragility evidence. No scenario may use zero slippage as the only official cost view.

## 5. D5 Gross R / SL Pips Resolution Proposal

D5_RESOLUTION_PROPOSAL:

| Field | Value |
| :--- | :--- |
| status | `D5_PROPOSED_OWNER_APPROVAL_REQUIRED` |
| selected_option | A - minimal additive behavior-neutral core telemetry |
| implementation_allowed_now | NO |
| requires_core_change | YES |
| requires_schema_change | NO |
| blocker_status | `BLOCKED_UNTIL_OWNER_APPROVES_ADDITIVE_CORE_TELEMETRY_CHANGE_REQUEST` |

Reason:

The F06 ledger requires `gross_r`, `sl_pips`, and `net_r`. The engine currently exports `pnl_r`, which is net R, plus basic trade timestamps/prices. During execution, the `Position` object holds the needed risk/cost telemetry (`sl`, `initial_risk_distance`, `entry_commission_usd`, `entry_spread_pips`, `entry_slippage_pips`), but those fields are discarded before the exported trade dict.

Institutional recommendation:

- Select Option A for future implementation.
- Do not implement it in this task.
- Do not weaken the ledger schema.
- Do not use `gross_r = net_r` as an official proxy.
- Do not reconstruct `gross_r` passively unless Option A is explicitly rejected and a separate high-risk exception is approved.

Required future tests:

| Test | Purpose |
| :--- | :--- |
| behavior-neutral regression | Same synthetic trade path produces identical `pnl_usd`, `pnl_r`, `exit_reason`, and timestamps before/after telemetry addition. |
| telemetry presence test | Future engine trade rows include `sl`, `initial_risk_distance`, `entry_spread_pips`, `entry_slippage_pips`, `entry_commission_usd`, `exit_commission_usd`. |
| ledger mapping test | Adapter maps `pnl_r -> net_r`, derives `sl_pips` from engine-emitted risk distance and pair pip size, and derives `gross_r` from emitted cost components. |
| fail-closed missing telemetry test | If any required telemetry field is missing, adapter blocks output creation. |
| schema validator test | Ledger, ranking, cost report, manifest, and hash checks remain strict. |
| no leakage guard | No validation/holdout/2025/2026 is read or emitted. |

## 6. Remaining Owner Approvals

The owner must explicitly approve:

1. D3 exact canonical params and the proposed parameter hash preimage.
2. D4 base/conservative/stress cost scenarios and explicit spread policy.
3. D5 Option A, including a future additive-only, behavior-neutral core telemetry change request.

No adapter implementation may start until these approvals are recorded.

## 7. What This Does NOT Authorize

- no adapter implementation
- no F06 real
- no backtest
- no strategy execution
- no optimization
- no sweep
- no validation
- no holdout
- no 2025
- no 2026
- no certification
- no production
- no incubation
- no paper/demo/funded/real trading

## 8. Next Step

Owner approval is the next gate. If the owner approves D3, D4, and D5 exactly as proposed, the next task may draft a strictly bounded implementation plan for adapter mocks/tests and the future Option A telemetry change request. It still must not run F06, validation, holdout, 2025, or 2026.
