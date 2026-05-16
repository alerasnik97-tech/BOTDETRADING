# OWNER FINAL APPROVAL D3-D5

## 1. Status

OWNER_D3_D5_APPROVED

This document records the owner's explicit final approval for D3, D4, and D5. This is a governance closure only. It does not authorize adapter implementation, core modification, F06 execution, backtest execution, validation, holdout, 2025, 2026, certification, or lab green light.

Precheck evidence:

| Check | Result |
| :--- | :--- |
| branch | `research/f06-clean-train-only-rerun-20260515` |
| local HEAD before this approval package | `f6df4271e48e04e3593e197019571f88e6aac206` |
| origin HEAD before this approval package | `f6df4271e48e04e3593e197019571f88e6aac206` |
| python backtest/sweep/optimization processes | none detected |
| tracked changes before this package | none detected |

Local reservation: ignored/untracked historical artifacts remain present in the worktree from earlier tasks. They were not used, modified, staged, or committed by this approval package.

## 2. Executive Summary

D1 remains `D1_VALIDATED_WITH_WARNINGS`: `keltner_volatility_expansion_simple` is the selected F06 candidate, but the engine calls `generate_signal` while the module exposes `signal`. That warning is still adapter/wrapper scope for a future task.

D2 remains `D2_VALIDATED`: the ranking policy is single-config frozen. No sweep, no optimization, and no multi-config comparison are authorized for this phase.

D3 is now owner-approved and hashed. The canonical config is frozen before any future result is read. It is not derived from `default_params()`, seed resolution, historical performance, validation, holdout, 2025, 2026, or any old output.

D4 is now owner-approved. Base, conservative, and stress cost scenarios are explicit and must be written into future manifest and cost report artifacts.

D5 is now owner-approved as a future change request only. Option A is the institutional resolution, but it does not modify core now. It requires a separate scoped prompt, tests, and Claude audit before any adapter work.

## 3. D3 Final Approved Canonical Config

| Field | Value |
| :--- | :--- |
| status | `D3_OWNER_APPROVED_AND_HASHED` |
| strategy_name | `keltner_volatility_expansion_simple` |
| strategy_path | `research_lab/strategies/keltner_volatility_expansion_simple.py` |
| config_id | `F06_PHASE3_CANONICAL_001` |
| parameter_hash_required | YES |
| no_post_result_change | YES |
| source of approval | owner final decision in this task |

Approved params:

```json
{
  "atr_mult_keltner": 2.0,
  "break_even_at_r": 1.0,
  "ema_base": 30,
  "ema_filter": 200,
  "expansion_atr_min": 1.2,
  "session_name": "research_08_1630",
  "stop_atr": 1.5,
  "target_rr": 1.5,
  "use_h1_context": false
}
```

Canonical parameter_preimage:

```json
{"atr_mult_keltner":2.0,"break_even_at_r":1.0,"ema_base":30,"ema_filter":200,"expansion_atr_min":1.2,"session_name":"research_08_1630","stop_atr":1.5,"target_rr":1.5,"use_h1_context":false}
```

parameter_sha256:

```text
779946fcd23c73003baa9d9810bf52f1edbe50eef30aca43976a83086725ff65
```

D3 consistency validation:

| Check | Result |
| :--- | :--- |
| All params are literal values | PASS |
| Values belong to documented domains | PASS |
| `session_name` exists in `SESSION_VARIANTS` | PASS (`research_08_1630`) |
| `use_h1_context=false` valid | PASS |
| `break_even_at_r=1.0` valid | PASS |
| `target_rr=1.5` valid | PASS |
| `stop_atr=1.5` valid | PASS |
| No `default_params()` random draw used | PASS |
| No result/performance dependency | PASS |

## 4. D4 Final Approved Cost Policy

Status: `D4_OWNER_APPROVED`

| scenario | spread_pips | slippage_pips | commission_roundturn_usd | max_spread_guard_pips | purpose | approved |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| base | 1.2 | 0.2 | 7.0 | 3.0 | Non-zero baseline grounded in EURUSD defaults. | YES |
| conservative | 1.62 | 0.5 | 7.0 | 3.0 | Main governance scenario with owner-required slippage floor. | YES |
| stress | 3.0 | 0.8 | 7.0 | 3.0 | Severe bounded friction and spread-guard sensitivity. | YES |

D4 consistency validation:

| Check | Result |
| :--- | :--- |
| Explicit spread in every scenario | PASS |
| Explicit slippage in every scenario | PASS |
| Explicit round-turn commission in every scenario | PASS |
| Explicit max spread guard in every scenario | PASS |
| Conservative slippage >= 0.5 pips | PASS |
| Stress spread > conservative spread | PASS |
| Stress slippage > conservative slippage | PASS |
| No zero-slippage-only policy | PASS |
| Future cost report compatible | PASS |

Future manifest and cost report must include all three scenarios and the applied components: spread, slippage, and round-turn commission. Conservative and stress must remain available for future sensitivity checks.

## 5. D5 Final Approved Resolution

Status: `D5_OWNER_APPROVED_AS_FUTURE_CHANGE_REQUEST`

Selected option:

`Option A - minimal additive behavior-neutral core telemetry`

Owner approval meaning:

- Option A is the preferred institutional resolution for `gross_r` / `sl_pips`.
- A future separate change request is required before any implementation.
- No core engine modification is authorized in this scope.
- No adapter implementation is authorized in this scope.
- No F06 real run, backtest, strategy run, validation, holdout, 2025, or 2026 is authorized.

Required future telemetry target:

- `gross_r` or equivalent pre-cost R source.
- `sl_pips` or risk-distance-in-pips source.
- Stop/risk telemetry needed for the output contract.
- Cost components needed to reconcile gross-to-net R.
- Any added fields must be additive-only and auditable.

Minimum future tests:

| Test | Required proof |
| :--- | :--- |
| behavior-neutral regression | Same inputs produce identical signals, entries, exits, `pnl_usd`, `pnl_r`, timestamps, and exit reasons before/after telemetry. |
| no strategy behavior change | Telemetry additions do not alter signal generation, order entry, stop/target logic, spread guard, news behavior, or cost calculation. |
| telemetry presence | Future trade rows expose the minimum fields required for ledger mapping. |
| ledger mapping | `net_r` maps from engine net R, `sl_pips` comes from emitted risk/stop telemetry, and `gross_r` is derived from emitted pre-cost/cost components. |
| fail-closed missing telemetry | Missing telemetry blocks adapter output. |
| no leakage | No validation, holdout, 2025, or 2026 access. |

## 6. What This Approval Authorizes

Only the following:

- documentation closure for D3, D4, and D5;
- future scoped telemetry change request planning;
- future adapter planning after telemetry gate;
- GitHub documentation update and PR comment.

## 7. What This Approval Does NOT Authorize

- adapter implementation now;
- core modification now;
- telemetry implementation now;
- F06 real run;
- backtest;
- strategy execution;
- optimization;
- sweep;
- validation;
- holdout;
- 2025;
- 2026;
- F06 certification;
- lab green light;
- PR ready conversion;
- merge;
- push to main;
- force push.

## 8. Remaining Gates Before Lab

1. D5 telemetry change request implemented in a separate scoped task.
2. Telemetry tests prove behavior neutrality.
3. Claude audit passes the telemetry change.
4. Safe adapter is implemented with mocks/tests only.
5. Claude audit passes the adapter.
6. A single train-only micro-run is explicitly authorized and executed.
7. Validator returns `READY_FOR_CLAUDE_AUDIT`.
8. Claude output audit passes the real train-only artifacts.
9. Validation, holdout, 2025, and 2026 remain untouched.
10. Local/repo hygiene is clean enough for a final lab readiness claim.
