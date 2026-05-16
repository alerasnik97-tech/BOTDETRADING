# D5 TELEMETRY CHANGE REQUEST REPORT

## 1. Status

D5_TELEMETRY_IMPLEMENTED_READY_FOR_CLAUDE_AUDIT

This change is additive telemetry only. It does not authorize adapter implementation, F06 real execution, real backtest execution, validation, holdout, 2025, 2026, certification, or lab green light.

## 2. Executive Summary

The engine now appends D5 telemetry fields to each exported trade record. Existing trade fields remain present and unchanged. The change exposes already available stop/risk/cost execution data from the `Position` object and close-time local variables.

`sl_pips` is available and emitted from the initial execution-risk distance divided by pair pip size. `net_r` is emitted as a duplicate of existing `pnl_r`.

`gross_r` is not faked. The current engine does not maintain an explicit pre-cost PnL source or exact per-component R decomposition. Therefore `gross_r=null`, `gross_r_available=false`, and the reason is recorded. Raw cost telemetry is emitted for future audit, while R-level cost breakdown fields remain null/unavailable.

## 3. Scope

Files changed:

| File | Purpose |
| :--- | :--- |
| `research_lab/engine.py` | Add append-only D5 telemetry fields to trade records. |
| `research_lab/tests/test_d5_core_telemetry.py` | Add synthetic behavior-neutral telemetry tests. |
| `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/f06_clean_train_only_rerun/D5_TELEMETRY_CHANGE_REQUEST_REPORT.md` | Record implementation and evidence. |
| `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/f06_clean_train_only_rerun/LAB_READINESS_GATE_CHECKLIST.md` | Update lab gate status. |
| `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/f06_clean_train_only_rerun/NEXT_PROMPT_CLAUDE_D5_TELEMETRY_AUDIT.md` | Prepare next audit prompt. |

Allowed in this scope:

- minimal core telemetry;
- synthetic unit tests;
- documentation;
- branch/commit/push/PR.

Forbidden and not done:

- adapter implementation;
- real F06 run;
- real backtest;
- strategy optimization or sweep;
- validation;
- holdout;
- 2025;
- 2026;
- raw/tick/parquet data reads;
- cost model behavior changes;
- signal/fill/exit/sizing behavior changes.

## 4. Telemetry Fields Added

Core inventory:

| field_needed | available_internal_source | source_file | source_line | safe_to_expose | requires_recompute | risk |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sl_pips` | `Position.initial_risk_distance / PAIR_META[pair]["pip_size"]` | `research_lab/engine.py` | `38`, `965`, `988` | YES | unit conversion only | LOW |
| `gross_r` | no explicit pre-cost PnL source | `research_lab/engine.py` | `955-963`, `985-987` | NO, emitted unavailable | no safe recompute | HIGH if faked |
| `net_r` | existing `pnl_r` calculation | `research_lab/engine.py` | `955-963`, `984` | YES | none | LOW |
| `spread_cost_r` | no exact R component source | `research_lab/engine.py` | `100-125`, `1012` | NO, emitted unavailable | unsafe decomposition | MEDIUM-HIGH |
| `slippage_cost_r` | no exact R component source | `research_lab/engine.py` | `134-169`, `1013` | NO, emitted unavailable | unsafe decomposition | MEDIUM-HIGH |
| `commission_cost_r` | entry/exit commission USD exists, R attribution not in `pnl_r` contract | `research_lab/engine.py` | `830`, `960`, `1014` | NO, emitted unavailable | unsafe decomposition | MEDIUM |
| `cost_total_r` | no exact R component source | `research_lab/engine.py` | `1015` | NO, emitted unavailable | unsafe decomposition | MEDIUM-HIGH |
| `stop_price` | `Position.sl` | `research_lab/engine.py` | `33`, `862`, `995` | YES | none | LOW |
| `risk_pips` | same as `sl_pips` | `research_lab/engine.py` | `965`, `990` | YES | unit conversion only | LOW |
| `risk_distance_price` | `Position.initial_risk_distance` | `research_lab/engine.py` | `38`, `870`, `991` | YES | none | LOW |
| `initial_stop_price` | `Position.sl` at entry | `research_lab/engine.py` | `862`, `996` | YES | none | LOW |
| `final_stop_price` | `Position.sl`; current engine does not mutate `position.sl` after entry | `research_lab/engine.py` | `862`, `997` | YES | none | LOW |
| `entry_price` | existing exported field | `research_lab/engine.py` | `974` | existing | none | LOW |
| `exit_price` | existing exported field | `research_lab/engine.py` | `976` | existing | none | LOW |

Telemetry field contract:

| field | meaning | source | nullable | reason | behavior impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `telemetry_version` | D5 telemetry schema version | constant `d5_core_telemetry_v1` | NO | traceability | none |
| `telemetry_behavior_neutral` | explicit marker for additive-only contract | literal `True` | NO | audit marker | none |
| `net_r` | existing net R | `pnl_r` local variable | NO | future ledger mapping | none |
| `gross_r` | pre-cost R | unavailable | YES | not safely separable | none |
| `gross_r_available` | availability flag | literal `False` | NO | prevents fake gross R | none |
| `gross_r_reason` | unavailable reason | literal reason string | NO | auditability | none |
| `sl_pips` | initial execution risk distance in pips | `initial_risk_distance / pip_size` | NO when pip size valid | future ledger mapping | none |
| `risk_pips` | synonym of `sl_pips` | same | NO when pip size valid | output clarity | none |
| `risk_distance_price` | initial execution risk distance in price units | `Position.initial_risk_distance` | NO | auditability | none |
| `risk_usd` | initial risk amount | `Position.risk_usd` | NO | sizing audit | none |
| `stop_price` / `initial_stop_price` / `final_stop_price` | stop trigger telemetry | `Position.sl` | NO | stop/risk audit | none |
| `entry_spread_pips` | spread at entry | `Position.entry_spread_pips` | NO | cost telemetry | none |
| `entry_slippage_pips` | slippage at entry | `Position.entry_slippage_pips` | NO | cost telemetry | none |
| `exit_slippage_pips` | slippage at exit | close-time `exit_slippage_pips` | NO | cost telemetry | none |
| `entry_commission_usd` | entry commission charged | `Position.entry_commission_usd` | NO | cost telemetry | none |
| `exit_commission_usd` | exit commission charged | close-time variable | NO | cost telemetry | none |
| `commission_total_usd` | entry + exit commission | local sum | NO | cost telemetry | none |
| `spread_cost_r` / `slippage_cost_r` / `commission_cost_r` / `cost_total_r` | R-level cost decomposition | unavailable | YES | not safely separable | none |
| `cost_breakdown_r_available` | availability flag | literal `False` | NO | prevents fake cost R | none |

## 5. Behavior-Neutrality Proof

The implementation only:

- adds a telemetry version constant;
- initializes `gap_reason=None` before possible assignment;
- stores `pnl_r` in a local variable instead of repeating the same expression;
- derives telemetry fields after `pnl_usd`, `exit_commission_usd`, cash update, and exit decision are already complete;
- appends additional keys to the trade dict.

No signal change:

- The strategy call remains `strategy_module.generate_signal(frame, i, params)`.
- Synthetic test confirms the same call indices and params are used.

No entry change:

- Entry fill logic, spread guard, slippage, stop entry, and position sizing code were not changed.

No exit change:

- Stop/target/news/final-close logic was not changed.

No PnL change:

- `pnl_usd` calculation remains the same.
- `pnl_r` is the same expression, now stored once as `pnl_r`.

No cost model change:

- Spread/slippage/commission functions and multipliers were not changed.

No sizing change:

- `risk_usd`, `units`, and `lots` calculations were not changed.

## 6. Tests

| test | purpose | result |
| :--- | :--- | :--- |
| `test_d5_telemetry_additive_fields_present` | New telemetry fields are emitted. | PASS |
| `test_d5_telemetry_existing_trade_fields_unchanged` | Existing field values remain at known synthetic expected values. | PASS |
| `test_d5_telemetry_does_not_change_pnl` | `pnl_usd`, `pnl_r`, `net_r` remain coherent. | PASS |
| `test_d5_telemetry_does_not_change_entry_exit_times` | Synthetic entry/exit timestamps stay unchanged. | PASS |
| `test_d5_telemetry_does_not_change_trade_count` | Trade count remains unchanged. | PASS |
| `test_d5_telemetry_no_strategy_call_change` | Strategy call path remains unchanged. | PASS |
| `test_d5_sl_pips_matches_initial_stop_distance_if_available` | `sl_pips` maps to initial risk distance. | PASS |
| `test_d5_gross_r_not_faked_if_unavailable` | `gross_r` and R-cost breakdown are not fabricated. | PASS |
| `test_d5_telemetry_version_present` | Version and behavior-neutral marker exist. | PASS |
| `test_no_real_data_paths_used` | Synthetic D5 test does not open forbidden data paths. | PASS |
| F06 pipeline unittest discovery | Existing F06 guard suite remains green. | PASS, 119 tests |

Commands run:

```powershell
python -m unittest research_lab.tests.test_d5_core_telemetry
python -m unittest discover -s "03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\pipelines\f06_evidence_rebuild\tests" -p "test_*.py"
```

Results:

- D5 synthetic tests: `10 passed`.
- F06 pipeline guard tests: `119 passed`.

Full `research_lab/tests` discovery was not run because that tree contains broader data/news/integration surfaces not required for this scoped D5 telemetry gate and could violate the no-real-data boundary.

## 7. Safety Verification

| item | status |
| :--- | :--- |
| adapter_implemented | NO |
| real_f06_run | NO |
| backtest_run | NO_REAL_BACKTEST; synthetic unit exercise only |
| validation_touched | NO |
| holdout_touched | NO |
| 2025_touched | NO |
| 2026_touched | NO |
| raw_data_loaded | NO |
| core behavior changed | NO |
| cost model changed | NO |
| strategy registry changed | NO |

## 8. Known Limitations

- `gross_r` remains unavailable because the engine does not compute or retain explicit pre-cost PnL.
- R-level `spread_cost_r`, `slippage_cost_r`, `commission_cost_r`, and `cost_total_r` remain unavailable because an exact per-component R decomposition is not currently maintained.
- Raw cost telemetry is now emitted, including entry spread/slippage, exit slippage, entry/exit commission, and commission total.
- `sl_pips` is available as initial execution-risk distance in pips, not as a recomputed signal-level ATR formula.
- Future adapter must fail closed if it requires `gross_r` as a numeric value before an approved exact gross-PnL source exists.

## 9. Decision

READY_FOR_CLAUDE_TELEMETRY_AUDIT

This branch is ready for Claude to audit the D5 telemetry diff, behavior-neutral tests, and unresolved `gross_r` limitation before any adapter work.

## 10. Copy-Paste Summary for ChatGPT

D5 telemetry change implemented on a scoped branch. Engine trade records now emit additive D5 telemetry: `telemetry_version`, `telemetry_behavior_neutral`, `net_r`, `sl_pips`, risk/stop fields, raw cost fields, execution metadata, and explicit unavailable flags for `gross_r` and R-level cost decomposition. Existing trade behavior is unchanged by code inspection and synthetic tests. D5 tests pass `10/10`; F06 pipeline guard tests pass `119/119`. No adapter, no real F06, no real backtest, no validation, no holdout, no 2025/2026, no raw data loaded. Decision: `READY_FOR_CLAUDE_TELEMETRY_AUDIT`, not lab green light.
