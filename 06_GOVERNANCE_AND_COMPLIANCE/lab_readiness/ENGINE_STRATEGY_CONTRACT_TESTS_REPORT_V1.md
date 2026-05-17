# ENGINE STRATEGY CONTRACT TESTS REPORT V1

> Test-only, fail-safe phase closing the systemic gaps from
> `ENGINE_STRATEGY_CONTRACT_AUDIT_VEORB_V1.md`. No engine/strategy/runner/data
> behavior changed. No backtest, validation, holdout, 2025/2026, optimization
> or sweep. No ZIP. No `git add .`. VE-ORB NOT revived.

---

## 1. Status

**CONTRACT_TESTS_GREEN_WITH_DOCUMENTED_MEDIUM_RISK**

All 17 new contract tests pass; all related existing contract suites pass with
**zero regressions** (no core code was touched). The engine↔strategy contract
is now **encoded as executable guards** instead of an unwritten assumption.
The residual MEDIUM risk is **documented, not eliminated**: the engine still
hands every strategy the full multi-year frame (causality remains the
strategy's responsibility — now explicitly tested), and the zero-activity
sentinel exists and is tested but is **not yet wired into the production seal
gate** (deliberately deferred — no core change in this phase).

## 2. Scope

- base branch: `audit/engine-strategy-contract-veorb-v1-20260517` @ `9a37b2b9b01a9bd5687881bc0969f93be78bfae9`
- test branch: `test/engine-strategy-contract-guards-v1-20260517`
- Read-only inputs: `engine.py`, `runners/formal_train_runner.py`, `data_loader.py`, `strategies/ve_orb_volatility_expansion.py`, `tests/test_engine.py` (fixture pattern), the prior audit doc.
- Changes: **3 new test files + this report only.** No engine/strategy/runner/data/config modification. No rerun. No `--execute`.

## 3. Files Added/Modified

| file | type |
|---|---|
| `03_RESEARCH_LAB/research_lab/tests/test_engine_strategy_contract.py` | NEW test (Groups A+B) |
| `03_RESEARCH_LAB/research_lab/tests/test_strategy_activity_gates.py` | NEW test (Groups C+D) |
| `03_RESEARCH_LAB/research_lab/tests/test_engine_time_contract.py` | NEW test (Group E) |
| `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/ENGINE_STRATEGY_CONTRACT_TESTS_REPORT_V1.md` | NEW report |

No engine/strategy/runner/core file modified.

## 4. Tests Added (17)

**A — Signal contract / invocation (`test_engine_strategy_contract.py`)**
- `test_engine_contract_exposes_full_frame_and_requires_strategy_causality` — spy proves the engine passes ONE full frame (len == full, single object id) and a contiguous `i` from `WARMUP_BARS`; documents "no causal sandbox".
- `test_signal_signature_is_frame_i_params` — locks the documented `(frame, i, params)` signature (introspection).
- `test_engine_prefers_generate_signal_when_present` — locks the engine.py:721-725 fallback.
- `test_entry_fill_is_t_plus_1_never_same_bar` — price-based proof the fill is the NEXT bar (~1.1050), never the signal bar (≤1.10050); immune to the cost model.

**B — Universal anti-lookahead harness**
- `lookahead_leak_indices()` reusable harness (future-row poisoning with NaN and wild values).
- `test_harness_detects_a_known_leaky_strategy` — proves the harness CATCHES a deliberately leaky strategy (non-decorative).
- `test_ve_orb_is_causal_under_future_poisoning` — VE-ORB invariant under future poisoning.

**C — Zero-activity sentinel (`test_strategy_activity_gates.py`)**
- `assess_activity()` pure helper.
- `test_zero_trades_is_flagged_degenerate`; `test_veorb_shape_15_trades_all_in_2015_is_flagged` (the exact incident shape trips it); `test_healthy_distribution_is_not_flagged` (no false positive).

**D — Performance-complexity guard**
- `scan_quadratic_risk()` heuristic lint.
- `test_scanner_flags_veorb_known_quadratic_pattern` (characterization); `test_scanner_is_quiet_on_a_clean_o1_strategy` (no false positive); `test_veorb_signal_smoke_completes_within_generous_budget` (catastrophic-regression smoke, generous stable bound).

**E — Timezone / index / cadence (`test_engine_time_contract.py`)**
- `test_entry_open_index_respects_dst_offsets` (EST -0500 / EDT -0400); `test_strategy_receives_tz_aware_ny_index`; `test_engine_localizes_naive_index_as_utc_then_ny`; `test_m5_with_weekend_gaps_infers_stable_cadence_5`; `test_irregular_cadence_silently_returns_none` (documents the C8 silent-kill fragility).

## 5. Contract Findings Encoded As Tests

| audit finding | now guarded by |
|---|---|
| C5 engine passes full frame, no sandbox | `test_engine_contract_exposes_full_frame_and_requires_strategy_causality` |
| C6 contract undocumented / duck-typed | `test_signal_signature_is_frame_i_params`, `test_engine_prefers_generate_signal_when_present` |
| C1 T+1 causal execution | `test_entry_fill_is_t_plus_1_never_same_bar` |
| C7 O(N²) pattern uncaught | `scan_quadratic_risk` + 3 perf tests |
| C8 cadence silent-kill fragility | `test_irregular_cadence_silently_returns_none`, `test_m5_with_weekend_gaps_infers_stable_cadence_5` |
| C9 gate blind to signal density | `assess_activity` + 3 sentinel tests |
| C3 NY tz / DST contract | 3 timezone tests |

## 6. Anti-Lookahead Coverage

Universal, reusable future-poisoning harness; **proven to detect** a real leak
(leaky fixture) and applied to VE-ORB (invariant). Note: on synthetic data
VE-ORB legitimately returns `None` at most indices; the property tested
(decision invariant under future poisoning) holds regardless and the harness's
detection power is independently proven. The harness is generic and can be
pointed at any registered strategy in future work.

## 7. Zero-Activity Coverage

`assess_activity` flags zero-trade, single-active-year-over-multi-year, low
month-coverage, and single-month concentration. The **exact VE-ORB incident
shape (15 trades, all 2015-01/02 over 2015-2024) is correctly flagged
degenerate.** **Gap (honest):** this sentinel is a tested pure helper but is
**NOT wired into the runner seal gate** — wiring would be a core change, out of
this phase's scope → `WARN_ZERO_ACTIVITY_SENTINEL_NOT_WIRED` (see §10).

## 8. Performance Coverage

Static heuristic lint (`scan_quadratic_risk`) flags VE-ORB's documented O(N²)
shape and stays quiet on a clean O(1) strategy; a generous, stable smoke test
guards against catastrophic per-call regression. No flaky benchmark introduced.
This is a regression/characterization guard, **not** an enforced perf budget on
existing strategies.

## 9. Timezone/Index Coverage

DST offsets (EST/EDT), strategy receives tz-aware NY index, naive→UTC→NY
localization, and cadence stability vs the documented silent-`None` fragility
are all guarded. Closes the diagnostic's residual timezone doubt as an
executable contract.

## 10. Remaining Risks

| id | risk | status |
|---|---|---|
| R1 | Engine still exposes full frame; causality is strategy-side | Documented + test-guarded; design-level, MEDIUM |
| R2 | Zero-activity sentinel not wired to seal gate | `WARN_ZERO_ACTIVITY_SENTINEL_NOT_WIRED`; needs authorized core change |
| R3 | Anti-lookahead harness not yet auto-applied to every registered strategy in CI | Harness ready; batch wiring deferred (could be heavy imports) |
| R4 | Perf guard is heuristic/characterization, not an enforced budget | Acceptable; documents risk |
| R5 | Timeframe traceability (runner `M1` vs effective M5) | Carried from audit C10; not in test scope |

None are blockers; all are documented MEDIUM/again-design-level.

## 11. Decision

`CONTRACT_TESTS_GREEN_WITH_DOCUMENTED_MEDIUM_RISK`. Research may proceed to new
strategy batches **with** these guards in place, **provided** that wiring the
zero-activity sentinel into the seal gate and auto-applying the anti-lookahead
harness across the registry are scheduled as a separate, explicitly-authorized
core-change phase **before any large-scale acceleration**. VE-ORB stays
rejected/non-viable — NO validation, NO holdout, NO demo/real/FTMO, NO edge.

## 12. Next Step

- Keep these tests in the standard suite (they are lightweight: ~0.4s total).
- Schedule a separate authorized phase to (a) wire `assess_activity` into the
  reconciliation/seal gate as a non-blocking WARN, (b) auto-run the
  anti-lookahead harness over `STRATEGY_REGISTRY`, (c) resolve C10 timeframe
  traceability. No backtest/validation/holdout in that phase either.
- Do NOT revive or modify VE-ORB.

---

*Test-only phase. No code/data/runner/engine/strategy behavior changed, no
rerun, no validation/holdout/2025-2026, no optimization/sweep, no ZIP, no heavy
output committed, no `git add .`, no destructive git. New tests + related
existing suites: all green, no regressions.*
