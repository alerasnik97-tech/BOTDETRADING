# OWNER DECISION VALIDATION REPORT

> Read-only audit of the proposed owner decision D1-D5. No adapter, no F06,
> no backtest, no strategy run, no validation/holdout/2025/2026, no core change.

- branch: `research/f06-clean-train-only-rerun-20260515`
- head / origin head: `a4a0dca9ff21270eefe8ee01f0f1b57db3971204` (in sync)
- repo: `alerasnik97-tech/bottrading` — PR #6
- date: 2026-05-16

---

## 1. Status

**OWNER_DECISION_PARTIAL_NEEDS_INPUT**

D1 validated (with warnings), D2 validated, D3 + D4 require explicit owner
inputs, D5 is blocked pending an explicit owner output-mapping decision. No code
defects — all blockers are governance/decision blockers.

---

## 2. Executive Summary

- The selected strategy `keltner_volatility_expansion_simple` is causally
  clean, registered, tracked, and free of leakage / future data / old-result
  dependency. It is **not directly engine-runnable** because the engine calls
  `generate_signal` and the module exposes only `signal` (no alias). This is
  precisely the documented reason the safe engine adapter must exist — an
  expected gap, not a defect.
- Single-config frozen ranking (D2) is structurally compatible with the
  existing fail-closed ranking / config-uniqueness guards. No sweep, no
  optimization, no p-hacking surface.
- The strategy has **no owner-safe defaults**: `default_params()` is a single
  numpy-seeded random draw over a 13,824-point grid. Pinning the canonical
  config is a genuine owner decision; resolving the seed was deliberately NOT
  done (read-only scope + p-hacking prevention).
- The cost model exists and is conservative/hardened, but the 3 numeric cost
  scenarios (incl. the owner-mandated ≥0.5-pip conservative slippage floor) are
  not encoded anywhere and require owner sign-off.
- The F06 ledger contract mandates `gross_r`, `sl_pips`, `net_r`; the engine
  emits only net `pnl_r`. `net_r`/`side` map cleanly; `gross_r`/`sl_pips`
  cannot be reconstructed exactly and audit-safely without either an additive
  core telemetry change or a fragile frame-rejoin reconstruction. This is the
  decisive blocker for adapter readiness.

---

## 3. D1 Validation

`D1_VALIDATED_WITH_WARNINGS`

- Existence / tracked / registry / `NAME` / `WARMUP_BARS` / causality:
  **all PASS** (evidence in `OWNER_DECISION_D1_D5_FILLED.md` §2).
- W1: engine entrypoint mismatch (`engine.py:980` calls `generate_signal`;
  module defines `signal`, no alias) — adapter's core purpose.
- W2: precomputed-feature dependency contract (`kc_upper_*`, `kc_lower_*`,
  `range_atr`, `ema{ema_filter}`).
- W3: general params (`session_name`, `break_even_at_r`, `use_h1_context`)
  injected via `add_general_params`.
- Verdict: strategy is sound; warnings define adapter scope, not blockers.

## 4. D2 Validation

`D2_VALIDATED`

- Single-config frozen ranking confirmed compatible with `ranking_schema.json`,
  `check_config_uniqueness` (`f06_rebuild_pipeline.py:246`), and
  `no_validation_columns_in_train_only`.
- Degeneracy guard is not tripped by a single config (it targets many configs
  collapsing to few result tuples).
- No sweep / no optimization / `config_id` unique
  (`F06_PHASE3_CANONICAL_001`). New configs later require a new PR +
  pre-registration. No blockers.

## 5. D3 Validation

`D3_OWNER_DECISION_REQUIRED`

- `default_params()` = `stratified_sample_combinations(parameter_space(),1,42)[0]`
  — one seeded PCG64 random pick over 13,824 combos. **Not** an owner-safe
  default.
- Seed deliberately **not** resolved: would require executing project code
  (out of read-only scope) and would launder a random draw into a "canonical"
  config (p-hacking risk D3 exists to prevent).
- Exact 9-parameter domain table delivered for owner pinning; nothing invented.
- Blocker: 9 literal param values + post-pin `parameter_hash`.

## 6. D4 Validation

`D4_OWNER_DECISION_REQUIRED`

- Cost model present, conservative, hardened (spread+slippage+round-turn
  commission applied: `engine.py:100-159,827,956`).
- Numeric scenario values are not in code/config; `cost_report_schema.json`
  needs ≥3 named scenarios with numeric fields.
- Recommended base/conservative/stress table provided, grounded in
  `config.py` constants; owner must confirm: spread policy (explicit vs auto),
  conservative slippage floor `0.5`, stress/`max_spread_pips=3.0` interaction,
  `$7.0` round-turn commission.
- No optimistic costs, no 0.0-slippage-only scenario proposed.

## 7. D5 Validation

`D5_PARTIAL_NEEDS_OWNER_DECISION` → `BLOCKED_OUTPUT_MAPPING_DECISION_REQUIRED`

- Ledger contract requires `gross_r,sl_pips,net_r`
  (`ledger_schema.json`; runtime `validate_ledger_schema:300-329`).
- Engine exports net `pnl_r` only (`engine.py:960-974`); `Position` cost/risk
  fields are discarded before export.
- `net_r`←`pnl_r`, `side`←`direction`: clean, HIGH auditability.
- `sl_pips`, `gross_r`: NOT passively reconstructable audit-safely under
  no-core-change. Owner must select Option A (recommended: minimal additive,
  behavior-neutral core telemetry), B (passive frame-rejoin + hard
  reconciliation), or C (proxy; not recommended).

---

## 8. Blocking Issues

| # | Decision | Blocking issue | Type |
| :- | :--- | :--- | :--- |
| B1 | D3 | 9 canonical params not pinned (no safe defaults; seed not resolvable in scope) | Owner input |
| B2 | D4 | 3 numeric cost scenarios + spread policy + 0.5-pip conservative floor not pinned | Owner input |
| B3 | D5 | `gross_r`/`sl_pips` not in engine export; no audit-safe passive reconstruction without owner-chosen option | Owner decision (blocking) |
| B4 | D1 (W1) | Engine `generate_signal` vs module `signal` (no alias) | Adapter scope (expected) |

None are code defects. B1–B3 are decision blockers; B4 defines adapter scope.

## 9. Adapter Implementation Readiness

**NEEDS_OWNER_INPUT**

Not `READY_TO_DRAFT_ADAPTER_IMPLEMENTATION_PROMPT` because D3, D4 require owner
inputs and D5 is blocked pending an explicit owner output-mapping decision.
Once §7 of `OWNER_DECISION_D1_D5_FILLED.md` is answered, a **design-only**
adapter prompt (mocks/tests; still no F06/backtest/data) can be drafted.

## 10. Safety Verification

| Item | Status |
| :--- | :--- |
| adapter_implemented | NO |
| real_f06_run | NO |
| backtest_run | NO |
| strategy_executed | NO |
| optimization_or_sweep | NO |
| validation_touched | NO |
| holdout_touched | NO |
| 2025_touched | NO |
| 2026_touched | NO |
| core_engine_modified | NO |
| parameters_invented | NO |
| raw/tick/parquet_touched | NO |
| quarantine/old-outputs used as SoT | NO |
| code/data changes | NONE (docs-only) |

## 11. Copy-Paste Summary for ChatGPT

```
F06 OWNER DECISION D1-D5 — VALIDATION RESULT (read-only, no execution)

STATUS: OWNER_DECISION_PARTIAL_NEEDS_INPUT
Branch research/f06-clean-train-only-rerun-20260515 @ a4a0dca (in sync, PR #6).

D1 keltner_volatility_expansion_simple = VALIDATED_WITH_WARNINGS
  - clean/causal/registered/tracked, no leakage/future/old-results.
  - WARNING: engine calls generate_signal(); module defines only signal()
    (no alias) -> not directly engine-runnable -> this is exactly why the
    safe engine adapter must exist. Also depends on precomputed features
    (kc_upper_*, kc_lower_*, range_atr, ema{filter}) + general params.
D2 single-config frozen ranking = VALIDATED. Compatible with ranking +
  config-uniqueness guards. No sweep/optimization. config_id unique.
D3 canonical config = OWNER_DECISION_REQUIRED. default_params() is a numpy
  seed=42 random pick over 13,824 combos -> NOT safe defaults. Owner must
  pin 9 literals: ema_base{20,30}, atr_mult_keltner{1.5,2.0},
  ema_filter{100,200}, expansion_atr_min{1.0,1.2,1.5}, stop_atr{1.0,1.5},
  target_rr{1.2,1.5,2.0}, session_name(16 variants), use_h1_context{F,T},
  break_even_at_r{null,1.0,1.2}. parameter_hash AFTER pinning.
D4 cost policy = OWNER_DECISION_REQUIRED. Model exists & hardened
  (spread 1.2 base / *1.35 stress, slippage 0.2 base, commission $7 RT).
  Owner must pin 3 scenarios (base/conservative/stress), confirm
  conservative slippage floor 0.5 pips, spread policy explicit-vs-auto,
  and stress vs max_spread_pips=3.0 entry rejection.
D5 gross_r/sl_pips = BLOCKED_OUTPUT_MAPPING_DECISION_REQUIRED. Ledger
  schema mandates gross_r/sl_pips/net_r; engine emits only net pnl_r;
  Position cost/risk fields discarded before export. net_r/side map
  cleanly. Owner must choose: A) minimal additive behavior-neutral core
  telemetry (recommended), B) passive frame-rejoin + hard reconciliation,
  C) proxy gross_r:=net_r (not recommended).

ADAPTER READINESS: NEEDS_OWNER_INPUT (not ready to draft impl prompt).
SAFETY: no adapter, no F06, no backtest, no strategy run, no validation,
  no holdout, no 2025/2026, no core change, no invented params. Docs-only.
NEXT: owner answers D3/D4/D5 inputs in OWNER_DECISION_D1_D5_FILLED.md §7.
```
