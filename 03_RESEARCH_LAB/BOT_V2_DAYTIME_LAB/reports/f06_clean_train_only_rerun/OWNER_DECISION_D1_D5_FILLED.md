# OWNER DECISION D1-D5 FILLED

> Read-only governance validation. NO adapter implemented. NO F06 run.
> NO backtest. NO strategy executed. NO validation/holdout/2025/2026 touched.
> NO core modified. NO parameters invented.

- branch: `research/f06-clean-train-only-rerun-20260515`
- head: `a4a0dca9ff21270eefe8ee01f0f1b57db3971204`
- origin head: `a4a0dca9ff21270eefe8ee01f0f1b57db3971204` (in sync)
- repo: `alerasnik97-tech/bottrading` (PR #6)
- validation date: 2026-05-16
- evidence base: source code read-only (`research_lab/`, F06 pipeline schemas/contracts)

---

## 1. Status

**OWNER_DECISION_PARTIAL_NEEDS_INPUT**

| Decision | Status |
| :--- | :--- |
| D1 — Strategy selected | `D1_VALIDATED_WITH_WARNINGS` |
| D2 — Ranking governance | `D2_VALIDATED` |
| D3 — Canonical config | `D3_OWNER_DECISION_REQUIRED` |
| D4 — Cost policy | `D4_OWNER_DECISION_REQUIRED` |
| D5 — gross_r / sl_pips | `D5_PARTIAL_NEEDS_OWNER_DECISION` → `BLOCKED_OUTPUT_MAPPING_DECISION_REQUIRED` |

Adapter implementation remains **BLOCKED**. The blockers are decision blockers
(owner inputs), not code defects.

---

## 2. D1 — Strategy Selected

- selected_strategy_name: `keltner_volatility_expansion_simple`
- selected_strategy_path: `research_lab/strategies/keltner_volatility_expansion_simple.py`
- reason: Cleanest causal volatility-expansion candidate; compatible-by-family
  with the engine; representative of a single volatility-expansion family (F06).
- rejected_alternatives: `campaign3b_session_expansion`,
  `pm_volatility_squeeze_retest_m5`, `bollinger_mean_reversion_simple`,
  `ema_trend_pullback`.

**Audit (read-only) — PASS with warnings:**

| Check | Result | Evidence |
| :--- | :--- | :--- |
| File exists | PASS | `research_lab/strategies/keltner_volatility_expansion_simple.py` |
| Git-tracked | PASS | `git ls-files --error-unmatch` returns the file |
| Registered in `STRATEGY_REGISTRY` | PASS | `research_lab/strategies/__init__.py:75` + import line 8 |
| `NAME` correct | PASS | `NAME = "keltner_volatility_expansion_simple"` (line 6) |
| `WARMUP_BARS` present | PASS | `WARMUP_BARS = 120` (line 7) |
| No future data (`iat[i+1]`) | PASS | only `iat[i]`, `iat[i-1]`, `iat[i-2]` (lines 33-43) |
| No `shift(-1)` | PASS | not present |
| No centered rolling | PASS | strategy reads precomputed cols only; no rolling in module |
| No old-results dependency | PASS | no I/O, no result imports |
| No validation/holdout dependency | PASS | none |
| No 2025/2026 reference | PASS | none |

**WARNING W1 (engine entrypoint contract — this is the adapter's reason to exist):**
The engine calls `strategy_module.generate_signal(frame, i, params)`
(`research_lab/engine.py:980`). The selected module exposes only
`def signal(frame, i, params)` and provides **no** `generate_signal` alias
(contrast `research_lab/strategies/am_silver_bullet_ny_v2.py:152`:
`generate_signal = signal`). The strategy is therefore **not directly
engine-runnable today**; bridging `signal → generate_signal` is exactly the
purpose of the future safe engine adapter. Expected gap, not a code defect.

**WARNING W2 (feature dependency contract):** `signal()` consumes precomputed
frame columns: `kc_upper_{ema_base}_{atr_mult_keltner|'.'→'_'}`,
`kc_lower_{...}`, `range_atr`, `ema{ema_filter}` (e.g. `kc_upper_20_1_5`,
`ema100`). The adapter/runner must guarantee these features exist with exact
naming before any future run. Not a leakage issue (causal).

**WARNING W3 (general params):** `signal()` also requires
`params['break_even_at_r']` and `params['session_name']`, injected by
`add_general_params()` (`research_lab/strategies/common.py:69-74`), plus
`use_h1_context`. Feeds directly into the D3 open item.

---

## 3. D2 — Ranking Governance

- single_config_ranking: **YES**
- multi_config_frozen: **NO** (not now)
- reason: Avoid sweep / optimization / p-hacking. The ranking is one frozen
  summary row for one pre-registered `config_id`, not a parameter search.

**Audit — VALIDATED.** Consistent with existing fail-closed guards:

| Governance rule | Enforcement (read-only evidence) |
| :--- | :--- |
| One config, one row | `ranking_schema.json` required cols `family_id,config_id,N_train,PF_train,Total_R_train,WR_train` |
| No degenerate ranking | `check_config_uniqueness` (`f06_rebuild_pipeline.py:246`) + `ranking_schema.degeneracy_rule` |
| No validation columns (train-only) | forbidden `N_val,PF_val,…`; `validate_ranking_schema:341` |
| Single run_id | `single_run_id_only: true` (F06 YAML) |
| `config_id` unique | `F06_PHASE3_CANONICAL_001` is unique by construction |
| New config later → new PR + pre-registration | governance rule; `no_post_result_change: YES` |

Single-config frozen ranking is structurally compatible and does not trip the
degeneracy guard (degeneracy only fails for many configs collapsing to few
result tuples). No blockers.

---

## 4. D3 — Canonical Config

- config_id: `F06_PHASE3_CANONICAL_001`
- params: **NOT PINNED — `D3_OWNER_DECISION_REQUIRED`**
- parameter_hash_required: YES (compute only AFTER params are pinned, before any result)
- no_post_result_change: YES

**Critical finding — no safe defaults exist in the module.**
`default_params()` = `parameter_grid(1)[0]` =
`stratified_sample_combinations(parameter_space(), 1, 42)[0]`
(`keltner_volatility_expansion_simple.py:27-28` →
`common.py:82-111`). This is **one seeded pseudo-random pick** (numpy PCG64,
seed=42) from a Cartesian grid of **13,824** combinations
(2·2·2·3·2·3·16·2·3). It is NOT an owner-curated canonical default. Resolving
it would require (a) executing project code (out of read-only scope) and
(b) laundering a random draw into a "canonical pre-registered config" — exactly
the p-hacking / reproducibility risk D3 exists to prevent. Parameters are
**not invented and not resolved**; the owner must pin literal values.

**Exact parameter domains (read-only facts, owner must choose ONE per row):**

| Param | Domain (from code) | Owner value | Source |
| :--- | :--- | :--- | :--- |
| `ema_base` | {20, 30} | _REQUIRED_ | `keltner…:13` |
| `atr_mult_keltner` | {1.5, 2.0} | _REQUIRED_ | `keltner…:14` |
| `ema_filter` | {100, 200} | _REQUIRED_ | `keltner…:15` |
| `expansion_atr_min` | {1.0, 1.2, 1.5} | _REQUIRED_ | `keltner…:16` |
| `stop_atr` | {1.0, 1.5} | _REQUIRED_ | `keltner…:17` |
| `target_rr` | {1.2, 1.5, 2.0} | _REQUIRED_ | `keltner…:18` |
| `session_name` | one of 16 `SESSION_VARIANTS` keys¹ | _REQUIRED_ | `common.py:71`, `config.py:134-151` |
| `use_h1_context` | {false, true} | _REQUIRED_ | `common.py:72` |
| `break_even_at_r` | {null, 1.0, 1.2} | _REQUIRED_ | `common.py:73` |

¹ `SESSION_VARIANTS` keys: `all_day, london_ny, ny_open, research_08_1630,
light_fixed, pm_11_12, pm_12_1330, pm_1330_16, pm_1630_19, pm_11_1630,
pm_silver_bullet, pm_11_16, pm_11_17, am_08_11, asia_19_03, london_03_07`.
Note the F06 pipeline session contract is NY `07:00–17:00`
(`F06_PHASE3_CLEAN_TRAIN_ONLY.yaml:7-10`); the chosen `session_name` should be
consistent with that institutional window — owner to reconcile.

**Result:** `D3_OWNER_DECISION_REQUIRED`. Canonical config draft cannot be
finalized until the owner pins all 9 literals. `parameter_hash` is computed
**after** pinning and frozen pre-result.

---

## 5. D4 — Cost Policy

**Code facts (read-only, `research_lab/config.py`):**

| Constant | Value | Line |
| :--- | :--- | :--- |
| `DEFAULT_SPREAD_PIPS` | 1.2 | 26 |
| `DEFAULT_SLIPPAGE_PIPS` | 0.2 | 27 |
| `DEFAULT_COMMISSION_ROUNDTURN_USD` | 7.0 | 28 |
| EURUSD `default_spread_pips` | 1.2 | 54 |
| `EngineConfig.assumed_spread_pips` | None → falls back to PAIR_META 1.2 | 183, `engine.py:73-76` |
| `stress_spread_multiplier` | 1.35 | 202 |
| `stress_slippage_multiplier` | 1.6 | 203 |
| `slippage_stop_multiplier` | 1.5 (HARDENED) | 196 |
| `slippage_stop_entry_multiplier` | 1.5 (HARDENED) | 197 |
| `spread_late_session_multiplier` | 3.0 (HARDENED) | 193 |
| `slippage_late_session_multiplier` | 2.0 (HARDENED) | 199 |
| `max_spread_pips` (entry guard) | 3.0 | 185 |

Cost model exists, is conservative/hardened, and applies spread + slippage +
round-turn commission (`engine.py:100-159`, `:827-828`, `:956-957`). The F06
contract requires the 3 components as booleans
(`F06_PHASE3_CLEAN_TRAIN_ONLY.yaml:35-38`) and `cost_report_schema.json`
requires **≥3 named scenarios** each with numeric `spread_pips`,
`slippage_pips`, `commission_round_turn_usd`. **The numeric scenario values are
NOT encoded in code/config — they are an owner decision.**

**RECOMMENDED 3-scenario table (grounded in code constants — requires owner sign-off, nothing silently invented):**

| scenario | spread_policy | slippage_pips | commission_roundturn | purpose | blocks_adapter? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| base | explicit `spread_pips = 1.2` (EURUSD PAIR_META) | 0.2 (`DEFAULT_SLIPPAGE_PIPS`) | $7.0 | realistic floor | NO |
| conservative | explicit `spread_pips = 1.62` (1.2 × 1.35 stress) | **0.5** (owner floor; code-derived 0.2×1.6=0.32 < 0.5) | $7.0 | owner-mandated conservative | NO |
| stress | explicit `spread_pips = 3.0` (capped by `max_spread_pips`; raw 1.2×1.35×… exceeds) | ≥ 0.8 (0.5 × 1.5 stop/entry) | $7.0 | worst-case friction | NO |

**Owner decisions still required (`D4_OWNER_DECISION_REQUIRED`):**
1. **Spread policy:** pin explicit per-scenario `spread_pips` (deterministic /
   audit-friendly, recommended for a frozen canonical config) **vs.**
   `cost_profile=auto` dynamic estimation (session/vol dependent, not a single
   reproducible number).
2. **Conservative slippage floor:** confirm `0.5` pips (the owner-stated
   minimum; it is NOT a code default — base is `0.2`).
3. **Stress definition:** confirm spread cap interaction —
   `max_spread_pips = 3.0` will **reject entries** when modeled spread > 3.0;
   the stress scenario must acknowledge this filtering effect.
4. **Commission:** confirm `$7.0` round-turn (code default) is the canonical value.

No optimistic costs. No `0.0`-slippage-only scenario. Round-turn commission
(engine halves it per leg: `engine.py:827`, `:956`).

---

## 6. D5 — Gross R / SL Pips Resolution

- selected_option (proposed): passive adapter reconstruction, no core change
- reason: engine does not emit `gross_r`/`sl_pips`; core change prohibited now
- schema_change_required: **NO** to F06 ledger schema (it already mandates them)
- core_change_required: **see decision matrix — currently UNRESOLVABLE passively**

**Contract gap (read-only evidence):**

`ledger_schema.json` `required_columns` (const):
`["run_id","family_id","config_id","signal_time","side","gross_r","sl_pips","net_r","month"]`.
Runtime guard `validate_ledger_schema` (`f06_rebuild_pipeline.py:300-329`)
requires these and that `gross_r/net_r/sl_pips` be numeric.

Engine `BacktestResult.trades` columns (`engine.py:960-974`):
`strategy_name, direction, signal_time, signal_price, entry_time, entry_price,
exit_time, exit_price, exit_reason, pnl_usd, pnl_r, lots, session_date`.
→ emits **net** `pnl_r` (= `pnl_usd / risk_usd`; `pnl_usd` already nets
commission+slippage+spread). Does **NOT** emit `gross_r`, `sl_pips`, or a
column literally named `net_r`. `Position` holds `sl`,
`initial_risk_distance`, `entry_commission_usd`, `entry_spread_pips`,
`entry_slippage_pips` (`engine.py:30-52`) but these are **discarded** (Position
set to `None` on close) and never written to the exported trades dict — a
passive adapter at the `BacktestResult` boundary cannot see them.

| required_field | available_source | derivation | auditability | risk | decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `run_id`,`family_id`,`config_id`,`month` | pipeline metadata | injected by runner | HIGH | low | OK |
| `side` | `direction` | rename long/short→side | HIGH | low | OK |
| `net_r` | `pnl_r` | direct (engine pnl_r is net) | HIGH | low | OK |
| `signal_time` | `signal_time` | direct | HIGH | low | OK |
| `sl_pips` | NOT exported | re-join frame @entry, `stop_atr × ATR@entry / pip_size` | MEDIUM-LOW | timestamp→bar off-by-one, feature recompute | OWNER DECISION |
| `gross_r` | NOT exported | `net_r + reconstructed per-trade cost` (cost pips not exported) | LOW | re-implements cost model, p-hacking surface | OWNER DECISION |

**Result:** `D5_PARTIAL_NEEDS_OWNER_DECISION` →
**`BLOCKED_OUTPUT_MAPPING_DECISION_REQUIRED`**. `net_r` and `side` map cleanly;
`gross_r` and `sl_pips` **cannot be reconstructed exactly and audit-safely**
from the engine's current exported trade schema under the hard constraints
(no core change, passive adapter, read-only). The owner must select ONE:

- **Option A (RECOMMENDED) — minimal additive, behavior-neutral core telemetry.**
  Append (no logic/behavior change) `sl`, `initial_risk_distance`,
  `entry_commission_usd`, `exit_commission_usd`, `entry_spread_pips`,
  `entry_slippage_pips` to the engine trades dict. Highest auditability.
  Requires explicitly relaxing D5's "no core change" **for additive-only
  telemetry columns** (no behavioral change, no schema break).
- **Option B — passive frame-rejoin adapter.** Recompute `sl_pips` from
  `stop_atr × ATR@entry` and `gross_r` from `net_r + re-derived per-trade cost`,
  gated by a hard reconciliation invariant (reconstructed net must equal engine
  `pnl_r` within ε, else BLOCK). No core change; higher complexity/audit risk.
- **Option C (NOT RECOMMENDED).** `gross_r := net_r` + provenance flag
  `gross_r_is_proxy=true`; `sl_pips` via frame re-join only. Schema-legal but
  governance-misleading (gross == net defeats the cost audit).

---

## 7. Remaining Owner Inputs

1. **D3:** Pin literal values for all 9 params (table §4) → enables
   `parameter_hash`. Nothing invented.
2. **D4:** Confirm (a) spread policy explicit-vs-auto, (b) conservative
   slippage floor `0.5`, (c) stress/`max_spread_pips=3.0` interaction,
   (d) commission `$7.0` round-turn.
3. **D5:** Choose Option A / B / C for `gross_r` + `sl_pips`. If A, explicitly
   authorize additive-only, behavior-neutral engine telemetry columns.

---

## 8. What This Does NOT Authorize

- NO adapter implementation
- NO real F06 run
- NO backtest
- NO strategy execution
- NO optimization / sweep
- NO validation
- NO holdout
- NO 2025 / NO 2026
- NO core engine modification
- NO F06 certification

---

## 9. Next Step

Owner provides the §7 inputs. Then a follow-up read-only pass produces the
finalized `OWNER_DECISION_D1_D5_FILLED` (all PINNED/RESOLVED) and only then may
a **design-only** adapter-implementation prompt (mocks/tests, still no F06/
backtest/data) be drafted. See
`NEXT_PROMPT_OWNER_DECISION_COMPLETION.md`.

**Final owner statement (to be countersigned):** *I understand this does not
authorize a real F06 run, validation, holdout, 2025/2026, or F06
certification.*
