# NEXT PROMPT — OWNER DECISION COMPLETION (D3 / D4 / D5)

Owner decision D1-D5 validation is **PARTIAL**. D1 validated (with warnings),
D2 validated. **D3 + D4 need explicit owner inputs; D5 is blocked pending an
explicit owner output-mapping decision.** Use this prompt only to **collect and
record the owner's literal answers** — still NO adapter, NO F06, NO backtest,
NO strategy run, NO validation/holdout/2025/2026, NO core change.

Authoritative context:
- `OWNER_DECISION_D1_D5_FILLED.md` (status, evidence, decision matrices)
- `OWNER_DECISION_VALIDATION_REPORT.md` (blocking issues B1-B4)

---

## What the owner must answer (verbatim, no invention by the agent)

### D3 — pin the 9 canonical parameters (config_id `F06_PHASE3_CANONICAL_001`)

Choose exactly ONE value per row (domains are code facts; do not invent):

| Param | Domain | Owner value |
| :--- | :--- | :--- |
| `ema_base` | 20 \| 30 | … |
| `atr_mult_keltner` | 1.5 \| 2.0 | … |
| `ema_filter` | 100 \| 200 | … |
| `expansion_atr_min` | 1.0 \| 1.2 \| 1.5 | … |
| `stop_atr` | 1.0 \| 1.5 | … |
| `target_rr` | 1.2 \| 1.5 \| 2.0 | … |
| `session_name` | one of 16 `SESSION_VARIANTS`¹ | … |
| `use_h1_context` | false \| true | … |
| `break_even_at_r` | null \| 1.0 \| 1.2 | … |

¹ `all_day, london_ny, ny_open, research_08_1630, light_fixed, pm_11_12,
pm_12_1330, pm_1330_16, pm_1630_19, pm_11_1630, pm_silver_bullet, pm_11_16,
pm_11_17, am_08_11, asia_19_03, london_03_07`. Reconcile with the F06 session
contract NY `07:00–17:00`.

`parameter_hash` is computed **after** these are pinned and frozen before any
result. `no_post_result_change = YES`.

### D4 — pin the cost policy

1. Spread policy: **explicit per-scenario `spread_pips`** (recommended) **or**
   `cost_profile=auto`? → …
2. Confirm 3 scenarios (numbers may follow the recommended table or be
   overridden by the owner — not by the agent):
   - base: spread_pips=… slippage_pips=… commission_round_turn_usd=…
   - conservative: spread_pips=… slippage_pips=… (≥ 0.5) commission_round_turn_usd=…
   - stress: spread_pips=… slippage_pips=… commission_round_turn_usd=…
3. Acknowledge `max_spread_pips = 3.0` rejects entries when modeled spread > 3.0
   (affects the stress scenario). → acknowledged? …
4. Commission round-turn USD canonical value (code default 7.0). → …

### D5 — choose the gross_r / sl_pips resolution

Pick ONE:
- [ ] **Option A (recommended)** — authorize a minimal, additive,
  behavior-neutral core telemetry change: append `sl`,
  `initial_risk_distance`, `entry_commission_usd`, `exit_commission_usd`,
  `entry_spread_pips`, `entry_slippage_pips` to the engine trades dict. No
  logic/behavior change. (Explicitly relaxes D5 "no core change" for
  additive-only telemetry.)
- [ ] **Option B** — authorize a passive frame-rejoin adapter computing
  `sl_pips = stop_atr × ATR@entry / pip_size` and
  `gross_r = net_r + reconstructed per-trade cost`, gated by a hard
  reconciliation invariant (reconstructed net == engine `pnl_r` within ε, else
  BLOCK). No core change.
- [ ] **Option C (not recommended)** — `gross_r := net_r` with
  `gross_r_is_proxy=true`; `sl_pips` via frame re-join only.

Owner statement (countersign): *I understand this does not authorize a real
F06 run, validation, holdout, 2025/2026, or F06 certification.*

---

## Agent instructions for the completion run (read-only + docs only)

1. Precheck (branch `research/f06-clean-train-only-rerun-20260515`, head
   `a4a0dca…`, in sync; no python backtest/sweep processes; abort →
   `SCOPE_ESCALATION_BLOCKED` on any execution attempt).
2. Transcribe the owner's literal D3/D4/D5 answers into
   `OWNER_DECISION_D1_D5_FILLED.md` (PINNED/RESOLVED). Do **not** invent or
   resolve seeds. If any answer is missing/ambiguous → keep
   `OWNER_DECISION_REQUIRED` for that item.
3. If (and only if) D3 PINNED + D4 PINNED + D5 RESOLVED: compute the
   `parameter_hash` over the pinned param dict (deterministic hash of literals
   only — no strategy/engine execution), record it, and draft
   `NEXT_PROMPT_SAFE_ENGINE_ADAPTER_IMPLEMENTATION_FROM_OWNER_DECISION.md`
   (design + mocks/tests only; still NO F06, NO backtest, NO data, NO
   validation/holdout/2025/2026; if D5=Option A, the only code change permitted
   later is the additive, behavior-neutral telemetry columns under its own PR
   with regression tests proving identical pnl).
4. Commit/push **docs only** to the branch. Comment PR #6. NO force push, NO
   merge, NO ready, NO certification.

## Hard prohibitions (unchanged)

NO adapter implementation · NO F06 run · NO backtest · NO strategy execution ·
NO optimization/sweep · NO validation · NO holdout · NO 2025 · NO 2026 ·
NO core change (except, only after D5=Option A is signed, additive
behavior-neutral telemetry under its own PR) · NO invented parameters ·
NO quarantine/old outputs as source of truth · NO main · NO force push ·
NO merge · NO PR ready · NO certification.
