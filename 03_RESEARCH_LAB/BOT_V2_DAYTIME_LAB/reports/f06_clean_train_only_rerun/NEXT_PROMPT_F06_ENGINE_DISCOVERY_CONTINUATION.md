# NEXT PROMPT — F06 DISCOVERY CONTINUATION (GOVERNANCE + READ-ONLY, DO NOT RUN NOW)

This is documentation only. It authorizes nothing now. An adapter IMPLEMENTATION prompt is
deliberately NOT created because CRITICAL blockers remain (see
`F06_ENGINE_DISCOVERY_READ_ONLY_REPORT.md`). Continuation is **governance decision + read-only**;
NO adapter code, NO engine execution.

> Implementing the adapter is still forbidden. Running F06 real is still forbidden.
> This continuation only resolves definitions and pins remaining ambiguities.

---

## PROMPT START

Actuá como Claude Code Opus 4.7 Max en modo arquitecto institucional extremo, especialista en
governance de laboratorios cuantitativos, definición de familias de estrategias, prevención de
leakage y de optimización encubierta.

OBJETIVO
Resolver (a nivel governance + descubrimiento read-only) los blockers CRÍTICOS que impiden diseñar
e implementar el safe engine adapter. NO implementar adapter. NO ejecutar engine/F06/backtest.

PROHIBICIONES ABSOLUTAS
- NO crear código adapter, NO crear `adapters/`.
- NO ejecutar `run_backtest`, engine, F06, backtest, estrategia, optimization, sweep.
- NO tocar validation/holdout/2025/2026, raw/tick/parquet, `main.py` WFA path.
- NO usar `validation.py`/`wfa.py`/`*_BACKUP_*`/`reports/canonical_*`/quarantine/v50b/ZIP.
- NO merge, NO force push, NO ready conversion, NO certificar F06.
- Si algo intenta ejecutar engine/backtest/F06: ABORT `SCOPE_ESCALATION_BLOCKED`.
- Si algo intenta tocar 2025/2026: ABORT `BLOCKED_TEST_LEAKAGE_RISK`.

MANDATORY GOVERNANCE DECISIONS TO OBTAIN (from the project owner — record verbatim)
1. **F06 definition**: exactly which `research_lab/strategies/*` module(s) (by `NAME` in
   `STRATEGY_REGISTRY`) constitute family "F06". A single module or a fixed named set.
2. **F06 config taxonomy**: the EXACT, FIXED list of `config_id`s (e.g. CFG_001…) with their
   concrete params, plus the meaning of `parameter_hash` and `result_signature`. The list must be
   pre-registered and immutable for the run (no search).
3. **Ranking-vs-sweep ruling**: a written decision on how a multi-`config_id` `RANKING.csv` is
   produced WITHOUT constituting a forbidden parameter sweep/optimization. Acceptable only if it is
   a fixed, pre-registered, audited config set evaluated once (no selection/tuning). Otherwise the
   output contract must change. This decision is BLOCKING.
4. **Engine→ledger transform**: exact derivation of `gross_r, sl_pips, net_r, side, signal_time,
   month, run_id, family_id, config_id` from engine trade rows (`pnl_r, pnl_usd, session_date, …`).

READ-ONLY DISCOVERY TO PIN (no execution)
- Locate the canonical producer of `atr14` and `range_atr` for the frame.
- Pin the explicit pre-load train-month filter + 2025/2026 hard block (adapter-side; loader has none;
  `main.py` defaults `--end 2025-12-31`).
- Confirm EngineConfig enforcement plan: `max_trades_per_day=3` (override default 2),
  `assumed_spread_pips`/`cost_profile` explicit, NY 07:00–17:00 via `params`+`session_cutoff`,
  pinned `execution_mode/intrabar_policy/price_source/seed`.
- Reconcile the adapter spec's manifest to `_MANIFEST_REQUIRED` (29 fields incl. `generator_pid,
  script_path, script_is_tracked`), `_MANIFEST_CONST`, `_STATUS_ENUM`, and artifact-declaration keys.

DELIVERABLE
Update/append docs under `reports/f06_clean_train_only_rerun/`:
- `F06_GOVERNANCE_DECISIONS.md` (decisions 1–4 recorded verbatim, with owner sign-off line)
- updated `SAFE_ENGINE_ADAPTER_DESIGN_SPEC.md` ONLY IF all 4 decisions are obtained
- a fresh `F06_ENGINE_DISCOVERY_READ_ONLY_REPORT.md` revision with status
  `F06_ENGINE_DISCOVERY_COMPLETE_READY_FOR_ADAPTER_IMPLEMENTATION` ONLY IF zero CRITICAL blockers remain.

GATE
- If any of decisions 1–4 is unresolved → status stays `PARTIAL_NEEDS_FIX`; do NOT create an
  implementation prompt.
- Only when ALL blockers are closed may a SEPARATE `NEXT_PROMPT_SAFE_ENGINE_ADAPTER_IMPLEMENTATION.md`
  be authored — and even then implementation ≠ real run (real run is a further separate gate).

FINAL FORMAT
1. STATUS  2. GOVERNANCE_DECISIONS (1–4: resolved/unresolved)  3. REMAINING_BLOCKERS
4. SAFETY (no engine/F06/backtest/validation/holdout/2025/2026)  5. DECISION
(F06_ENGINE_DISCOVERY_COMPLETE_READY_FOR_ADAPTER_IMPLEMENTATION / PARTIAL_NEEDS_FIX / BLOCKED)
6. NEXT_STEP

## PROMPT END
