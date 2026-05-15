# NEXT PROMPT — F06 GOVERNANCE OWNER DECISION (DECISION REQUEST, NOT EXECUTION)

This document requests explicit decisions from the PROJECT OWNER. It authorizes nothing. No adapter
implementation prompt exists yet — CRITICAL blockers remain (see
`F06_GOVERNANCE_RESOLUTION_REPORT.md` and `F06_DISCOVERY_CONTINUATION_REPORT.md`).

> Implementing the adapter ≠ running F06 real. Both remain FORBIDDEN until every decision below is
> made, frozen, and re-audited by Claude.

## Why this is needed
"F06" has NO safe code definition: `STRATEGY_REGISTRY` (60+ strategies) has no F06; F06/F08/F12 are
NOT-CERTIFIED V50B-era family labels whose evidence was invalidated. Reconstructing F06 from V50B is
forbidden. The output contract also requires ledger fields (`gross_r`, `sl_pips`) the engine does
not emit. These are governance decisions, not code.

## DECISIONS REQUIRED FROM OWNER (each must be explicit, written, and frozen)

### D1 — F06 strategy identity
Choose EXACTLY ONE tracked `STRATEGY_REGISTRY` `NAME` that IS F06 for Phase 3.
- Owner answer: `F06_STRATEGY_NAME = "<one registry NAME>"`
- Constraint: must be a tracked `research_lab/strategies/*` module; NOT reconstructed from V50B.

### D2 — Ranking model
Pick one:
- (Recommended) `SINGLE_CONFIG`: one `config_id = F06_PHASE3_CANONICAL_001`; RANKING.csv is a
  one-row summary (no sweep risk).
- `FROZEN_MULTI_CONFIG`: an owner-supplied, pre-registered, hash-stamped immutable config list
  created BEFORE any run, no tuning during/after. (Heavier; defer unless explicitly needed.)

### D3 — Canonical F06 params
Provide the exact, frozen `params` dict for `run_backtest` (including the session window that yields
NY 07:00–17:00) and risk params consistent with Phase 3. This becomes `CONFIG_USED.yaml` +
`parameter_hash` BEFORE any run.

### D4 — Cost model method
Pick one and freeze: explicit `assumed_spread_pips = <value>` OR an audited `cost_profile =
<base|precision|...>`. Also provide the ≥3 cost `scenarios` (name/spread_pips/slippage_pips/
commission_round_turn_usd) required by `cost_report_schema.json`.

### D5 — gross_r / sl_pips resolution (engine does not emit them)
The ledger schema REQUIRES `gross_r` and `sl_pips`; the engine trade record emits neither (core
modification is FORBIDDEN). Choose one:
- (a) Owner-approved adapter-side derivation rule (define exactly: e.g. `sl_pips` from
  `Position.initial_risk_distance`/pip_size; `gross_r` from cost-free reconstruction) — must be
  precisely specified and audited; OR
- (b) A contract/schema amendment (separate, audited change to `ledger_schema.json` +
  `validate_output_dir`), which is its own gated task; OR
- (c) Keep BLOCKED.

## RULES (unchanged)
NO adapter implementation. NO `adapters/`. NO engine/F06/backtest/strategy/optimization/sweep.
NO validation/holdout/2025/2026. NO main.py WFA path. NO V50B/quarantine/backup as source of truth.
NO merge/force-push/ready. NO F06 certification. Scope escalation ⇒ `SCOPE_ESCALATION_BLOCKED`.
2025/2026 touch attempt ⇒ `BLOCKED_TEST_LEAKAGE_RISK`.

## GATE / NEXT STEP AFTER OWNER ANSWERS
1. Record D1–D5 verbatim in a new `F06_GOVERNANCE_DECISIONS.md` (owner sign-off line).
2. Run a READ-ONLY validation pass confirming D1 NAME exists & is tracked, D3 params produce the
   NY window, D4 satisfies the cost schema, D5 rule is schema-consistent.
3. ONLY if zero CRITICAL blockers remain → author a SEPARATE
   `NEXT_PROMPT_SAFE_ENGINE_ADAPTER_IMPLEMENTATION.md`.
4. Implementation (code + mocked tests, NO real run) is still a further gate; real F06 run is a
   later, separate gate again.

## FINAL FORMAT (for the decision-recording task)
1. STATUS  2. D1..D5 (resolved/unresolved + verbatim answers)  3. REMAINING_BLOCKERS
4. SAFETY (no engine/F06/backtest/validation/holdout/2025/2026)  5. DECISION
(F06_GOVERNANCE_PINNED / NEEDS_OWNER_DECISION / BLOCKED)  6. NEXT_STEP
