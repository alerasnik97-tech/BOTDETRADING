# FTMO FLOATING DRAWDOWN RISK AUDIT

Status: COMPLETE_STATIC_READ
Decision class: BLOCKS_PAPER / BLOCKS_DEMO_FUNDED

## Scope read
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v7_engine/ftmo_compliance.py`
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v7_engine/engine.py`
- selected FTMO and engine cost tests

## Findings

### FTMO-001 - FTMO module supports floating PnL but engine does not feed intratrade path
- Severity: HIGH
- Status: CONFIRMED_ACTIVE
- Evidence: `ftmo_compliance.py` evaluates `current_balance + floating_pnl`; `engine.py` calls `update_state(... floating_pnl=0.0)` at signal and after close.
- Impact: closed-trade equity is net, but intratrade MAE/floating drawdown is not evaluated against daily/absolute FTMO limits.
- Classification: BLOCKS_PAPER, BLOCKS_DEMO_FUNDED
- Required correction: add path-level floating equity checks during open trades using bid/ask tick path and NY daily reset logic.

### FTMO-002 - Net PnL is integrated after cost model
- Severity: INFO
- Status: CONFIRMED_ACTIVE
- Evidence: `engine.py` applies `CostModel.apply_costs_to_trade` before computing `net_pnl_usd` and FTMO closed PnL update.
- Impact: closed ledger is net-aware; this is not the weak point.
- Classification: INFO
- Required correction: preserve this behavior while adding floating DD.

### FTMO-003 - Simplified FTMO rule model is not enough for funded/demo authorization
- Severity: HIGH
- Status: CONFIRMED_ACTIVE
- Evidence: implementation covers daily/absolute equity thresholds and minimum trading days, but not broker/server-time operational nuances, order rejection, partial fill, latency, or platform-specific equity behavior.
- Impact: adequate for research gating only; insufficient for paper/demo/funded.
- Classification: BLOCKS_DEMO_FUNDED
- Required correction: create a separate FTMO operational simulator before any funded claim.
