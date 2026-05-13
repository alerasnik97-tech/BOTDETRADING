# PARADIGM RECONCILIATION AUDIT

## Live Modules (Gate 6 Mini)
- **Engine**: `v7_engine/engine.py` (UnifiedV7Engine) — canonical
- **Signal Gen**: `phase18_h1_fractal_sweep.py` + `phase18_first_3m_choch.py`
- **Bar Builder**: `v6_utils/bars.py`
- **Cost Model**: `v7_engine/cost_model.py` (FTMO mode, round-turn)
- **Metrics**: `v7_engine/metrics.py` (net_r based)
- **Execution**: `v6_utils/execution.py` (next_bar_execute)

## Deprecated (NOT used in Gate 6 Mini)
- `sweep_direct.py` — MOCK DATA, replaced by gate6_mini_runner
- `phase14_signals.py` — superseded by phase18
- `phase19_repaired_engine.py` — superseded by V7
- `phase27-32` scripts — legacy analysis, pre-V7

## MANIPULANTE 2.0 Days 1-3 Status
Days 1-3 produced the V7 engine architecture. The signal generation remains from phase18.
ANCHOR_CONFIG was superseded by V7 engine constructor params.

## Column Mapping Note
phase18 modules expect: `high_bid`, `low_bid`, `close_bid`, `timestamp_ny`
bars.py produces: `high`, `low`, `close` with UTC index
Gate 6 Mini runner must rename + convert timezone.
