# SLIPPAGE REALISM RISK AUDIT

Status: COMPLETE_STATIC_READ
Decision class: BLOCKS_PAPER / BLOCKS_DEMO_FUNDED

## Scope read
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v7_engine/cost_model.py`
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v6_utils/execution.py`
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v7_engine/tests/test_engine_cost_integration.py`
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v7_engine/tests/test_ftmo_commission_integration.py`

## Findings

### SLIP-001 - Slippage is a scalar R penalty, not execution-path slippage
- Severity: HIGH
- Status: CONFIRMED_ACTIVE
- Evidence: `cost_model.py` computes `slippage_r = slippage_pips / sl_pips` and subtracts it from gross R after exit.
- Impact: it stress-tests metrics, but does not alter entry/exit prices, stop triggering, BE triggering, missed fills, or adverse gap sequence.
- Classification: BLOCKS_PAPER, BLOCKS_DEMO_FUNDED
- Required correction: keep scalar stress as conservative research overlay, but add a path-level slippage/fill model before paper/demo/funded.

### SLIP-002 - Commission is integrated and net metrics are available
- Severity: INFO
- Status: CONFIRMED_ACTIVE
- Evidence: `cost_model.py` calculates commission R/USD; `metrics.py` defaults to `net_r`.
- Impact: net metric reporting is better than prior gross-only backtests.
- Classification: INFO
- Required correction: preserve net ledger discipline.

### SLIP-003 - Rollover/spread stress is documented outside engine
- Severity: MEDIUM
- Status: CONFIRMED_ACTIVE
- Evidence: governance data quality audit flags rollover spread degradation up to extreme pips; engine cost model does not enforce a dynamic no-entry layer by itself.
- Impact: research may know the risk, but operational enforcement is not guaranteed in the core path.
- Classification: BLOCKS_DEMO_FUNDED
- Required correction: make rollover/spread exclusion an enforced gate in execution runners before demo/funded.
