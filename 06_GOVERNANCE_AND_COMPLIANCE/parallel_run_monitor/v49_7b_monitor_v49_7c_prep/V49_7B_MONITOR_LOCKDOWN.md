# V49.7B — MONITOR LOCKDOWN PROTOCOL

## Role: Parallel Governance Watchdog
**Status**: ACTIVE
**Agent ID**: AGENT_PARALLEL_3

## Strict Constraints
- **NO EXECUTION**: Prohibido iniciar o detener procesos de backtest.
- **NO MODIFICATION**: Prohibido tocar runners, engine, core o datos crudos.
- **READ-ONLY DATA**: Acceso a `05_MARKET_DATA_VAULT` restringido a lectura.
- **TEST LOCKDOWN**: Prohibido cualquier acceso a periodos 2025-2026.
- **SOURCE OF TRUTH**: GitHub `clean-sync-branch`. No se permiten ZIPs.

## Objectives
1. Passive monitoring of V49.7B Representative Stability Run.
2. Leakage and Core Drift auditing.
3. Preparation of V49.7C Full Scope Run.

## Lockdown Compliance
- `v7_engine`: UNTOUCHED
- `market_data`: READ-ONLY
- `test_oos`: ABSOLUTELY PROTECTED
