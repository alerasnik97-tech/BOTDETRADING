# MULTI-AGENT CONTROL PLANE

## Philosophy
Each agent operates within its designated folder. No cross-contamination.
Research reads data but never writes to production.
Production only receives approved releases.
Infrastructure observes but never decides trades.

## Permitted Data Flow
- Research → reads 05_MARKET_DATA, reads 06_GOVERNANCE
- Incubation → clones from 01_CORE (approved only)
- Production → receives from Incubation (validated only)
- Infrastructure → reads all, writes 04 only
- Governance → writes 06, reads all

## Prohibited Flows
- Research → Production (must go through Incubation)
- Infrastructure → Strategy logic
- Any agent → delete data/backups
- Any agent → push without user approval

## Agent Roles
1. Research Agent: backtests, gates, motor development
2. Engine/Testing Agent: test suite, cost model validation
3. Infrastructure Agent: packaging, dashboards, maintenance
4. Governance Agent: rules, protocols, architecture
5. Data Quality Agent: data integrity, calendars
6. Production Release Agent: frozen releases only
7. Incubation Agent: forward/paper testing
8. ChatGPT/Claude Audit Agent: external audit, read-only

## Git Policy
- Each agent uses its own branch
- Merge to main requires: tests pass + manifest + audit
- No push without explicit user authorization
