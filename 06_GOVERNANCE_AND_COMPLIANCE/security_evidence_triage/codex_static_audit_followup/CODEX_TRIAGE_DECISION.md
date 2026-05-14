# CODEX TRIAGE DECISION

Final state: CODEX_TRIAGE_SECURITY_CRITICAL

Secondary state: CODEX_TRIAGE_ACTIVE_BLOCKERS_FOUND

## Basis
1. A tracked Telegram bot token is present in the current tree and historical backup report. This triggers `CODEX_TRIAGE_SECURITY_CRITICAL`.
2. Current V50B evidence is not valid research evidence: scripts generate synthetic/dummy trades/rankings and the runner logs month completion without real backtest execution. This triggers `CODEX_TRIAGE_ACTIVE_BLOCKERS_FOUND`.
3. The ZIP contradiction is now legacy because GitHub source-of-truth policy deprecates ZIP workflow. Status: `LEGACY_ZIP_WORKFLOW_DEPRECATED`.
4. R1/R2 findings are superseded by later R1 freeze and V50 commits. Status: `SUPERSEDED_BY_LATER_COMMIT`.
5. Data/news manifests are partial: governance audits exist, but vault-local files promised by README are absent. This does not by itself block early research, but it blocks paper/demo/funded.

## Authorizations
- V50B accepted evidence: NOT AUTHORIZED
- TEST 2025-2026: NOT AUTHORIZED
- Paper/demo/funded/real: NOT AUTHORIZED
- ZIP creation: NOT AUTHORIZED

## Safe continuation boundary
Only security cleanup, evidence invalidation, and rebuilding research from real engine outputs are allowed. No TEST, no paper, no demo, no funded, no live.
