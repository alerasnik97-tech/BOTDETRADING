# REMEDIATION PRIORITY BOARD

Status: COMPLETE

## P0 - Security stop
1. Revoke the exposed Telegram bot token in BotFather immediately.
2. After revocation, decide whether to mask/remove tracked reports and whether to rewrite history. Do not rewrite history without explicit authorization.
3. Install or enforce an automatic content scanner for staged files, including `.md`, `.json`, `.csv`, `.bak`, and reports.

## P0 - Evidence stop
4. Invalidate V50B generated `V50B_MASTER_RANKING.csv`, `V50B_TRADES_ALL.csv`, `V50B_FAMILY_SCOREBOARD.csv`, and `V50B_DECISION.md` as accepted evidence because the source scripts are synthetic/dummy.
5. Block V50C, TEST, paper, demo, and funded claims until V50B is rebuilt from real engine outputs.

## P1 - Research hardening
6. Replace or quarantine active/legacy mock surfaces that can generate official-looking reports.
7. Repair timestamp conversion and legacy loader path before any paper/demo path.
8. Add vault-local manifest/schema or update README to point to the governance source of truth.

## P2 - Operational hardening
9. Add intratrade FTMO floating drawdown checks.
10. Add path-level slippage/fill modeling and enforce rollover/spread no-entry gates.
11. Strengthen multi-agent enforcement beyond local docs/hooks.

## Explicit non-authorizations
- TEST 2025-2026: NOT AUTHORIZED
- Paper/demo/funded: NOT AUTHORIZED
- ZIP creation: NOT AUTHORIZED in this triage
