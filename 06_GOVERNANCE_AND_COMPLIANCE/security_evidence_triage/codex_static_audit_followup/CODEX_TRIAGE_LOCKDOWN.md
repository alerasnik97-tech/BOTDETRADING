# Codex Triage Lockdown - V50.1
Fecha: 2026-05-14

## Estado de Bloqueo
- **READ-ONLY**: No se modifican archivos de código (`src/`) ni datos (`05_MARKET_DATA_VAULT`).
- **NO EXECUTION**: No se corren backtests ni sweeps.
- **SECURITY FIRST**: No se imprimen secretos completos. Enmascaramiento obligatorio.
- **BRANCH LOCK**: Solo se permiten commits en `clean-sync-branch` dentro de `06_GOVERNANCE_AND_COMPLIANCE`.

## Prohibiciones Activas
- NO tocar `v50b` outputs (fase activa).
- NO tocar TEST (2025-2026).
- NO borrar evidencia, incluso si es contradictoria.
- NO reescribir historial Git.

## Mandato
Realizar un triage estático de seguridad y evidencia para informar a la fase V50B sobre riesgos estructurales o filtraciones de credenciales.
