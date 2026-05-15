# PRE-CLAUDE FREEZE NOTICE

**Status:** BLOCKED_MAJOR_RISK_PENDING_REMEDIATION
**Fecha:** 2026-05-15
**Emitido por:** Institutional Integrity Remediation (pre-Claude nightly audit)
**Disparador:** CLAUDE EXTREME AUDIT — veredicto BLOCKED_MAJOR_RISK sobre evidencia V50B / F06

---

## 1. Reglas de Freeze (vinculantes)

- F06 **NO certificada**.
- F08 **NO certificada**.
- F12 **NO certificada**.
- Cost Hardening anterior queda **SUPERSEDED / NOT CERTIFIED**.
- **NO** V50C.
- **NO** validation.
- **NO** 2025.
- **NO** 2026.
- **NO** holdout.
- **NO** demo.
- **NO** FTMO.
- **NO** real.
- **NO** nuevas familias.
- **NO** sweeps.
- **NO** optimización.

## 2. Razón

La auditoría extrema detectó evidencia inválida o no suficientemente reproducible:

- El ledger usado para certificar F06 (`V50B_RERUN_TRADES.csv`) contiene RunIDs
  contaminados (`1fa40f18` ABORTED, `129e106b` PREFLIGHT) → la contaminación
  Multi-RunID **no** quedó remediada.
- El RunID canónico `68fa2280` aporta una muestra trivial (~N=10) al Cost
  Hardening de F06.
- `V50B_RERUN_MASTER_RANKING.csv` es degenerado (configs duplicadas, pocas
  tuplas únicas) y contiene columnas de validación pobladas pese a las
  atestaciones `validation_touched: NO`.
- El script generador declarado (`v50b_limited_rerun_ultra.py`) no está
  presente/trackeado → reproducibilidad rota.
- La fuente de verdad reside dentro de un directorio marcado
  `QUARANTINED_DO_NOT_USE.md` nunca levantado.

Detalle forense completo: `FORENSIC_VERIFICATION_REPORT.md` (mismo directorio).

## 3. Estado de la Evidencia V50B

| Artefacto | Estado |
| :--- | :--- |
| `FULL_RERUN_TRAIN_ONLY_REPORT.md` | SUPERSEDED — no usar para promoción |
| `EVIDENCE_RECONCILIATION_REPORT.md` | SUPERSEDED — no usar para promoción |
| `COST_HARDENING_REPORT.md` | SUPERSEDED / NOT CERTIFIED |
| `V50B_RERUN_MASTER_RANKING.csv` | INVALID — degenerado + columnas val |
| `V50B_RERUN_TRADES.csv` | INVALID — multi-RunID contaminado |
| Decisión "F06 COST_ROBUST" | INVALID |
| Decisión "READY_FOR_VAL" (F06/F08/F12) | INVALID |
| Decisión "CONTROLLED_VALIDATION_PLAN" | BLOQUEADA |
| Recomendación V50C | BLOQUEADA |

## 4. Qué SÍ está permitido bajo este freeze

- Verificación forense **read-only** de la evidencia existente.
- Marcado institucional de certificaciones incorrectas (supersede).
- Redacción de un plan de reconstrucción de evidencia (sin ejecutarlo).
- Publicación de un audit pack liviano en GitHub.

## 5. Levantamiento del Freeze

Este freeze solo puede levantarse cuando exista evidencia reconstruida que
cumpla `FULL_EVIDENCE_REBUILD_PLAN.md` y supere una nueva auditoría Claude.
No se levanta por re-ejecución correctiva ni por re-interpretación de la
evidencia actual.

---
**No defender resultados anteriores. La acción profesional correcta hoy es
bloquear, congelar y dejar el camino limpio para reconstruir evidencia real.**
