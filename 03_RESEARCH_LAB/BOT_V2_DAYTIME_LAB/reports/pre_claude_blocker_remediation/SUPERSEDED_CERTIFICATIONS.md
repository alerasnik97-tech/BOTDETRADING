# SUPERSEDED CERTIFICATIONS

**Fecha:** 2026-05-15
**Estado institucional:** SUPERSEDED_BY_CLAUDE_EXTREME_AUDIT
**Base:** `FORENSIC_VERIFICATION_REPORT.md` → BLOCKED_MAJOR_RISK_CONFIRMED

> Ningún archivo previo se borra ni se reescribe. Este documento solo declara
> el **estado institucional vigente**. Los artefactos originales se conservan
> como evidencia del incidente.

---

## 1. Reportes marcados NO CERTIFICADOS / SUPERSEDED

| Artefacto | Estado nuevo | Motivo |
| :--- | :--- | :--- |
| `…/v50b_train_only_full_rerun_20260515_0926/FULL_RERUN_TRAIN_ONLY_REPORT.md` | **SUPERSEDED / NOT CERTIFIED** | Ledger 91.7% contaminado; "output_isolation: SUCCESS" falso; columnas val |
| `…/v50b_train_only_full_rerun_20260515_0926/EVIDENCE_RECONCILIATION_REPORT.md` | **SUPERSEDED / NOT CERTIFIED** | "Reconciliación" no resolvió contaminación física; provenance git errónea |
| `…/cost_hardening_v50b_train_only_20260515_1020/COST_HARDENING_REPORT.md` | **SUPERSEDED / NOT CERTIFIED** | Input cuarentenado; N=10; sin spread |
| `…/v50b_limited_real_gauntlet_rerun_sw/results/V50B_RERUN_MASTER_RANKING.csv` | **INVALID** | Degenerado (150 configs / 6 tuplas) + columnas val |
| `…/v50b_limited_real_gauntlet_rerun_sw/trades/V50B_RERUN_TRADES.csv` | **INVALID** | 91.7% RunIDs cuarentenados |
| `MANIFEST_68fa2280.json` | **INSUFICIENTE** | Stub de 4 líneas; sin hashes/scope/config/row counts |

## 2. Decisiones revocadas

| Decisión previa | Estado nuevo |
| :--- | :--- |
| F06 = COST_ROBUST | **REVOCADA — F06 NOT CERTIFIED** |
| F06 / F08 / F12 = READY_FOR_VAL | **REVOCADA — NOT CERTIFIED** |
| F08 / F12 = "rechazadas por costos" | **RECLASIFICADA** — descartadas, pero por señal de motor/sweep/overfit/leakage roto, no solo por costos |
| CERTIFIED_FOR_TRAIN_RESEARCH_ONLY | **REVOCADA** |
| CONTROLLED_VALIDATION_PLAN (próximo paso) | **BLOQUEADA** |
| Recomendación V50C | **BLOQUEADA** |

## 3. Estado de familias

- **F06:** NOT CERTIFIED. Sin evidencia válida de edge. No avanza.
- **F08:** NOT CERTIFIED. Descartada de línea principal.
- **F12:** NOT CERTIFIED. Descartada de línea principal.

Ninguna familia pasa a watchlist: la evidencia que las evaluó es inválida,
por lo que no hay nada estadísticamente vigilable hasta la reconstrucción.

## 4. Qué NO implica este documento

- NO implica que las estrategias sean malas o buenas — implica que **no hay
  evidencia válida** para afirmar nada sobre ellas.
- NO autoriza re-correr para "arreglar".
- NO autoriza tocar validation / holdout / 2025 / 2026 / V50C.

## 5. Camino de salida

Solo `FULL_EVIDENCE_REBUILD_PLAN.md` (mismo directorio), ejecutado en su
totalidad y re-auditado por Claude, puede producir certificaciones nuevas.
Hasta entonces el estado vigente es **BLOCKED_MAJOR_RISK_PENDING_REMEDIATION**.
