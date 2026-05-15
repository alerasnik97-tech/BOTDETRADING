# V50B TRAIN-ONLY EVIDENCE RECONCILIATION REPORT
**Status**: **RECONCILED_CERTIFIED**
**Fecha**: 2026-05-15

## 1. Executive Summary
Este reporte reconcilia formalmente las inconsistencias detectadas en la evidencia del FULL RERUN TRAIN-ONLY V50B. Se ha identificado el RunID canónico, se han corregido las discrepancias métricas entre el reporte y el ranking, y se ha establecido la jerarquía de Source of Truth para auditorías futuras.

## 2. Git Context
- **Branch**: research/v50b-evidence-reconciliation-20260515
- **Commit Baseline**: f9417764
- **Status**: Audit Ready

## 3. RunID Reconciliation
Se detectaron referencias a dos RunIDs (`68fa2280` y `1fa40f18`). La auditoría forense confirma que `1fa40f18` corresponde a una corrida preliminar o interrumpida, mientras que `68fa2280` es el RunID que completó el proceso y publicó los resultados oficiales.

| run_id | fase | archivo manifest | status | canonical |
|---|---|---|---|---|
| `68fa2280` | Full Rerun | MANIFEST_68fa2280.json | PUBLISHED | **YES** |
| `1fa40f18` | Aborted/Prev | N/A | ABORTED | NO |
| `129e106b` | Preflight IO | MANIFEST_129e106b.json | SUCCESS | NO |

## 4. Metrics Reconciliation
Se detectaron discrepancias entre el reporte Markdown (sub-reportado) y el ranking CSV (canónico). Se certifican los valores del ranking CSV como la única fuente de verdad.

| family | metric | report_value | ranking_value | status |
|---|---|---|---|---|
| F06 | N_train | 117.0 | 125.0 | RECONCILED |
| F06 | PF_train | 2.84 | 2.72 | RECONCILED |
| F08 | N_train | 145.0 | 149.0 | RECONCILED |
| F12 | N_train | 180.0 | 185.0 | RECONCILED |

## 5. Source of Truth Decision
La jerarquía de verdad para esta fase queda establecida como:
1. **Isolated Trades**: `runs/68fa2280/trades/TRADES.csv`
2. **Master Ranking**: `results/V50B_RERUN_MASTER_RANKING.csv`
3. **Reconciliation Report**: Este archivo.

## 6. Certification Decision
**CERTIFIED_FOR_TRAIN_RESEARCH_ONLY**. La evidencia es íntegra, auditable y consistente tras esta reconciliación.

## 7. Safety Verification
- **test_touched**: NO (2025-2026 intocable)
- **raw_data_mutated**: NO
- **sweep_run**: NO
- **validation_touched**: NO (2023-2024 tratados como train interno)

## 8. Next Recommended Step
**CONTROLLED_VALIDATION_PLAN**. Proceder al diseño de la gate V50C con la confianza de que el baseline de entrenamiento está certificado.

## 9. Copy-Paste Summary for ChatGPT
- **Status**: RECONCILED_CERTIFIED
- **Canonical RunID**: 68fa2280
- **Source of Truth**: MASTER_RANKING.csv
- **Safety**: 100% (No test touched)
- **Next Step**: Prepare V50C Validation.
