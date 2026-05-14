# R1 REJECTION EVIDENCE INDEX

**Resumen de Evidencia Acumulada contra R1**

| Fase | Decisin | Hallazgo Principal |
| :--- | :--- | :--- |
| **V49.5** | BLOCKED | Bug crtico de inyeccin de parǭmetros detectado. |
| **V49.6** | PASSED | Saneamiento de parǭmetros y deduplicacin de configs. |
| **V49.7B** | CRASHED | Descubrimiento de saturacin de RAM/OOM por falta de batching. |
| **V49.7B-CR** | PASSED (Tech) | Estabilidad de memoria lograda via Batching 50. |
| **V49.7B-R2** | PASSED (Full) | Cobertura real de VAL 2023-2024 lograda. |
| **V49.7B-R2B** | **REJECTED** | 0/800 configs pasan gate combined. TRAIN perdedor (PF 0.65). |

## Evidencia NumǸrica Irrefutable
- **Configs Evaluadas**: 800
- **Trades Auditados**: 74,994
- **Pass Rate (Gate A)**: **0.00%**
- **Concentracin VAL**: > 80% en meses aislados para los "mejores" perdedores.

**Estado**: ARCHIVED AS FAILURE.
