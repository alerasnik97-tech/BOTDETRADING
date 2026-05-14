# R1 V49.7B-R2 ?" METRIC RECALC AUDIT

**Objetivo**: Certificar que las mǸtricas del ranking coinciden con la evidencia fscia de trades.

## Auditora Forense
- **Muestra**: Top 1 (V49_7B_0001).
- **Recalculo**: 
  - PF_train = sum(win) / sum(loss) en fase TRAIN.
  - PF_val = sum(win) / sum(loss) en fase VAL.
- **Resultado**: MATCH. Las mǸtricas son 100% recalculables desde `R1_V49_7B_R2_TRADES.csv`.

**Veredicto**: METRIC_MATCH_OK.
