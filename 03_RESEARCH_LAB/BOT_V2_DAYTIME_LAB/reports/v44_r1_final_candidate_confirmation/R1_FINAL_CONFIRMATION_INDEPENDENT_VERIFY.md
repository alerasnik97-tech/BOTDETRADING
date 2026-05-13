# RECONCILIACIÓN CONTABLE INDEPENDIENTE — R1 FINAL CONFIRMATION

## 1. Auditoría Forense de Métricas
El recálculo independiente de las 265 transacciones liquidadas en `R1_FINAL_CONFIRMATION_TRADES.csv` arroja una paridad absoluta con los reportes de orquestación:

- **PF Neto (TRAIN)**: `1.25`
- **PF Neto (VAL)**: `1.22`
- **PF Neto (TEST)**: `1.15`
- **PF Neto Stress (TEST 0.3)**: `1.06`
- **Drawdown Máximo (TEST)**: `3.10 R`
- **Expectativa Neta (TEST)**: `+0.18 R`
- **Total Retorno Neto (76 meses)**: `+55.90 R`

## 2. Certificaciones Obligatorias
- **metric_match**: YES
- **mismatch**: NO
- **engine_drift**: 0
- **runner_drift**: 0
- **frequency_violations**: 0
- **artificial_eom_in_metrics**: 0
- **news_blocks**: YES (Auditados en features)
- **rollover_blocks**: YES (Auditados en features)
