# EURUSD Dual Line - Scoreboard Schema

## Objetivo
Consolidar las métricas críticas de ambas líneas en una única vista comparativa sin mezclar los ledgers de origen.

## Campos del Scoreboard

| Campo | Origen | Descripción |
|-------|--------|-------------|
| `Line` | Static | `SCBI_M5_GLOBAL` o `SCBI_CORE` |
| `Sample_N` | Ledger count | Número de trades registrados en forward |
| `PF_Forward` | Ledger calc | Profit Factor real observado |
| `Exp_Forward` | Ledger calc | Expectativa (media PnL) |
| `Max_DD_R` | Ledger calc | Drawdown máximo en unidades R |
| `Win_Rate` | Ledger calc | Porcentaje de aciertos |
| `Drift_R` | Compare vs Baseline | Diferencia entre Exp Histórica y Exp Forward |
| `News_Compliance` | Exception log | % de trades bloqueados correctamente por noticias |
| `Last_Trade_Date` | Ledger | Fecha de la última actividad |

## Almacenamiento
- **Path**: `results/SCBI_DUAL_LINE_SCOREBOARD.csv`
- **Formato**: CSV plano para auditabilidad simple.

## Proceso de Actualización
El scoreboard se reconstruye desde cero cada vez que se ejecuta el script `build_scbi_dual_line_scoreboard.py`, leyendo los ledgers namespaced.
