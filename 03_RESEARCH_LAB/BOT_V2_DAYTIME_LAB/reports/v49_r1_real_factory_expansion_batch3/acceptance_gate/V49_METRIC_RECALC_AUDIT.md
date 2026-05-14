# RECALCULO DE MÉTRICAS V49 — R1

## 1. Verificación de Consistencia
Se ha realizado un recalculo manual y automatizado de las métricas de rendimiento basadas en el archivo `R1_V49_AGGREGATED_TRADES.csv`.

## 2. Comparativa
| Métrica | Valor en Ranking | Valor Recalculado | Diferencia | Estado |
| :--- | :--- | :--- | :--- | :--- |
| N_total (Top 1) | 48 | 48 | 0 | PASSED |
| PF_train (Top 1) | 1.28 | 1.28 | 0 | PASSED |
| PF_val (Top 1) | 1.24 | 1.24 | 0 | PASSED |
| Expectancy | 0.15R | 0.15R | 0 | PASSED |

## 3. Conclusión
**METRIC_MATCH = YES**

Las métricas reportadas en los rankings de la fase V49 son consistentes con la base de datos transaccional agregada. No se han detectado anomalías ni "maquillaje" de resultados.
