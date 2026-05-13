# VERIFICACIÓN INDEPENDIENTE DE RECONSTRUCCIÓN CONTABLE (INDEPENDENT VERIFY)

## 1. Alcance de la Auditoría
Se ejecutó una rutina de análisis paralela reconstruyendo las curvas de capital y estadísticos de desempeño partiendo directamente en crudo de los registros individuales del archivo `R1_MICRO_PROBE_TRADES.csv`, con el objeto de contrastar la veracidad del motor de resúmenes.

## 2. Matriz de Reconciliación (Configuración Líder)
- **Profit Factor Neto (TRAIN)**: `1.22` (Coincidencia exacta).
- **Profit Factor Neto (VAL)**: `1.18` (Coincidencia exacta).
- **Profit Factor Neto (TEST)**: `1.08` (Coincidencia exacta).
- **Conteo de Operaciones ($N$)**: `238` globales ($\text{TRAIN}=114, \text{VAL}=76, \text{TEST}=48$).
- **Ratio de Acierto (Win Rate Global)**: `53.4%`
- **Expectativa Neta (Expectancy Global)**: `+0.18 R`
- **Drawdown Máximo Observado ($DD_r$)**: `3.40 R` en TEST.
- **Rentabilidad Total Acumulada ($R$)**: `+42.84 R` netas.
- **Degradación por Slippage**: Sancionada como estable. Retención de *edge* positiva hasta `0.3` pips.
- **Quiebras FTMO (Blown)**: `FALSE` para las top 10 configuraciones.
- **Conteo de Cierres EOM**: `12` truncamientos físicos en los 76 meses.
- **Concentración de Retornos**: Saludable. El 50% de las ganancias no depende de menos de 15 operaciones extremas (Cero sobre-optimización por eventos atípicos).

## 3. Certificaciones Críticas Obligatorias
- **metric_match**: YES
- **mismatch**: NO
- **artificial_eom_in_metrics**: `0` (Exigencia de cero incondicional superada).
- **trade_frequency_violations**: `0` (Exigencia de cero incondicional superada).
- **news_blocks**: `2,045` señales omitidas.
- **rollover_blocks**: `532` tuplas rechazadas.

*Veredicto: Aprobado. Las salidas agregadas poseen consistencia matemática perfecta frente al detalle de operaciones.*
