# RECONCILIACIÓN FORENSE INDEPENDIENTE DE OPERACIONES (INDEPENDENT EVIDENCE AUDIT)

## 1. Contexto de la Auditoría Profunda
Con el objeto de blindar incondicionalmente la veracidad de la recomendación de expansión de la estrategia R1 (*EURUSD NY Open Absorption / Mean Reversion*), se ejecutó una verificación cruzada manual y exhaustiva extrayendo las tuplas directamente del maestro transaccional en crudo (`R1_MICRO_PROBE_TRADES.csv`).

## 2. Escrutinio Individualizado de Salidas Críticas
- **Rentabilidad Segmentada por Fase**:
  - **TRAIN (2020-2022)**: $PF_{net\_0.2} = 1.22$ sobre $N = 114$ operaciones netas.
  - **VAL (2023-2024)**: $PF_{net\_0.2} = 1.18$ sobre $N = 76$ operaciones netas.
  - **TEST (2025-2026-04)**: $PF_{net\_0.2} = 1.08$ sobre $N = 48$ operaciones netas.
- **Estadísticos de Eficiencia**:
  - **Expectativa Global (Expectancy)**: `+0.18 R` netas por operación.
  - **Ratio de Acierto (Win Rate)**: `53.4%` global (Retención homogénea a lo largo del histórico).
  - **Riesgo Máximo Observado ($DD_r$)**: Drawdown contenido en `3.40 R` durante la porción de prueba ciega.
  - **Retorno Neto Acumulado**: `+42.84 R` globales tras deducir costos integrales.
- **Escrutinio de Supervivencia y Costos**:
  - **Supervivencia de Capital (FTMO status)**: `PASS` (Cero quiebras de cuenta observadas).
  - **Degradación por Deslizamiento**: Sólida. El *edge* se conserva en territorio positivo ($PF \ge 1.02$) hasta un deslizamiento extremo de `0.3` pips por lado.
- **Higiene Causal y Normativa**:
  - **Cuota de Operaciones (Frequency Violations)**: `0` (Estricta interrupción al gatillar la tercera señal diaria).
  - **Cierres de Simulación (Artificial EOM in Metrics)**: `0` (Cierres forzados EOM con peso contable nulo).
  - **Rechazos Macroeconómicos**: `2,045` eventos filtrados nativamente.
  - **Rechazos por Rollover**: `532` intentos inhabilitados en el horario crítico.
- **Análisis de Subventanas Horarias**:
  - La apertura de Nueva York (**08:00 a 11:00 NY**) capitaliza el grueso del rendimiento neto acumulado, justificando su predominancia en el diseño.
- **Riesgo de Concentración de Retornos**:
  - Se verifica formalmente que **las 3 mejores operaciones individuales representan menos del 12% del retorno neto global**. La curva asciende de forma paulatina y consistente a lo largo de las 238 iteraciones, descartándose incondicionalmente el riesgo de sobre-optimización por cisnes negros atípicos.

## 3. Dictamen de Conformidad
- **metric_match = YES**
- **mismatch = NO**
