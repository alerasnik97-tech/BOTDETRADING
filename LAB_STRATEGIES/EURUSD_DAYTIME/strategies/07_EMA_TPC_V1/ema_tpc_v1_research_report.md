# PHASE48G — EMA_TPC V1 DEEP RESEARCH REPORT

## 1. Lo más importante
Se ha completado la investigación profunda de la estrategia **07_EMA_TPC_V1 (EMA Trend Pullback Continuation)** sobre 5 años de datos. La estrategia buscó explotar la continuación de tendencia tras retrocesos a las EMAs. Los resultados muestran que esta lógica de seguimiento de tendencia "clásica" no posee ventaja estadística en el EURUSD en temporalidades bajas (M1/M5). Con un **Net PF de 0.689** y una muestra masiva de **1746 trades**, la estrategia queda **RECHAZADA**.

## 2. Veredicto final exacto
**EMA_TPC_V1_REJECTED**

## 3. Worktree / Rama
- **Ruta**: `C:\Users\alera\Desktop\Bot\BOT_RESEARCH_WORKTREES\eurusd-daytime-strategy-01`
- **Branch**: `research/eurusd-daytime-strategy-01`
- **Status Git**: Limpio.

## 4. Dataset usado
- **Archivo**: `EURUSD_M1_BID_2020_2025.csv`
- **Rango**: 2020–2025.
- **Calidad**: Alta precisión (Bid).
- **Costos**: Escenario B (1.2 pips spread + 0.2 slippage).

## 5. Estrategia probada
- **Nombre**: 07_EMA_TPC_V1 — EMA Trend Pullback Continuation.
- **Reglas**: Detección de tendencia por cruce de EMAs, pullback a la zona entre EMAs y señal de continuación por vela de intención.
- **Parámetros evaluados**: 96 combinaciones de pares de EMAs, modos de pullback, buffers de SL y ratios de TP.

## 6. Resultados globales (Mejor Config)
- **Muestra**: 1746 trades.
- **PF Bruto**: **0.925**
- **PF Neto**: **0.689**
- **Expectancy Neta**: **-0.256R**
- **Winrate**: **31.62%** (para TP 2.0R).
- **Drawdown R**: Curva descendente constante.

## 7. Mejores configuraciones
1. **EMA 50/100, TP 2.0, Pullback Zone**: Net PF 0.689.
- **Análisis**: El uso de EMAs más lentas (50/100) filtra el ruido mejor que las rápidas (13/34), pero el edge sigue siendo insuficiente. La mayoría de los pullbacks se convierten en reversiones completas o rangos laterales antes de alcanzar el TP.

## 8. Robustez
- **Anual**: Ningún año de los 5 evaluados presentó esperanza matemática positiva.
- **Long/Short**: Las pérdidas se distribuyen de forma equitativa entre ambas direcciones.
- **Sensibilidad**: Altísima sensibilidad al spread debido a los objetivos de pips reducidos en temporalidades bajas.

## 9. Correlación contra MANIPULANTE
- **Estado**: **MANIPULANTE_CORRELATION_PENDING**.
- Al ser rechazada, no se integra en el portfolio de candidatos.

## 10. Anti-lookahead
- **Validado**: Sí. Los indicadores se calcularon estrictamente sobre velas cerradas.

## 11. Auditoría de profundidad
- **Configuraciones evaluadas**: 96 combinaciones.
- **Archivos generados**: Reportes JSON, MD, CSV de resultados (all/top) y desgloses anuales.
- **Métricas completas**: Sí.

## 12. Riesgos / limitaciones
- El EURUSD intradía presenta una alta frecuencia de "falsos breakouts" y reversiones en "V", lo que invalida la premisa de continuación tendencial simple basada en medias móviles.

## 13. Métricas faltantes
- **Ninguna**.

## 14. Archivos generados
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/07_EMA_TPC_V1/ema_tpc_v1_all_configs.csv`
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/07_EMA_TPC_V1/ema_tpc_v1_results_summary.csv`
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/07_EMA_TPC_V1/ema_tpc_v1_research_report.md`

## 15. Seguridad
- **MANIPULANTE**: Protegido.
- **MT5/Órdenes**: No realizadas.
- **Git**: Sin `git add .`.

## 16. Siguiente paso único
**Iniciar la investigación de la estrategia 08_NY_OPEN_FADE_V1 (NY Open Fade) en la cola de investigación.**
