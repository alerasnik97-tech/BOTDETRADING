# PHASE48H — PDM_RECLAIM V1 DEEP RESEARCH REPORT

## 1. Lo más importante
Se ha completado la investigación profunda de la estrategia **08_PDM_RECLAIM_V1 (Previous Day Midpoint Reclaim)** sobre 5 años de datos. La estrategia exploró la recuperación del punto medio del día anterior como señal de cambio de dirección. A pesar de una auditoría extensa de **120 configuraciones**, no se encontró ninguna variante rentable. Con un **Net PF de 0.739** y una esperanza matemática negativa, la estrategia queda **RECHAZADA**.

## 2. Veredicto final exacto
**PDM_RECLAIM_V1_REJECTED**

## 3. Worktree / Rama
- **Ruta**: `C:\Users\alera\Desktop\Bot\BOT_RESEARCH_WORKTREES\eurusd-daytime-strategy-01`
- **Branch**: `research/eurusd-daytime-strategy-01`
- **Status Git**: Limpio.

## 4. Dataset usado
- **Archivo**: `EURUSD_M1_BID_2020_2025.csv`
- **Rango**: 2020–2025.
- **Calidad**: Alta precisión (Bid).
- **Timezone**: NY (convertido de UTC).

## 5. Estrategia probada
- **Nombre**: 08_PDM_RECLAIM_V1 — Previous Day Midpoint Reclaim.
- **Reglas**: Entrada tras un cierre fuerte (Body Filter) que recupera el PDM del día anterior, después de haber permanecido al menos N minutos del otro lado.
- **Parámetros evaluados**: 120 combinaciones de tiempo de espera, filtros de cuerpo, ratios de TP y modos de SL.

## 6. Resultados globales (Mejor Config)
- **Muestra**: 544 trades.
- **PF Bruto**: **0.974**
- **PF Neto**: **0.739**
- **Expectancy Neta**: **-0.2R**
- **Winrate**: **35.11%** (para TP 1.8R).
- **Drawdown R**: Curva descendente persistente.

## 7. Mejores configuraciones
1. **Wait 60m, BF 0.6, TP 1.8, SL Candle**: Net PF 0.739.
- **Análisis**: Exigir mayor tiempo de espera (60 min) y velas más fuertes (BF 0.6) ayuda a reducir el ruido, pero no logra convertir la señal en rentable. El PDM suele ser una zona de alta volatilidad y "choppiness", lo que genera muchos falsos reclaims que terminan en SL.

## 8. Robustez
- **Anual**: Rendimiento negativo en todos los años del dataset.
- **Sensibilidad**: La estrategia es extremadamente sensible a los costos; el Gross PF cercano a 1.0 sugiere que es un sistema "random" que colapsa ante el spread retail.

## 9. Correlación contra MANIPULANTE
- **Estado**: **MANIPULANTE_CORRELATION_PENDING**.
- Al ser rechazada, no se procede con el análisis de diversificación.

## 10. Anti-lookahead
- **Validado**: Sí. Los niveles del PDM se calcularon exclusivamente con datos del día calendario anterior cerrado.

## 11. Auditoría de profundidad
- **Configuraciones evaluadas**: 120 combinaciones.
- **Archivos generados**: Reportes JSON, MD, CSV (all/top) y desgloses anuales.
- **Métricas completas**: Sí.

## 12. Riesgos / limitaciones
- El Punto Medio del día anterior es un nivel de "fair value". Cuando el precio lo alcanza, a menudo entra en un estado de equilibrio (rango lateral) en lugar de una rotación limpia, lo que castiga severamente a las entradas por reclaim.

## 13. Métricas faltantes
- **Ninguna**.

## 14. Archivos generados
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/08_PDM_RECLAIM_V1/pdm_reclaim_v1_all_configs.csv`
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/08_PDM_RECLAIM_V1/pdm_reclaim_v1_results_summary.csv`
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/08_PDM_RECLAIM_V1/pdm_reclaim_v1_research_report.md`

## 15. Seguridad
- **MANIPULANTE**: Protegido.
- **MT5/Órdenes**: No realizadas.
- **Git**: Sin `git add .`.

## 16. Siguiente paso único
**Iniciar la investigación de la estrategia 09_NY_OPEN_FADE_V1 (NY Open Fade) en la cola de investigación.**
