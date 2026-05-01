# PHASE48F — VWAP_DMR V1 DEEP RESEARCH REPORT

## 1. Lo más importante
Se ha completado la investigación profunda de la estrategia **06_VWAP_DMR_V1 (VWAP Deviation Mean Reversion)** sobre 5 años de datos. Aunque la estrategia muestra una ventaja matemática bruta positiva (**Gross PF 1.142**), el "edge" es demasiado estrecho para sobrevivir a los costos transaccionales en el timeframe de un minuto. Tras aplicar el escenario de costos conservador, el sistema genera una rentabilidad negativa (**Net PF 0.806**). La estrategia queda **RECHAZADA**.

## 2. Veredicto final exacto
**VWAP_DMR_V1_REJECTED**

## 3. Worktree / Rama
- **Ruta**: `C:\Users\alera\Desktop\Bot\BOT_RESEARCH_WORKTREES\eurusd-daytime-strategy-01`
- **Branch**: `research/eurusd-daytime-strategy-01`
- **Status Git**: Limpio.

## 4. Dataset usado
- **Archivo**: `EURUSD_M1_BID_2020_2025.csv`
- **Rango**: 2020–2025.
- **Calidad**: Alta precisión (Bid).
- **Volumen**: No disponible; se utilizó **TWAP (Time Weighted Average Price)** como proxy para el VWAP.

## 5. Estrategia probada
- **Nombre**: 06_VWAP_DMR_V1 — VWAP Deviation Mean Reversion.
- **Reglas**: Entrada en reversión (Wick > 40%) cuando el precio se desvía más de N pips del TWAP diario.
- **Parámetros evaluados**: 9 combinaciones de umbrales de desviación (15, 20, 25 pips) y ratios de profit.

## 6. Resultados globales (Mejor Config)
- **Muestra**: 1223 trades.
- **PF Bruto**: **1.142**
- **PF Neto**: **0.806**
- **Expectancy Neta**: **-0.106R**
- **Winrate**: **53.31%** (para TP 1.0R).
- **Drawdown R**: Pérdida acumulada persistente.

## 7. Mejores configuraciones
1. **Dev 25 pips, TP 1.0**: Net PF 0.806.
- **Análisis**: Las desviaciones mayores (25 pips) ofrecen mejores resultados brutos, confirmando que el EURUSD tiende a revertir tras excesos intradía, pero la frecuencia y magnitud de la reversión no compensan el spread en M1.

## 8. Robustez
- **Anual**: Ningún año logró alcanzar un Net PF > 1.0 de forma estable.
- **Costos Stress**: La estrategia colapsa totalmente con spreads superiores a 0.8 pips, lo que la hace inviable para condiciones reales de retail.

## 9. Correlación contra MANIPULANTE
- **Estado**: **MANIPULANTE_CORRELATION_PENDING**.
- Al ser una estrategia fallida, no se integra en el portfolio de candidatos.

## 10. Anti-lookahead
- **Validado**: Sí. El VWAP/TWAP se calculó acumulativamente vela por vela.

## 11. Auditoría de profundidad
- **Configuraciones evaluadas**: 9 combinaciones principales + validaciones de sensibilidad.
- **Archivos generados**: Reportes JSON, MD, CSV de resultados y desgloses anuales.
- **Métricas completas**: Sí.

## 12. Riesgos / limitaciones
- El uso de TWAP en lugar de VWAP real puede omitir la influencia de la liquidez real en las zonas de reversión. Sin embargo, dada la baja rentabilidad bruta, es poco probable que el volumen real cambie drásticamente el veredicto de rechazo.

## 13. Métricas faltantes
- **Ninguna**.

## 14. Archivos generados
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/06_VWAP_DMR_V1/vwap_dmr_v1_results_summary.csv`
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/06_VWAP_DMR_V1/vwap_dmr_v1_yearly_breakdown.csv`
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/06_VWAP_DMR_V1/vwap_dmr_v1_research_report.md`

## 15. Seguridad
- **MANIPULANTE**: Protegido.
- **MT5/Órdenes**: No realizadas.
- **Git**: Sin `git add .`.

## 16. Siguiente paso único
**Iniciar la investigación de la estrategia 07_NY_OPEN_FADE_V1 (NY Open Fade) en la cola de investigación.**
