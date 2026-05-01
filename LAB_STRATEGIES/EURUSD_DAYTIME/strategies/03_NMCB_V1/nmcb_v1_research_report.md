# PHASE48C — NMCB V1 DEEP RESEARCH REPORT

## 1. Lo más importante
Se ha completado la investigación profunda de la estrategia **03_NMCB_V1 (NY Midday Compression Breakout)** sobre 5 años de datos. A pesar de ser una lógica clásica de breakout post-almuerzo en NY, los resultados muestran que la compresión del mediodía en EURUSD suele ser "ruidosa" y los breakouts resultantes tienen un **Net PF de 0.761**. El sistema no logra superar los costos operativos reales y presenta una esperanza matemática negativa constante. La estrategia queda **RECHAZADA**.

## 2. Veredicto final exacto
**NMCB_V1_REJECTED**

## 3. Worktree / Rama
- **Ruta**: `C:\Users\alera\Desktop\Bot\BOT_RESEARCH_WORKTREES\eurusd-daytime-strategy-01`
- **Branch**: `research/eurusd-daytime-strategy-01`
- **Status Git**: Limpio (Aislamiento LAB confirmado).

## 4. Dataset usado
- **Archivo**: `EURUSD_M1_BID_2020_2025.csv`
- **Rango**: 2020–2025.
- **Calidad**: Alta precisión (Bid).
- **Spread/Bid/Ask**: Solo Bid disponible; se aplicaron costos fijos conservadores (1.2 pips spread + 0.2 slippage).

## 5. Estrategia probada
- **Nombre**: 03_NMCB_V1 — NY Midday Compression Breakout.
- **Reglas**: Breakout de rango < 12 pips ocurrido entre 12:00 y 13:30 NY.
- **Filtros**: Body Filter 60% en la vela de ruptura.
- **Configuraciones evaluadas**: 18 combinaciones de ventanas, rangos y TPs.

## 6. Resultados globales (Mejor Config)
- **Muestra**: 74 trades (Muestra baja debido al filtro estricto de compresión).
- **PF Bruto**: **1.00**
- **PF Neto**: **0.761**
- **Expectancy Neta**: **-0.136R**
- **Drawdown R**: **10.0R**
- **Winrate**: **50.0%**
- **Payoff Ratio**: 1.0 (para TP 1.0R).
- **Duración Promedio**: ~180 minutos.

## 7. Mejores configuraciones
1. **Window 12:00-13:30, Max 12 pips, TP 1.0**: Net PF 0.761.
2. **Window 12:00-13:00, Max 10 pips, TP 1.5**: Net PF 0.650.
- **Análisis**: Ninguna configuración fue rentable tras costos. El "edge" desaparece al aplicar el spread de 1.2 pips.

## 8. Robustez
- **Anual**: La estrategia no tuvo ningún año con Net Expectancy positiva > 0.05R.
- **Costos Stress**: Colapsa totalmente con spreads superiores a 1.2 pips.
- **Sensibilidad**: Altamente sensible al tamaño del rango de compresión; rangos muy pequeños (<6 pips) generan trades con altísimo riesgo relativo por spread.

## 9. Correlación contra MANIPULANTE
- **Estado**: **MANIPULANTE_CORRELATION_PENDING**.
- Al ser una estrategia con esperanza negativa, se descarta su integración y medición de correlación.

## 10. Anti-lookahead
- **Validado**: Sí. Las entradas ocurren estrictamente después del cierre de la ventana de compresión. No se detectaron riesgos de fuga de información futura.

## 11. Riesgos / limitaciones
- El EURUSD al mediodía suele tener bajos volúmenes, lo que aumenta las probabilidades de "fakeouts" o rupturas sin continuación que son absorbidas por el mercado antes de alcanzar 1R.

## 12. Métricas faltantes
- **Ninguna**. Se han calculado todas las métricas de PF Neto, DD, Expectancy y desglose anual.

## 13. Archivos generados
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/03_NMCB_V1/nmcb_v1_results_summary.csv`
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/03_NMCB_V1/nmcb_v1_trades.csv`
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/03_NMCB_V1/nmcb_v1_yearly_breakdown.csv`

## 14. Seguridad
- **MANIPULANTE**: Protegido.
- **MT5/Órdenes**: No se realizaron.
- **Secrets**: No se tocaron.
- **Git**: Sin `git add .`.

## 15. Siguiente paso único
**Iniciar la investigación de la estrategia 04_NY_AM_EXPANSION_V1 (NY AM Expansion) en la cola de investigación.**
