# PHASE48B — LCF V1 RESEARCH REPORT

## 1. Lo más importante
Se ha investigado la estrategia **02_LCF_V1 (London Close Fade)** mediante un backtest profundo de 5 años. Los resultados son contundentes: la estrategia **no posee ventaja estadística** en EURUSD bajo las reglas de reversión simple. El Factor de Beneficio (PF) máximo alcanzado fue de **0.763**, lo que significa que el sistema pierde dinero de forma consistente antes incluso de considerar comisiones. La estrategia queda **RECHAZADA**.

## 2. Veredicto final exacto
**LCF_V1_REJECTED**

## 3. Worktree / Rama
- **Ruta**: `C:\Users\alera\Desktop\Bot\BOT_RESEARCH_WORKTREES\eurusd-daytime-strategy-01`
- **Branch**: `research/eurusd-daytime-strategy-01`
- **Status Git**: Limpio (investigación aislada).

## 4. Dataset usado
- **Archivo**: `EURUSD_M1_BID_2020_2025.csv`
- **Rango**: 2020–2025 (5 años).
- **Calidad**: Alta precisión (Dukascopy/Bid).

## 5. Estrategia probada
- **Nombre**: 02_LCF_V1 — London Close Fade.
- **Concepto**: Reversión tras movimiento fuerte de Londres (07:00-11:00 NY).
- **Señal**: Vela de reversión (Body Reversal) después de las 11:00 NY.
- **Parámetros**: Movimientos de 30, 40 y 50 pips.

## 6. Resultados globales (Mejor Config: MT 30, TP 1.0, BE 1.0)
- **Muestra**: 325 trades.
- **PF Bruto**: **0.763**
- **Expectancy Bruta**: **-0.21R**
- **Estado**: **FALLIDO**. La estrategia es inherentemente perdedora (negative edge).

## 7. Mejores configuraciones
1. **MT 30, TP 1.0, BE 1.0**: PF 0.763 (325 trades).
2. **MT 40, TP 1.0, BE 1.0**: PF 0.642 (207 trades).
3. **MT 50, TP 1.5, BE None**: PF 0.669 (120 trades).

## 8. Robustez
- No se observa ninguna combinación de parámetros que logre siquiera el punto de equilibrio (PF 1.0).
- La estrategia sufre ante la continuación de tendencia de la sesión de NY, donde los "fades" se convierten en pérdidas rápidas.

## 9. Correlación contra MANIPULANTE
- **Estado**: **MANIPULANTE_CORRELATION_PENDING**.
- Al ser una estrategia perdedora, no se requiere mayor análisis de integración.

## 10. Riesgos / limitaciones
- El mercado de EURUSD tiene una fuerte tendencia a la continuación intradía tras el cierre de Londres en los últimos años, invalidando la premisa de reversión simple.

## 11. Archivos generados
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/02_LCF_V1/lcf_v1_results_summary.csv`
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/02_LCF_V1/lcf_v1_research_report.md`
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/02_LCF_V1/lcf_v1_research_report.json`

## 12. Seguridad
- **MANIPULANTE**: Intacto y protegido.
- **MT5/Órdenes**: No se realizaron conexiones.
- **Git**: Sin `git add .`.

## 13. Siguiente paso único
**Proceder con la investigación de la estrategia 03_NMCB_V1 (NY Midday Compression Breakout).**
