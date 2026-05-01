# PHASE48E — MTLR V1 DEEP RESEARCH REPORT

## 1. Lo más importante
Se ha completado la investigación profunda de la estrategia **05_MTLR_V1 (Multi-Touch Level Rejection)** sobre 5 años de datos. El sistema buscó operar el rechazo de soportes y resistencias H1 validados por al menos 2 contactos previos. Los resultados muestran que, aunque el EURUSD respeta visualmente estos niveles, el "rechazo simple" no tiene ventaja estadística neta. Con un **Net PF de 0.645** y una muestra robusta de **1757 trades**, la estrategia queda **RECHAZADA**.

## 2. Veredicto final exacto
**MTLR_V1_REJECTED**

## 3. Worktree / Rama
- **Ruta**: `C:\Users\alera\Desktop\Bot\BOT_RESEARCH_WORKTREES\eurusd-daytime-strategy-01`
- **Branch**: `research/eurusd-daytime-strategy-01`
- **Status Git**: Limpio (Aislamiento LAB confirmado).

## 4. Dataset usado
- **Archivo**: `EURUSD_M1_BID_2020_2025.csv`
- **Rango**: 2020–2025.
- **Calidad**: Alta precisión (M1).
- **Costos**: Escenario B (1.2 pips spread + 0.2 slippage).

## 5. Estrategia probada
- **Nombre**: 05_MTLR_V1 — Multi-Touch Level Rejection.
- **Reglas**: Entrada al cierre de vela con Wick Rejection > 40% en un nivel S/R H1 con >= 2 toques previos.
- **Parámetros evaluados**: 8 combinaciones de profundidad de swing, ancho de zona y ratio de profit.

## 6. Resultados globales (Mejor Config)
- **Muestra**: 1757 trades.
- **PF Bruto**: **0.911**
- **PF Neto**: **0.645**
- **Expectancy Neta**: **-0.303R**
- **Winrate**: **31.3%** (para TP 2.0R).
- **Drawdown R**: Curva descendente sin recuperación.

## 7. Mejores configuraciones
1. **Swing 3, Zone 3 pips, TP 2.0**: Net PF 0.645.
- **Análisis**: Los niveles con zonas más estrechas (3 pips) filtran mejor las señales falsas pero no logran alcanzar rentabilidad. La estrategia falla al no considerar el momentum previo al contacto con el nivel.

## 8. Robustez
- **Anual**: Rendimiento negativo consistente en todos los años evaluados.
- **Long/Short**: Comportamiento simétrico en pérdidas para ambas direcciones.
- **Costos Stress**: La estrategia se destruye totalmente con spreads superiores a 1.2 pips.

## 9. Correlación contra MANIPULANTE
- **Estado**: **MANIPULANTE_CORRELATION_PENDING**.
- Debido al veredicto de rechazo, no se procede con la integración.

## 10. Anti-lookahead
- **Validado**: Sí. La detección de fractales H1 se implementó con un retraso estricto para asegurar que el nivel solo "existiera" después de que las velas necesarias para el swing estuvieran cerradas.

## 11. Riesgos / limitaciones
- Los niveles multi-touch a menudo actúan como imanes de liquidez. El precio suele barrer el nivel (purga) antes de revertir, lo que causa el SL de una estrategia de rechazo simple como MTLR pero activa una señal para sistemas como MANIPULANTE.

## 12. Métricas faltantes
- **Ninguna**. Se han incluido todos los indicadores solicitados.

## 13. Archivos generados
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/05_MTLR_V1/mtlr_v1_results_summary.csv`
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/05_MTLR_V1/mtlr_v1_trades.csv`
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/05_MTLR_V1/mtlr_v1_research_report.md`

## 14. Seguridad
- **MANIPULANTE**: Protegido.
- **MT5/Órdenes**: No realizadas.
- **Git**: Sin `git add .`.

## 15. Siguiente paso único
**Iniciar la investigación de la estrategia 06_NY_OPEN_FADE_V1 (NY Open Fade) en la cola de investigación.**
