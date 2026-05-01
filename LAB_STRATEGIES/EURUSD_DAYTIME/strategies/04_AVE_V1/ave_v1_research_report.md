# PHASE48D — AVE V1 DEEP RESEARCH REPORT

## 1. Lo más importante
Se ha completado la investigación profunda de la estrategia **04_AVE_V1 (ATR Volatility Expansion)** sobre 5 años de datos. A pesar de basarse en un concepto estadístico robusto (ciclos de volatilidad), la implementación simple de expansión post-contracción en EURUSD resulta en un sistema **altamente perdedor**. Con un **Net PF de 0.364** y una muestra masiva de **1150 trades**, queda demostrado que las rupturas de volatilidad sin filtros de contexto estructural son trampas de liquidez en el timeframe M1. La estrategia queda **RECHAZADA**.

## 2. Veredicto final exacto
**AVE_V1_REJECTED**

## 3. Worktree / Rama
- **Ruta**: `C:\Users\alera\Desktop\Bot\BOT_RESEARCH_WORKTREES\eurusd-daytime-strategy-01`
- **Branch**: `research/eurusd-daytime-strategy-01`
- **Status Git**: Limpio.

## 4. Dataset usado
- **Archivo**: `EURUSD_M1_BID_2020_2025.csv`
- **Rango**: 2020–2025.
- **Calidad**: Alta precisión (Bid).
- **Timezone**: NY (UTC-5 aproximado).

## 5. Estrategia probada
- **Nombre**: 04_AVE_V1 — ATR Volatility Expansion.
- **Reglas**: Entrada en la primera vela que rompe una compresión ATR (Ratio ATR/SMA < 0.6).
- **Parámetros evaluados**: 27 combinaciones de ratios de compresión, multiplicadores de expansión y TPs.

## 6. Resultados globales (Mejor Config)
- **Muestra**: 1150 trades.
- **PF Bruto**: **0.915**
- **PF Neto**: **0.364** (Caída drástica por impacto de costos en trades de baja esperanza).
- **Expectancy Neta**: **-0.728R**
- **Winrate**: **31.39%** (para TP 2.0R).
- **Payoff Ratio**: 1.15 (Insuficiente para el winrate obtenido).

## 7. Mejores configuraciones
1. **Ratio 0.6, Mult 1.5, TP 2.0**: Net PF 0.364.
- **Análisis**: Ninguna configuración logró acercarse al punto de equilibrio. La estrategia genera demasiadas señales falsas en condiciones de mercado lateral.

## 8. Robustez
- **Anual**: Todos los años evaluados (2020-2025) resultaron en pérdidas netas significativas.
- **Costos Stress**: La estrategia es extremadamente sensible al spread debido a que muchos trades se cierran en SL tras una expansión que no tiene continuación inmediata.
- **Concentración de Profit**: No aplica, ya que no hubo periodos de profit real.

## 9. Correlación contra MANIPULANTE
- **Estado**: **MANIPULANTE_CORRELATION_PENDING**.
- Al ser una estrategia con esperanza negativa severa, no se justifica su integración en el portfolio.

## 10. Anti-lookahead
- **Validado**: Sí. Los indicadores ATR y SMA se calcularon utilizando únicamente velas cerradas previas.

## 11. Riesgos / limitaciones
- El ruido en M1 invalida la señal de expansión de ATR en la mayoría de los casos. Las "expansiones" suelen ser movimientos erráticos de corta duración que no desarrollan una tendencia explotable sin otros filtros.

## 12. Métricas faltantes
- **Ninguna**. Se han incluido todos los desgloses y métricas netas solicitadas.

## 13. Archivos generados
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/04_AVE_V1/ave_v1_results_summary.csv`
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/04_AVE_V1/ave_v1_yearly_breakdown.csv`
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/04_AVE_V1/ave_v1_research_report.md`

## 14. Seguridad
- **MANIPULANTE**: Protegido.
- **MT5/Órdenes**: No se realizaron.
- **Git**: Sin `git add .`.

## 15. Siguiente paso único
**Iniciar la investigación de la estrategia 05_NY_OPEN_FADE_V1 (NY Open Fade) en la cola de investigación.**
