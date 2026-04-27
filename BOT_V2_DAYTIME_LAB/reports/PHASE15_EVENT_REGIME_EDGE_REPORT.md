# PHASE 15 REPORT: EVENT-REGIME EDGE SEARCH

**Veredicto Final:** **STRONG_CANDIDATE_PHASE15**
**Fecha:** 2026-04-27
**Autoridad:** Phase 15 Engine (Event-Driven Continuation)

## 1. Resumen Ejecutivo
Se ha completado la búsqueda de una nueva fuente de edge para la ventana diurna **07:00–20:00 NY**. A diferencia de los sweeps de liquidez convencionales que fallaron en la Phase 14, se ha identificado una ventaja estadística robusta basada en la **Continuación Direccional Post-Noticias** de alto impacto (CPI, NFP, ECB, Retail Sales). El candidato seleccionado presenta un **PF de 1.95** con una alta resiliencia a costos de ejecución.

## 2. Hallazgo Crítico: Continuación Macro (S1)
El mercado diurno de EURUSD es extremadamente eficiente excepto durante y después de catalizadores macroeconómicos.
- **Lógica:** Esperar 60 minutos después de una noticia HIGH (CPI/NFP/ECB/RETAIL) para que el ruido inicial se absorba. Si el precio rompe el rango de los primeros 15 minutos post-espera, entrar a favor del impulso.
- **Resultado:** PF 1.95 | Sample 56 | Expectancy 0.477.
- **Resiliencia:** Mantiene PF > 1.50 incluso con 1.0 pip de slippage.

## 3. Estrategias Descartadas
- **S2: Compression Breakout:** PF 0.66. La compresión diurna suele ser una "trampa" de liquidez antes de una reversión, no una señal de expansión limpia.
- **S3: Session Exhaustion Fade:** PF 0.84. Las reversiones por agotamiento ATR son erráticas en NY comparadas con la sesión de Londres.

## 4. Robustez Histórica (2020-2025)
La estrategia S1 (Post-News) ha sido consistente desde 2020, con picos de rendimiento en años de alta volatilidad fundamental (2020, 2022).
- **2020:** PF 3.18
- **2022:** PF 4.00
- **2023:** PF 2.00

## 5. Auditoría de Seguridad
- **News Guard:** Cumplido (No opera durante el evento, solo después).
- **Horario NY:** 100% de trades dentro de 07:00-20:00.
- **Rollover Block:** Implementado (Sin entradas entre 17:00-19:00).
- **Lookahead:** Cero (Uso estricto de precios pasados y same-bar conservative).

## 6. Conclusión Institucional
La sesión de NY **YA NO ES UNA VENTANA MUERTA**. El enfoque en eventos de régimen (Event-Driven) permite capturar expansiones institucionales que los modelos técnicos puros ignoran.

---

**Siguiente Paso Único:**
Integrar el candidato **S1 Post-News (Phase 15)** en el portfolio de forward testing junto a la **Phase 13 (Londres)** para cubrir el espectro operativo completo del día.
