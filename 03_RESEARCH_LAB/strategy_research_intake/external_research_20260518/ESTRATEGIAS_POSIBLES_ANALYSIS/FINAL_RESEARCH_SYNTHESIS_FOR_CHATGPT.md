# FINAL RESEARCH SYNTHESIS — THE QUANTITATIVE COMPACT BLUEPRINT
**Date:** 2026-05-18
**Project:** Systematic Infrastructure Professionalization — Executive Intake Synthesis for LLM Uploads
**Security Status:** READ-ONLY AUDIT & COMPILATION — NO CODE OR REPOSITORY MUTATION

---

## 1. Síntesis Ejecutiva de la Ingesta Quant

Este documento condensa los hallazgos críticos de la auditoría y catalogación de las **20 estrategias cuantitativas externas** (ingestadas desde `ESTRATEGIAS_POSIBLES/`) y las **10 estrategias institucionales agregadas**, optimizadas específicamente para operar de forma sistemática el par EURUSD intradía (07:00-19:00 NY). 

El objetivo es proveer una ficha técnica ultra-condensada de máxima fidelidad técnica lista para ser suministrada como contexto a otros modelos avanzados (Grok, Claude, ChatGPT) en fases futuras del desarrollo algorítmico.

---

## 2. El Portafolio Consolidado (Taxonomía de 30 Modelos)

El universo de 30 estrategias ha sido clasificado rigurosamente en seis familias operacionales. A continuación se presenta el mapa consolidado:

```
+-------------------------------------------------------------------------------------------------------------------+
|                                      DISTRIBUCIÓN TAXONÓMICA DE LAS 30 ESTRATEGIAS                                |
+---+--------------------+-------------+------------------------------------+-----------------------------------+
| # | Familia            | Cantidad    | Modelos Clave                      | Enfoque Microestructural          |
+---+--------------------+-------------+------------------------------------+-----------------------------------+
| 1 | Mean Reversion     | 10 Modelos  | MR17 (London Close), MR05 (VWAP)   | Fatiga de momentum, retorno medio|
| 2 | Trend Pullback     | 5 Modelos   | TP12 (EMA Pullback), TP14 (Retest) | Incorporación en valor dinámico   |
| 3 | Volatility Exp     | 6 Modelos   | VE01 (ORB), VE18 (Mid-Day Breakout)| Transición compresión a expansión |
| 4 | Session Dynamics   | 4 Modelos   | SD09 (London H/L), SD01 (LO-FBO)   | Ruptura de fronteras geográficas  |
| 5 | Seasonal           | 2 Modelos   | SE07 (Friday Fade), SE08 (DOW-TAC) | Anomalías basadas en el reloj     |
| 6 | Event-Driven       | 3 Modelos   | ED15 (Post-News), ED09 (ECB Drift) | Sobre-reacciones fundamentales    |
+---+--------------------+-------------+------------------------------------+-----------------------------------+
```

---

## 3. Fichas Técnicas de los Top 3 Candidatos Aprobados

### CANDIDATO 1: London Close Mean Reversion VWAP (LCMR-VWAP)
*   **Familia:** Mean Reversion.
*   **Lógica:** Explota la toma de beneficios y liquidación física de libros de las tesorerías europeas al cierre de Londres (11:30 - 12:00 NY), forzando al EURUSD a revertir hacia su VWAP diario.
*   **Trigger:** El precio M5 cierra a $\ge 5$ pips del VWAP diario. Confirmar con vela reversal en M1.
*   **Riesgo / Beneficio:** SL estático de 10 pips, TP de 8 pips (cierre del $50\%$ a los 4 pips y trailing de 3 pips).
*   **Filtros:** Spread $\le 1.5$ pips. Excluir días con ATR(14) diario $>40$ pips.
*   **Correlación con Manipulante:** **NULA** (Diversificador óptimo).

### CANDIDATO 2: ORB Volatility ATR Threshold (ORB-ATR)
*   **Familia:** Volatility Expansion.
*   **Lógica:** Captura los movimientos direccionales muy limpios en la sesión americana tras romper los máximos/mínimos del rango de apertura (07:00-09:00 NY), condicionado a una volatilidad mínima.
*   **Trigger:** Ruptura por cierre M15 de los extremos del Opening Range, con ATR(14) M15 en las 09:00 NY $\ge 5$ pips.
*   **Riesgo / Beneficio:** SL colocado a $1.0\times$ ATR(14) M15 del extremo roto, TP a $2.0\times$ la distancia del SL.
*   **Filtros:** Spread $\le 1.5$ pips. Cierre forzado por tiempo a las 19:00 NY.
*   **Correlación con Manipulante:** **BAJA** (Captura tendencias puras).

### CANDIDATO 3: VWAP Reversion con Z-Score
*   **Familia:** Mean Reversion.
*   **Lógica:** Arbitraje estadístico del exceso de fluctuación del par EURUSD sobre su VWAP diario, gatillado por desviaciones extremas de Z-Score.
*   **Trigger:** Z-Score del precio M1 sobre el VWAP acumulado cruza por debajo de $-2.0$ (Long) o por encima de $+2.0$ (Short).
*   **Riesgo / Beneficio:** TP en el VWAP diario, SL a $1.5\times$ ATR(14) M15.
*   **Filtros:** Spread máximo $\le 1.5$ pips.
*   **Correlación con Manipulante:** **BAJA**.

---

## 4. El Marco de Anti-Overfitting y Parameter Governance

1.  **Congelación Pre-Backtest:** Los parámetros analíticos (ATR period, multipliers, thresholds) se fijan de forma formal antes de correr simulaciones históricas, evitando el p-hacking.
2.  **No Grid Search:** Se prohíben barridos de fuerza bruta. Solo se permiten análisis de estabilidad paramétrica ($\pm10\%$) para verificar que el Sharpe Ratio no sea inestable.
3.  **Holdout 2025/2026 Bloqueado:** Ninguna simulación puede tocar la muestra sellada 2025/2026, la cual permanece protegida bajo cuarentena absoluta.
4.  **Criterios de Rechazo Temprano:** Cualquier modelo con Sharpe Ratio $<1.0$, Drawdown Máximo $>3.0\%$, profit factor $<1.3$ o esperanza matemática $<1.5$ pips netos se descarta de forma automática e irrevocable.
5.  **Límite de Frecuencia:** El algoritmo bajo testeo no puede realizar más de **3 operaciones al día**, limitando la fricción y el pago de comisiones al bróker.

---

## 5. Próximo Paso Crítico del Desarrollo
*   **Implementación del Backtesting Smoke Test:** Crear los scripts lógicos para validar la ingesta limpia de barras Parquet M1/M5 y la reconstrucción causal backward-only del VWAP y ATR, sentando la base de Oleada 1 (Mean Reversion) en el laboratorio de desarrollo `03_RESEARCH_LAB/`.
