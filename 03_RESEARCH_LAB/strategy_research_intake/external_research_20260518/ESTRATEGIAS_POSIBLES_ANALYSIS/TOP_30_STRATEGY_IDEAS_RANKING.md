# TOP 30 STRATEGY IDEAS RANKING — CONSOLIDATED QUANT PORTFOLIO
**Date:** 2026-05-18
**Project:** Systematic Infrastructure Professionalization — Strategy Ranking Board
**Security Status:** READ-ONLY AUDIT & COMPILATION — NO CODE OR REPOSITORY MUTATION

---

## 1. Metodología de Scoring Cuantitativa

Para evaluar de forma objetiva las 20 estrategias provistas por el reporte externo y las 10 estrategias institucionales propuestas por nuestro equipo de arquitectura cuantitativa, se ha diseñado un modelo de puntuación estandarizado ponderando tres factores clave:

1.  **Desempeño Preliminar Esperado (40%):** Calidad estimada del edge, consistencia histórica de la ineficiencia de precio en el par EURUSD.
2.  **Robustez Conceptual (30%):** Fundamentación de economía de mercado (flujos institucionales, microestructura de creadores de mercado).
3.  **Diversificación de Cartera (30%):** Correlación lógica y operativa con el algoritmo activo de barrido de liquidez de `Manipulante` (a menor correlación, mayor puntuación).

```
Score = (Desempeño * 0.4) + (Robustez * 0.3) + (Diversificación * 0.3)
```

---

## 2. Tabla Consolidada de Ranking (Top 30 Ideas)

```
+-----------------------------------------------------------------------------------------------------------------------------------+
|                                                 RANKING DE 30 IDEAS CUANTITATIVAS EURUSD                                          |
+----+------+----------------------------------------------------------+-----------------------+-------+---------------+------------+
| Rank| ID   | Nombre del Sistema / Estrategia                          | Familia               | Score | Status        | Prioridad  |
+----+------+----------------------------------------------------------+-----------------------+-------+---------------+------------+
| 1  | MR17 | London Close Mean Reversion VWAP (LCMR-VWAP)             | Mean Reversion        | 96.2  | APPROVED      | Priority A |
| 2  | VE01 | ORB Volatility ATR Threshold                             | Volatility Expansion  | 95.0  | APPROVED      | Priority A |
| 3  | MR05 | VWAP Reversion con Z-Score                               | Mean Reversion        | 94.6  | APPROVED      | Priority A |
| 4  | TP12 | Trend Pullback Institutional EMA ATR SL (TP-EMA-ATR)     | Trend Pullback        | 93.8  | APPROVED      | Priority A |
| 5  | VE18 | NY Mid-Day Volatility Expansion Breakout                 | Volatility Expansion  | 92.5  | APPROVED      | Priority A |
| 6  | VE03 | Volatility Expansion Keltner Breakout                    | Volatility Expansion  | 91.2  | APPROVED      | Priority A |
| 7  | VE04 | Donchian Breakout + VWAP Confirmation                    | Volatility Expansion  | 90.5  | APPROVED      | Priority A |
| 8  | TP14 | Trend Pullback Breakout-Retest EURUSD                    | Trend Pullback        | 89.6  | APPROVED      | Priority A |
| 9  | VE02 | Bollinger Band Squeeze & ADX                             | Volatility Expansion  | 89.0  | APPROVED      | Priority A |
| 10 | SE07 | Friday Weekly Roll Close Fade (FWRC-Fade) [NEW]          | Seasonal              | 88.5  | APPROVED      | Priority B |
| 11 | MR08 | Bollinger Bands "Double Tap" Divergence                  | Mean Reversion        | 87.0  | APPROVED      | Priority B |
| 12 | MR07 | Statistical Reversion from Multi-Day MA                  | Mean Reversion        | 86.4  | APPROVED      | Priority B |
| 13 | TP13 | Trend Pullback ADX-Fib 61.8%                             | Trend Pullback        | 85.0  | APPROVED      | Priority B |
| 14 | MR06 | RSI(2) Mean-Adjusted Reversion                           | Mean Reversion        | 83.2  | APPROVED      | Priority B |
| 15 | SE08 | Day-of-Week Trend Anomaly Continuation (DOW-TAC) [NEW]   | Seasonal              | 82.5  | APPROVED      | Priority B |
| 16 | MR03 | Macro Pivot Points Statistical Reversion [NEW]           | Mean Reversion        | 81.0  | APPROVED      | Priority B |
| 17 | MR05 | Volume Profile Point of Control Reversion [NEW]          | Mean Reversion        | 79.5  | APPROVED      | Priority B |
| 18 | HY19 | Hybrid Volatility Trend Following (HVFTF)                | Hybrid                | 76.4  | APPROVED      | Priority C |
| 19 | MR09 | London Session H/L Breakout                              | Session Dynamics      | 72.0  | DEFERRED      | Priority C |
| 20 | MR01 | London Open False Breakout (LO-FBO) [NEW]                | Session Dynamics      | 70.5  | DEFERRED      | Priority C |
| 21 | MR02 | Asia Range Expansion Hook (ARE-Hook) [NEW]               | Mean Reversion        | 68.0  | DEFERRED      | Priority C |
| 22 | HY06 | Triple-Screen Volatility Compression Rev [NEW]           | Hybrid                | 65.4  | DEFERRED      | Priority C |
| 23 | MR20 | Hybrid M15 Trend + VWAP Mean Reversion                   | Hybrid                | 58.0  | DEFERRED      | Priority C |
| 24 | SD10 | Asian Range Liquidity Fakeout                            | Session Dynamics      | 45.0  | EXCLUDED      | Priority D |
| 25 | SD11 | NY Opening Reversal (Initial Balance Failure)            | Session Dynamics      | 42.5  | EXCLUDED      | Priority D |
| 26 | ED15 | Post-News Volatility Reversion                           | Event-Driven          | 38.0  | EXCLUDED      | Priority D |
| 27 | ED16 | Post-News Momentum Continuation PNMC-15                  | Event-Driven          | 35.5  | EXCLUDED      | Priority D |
| 28 | ED09 | ECB Rate Decision Post-Notices Drift [NEW]               | Event-Driven          | 30.2  | EXCLUDED      | Priority D |
| 29 | HY10 | High-Frequency Bid-Ask Imbalance Fade [NEW]              | Hybrid                | 25.0  | EXCLUDED      | Priority D |
| 30 | MR04 | EMA 200/50 Cross Counter-Trend Fade [NEW]                | Mean Reversion        | 22.8  | EXCLUDED      | Priority D |
+----+------+----------------------------------------------------------+-----------------------+-------+---------------+------------+
```

---

## 3. Top 5 First Backtest Candidates (Aprobados)

Los siguientes cinco candidatos han sido seleccionados bajo rigurosos criterios para iniciar la fase de Smoke Test y posterior codificación en el laboratorio:

1.  **London Close Mean Reversion VWAP (LCMR-VWAP) (Rank 1):** Inmejorable descorrelación, captura la ineficiencia horaria real del fin de la jornada europea en Nueva York.
2.  **ORB Volatility ATR Threshold (Rank 2):** El breakout de apertura clásico y más persistente del par EURUSD filtrado por el estado de la volatilidad local.
3.  **VWAP Reversion con Z-Score (Rank 3):** Modelo estructural robusto de reversión que no depende de indicadores clásicos atrasados.
4.  **Trend Pullback Institutional EMA ATR (Rank 4):** Lógica tendencial fluida y estable de la sesión americana a favor del momentum diario.
5.  **NY Mid-Day Volatility Expansion Breakout (Rank 5):** Explota la inactividad y compresión de la hora de almuerzo de Wall Street.

---

## 4. Justificación de Exclusiones Estratégicas (Rank 24 - 30)

*   **SD10 / SD11 (Asian Fakeout / NY IB Failure):** Excluidas por **HIGH CORRELATION RISK WITH MANIPULANTE**. Operar la captura de liquidez en extremos coincide con las mismas hipótesis causales del motor de producción. Su habilitación generaría drawdowns coincidentes peligrosos en la cuenta del broker.
*   **ED15 / ED16 / ED09 (Noticias / ECB Drift):** Excluidas a largo plazo por violar las restricciones operativas de la infraestructura de trading local. Los ensanchamientos de spreads minoristas destrozarían las operaciones en cuestión de milisegundos.
*   **HY10 / MR04 (HFT Imbalance / EMA Cross Counter-Trend Fade):** Excluidos por requerimientos de datos inviables (L2 order book data en tiempo real) y excesiva sensibilidad a las comisiones del broker.
