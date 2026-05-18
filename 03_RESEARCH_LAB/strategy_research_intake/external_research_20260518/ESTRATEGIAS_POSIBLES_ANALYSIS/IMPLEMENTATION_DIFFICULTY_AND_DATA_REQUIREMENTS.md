# IMPLEMENTATION DIFFICULTY AND DATA REQUIREMENTS — INFRASTRUCTURE AUDIT
**Date:** 2026-05-18
**Project:** Systematic Infrastructure Professionalization — Data Requirements and Complexity Audit
**Security Status:** READ-ONLY AUDIT & COMPILATION — NO CODE OR REPOSITORY MUTATION

---

## 1. Mapeo de Requerimientos de Datos por Familia

La viabilidad técnica de una estrategia en el laboratorio algorítmico depende directamente del tipo, frecuencia y calidad de los datos requeridos para alimentar sus modelos matemáticos.

```
+-------------------------------------------------------------------------------------------------------------------+
|                                     MATRIZ DE REQUERIMIENTOS Y FUENTES DE DATOS                                   |
+---+--------------------+------------------------+---------------------------------------+---------------------+
| # | Familia            | Datos Requeridos       | Frecuencia Requerida                  | Fuente Local        |
+---+--------------------+------------------------+---------------------------------------+---------------------+
| 1 | Mean Reversion     | OHLCV, VWAP diario     | M5 para señales, M1 para VWAP         | 05_MARKET_DATA_VAULT|
| 2 | Trend Pullback     | OHLCV                  | M5, M15 multi-timeframe               | 05_MARKET_DATA_VAULT|
| 3 | Volatility Exp     | OHLCV, Volúmenes/Ticks | M5 para canales, M15 para ATR         | 05_MARKET_DATA_VAULT|
| 4 | Session Dynamics   | OHLCV                  | M5/M1 con timestamp exacto            | 05_MARKET_DATA_VAULT|
| 5 | Seasonal           | OHLCV                  | M15 con reloj de sesión NY            | 05_MARKET_DATA_VAULT|
| 6 | Event-Driven       | Bid/Ask Ticks, News DB | Ticks agregados millisecond y News DB | NO DISPONIBLE       |
+---+--------------------+------------------------+---------------------------------------+---------------------+
```

---

## 2. Clasificación de Dificultad Técnica de Codificación

Evaluamos la complejidad de programar y probar cada modelo de forma robusta e institucional, evitando trampas de latencia y lookahead bias.

### A. Complejidad BAJA (Triviales): *Viables en horas*
*   **Sistemas:** `TP12` (Trend Pullback EMA), `SE07` (Friday Weekly Roll Close Fade), `SE08` (Day-of-Week Trend Anomaly).
*   **Características:** Utilizan indicadores clásicos (EMAs, ATR, filtros de tiempo) sobre barras estándar de M15 o M5. No requieren almacenamiento de estados intrabarra complejos ni anclajes acumulativos dinámicos.
*   **Infraestructura de Cómputo:** Local CPU estándar.

### B. Complejidad MEDIA: *Viables en días*
*   **Sistemas:** `MR17` (London Close VWAP), `MR05` (VWAP Z-Score), `VE01` (ORB Volatility), `VE18` (NY Mid-Day Breakout), `VE04` (Donchian + VWAP).
*   **Características:** Requieren la inicialización y el cálculo dinámico del VWAP acumulativo desde las 07:00 NY, integrando de forma causal barras M1 en marcos de M5. Requieren motores de cálculo rodante backward-only para evitar lookahead bias en las Bandas de Bollinger y canales Donchian.
*   **Infraestructura de Cómputo:** Kaggle Cloud sweep pre-configurado para pruebas de sensibilidad.

### C. Complejidad ALTA: *Viables en semanas / meses*
*   **Sistemas:** `MR08` (BB Double Tap Divergence), `MR05_2` (Volume Profile POC Reversion).
*   **Características:** Requieren algoritmos matemáticos complejos para la detección formal y causal de divergencias en tiempo real (mínimos crecientes de precio con máximos decrecientes de RSI) sin sesgo retrospectivo. El modelado del perfil de volumen requiere consolidar ticks crudos a diario y mapear nodos de alto volumen (HVN).
*   **Infraestructura de Cómputo:** Kaggle Cloud sweep y bases de datos NoSQL dedicadas.

### D. Complejidad EXTREMA: *Meses de Ingeniería de Datos y Fricción Operativa*
*   **Sistemas:** `ED15` (Post-News Reversion), `ED16` (Post-News Momentum), `HY10` (High-Frequency Bid-Ask Imbalance Fade).
*   **Características:** Requieren la adquisición, limpieza y almacenamiento estructurado de una base de datos histórica de anuncios económicos con marcas de tiempo en milisegundos. Operar a este nivel exige un feed de datos L2 (Order Book Depth) bid/ask continuo y un motor de execution ultra-rápido para evadir el ensanchamiento masivo de spreads.
*   **Infraestructura de Cómputo:** Servidores VPS dedicados de ultrabaja latencia conectados por fibra óptica en centros de datos de Londres/Nueva York.

---

## 3. Diagnóstico de Infraestructura Local

Para maximizar la agilidad del laboratorio sin violar las directrices del owner:

1.  **Sistemas Triviales / Medios (Mean Reversion y Volatility Expansion):** Son 100% compatibles con la infraestructura de Kaggle Cloud y del Data Vault actual. Las bases de datos locales en `05_MARKET_DATA_VAULT/` contienen las barras M1 y M5 en formato Parquet necesarias para reconstruir de forma causal el VWAP, ATR y rangos diarios.
2.  **Sistemas de Alta/Extrema Complejidad (Event-Driven y HFT):** **No son viables en la fase actual**. El intento de codificarlos consumiría meses enteros de ingeniería de infraestructura de datos, desviando la agilidad investigadora y arriesgando la seguridad y agilidad operacional exigida por el owner. Se decretan diferidos y congelados indefinidamente.
