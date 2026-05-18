# TOP STRATEGY IDEAS PRELIMINARY RANKING
**Date:** 2026-05-18
**Project:** EURUSD Intraday Research (NY Session)
**Security Status:** READ-ONLY AUDIT & COMPILATION — NO CODE OR REPOSITORY MUTATION

---

## 1. Executive Summary

Este documento presenta el **Ranking de Prioridades Cuantitativas** y el **Backlog Consolidado de 20 Ideas** para el laboratorio de trading. Se estructura bajo el principio de "anti-autoengaño", clasificando las estrategias no por su "promesa de rentabilidad" (lo cual constituiría una mala práctica de sobreajuste antes del backtest), sino por su **viabilidad de implementación, complementariedad con `Manipulante` y solidez lógica**.

---

## 2. Priority A candidates (Aprobadas para Codificación de Señal Base)

A continuación se detallan las cuatro estrategias de **Priority A** aprobadas para su desarrollo experimental de señales base en `03_RESEARCH_LAB`, respetando la exclusión estricta de noticias y alta precisión.

---

### #1. MR-01: Anchor Elastic Mean Reversion
*   **Nombre Tentativo:** MR-01 Anchor Elastic
*   **Lógica de Mercado:** Reversión estadística hacia el Average Median Price (APM) calculado desde las 07:00 NY, cuando el precio se desvía más de un umbral dinámico basado en volatilidad intradía extrema.
*   **Aplica a EURUSD Intraday:** Sí.
*   **Horario NY Probable:** 07:00 – 19:00 NY (Foco activo en la sesión americana).
*   **Correlación Esperada contra `Manipulante`:** **BAJA** (No depende de barridos ni de fakeouts).
*   **Dificultad de Programar:** **Baja** (OHLCV-only, anclaje temporal simple).
*   **Riesgo de Overfitting:** **Bajo** (Parámetros y anclajes lógicos basados en el autor de la fuente [F1]).
*   **Riesgo de Baja Muestra:** **Bajo** (Genera señales frecuentes en mercados de rango diario).
*   **Compatibilidad FTMO:** **Alta** (Uso de stop temporal a 45 minutos y relaciones R:R de 1.5R claras).
*   **Datos Necesarios:** OHLCV en barras M5 y M1.
*   **Estado:** **Candidata**
*   **Prioridad:** **Alta**
*   **Motivo para Testear:** El desvío estadístico con límites dinámicos (`max(1.2*ATR14_M5, 1.8σ_M1)`) y filtro de tendencia `ADX < 22` es el modelo MR más limpio y con menor correlación a la estrategia activa.
*   **Motivo para Descartar:** Ninguno en esta fase.
*   **Gate Mínimo Recomendado:** **Gate 1** (Prueba de señal en train).

---

### #2. MR-02: VWAP Stretch Reversion
*   **Nombre Tentativo:** MR-02 VWAP Stretch
*   **Lógica de Mercado:** Captura reversiones rápidas cuando la cotización se desvía más de 2.25 desviaciones estándar del VWAP intradía (calculado con fallback TWAP si no hay volumen de ticks confiable) acoplado a un RSI-14 en niveles de sobrecompra/sobreventa extrema.
*   **Aplica a EURUSD Intraday:** Sí.
*   **Horario NY Probable:** 07:00 – 19:00 NY.
*   **Correlación Esperada contra `Manipulante`:** **BAJA** (Reversión a la media basada en bandas estadísticas).
*   **Dificultad de Programar:** **Baja** (Uso de VWAP estándar y RSI).
*   **Riesgo de Overfitting:** **Medio-Bajo** (Doble filtro de banda + oscilador).
*   **Riesgo de Baja Muestra:** **Bajo**.
*   **Compatibilidad FTMO:** **Alta** (Estrategia estructurada con stop loss basado en multiplicador ATR).
*   **Datos Necesarios:** M5 OHLCV + VWAP.
*   **Estado:** **Candidata**
*   **Prioridad:** **Alta** (Promovida desde B por Claude Opus para cubrir vacíos de desestimación).
*   **Motivo para Testear:** Excelente diversificador que aprovecha desvíos dinámicos en lugar de anclajes puramente temporales.
*   **Motivo para Descartar:** Ninguno.
*   **Gate Mínimo Recomendado:** **Gate 1** (Prueba de señal en train).

---

### #3. TP-01: Trend Day EMA Pullback
*   **Nombre Tentativo:** TP-01 Trend Day Pullback
*   **Lógica de Mercado:** Identificación formal de un "Día de Tendencia" (Trend Day) mediante la acción del precio de las primeras 2.5 horas de la mañana de NY. Si se confirma la fuerza, se entra en retrocesos a la EMA20 o al APM a favor del momentum dominante.
*   **Aplica a EURUSD Intraday:** Sí.
*   **Horario NY Probable:** 09:30 – 16:00 NY (Filtro cerrado hasta las 09:30 para clasificar).
*   **Correlación Esperada contra `Manipulante`:** **BAJA** (Estrategia puramente tendencial/continuación).
*   **Dificultad de Programar:** **Media** (Requiere lógica de clasificación de estado de día antes de habilitar gatillos).
*   **Riesgo de Overfitting:** **Medio-Alto** (Peligro de ajustar demasiado las condiciones de confirmación del Trend Day).
*   **Riesgo de Baja Muestra:** **Medio** (EURUSD solo presenta días de tendencia fuerte un 15-20% del tiempo).
*   **Compatibilidad FTMO:** **Alta** (Alinear operaciones con la dirección del flujo institucional reduce dramáticamente el drawdown diario).
*   **Datos Necesarios:** M5/M15 OHLCV.
*   **Estado:** **Candidata**
*   **Prioridad:** **Alta**
*   **Motivo para Testear:** Primera hipótesis formal que explota la continuación de tendencia pura en lugar de la reversión de rangos.
*   **Motivo para Descartar:** Si la definición fuente es tan restrictiva que genera menos de 30 muestras por año en el periodo de TRAIN.
*   **Gate Mínimo Recomendado:** **Gate 2** (Requiere especificación anti-lookahead validada).

---

### #4. VE-01: RV Shock Donchian Breakout
*   **Nombre Tentativo:** VE-01 RV Shock Breakout
*   **Lógica de Mercado:** Ruptura de canales Donchian de 30 minutos gatillada únicamente cuando el volumen relativo o actividad de ticks indica una expansión inusual frente a la mediana de los últimos 12 bloques y el percentil histórico.
*   **Aplica a EURUSD Intraday:** Sí.
*   **Horario NY Probable:** 07:00 – 19:00 NY.
*   **Correlación Esperada contra `Manipulante`:** **BAJA** (Ruptura/continuación de rango).
*   **Dificultad de Programar:** **Media** (Requiere cálculos de percentiles históricos de volumen por tramos horarios).
*   **Riesgo de Overfitting:** **Alto** (El stack de 3 condiciones independientes de volumen y Donchian presenta alta vulnerabilidad al ajuste de curvas).
*   **Riesgo de Baja Muestra:** **Medio**.
*   **Compatibilidad FTMO:** **Alta** (Stop loss acoplado al extremo opuesto del canal Donchian).
*   **Datos Necesarios:** M5/M30 OHLCV + Volumen relativo o Ticks negociados.
*   **Estado:** **Candidata**
*   **Prioridad:** **Alta**
*   **Motivo para Testear:** Valida si el volumen es un filtro eficaz para rechazar falsas rupturas en EURUSD.
*   **Motivo para Descartar:** Descartar si el cálculo de percentiles históricos de volumen no es estable a través de diferentes años en TRAIN.
*   **Gate Mínimo Recomendado:** **Gate 2** (Requiere motor de percentiles rodantes backward-only).

---

## 3. Backlog de 20 Ideas para Testear Posteriormente

A continuación se presenta el inventario consolidado de 20 hipótesis de trading, organizadas por orden de prioridad cuantitativa y estado de gobernanza.

```
| Rank | ID     | Nombre de Estrategia                  | Familia            | Prioridad | Estado      | Gate Mínimo |
|------|--------|----------------------------------------|--------------------|-----------|-------------|-------------|
| 1    | MR-01  | Anchor Elastic Mean Reversion          | Mean Reversion     | ALTA      | Candidata   | Gate 1      |
| 2    | MR-02  | VWAP Stretch Reversion                 | Mean Reversion     | ALTA      | Candidata   | Gate 1      |
| 3    | TP-01  | Trend Day EMA Pullback                 | Trend Pullback     | ALTA      | Candidata   | Gate 2      |
| 4    | VE-01  | RV Shock Donchian Breakout             | Volatility Expand  | ALTA      | Candidata   | Gate 2      |
| 5    | SD-02  | London Session Breakout                | Session Dynamics   | MEDIA     | Idea        | Gate 1      |
| 6    | SE-01  | Friday Reversion Flow                  | Seasonal           | MEDIA     | Idea        | Gate 1      |
| 7    | VE-02  | Bollinger Bands Squeeze Momentum       | Volatility Expand  | MEDIA     | Idea        | Gate 1      |
| 8    | TP-02  | Institutional EMA Pullback             | Trend Pullback     | MEDIA     | Idea        | Gate 2      |
| 9    | MR-04  | Keltner Snapback                       | Mean Reversion     | MEDIA     | Idea        | Gate 1      |
| 10   | VE-04  | NY Mid-Day Volatility Expansion        | Volatility Expand  | MEDIA     | Idea        | Gate 1      |
| 11   | TP-04  | Breakout-Retest Structural             | Trend Pullback     | MEDIA     | Idea        | Gate 2      |
| 12   | VE-03  | ATR Compression-Expansion              | Volatility Expand  | BAJA      | Idea        | Gate 2      |
| 13   | HY-02  | HVFTF Trend Following (SuperTrend)      | Hybrid             | BAJA      | Idea        | Gate 1      |
| 14   | HY-03  | M15 Trend + VWAP MR                    | Hybrid             | BAJA      | Idea        | Gate 2      |
| 15   | SD-01  | Europe Extreme Failure (Bid-Based)     | Session Dynamics   | MEDIA     | Diferida    | Gate 3      |
| 16   | SD-03  | Asian Range Fakeout                    | Session Dynamics   | BAJA      | Diferida    | Gate 3      |
| 17   | SD-04  | Initial Balance Failure                | Session Dynamics   | BAJA      | Diferida    | Gate 3      |
| 18   | MR-03  | London Close Mean Reversion            | Mean Reversion     | BAJA      | Diferida    | Gate 3      |
| 19   | SE-02  | London Lunch Fade                      | Seasonal           | BAJA      | Diferida    | Gate 3      |
| 20   | ED-01  | Post-News Stabilization (Event-Driven) | Event Driven       | MEDIA     | Diferida    | Gate 4      |
```

---

## 4. Major Warnings (Alertas Críticas de Investigación)

> [!WARNING]
> **1. RIESGO DE CORRELACIÓN EN SESSION DYNAMICS (SD):**
> Las ideas SD-01, SD-03 y SD-04 presentan una correlación teórica extrema con `Manipulante` debido a que todas explotan el mismo desbalance microestructural (falso breakout de extremos). Su testeo queda **diferido a Gate 3** para evitar canibalizar el margen de drawdowns de la cuenta.

> [!WARNING]
> **2. FILTRO DE SPREAD DE ALTA PRECISIÓN:**
> Muchas ideas de mean reversion de la tarde (p. ej. MR-03) o post-noticias (ED-01) optimizan sus resultados exigiendo que el `spread < 1.5 pips`. En simulación OHLCV estándar esto pasa como aprobado, pero en la realidad de FTMO, los spreads durante noticias o rollover se expanden hasta 10 pips, destruyendo la estrategia. Se prohíbe asumir spreads fijos en simulación para estas familias.

> [!CAUTION]
> **3. LOOKAHEAD BIAS EN CLASIFICACIONES HORARIAS:**
> Estrategias como TP-01 y VE-03 dependen de saber si la sesión de Londres fue "de rango" o si el día de NY será "de tendencia". Si el script de backtest utiliza datos del día completo para clasificar la mañana, los resultados serán espectacularmente falsos. El motor de clasificación debe ser estrictamente causal y cerrarse antes de permitir señales de entrada.
