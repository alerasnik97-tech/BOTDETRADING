# STRATEGY IDEA CARDS — THE COMPLETE SPECS SHEET
**Date:** 2026-05-18
**Project:** Systematic Infrastructure Professionalization — Strategy Specification Sheets
**Security Status:** READ-ONLY AUDIT & COMPILATION — NO CODE OR REPOSITORY MUTATION

---

## 1. Estructura de Ficha Estratégica

Cada ficha estratégica detalla los requerimientos e hipótesis necesarias para codificar y evaluar el modelo en el laboratorio cuantitativo de forma unívoca, sin dejar margen a la ambigüedad discrecional.

---

## 2. Fichas de Estrategias: Mean Reversion (MR)

### CARD MR05: VWAP Reversion con Z-Score
*   **ID:** MR05
*   **Nombre:** VWAP Reversion con Z-Score
*   **Hipótesis:** Las desviaciones estadísticas extremas del precio intradía respecto al VWAP acumulativo diario representan excesos temporales que tienden a corregirse rápidamente en un par altamente líquido como EURUSD.
*   **Reglas de Entrada:** 
    *   **Long:** El Z-Score del precio M1 (calculado sobre el VWAP acumulativo desde las 07:00 NY) cruza por debajo de $-2.0$. Confirmar que la vela M1 actual cierre alcista.
    *   **Short:** El Z-Score del precio M1 cruza por encima de $+2.0$. Confirmar que la vela M1 actual cierre bajista.
*   **Reglas de Salida:**
    *   **Take Profit:** Retorno exacto al nivel actual del VWAP diario.
    *   **Stop Loss:** Colocado de forma fija a una distancia de $1.5\times$ ATR(14) en M15.
    *   **Gestión:** Mover stop a Break Even cuando el precio recorra el $50\%$ de la distancia al VWAP.
*   **Parámetros:** VWAP diario reiniciado a las 07:00 NY, Z-Score Threshold: $\pm2.0$, ATR periodo: 14.
*   **Timeframes:** M1 para ejecución y Z-Score, M15 para ATR.
*   **Ventana Horaria NY:** 09:30 - 16:30 NY.
*   **Filtros:** Spread máximo permitido $\le 1.5$ pips. No operar si el ATR(14) diario es $>40$ pips.
*   **Riesgo de Overfitting:** **Bajo**. Se basa en un cálculo estadístico estándar.
*   **Sensibilidad a Costos:** **Media**. TP moderadamente corto.
*   **Clasificación de Prioridad:** **Priority A (ALTA)**.

---

### CARD MR06: RSI(2) Mean-Adjusted Reversion
*   **ID:** MR06
*   **Nombre:** RSI(2) Mean-Adjusted Reversion
*   **Hipótesis:** La sobreventa o sobrecompra en horizontes extremadamente cortos (RSI de 2 periodos) genera micro-picos que pueden ser explotados si el precio está desalineado de su tendencia inmediata de 5 minutos (EMA20).
*   **Reglas de Entrada:**
    *   **Long:** RSI(2) en M5 por debajo de $10$, y el precio M5 está cotizando por debajo de la EMA20 M5.
    *   **Short:** RSI(2) en M5 por encima de $90$, y el precio M5 está por encima de la EMA20 M5.
*   **Reglas de Salida:**
    *   **Take Profit:** Cruce del precio con la EMA20 M5.
    *   **Stop Loss:** Colocado a $1.0\times$ ATR(14) M5 de la entrada.
    *   **Gestión:** Cierre del $50\%$ de la posición si el precio alcanza la EMA10.
*   **Parámetros:** RSI periodo: 2, EMA periodo: 20, ATR periodo: 14.
*   **Timeframes:** M5 para todo.
*   **Ventana Horaria NY:** 08:00 - 17:00 NY.
*   **Filtros:** Operar solo si ATR(14) M5 es $\ge 5$ pips para evitar mercados planos.
*   **Riesgo de Overfitting:** **Medio**. Alta sensibilidad a los parámetros del RSI.
*   **Sensibilidad a Costos:** **Alta**. Gran frecuencia y márgenes pequeños.
*   **Clasificación de Prioridad:** **Priority B (MEDIA)**.

---

### CARD MR07: Statistical Reversion from Multi-Day MA
*   **ID:** MR07
*   **Nombre:** Statistical Reversion from Multi-Day MA
*   **Hipótesis:** El EURUSD intradía tiende a revertir hacia su media móvil simple de 20 días cuando se estira excesivamente, ofreciendo una resistencia dinámica de gran fiabilidad.
*   **Reglas de Entrada:**
    *   **Long:** Cierre de la vela M15 a una distancia $\ge 2.0$ desviaciones estándar por debajo de la SMA de 20 días.
    *   **Short:** Cierre de la vela M15 a una distancia $\ge 2.0$ desviaciones estándar por encima de la SMA de 20 días.
*   **Reglas de Salida:**
    *   **Take Profit:** Retorno exacto a la SMA de 20 días o salida horaria a las 19:00 NY.
    *   **Stop Loss:** Stop estático a $50$ pips de la entrada.
*   **Parámetros:** SMA periodo: 20 días, Desviaciones estándar: 2.0.
*   **Timeframes:** M15.
*   **Ventana Horaria NY:** 07:00 - 19:00 NY.
*   **Filtros:** Excluir días festivos bancarios.
*   **Riesgo de Overfitting:** **Bajo**. Parámetros de largo plazo reducen el ajuste.
*   **Sensibilidad a Costos:** **Baja**. TP muy amplio.
*   **Clasificación de Prioridad:** **Priority B (MEDIA)**.

---

### CARD MR08: Bollinger Bands "Double Tap" Divergence
*   **ID:** MR08
*   **Nombre:** Bollinger Bands "Double Tap" Divergence
*   **Hipótesis:** Un doble toque en las bandas de Bollinger extremas acompañado por una divergencia alcista/bajista en el RSI de 15 minutos marca un agotamiento severo del flujo de órdenes.
*   **Reglas de Entrada:**
    *   **Long:** El precio M5 toca la banda inferior de Bollinger, rebota, y vuelve a tocarla (Double Tap) en un rango de 15-30 minutos, mientras el RSI(14) en M15 registra un mínimo más alto que en el primer toque.
    *   **Short:** El precio M5 toca la banda superior de Bollinger, rebota, y vuelve a tocarla, mientras el RSI(14) M15 registra un máximo más bajo.
*   **Reglas de Salida:**
    *   **Take Profit:** Cierre del precio en la banda media de Bollinger.
    *   **Stop Loss:** Colocado 3 pips por debajo del mínimo del doble toque.
*   **Parámetros:** Bollinger Bands (20, 2), RSI(14) en M15.
*   **Timeframes:** M5 y M15.
*   **Ventana Horaria NY:** 08:30 - 16:30 NY.
*   **Filtros:** Spread máximo $\le 1.2$ pips.
*   **Riesgo de Overfitting:** **Alto**. Programar la divergencia matemática sin lookahead bias es complejo.
*   **Sensibilidad a Costos:** **Media**.
*   **Clasificación de Prioridad:** **Priority B (MEDIA)**.

---

### CARD MR17: London Close Mean Reversion VWAP (LCMR-VWAP)
*   **ID:** MR17
*   **Nombre:** London Close Mean Reversion VWAP (LCMR-VWAP)
*   **Hipótesis:** El cierre de la sesión de Londres (11:30 - 12:00 NY) genera la liquidación de carteras y el reequilibrio de posiciones de divisas por parte de tesorerías europeas, lo que fuerza estadísticamente al EURUSD a retornar a su VWAP diario.
*   **Reglas de Entrada:**
    *   **Long:** En la ventana de liquidación, el precio M5 cierra $\ge 5$ pips por debajo del VWAP diario. Confirmar vela M1 alcista.
    *   **Short:** El precio M5 cierra $\ge 5$ pips por encima del VWAP diario. Confirmar vela M1 bajista.
*   **Reglas de Salida:**
    *   **Take Profit:** Objetivo fijo de 8 pips (con cierre parcial del $50\%$ a los 4 pips y trailing stop de 3 pips).
    *   **Stop Loss:** Fijo e irrevocable a 10 pips de la entrada.
    *   **Time Stop:** Cierre obligatorio de toda posición a las 16:30 NY.
*   **Parámetros:** Ventana operativa: 09:30 - 16:30 NY, Desviación mínima: 5 pips.
*   **Timeframes:** M5 para señales, M1 para confirmación.
*   **Ventana Horaria NY:** 09:30 - 16:30 NY.
*   **Filtros:** Spread $\le 1.5$ pips. ATR(14) diario $< 40$ pips.
*   **Riesgo de Overfitting:** **Bajo**. Fenómeno basado en flujos físicos de liquidación.
*   **Sensibilidad a Costos:** **Media-Alta** (TP corto de 8 pips).
*   **Clasificación de Prioridad:** **Priority A (ALTA)**.

---

### CARD MR03: Macro Pivot Points Statistical Reversion [NEW]
*   **ID:** MR03
*   **Nombre:** Macro Pivot Points Statistical Reversion
*   **Hipótesis:** Los Pivot Points calculados sobre el marco de tiempo semanal actúan como potentes imanes y soportes/resistencias en el EURUSD intradía. Una excursión rápida hasta el soporte S2 o resistencia R2 tiende a revertir.
*   **Reglas de Entrada:**
    *   **Long:** El precio M5 toca el soporte semanal S2 y el RSI(14) en M15 está por debajo de $20$.
    *   **Short:** El precio M5 toca la resistencia semanal R2 y el RSI(14) M15 está por encima de $80$.
*   **Reglas de Salida:**
    *   **Take Profit:** Retorno al Pivot Point Semanal Central.
    *   **Stop Loss:** Colocado a $15$ pips del nivel S2/R2.
*   **Parámetros:** Pivot Points Semanales Clásicos, RSI: 14 periodos.
*   **Timeframes:** M5 y M15.
*   **Ventana Horaria NY:** 07:00 - 19:00 NY.
*   **Filtros:** No operar si el mercado cotiza con spread $> 1.5$ pips.
*   **Riesgo de Overfitting:** **Bajo**. Pivots semanales no cambian dinámicamente durante la sesión.
*   **Sensibilidad a Costos:** **Baja**. TP amplios.
*   **Clasificación de Prioridad:** **Priority B (MEDIA)**.

---

### CARD MR05_2: Volume Profile Point of Control Reversion [NEW]
*   **ID:** MR05_2
*   **Nombre:** Volume Profile Point of Control Reversion
*   **Hipótesis:** El Point of Control (POC) del perfil de volumen del día anterior representa el precio más justo y aceptado. Las excursiones del precio al inicio del día tienden a ser atraídas de vuelta a este nivel.
*   **Reglas de Entrada:**
    *   **Long:** El precio cotiza a una distancia $\ge 15$ pips por debajo del POC del día anterior durante la sesión americana y el precio M15 cierra con vela de martillo alcista.
    *   **Short:** El precio cotiza $\ge 15$ pips por encima del POC del día anterior y cierra con vela de estrella fugaz M15.
*   **Reglas de Salida:**
    *   **Take Profit:** Retorno exacto al POC del día anterior.
    *   **Stop Loss:** Colocado a $10$ pips de la entrada.
*   **Parámetros:** POC diario calculado a las 17:00 NY del día anterior.
*   **Timeframes:** M15.
*   **Ventana Horaria NY:** 08:00 - 15:00 NY.
*   **Filtros:** Solo operar si la sesión anterior fue de perfil equilibrado (rango).
*   **Riesgo de Overfitting:** **Medio**. Requiere perfiles de volumen dinámicos.
*   **Sensibilidad a Costos:** **Media**.
*   **Clasificación de Prioridad:** **Priority B (MEDIA)**.

---

### CARD MR04: EMA 200/50 Cross Counter-Trend Fade [NEW]
*   **ID:** MR04
*   **Nombre:** EMA 200/50 Cross Counter-Trend Fade
*   **Hipótesis:** Los cruces de la EMA50 y EMA200 en gráficos M5 a menudo marcan el clímax y agotamiento de un movimiento de tendencia intradía en EURUSD cuando el ADX general muestra debilidad extrema.
*   **Reglas de Entrada:**
    *   **Long:** Ocurre un cruce bajista de la EMA50 por debajo de la EMA200 en M5, pero el ADX(14) en M15 es $< 15$. Entrar largo inmediatamente.
    *   **Short:** Ocurre un cruce alcista de la EMA50 por encima de la EMA200 en M5, pero el ADX(14) M15 es $< 15$. Entrar corto.
*   **Reglas de Salida:**
    *   **Take Profit:** Retorno exacto a la EMA50 M5.
    *   **Stop Loss:** Colocado a $1.5\times$ ATR(14) M5 de la entrada.
*   **Parámetros:** EMA50, EMA200, ADX(14).
*   **Timeframes:** M5 y M15.
*   **Ventana Horaria NY:** 09:00 - 16:00 NY.
*   **Filtros:** Spread $\le 1.2$ pips.
*   **Riesgo de Overfitting:** **Alto** (Fading de cruces es de alta peligrosidad).
*   **Sensibilidad a Costos:** **Alta**.
*   **Clasificación de Prioridad:** **Priority D (EXCLUDED)**.

---

### CARD MR01: London Open False Breakout (LO-FBO) [NEW]
*   **ID:** MR01
*   **Nombre:** London Open False Breakout (LO-FBO)
*   **Hipótesis:** La apertura de la sesión de Londres barre el máximo o mínimo de la consolidación de Tokio antes de revertir con violencia hacia el valor real del día.
*   **Reglas de Entrada:**
    *   **Long:** El precio en M5 rompe por debajo del mínimo de la sesión de Tokio en $\le 5$ pips en la primera media hora de Londres, cerrando una vela M5 dentro del rango previo.
    *   **Short:** El precio en M5 rompe por encima del máximo de Tokio en $\le 5$ pips y cierra una vela M5 dentro del rango previo.
*   **Reglas de Salida:**
    *   **Take Profit:** Cierre en el punto medio del rango de Tokio.
    *   **Stop Loss:** Colocado 5 pips por debajo del mínimo/máximo del barrido.
*   **Parámetros:** Rango de Tokio definido entre las 19:00 y las 02:00 NY.
*   **Timeframes:** M5.
*   **Ventana Horaria NY:** 03:00 - 05:00 NY.
*   **Filtros:** Operar solo si el rango de Tokio fue $<25$ pips.
*   **Riesgo de Overfitting:** **Medio**.
*   **Sensibilidad a Costos:** **Baja**.
*   **Clasificación de Prioridad:** **Priority C (DEFERRED)**.

---

### CARD MR02: Asia Range Expansion Hook (ARE-Hook) [NEW]
*   **ID:** MR02
*   **Nombre:** Asia Range Expansion Hook (ARE-Hook)
*   **Hipótesis:** Si la sesión de Nueva York abre sin noticias económicas clave, cualquier expansión rápida fuera del rango asiático tiende a corregir hacia el VWAP asiático debido a la falta de flujos direccionales.
*   **Reglas de Entrada:**
    *   **Long:** El precio cae $>15$ pips por debajo del mínimo asiático a las 08:30 NY sin noticias macro en agenda. Entrar largo en vela M5 alcista.
    *   **Short:** El precio sube $>15$ pips por encima del máximo asiático a las 08:30 NY. Entrar corto en vela M5 bajista.
*   **Reglas de Salida:**
    *   **Take Profit:** Cierre en el VWAP asiático.
    *   **Stop Loss:** Colocado a $10$ pips.
*   **Parámetros:** Rango asiático (22:00-07:00 NY).
*   **Timeframes:** M5.
*   **Ventana Horaria NY:** 08:30 - 11:30 NY.
*   **Filtros:** Spread máximo $\le 1.5$ pips.
*   **Riesgo de Overfitting:** **Medio**.
*   **Sensibilidad a Costos:** **Media**.
*   **Clasificación de Prioridad:** **Priority C (DEFERRED)**.

---

## 3. Fichas de Estrategias: Trend Pullback (TP)

### CARD TP12: Trend Pullback Institutional EMA ATR
*   **ID:** TP12
*   **Nombre:** Trend Pullback Institutional EMA ATR (TP-EMA-ATR)
*   **Hipótesis:** Durante tendencias intradiarias sólidas en gráficos de 15 minutos, los retrocesos del precio a la EMA de 50 períodos en gráficos de 5 minutos ofrecen entradas de alta probabilidad a favor del momentum global.
*   **Reglas de Entrada:**
    *   **Long:** La tendencia en M15 es alcista (EMA50 > EMA200). El precio en M5 retrocede y toca o cruza la EMA50 en M5 (sincronizada desde M15). La vela M5 cierra por encima de su mínimo y el precio M5 supera el máximo de las últimas 3 velas M5.
    *   **Short:** Tendencia M15 bajista (EMA50 < EMA200). El precio M5 toca la EMA50 en M5. Vela M5 cierra por debajo de su máximo y el precio M5 cae por debajo del mínimo de las últimas 3 velas.
*   **Reglas de Salida:**
    *   **Take Profit:** Objetivo fijo a $2.5\times$ la distancia del Stop Loss (R:R 2.5:1).
    *   **Stop Loss:** Colocado a una distancia basada en ATR en M5:
        *   Long: Mínimo local de las últimas 3 velas M5 $- (1.5\times$ ATR(14) M5).
        *   Short: Máximo local de las últimas 3 velas M5 $+ (1.5\times$ ATR(14) M5).
    *   **Gestión:** Tras alcanzar $1.5\times$ de beneficio, mover stop a Break Even.
*   **Parámetros:** EMA50, EMA200, ATR periodo: 14.
*   **Timeframes:** M5 para señales y disparos, M15 para tendencia global.
*   **Ventana Horaria NY:** 08:00 - 17:00 NY.
*   **Filtros:** Spread máximo $\le 2.0$ pips. ATR(14) M5 $\ge 6$ pips. Evitar noticias $\pm30$ minutos.
*   **Riesgo de Overfitting:** **Medio**. Estrategia clásica institucional.
*   **Sensibilidad a Costos:** **Media**.
*   **Clasificación de Prioridad:** **Priority A (ALTA)**.

---

### CARD TP13: Trend Pullback ADX-Fib 61.8%
*   **ID:** TP13
*   **Nombre:** Trend Pullback ADX-Fib 61.8%
*   **Hipótesis:** Las correcciones que alcanzan exactamente el retroceso de Fibonacci del $61.8\%$ del impulso intradía anterior ofrecen un punto de pivote estructural con una probabilidad de continuación del $60\%$ si el ADX M15 marca tendencia fuerte.
*   **Reglas de Entrada:**
    *   **Long:** ADX(14) en M15 es $>25$ y DI+ > DI-. Identificar el impulso M5 desde el Swing Low A al Swing High B. El precio en M5 retrocede y toca la zona del $61.8\%$ de Fib. Entrar largo al cierre de la primera vela M5 alcista confirmatoria.
    *   **Short:** ADX(14) M15 es $>25$ y DI- > DI+. Impulso bajista M5 A-B. El precio retrocede al $61.8\%$. Entrar corto en vela M5 bajista.
*   **Reglas de Salida:**
    *   **Take Profit:** Objetivo a $1.5\times$ la distancia del Stop Loss (7.5 pips mínimo).
    *   **Stop Loss:** Fijo a 5 pips por debajo del Swing Low A (para largos) o 5 pips por encima del Swing High A (para cortos).
*   **Parámetros:** ADX(14), Nivel Fib: 0.618, SL offset: 5 pips.
*   **Timeframes:** M5 para impulsos, M15 para ADX.
*   **Ventana Horaria NY:** 09:30 - 16:30 NY.
*   **Filtros:** Spread máximo $\le 1.5$ pips. ATR(14) M5 $> 3$ pips.
*   **Riesgo de Overfitting:** **Medio-Alto** (Definición automática de Swings).
*   **Sensibilidad a Costos:** **Media**.
*   **Clasificación de Prioridad:** **Priority B (MEDIA)**.

---

### CARD TP14: Trend Pullback Breakout-Retest
*   **ID:** TP14
*   **Nombre:** Trend Pullback Breakout-Retest
*   **Hipótesis:** La ruptura de los extremos de la sesión anterior (máximos/mínimos diarios) confirma un cambio de flujo institucional; el retesteo posterior del nivel roto funciona como el pullback perfecto para incorporarse a la tendencia.
*   **Reglas de Entrada:**
    *   **Long:** Cierre de vela M5 por encima del máximo diario de la sesión anterior. Esperar que el precio retroceda y retestee el nivel roto en un rango de $\pm3$ pips. Entrar largo al cierre de la primera vela M1 alcista con sombra inferior $\le 50\%$.
    *   **Short:** Cierre de vela M5 por debajo del mínimo diario de la sesión anterior. Esperar retesteo del nivel en $\pm3$ pips. Entrar corto al cierre de vela M1 bajista con sombra superior $\le 50\%$.
*   **Reglas de Salida:**
    *   **Take Profit:** Fijo en 15 pips de la entrada.
    *   **Stop Loss:** Fijo en 10 pips de la entrada.
    *   **Gestión:** Mover stop a Break Even al alcanzar +7.5 pips.
*   **Parámetros:** Niveles diarios calculados a las 17:00 NY. Tolerancia: 3 pips.
*   **Timeframes:** M5 para breakouts, M1 para retesteo.
*   **Ventana Horaria NY:** 08:00 - 17:00 NY.
*   **Filtros:** ATR(14) M5 $\ge 5$ pips. Spread $\le 2.0$ pips.
*   **Riesgo de Overfitting:** **Bajo-Medio**. Basado en niveles estructurales estables.
*   **Sensibilidad a Costos:** **Media-Alta** (Stops cortos).
*   **Clasificación de Prioridad:** **Priority A (ALTA)**.

---

## 4. Fichas de Estrategias: Volatility Expansion (VE)

### CARD VE01: ORB Volatility ATR Threshold
*   **ID:** VE01
*   **Nombre:** ORB Volatility ATR Threshold (ORB-ATR)
*   **Hipótesis:** La ruptura del Opening Range (rango de las primeras 2 horas de negociación en Nueva York, 07:00-09:00 NY) indica la dirección predominante de los flujos institucionales del día si se acompaña de una volatilidad inicial suficiente.
*   **Reglas de Entrada:**
    *   **Long:** Ruptura alcista del máximo del Opening Range (07:00-09:00 NY) por cierre de vela M15. Confirmar que el ATR(14) en M15 al cierre de las 09:00 sea superior a 5 pips.
    *   **Short:** Ruptura bajista del mínimo del Opening Range por cierre M15. Confirmar que el ATR(14) M15 sea $\ge 5$ pips.
*   **Reglas de Salida:**
    *   **Take Profit:** $2.0\times$ la distancia del Stop Loss (aproximadamente 20 pips).
    *   **Stop Loss:** $1.0\times$ ATR(14) M15 por debajo del máximo del rango (para largos) o por encima del mínimo (para cortos).
    *   **Time Stop:** Cierre forzado de la posición a las 19:00 NY.
*   **Parámetros:** Opening Range: 07:00-09:00 NY, ATR periodo: 14, Multiplicador: 1.0.
*   **Timeframes:** M15.
*   **Ventana Horaria NY:** 09:00 - 12:00 NY.
*   **Filtros:** Spread máximo $\le 1.5$ pips. No operar si el rango es $>30$ pips.
*   **Riesgo de Overfitting:** **Bajo**. Modelo clásico muy robusto.
*   **Sensibilidad a Costos:** **Baja**. Captura tendencias amplias de la tarde.
*   **Clasificación de Prioridad:** **Priority A (ALTA)**.

---

### CARD VE02: Bollinger Band Squeeze & ADX
*   **ID:** VE02
*   **Nombre:** Bollinger Band Squeeze & ADX
*   **Hipótesis:** Los períodos de extrema compresión de precios en EURUSD se resuelven mediante expansiones rápidas y direccionales de volatilidad que pueden ser capturadas al inicio del momentum.
*   **Reglas de Entrada:**
    *   **Long:** Las bandas de Bollinger M5 se comprimen a un ancho de banda (Bandwidth) por debajo del $20\%$ del promedio de las últimas 100 velas M5. El precio en M5 rompe la banda superior y el ADX(14) en M15 es $>25$.
    *   **Short:** Bandwidth M5 por debajo de su percentil $20\%$. El precio rompe la banda inferior M5 y el ADX(14) M15 es $>25$.
*   **Reglas de Salida:**
    *   **Take Profit:** Múltiplo de $2.0\times$ la distancia de las bandas en el momento de la ruptura.
    *   **Stop Loss:** Colocado en la banda de Bollinger contraria en el momento de la entrada.
*   **Parámetros:** Bollinger Bands (20, 2), ADX(14) en M15, percentil del bandwidth: $20\%$.
*   **Timeframes:** M5 y M15.
*   **Ventana Horaria NY:** 08:30 - 15:00 NY.
*   **Filtros:** Spread máximo $\le 1.5$ pips.
*   **Riesgo de Overfitting:** **Alto** (Peligro de ajustar demasiado los percentiles del squeeze).
*   **Sensibilidad a Costos:** **Media**.
*   **Clasificación de Prioridad:** **Priority A (ALTA)**.

---

### CARD VE03: Volatility Expansion Keltner Breakout
*   **ID:** VE03
*   **Nombre:** Volatility Expansion Keltner Breakout
*   **Hipótesis:** Las rupturas dinámicas del Canal de Keltner acompañadas por volumen relativo alto confirman la presencia de dinero institucional activo y momentum sostenido.
*   **Reglas de Entrada:**
    *   **Long:** Cierre de vela M5 por encima de la banda superior del Keltner Channel durante la superposición Londres-NY. Confirmar que el volumen relativo de la vela sea $>1.5\times$ el volumen promedio de las últimas 20 velas M5.
    *   **Short:** Cierre de vela M5 por debajo del Keltner Channel inferior. Volumen relativo $>1.5\times$ el promedio de las últimas 20 velas.
*   **Reglas de Salida:**
    *   **Take Profit:** Fijo en $2.5\times$ la distancia ATR del canal.
    *   **Stop Loss:** Cierre de vela M5 por debajo de la línea central de Keltner (20 EMA).
*   **Parámetros:** Keltner Channel (20 EMA, 10 ATR, 1.5 multiplier), Volumen periodo: 20.
*   **Timeframes:** M5.
*   **Ventana Horaria NY:** 07:00 - 11:00 NY.
*   **Filtros:** Spread máximo $\le 1.5$ pips.
*   **Riesgo de Overfitting:** **Medio**. Canales dinámicos reducen el ajuste.
*   **Sensibilidad a Costos:** **Media**.
*   **Clasificación de Prioridad:** **Priority A (ALTA)**.

---

### CARD VE04: Donchian Breakout + VWAP Confirmation
*   **ID:** VE04
*   **Nombre:** Donchian Breakout + VWAP Confirmation
*   **Hipótesis:** Las rupturas de rangos de 20 períodos en Donchian son altamente fiables cuando la pendiente intrabarra del VWAP confirma que la acumulación de volumen institucional apoya el movimiento.
*   **Reglas de Entrada:**
    *   **Long:** El precio en M5 rompe el Donchian Channel superior de 20 periodos, y el precio M1 actual está cotizando por encima del VWAP diario y este tiene pendiente alcista.
    *   **Short:** El precio M5 rompe el Donchian inferior de 20 periodos, cotizando el precio M1 por debajo del VWAP con pendiente bajista.
*   **Reglas de Salida:**
    *   **Take Profit:** Fijo en $1.5\times$ la distancia del canal en el breakout.
    *   **Stop Loss:** Cierre de vela M5 por debajo del Donchian central (punto medio).
*   **Parámetros:** Donchian Channel: 20 periodos, VWAP diario.
*   **Timeframes:** M5 para Donchian, M1 para VWAP.
*   **Ventana Horaria NY:** 08:00 - 15:00 NY.
*   **Filtros:** Spread $\le 1.5$ pips.
*   **Riesgo de Overfitting:** **Bajo-Medio**.
*   **Sensibilidad a Costos:** **Media**.
*   **Clasificación de Prioridad:** **Priority A (ALTA)**.

---

### CARD VE18: NY Mid-Day Volatility Expansion Breakout
*   **ID:** VE18
*   **Nombre:** NY Mid-Day Volatility Expansion Breakout
*   **Hipótesis:** La consolidación de precios y volumen que ocurre durante el almuerzo de Wall Street (11:30-12:00 NY) crea un rango comprimido cuya ruptura posterior marca la dirección de la tarde con gran fiabilidad.
*   **Reglas de Entrada:**
    *   **Long:** El precio en M1 rompe el máximo del rango consolidado entre las 11:30 y 12:00 NY. Confirmar que el rango total haya sido $< 0.5\times$ ATR(14) M5.
    *   **Short:** El precio en M1 rompe el mínimo del rango consolidado entre las 11:30 y 12:00 NY.
*   **Reglas de Salida:**
    *   **Take Profit:** Objetivo inicial a $2.0\times$ ATR(14) M5.
    *   **Stop Loss:** $1.0\times$ ATR(14) M5 de la entrada.
    *   **Gestión:** Mover stop a Break Even al alcanzar +1.0 ATR de ganancia. Trailing de 0.5 ATR.
    *   **Time Stop:** Cierre obligatorio a las 15:00 NY.
*   **Parámetros:** Ventana rango: 11:30-12:00 NY, Ventana trading: 12:00-14:00 NY, ATR periodo: 14.
*   **Timeframes:** M5 para ATR y rango, M1 para disparo.
*   **Ventana Horaria NY:** 12:00 - 14:00 NY.
*   **Filtros:** Spread máximo $\le 1.5$ pips. Evitar noticias de alta volatilidad.
*   **Riesgo de Overfitting:** **Bajo**. Estructura basada puramente en el reloj biológico del mercado.
*   **Sensibilidad a Costos:** **Media**.
*   **Clasificación de Prioridad:** **Priority A (ALTA)**.

---

## 5. Fichas de Estrategias: Session Dynamics & Seasonal

### CARD SD09: London Session H/L Breakout
*   **ID:** SD09
*   **Nombre:** London Session H/L Breakout
*   **Hipótesis:** La ruptura de los extremos de la sesión europea de Londres indica la fuerza real de los participantes del día antes de la apertura de Nueva York.
*   **Reglas de Entrada:**
    *   **Long:** Ruptura por cierre M5 del máximo de la sesión de Londres (03:00-12:00 UTC) filtrada por un rango de pre-sesión estrecho ($\le0.3\times$ ATR).
    *   **Short:** Ruptura del mínimo de la sesión de Londres.
*   **Reglas de Salida:**
    *   **Take Profit:** $1.5\times$ la distancia del rango de Londres.
    *   **Stop Loss:** Colocado a $10$ pips.
*   **Parámetros:** Sesión de Londres (03:00-12:00 UTC).
*   **Timeframes:** M5.
*   **Ventana Horaria NY:** 07:00 - 10:00 NY.
*   **Filtros:** Spread $\le 1.5$ pips.
*   **Riesgo de Overfitting:** **Medio**.
*   **Sensibilidad a Costos:** **Media**.
*   **Clasificación de Prioridad:** **Priority C (DEFERRED)**.

---

### CARD SD10: Asian Range Liquidity Fakeout
*   **ID:** SD10
*   **Nombre:** Asian Range Liquidity Fakeout
*   **Hipótesis:** Las falsas rupturas de los límites de la consolidación de Tokio son generadas por instituciones para recolectar stop-loss minoristas antes de arrancar la sesión de Nueva York.
*   **Reglas de Entrada:**
    *   **Long:** Ruptura a la baja del mínimo asiático por $\le 5$ pips seguido por vela M1 alcista de fuerte cuerpo.
    *   **Short:** Ruptura al alza del máximo asiático por $\le 5$ pips seguido por vela M1 bajista de fuerte cuerpo.
*   **Reglas de Salida:**
    *   **Take Profit:** Cierre en el VWAP de la sesión asiática.
    *   **Stop Loss:** Colocado 3 pips por debajo del mínimo del barrido.
*   **Parámetros:** Horario Tokio: 22:00-07:00 NY. Buffer: 5 pips.
*   **Timeframes:** M1 para entradas de precisión.
*   **Ventana Horaria NY:** 07:00 - 11:30 NY.
*   **Filtros:** Spread máximo $\le 1.5$ pips.
*   **Riesgo de Overfitting:** **Alto** (Peligro de sesgo retrospectivo).
*   **Sensibilidad a Costos:** **Alta** (Entrada en momentos tensos).
*   **Clasificación de Prioridad:** **Priority D (EXCLUDED)**.

---

### CARD SD11: NY Opening Reversal (Initial Balance Failure)
*   **ID:** SD11
*   **Nombre:** NY Opening Reversal (Initial Balance Failure)
*   **Hipótesis:** El Initial Balance (primera hora y media de Wall Street) define la frontera de liquidez. Las rupturas fallidas que retornan velozmente dentro del rango marcan la reversión del día.
*   **Reglas de Entrada:**
    *   **Long:** El precio rompe el mínimo del Initial Balance y en las siguientes 2 velas M5 vuelve a cerrar dentro del Initial Balance. Entrar largo.
    *   **Short:** El precio rompe el máximo del Initial Balance y vuelve a cerrar por debajo del máximo en 2 velas M5. Entrar corto.
*   **Reglas de Salida:**
    *   **Take Profit:** Target de $1.0\times$ ATR M5 al extremo opuesto del rango.
    *   **Stop Loss:** Colocado a $0.5\times$ ATR M5 del extremo roto.
*   **Parámetros:** Initial Balance start: 07:00 NY, end: 08:30 NY. Velas fallo: 2 M5.
*   **Timeframes:** M5.
*   **Ventana Horaria NY:** 08:30 - 12:00 NY.
*   **Filtros:** ATR(14) M5 $\ge 5$ pips. Spread $\le 1.5$ pips.
*   **Riesgo de Overfitting:** **Medio-Alto**.
*   **Sensibilidad a Costos:** **Media**.
*   **Clasificación de Prioridad:** **Priority D (EXCLUDED)**.

---

### CARD SE07: Friday Weekly Roll Close Fade (FWRC-Fade) [NEW]
*   **ID:** SE07
*   **Nombre:** Friday Weekly Roll Close Fade (FWRC-Fade)
*   **Hipótesis:** Las tardes de los viernes (14:00 - 16:30 NY) se caracterizan por el cierre masivo de libros y la toma de beneficios por parte de traders institucionales, induciendo a un "fade" o reversión estadística de la tendencia semanal dominante.
*   **Reglas de Entrada:**
    *   **Long:** El par EURUSD cotiza en una semana fuertemente bajista (precio actual por debajo del Open de la semana en $>60$ pips). Entrar largo en la apertura de las 14:00 NY del viernes.
    *   **Short:** El par cotiza en una semana fuertemente alcista (precio actual por encima del Open de la semana en $>60$ pips). Entrar corto en la apertura de las 14:00 NY del viernes.
*   **Reglas de Salida:**
    *   **Take Profit:** Retorno equivalente al $30\%$ del rango total de la semana.
    *   **Stop Loss:** Colocado de forma fija a una distancia de $20$ pips de la entrada.
    *   **Time Stop:** Cierre forzado irrevocable a las 16:30 NY (cierre de mercado antes del fin de semana).
*   **Parámetros:** Open semanal tomado a las 17:00 NY del domingo anterior. Rango mínimo: 60 pips.
*   **Timeframes:** M15.
*   **Ventana Horaria NY:** Viernes 14:00 - 16:30 NY.
*   **Filtros:** No operar si el spread del viernes por la tarde supera los 2.0 pips.
*   **Riesgo de Overfitting:** **Bajo**. Fenómeno conductual institucional muy persistente.
*   **Sensibilidad a Costos:** **Baja** (Opera rangos medianos).
*   **Clasificación de Prioridad:** **Priority B (MEDIA)**.

---

### CARD SE08: Day-of-Week Trend Anomaly Continuation (DOW-TAC) [NEW]
*   **ID:** SE08
*   **Nombre:** Day-of-Week Trend Anomaly Continuation
*   **Hipótesis:** Históricamente, los martes y jueves son los días de mayor momentum y direccionalidad real en el par EURUSD. Las rupturas iniciales de la sesión de Londres en estos días tienden a prolongarse con fuerza durante la sesión NY sin retrocesos profundos.
*   **Reglas de Entrada:**
    *   **Long:** Solo en días martes o jueves. El precio M15 rompe el máximo del rango asiático durante la sesión europea y se mantiene por encima del rango a las 08:00 NY. Entrar largo.
    *   **Short:** Solo en días martes o jueves. El precio M15 rompe el mínimo del rango asiático y se mantiene por debajo a las 08:00 NY. Entrar corto.
*   **Reglas de Salida:**
    *   **Take Profit:** Fijo a 25 pips de la entrada.
    *   **Stop Loss:** Fijo a 15 pips de la entrada.
*   **Parámetros:** Días de operación: Martes (2) y Jueves (4) en calendario.
*   **Timeframes:** M15.
*   **Ventana Horaria NY:** 08:00 - 15:00 NY.
*   **Filtros:** Spread $\le 1.5$ pips. No operar si hay noticias de alto impacto el resto del día.
*   **Riesgo de Overfitting:** **Medio**. La anomalía horaria/semanal debe validarse en out-of-sample amplio.
*   **Sensibilidad a Costos:** **Baja**.
*   **Clasificación de Prioridad:** **Priority B (MEDIA)**.

---

## 6. Fichas de Estrategias: Event-Driven & Hybrids

### CARD ED15: Post-News Volatility Reversion
*   **ID:** ED15
*   **Nombre:** Post-News Volatility Reversion
*   **Hipótesis:** Tras la inyección de volatilidad de una noticia macro catalogada como "High Impact", el precio sobrerreacciona de forma temporal y tiende a revertir hacia la media local pre-noticia a los 10-120 minutos.
*   **Reglas de Entrada:**
    *   **Long:** Noticia relevante gatillada a t0. ATR(14) M5 en los 10 min posteriores se duplica en comparación con el ATR_pre. En M1, una vela cerrada bajista con rango $> 1.5\times$ ATR_pre es rota en su máximo por la siguiente vela M1. Entrar largo en M1 close.
    *   **Short:** Noticia relevante t0. Spike de ATR $> 2\times$. Vela M1 cerrada alcista con rango $>1.5\times$ es rota en su mínimo por la siguiente vela M1. Entrar corto en M1 close.
*   **Reglas de Salida:**
    *   **Take Profit:** $0.75\times$ ATR_pre de la entrada (R:R 1:0.75).
    *   **Stop Loss:** $1.0\times$ ATR_pre de la entrada.
    *   **Gestión:** Mover stop a BE al alcanzar $+0.25\times$ ATR_pre. Cierre por tiempo a los 120 minutos.
*   **Parámetros:** ATR(14) M5, Multiplicador spike: 2.0, Multiplicador vela: 1.5.
*   **Timeframes:** M1 para entrada, M5 para ATR.
*   **Ventana Horaria NY:** 10 a 120 minutos post-noticia.
*   **Filtros:** Spread máximo $\le 1.5$ pips.
*   **Riesgo de Overfitting:** **Medio-Alto**.
*   **Sensibilidad a Costos:** **Alta** (Entrada en momentos tensos).
*   **Clasificación de Prioridad:** **Priority D (EXCLUDED)**.

---

### CARD ED16: Post-News Momentum Continuation (PNMC-15)
*   **ID:** ED16
*   **Nombre:** Post-News Momentum Continuation (PNMC-15)
*   **Hipótesis:** El primer impulso posterior a una noticia relevante a menudo se consolida y descansa durante aproximadamente 15 minutos (fase de absorción de spread y liquidez) antes de reanudarse en la misma dirección.
*   **Reglas de Entrada:**
    *   **Long:** La noticia M1 cierra con rango $>1.5\times$ ATR_1m alcista. Esperar 15 minutos de estabilización (desviación estándar de M1 en ese lapso $\le 0.5\times$ ATR_1m). El precio M15 posterior cierra por encima del máximo de los 15 minutos de estabilización. Entrar largo.
    *   **Short:** La noticia M1 cierra con rango bajista $>1.5\times$ ATR_1m. Estabilización de 15 minutos con desviación estándar $\le 0.5\times$ ATR_1m. Cierre M15 posterior por debajo del mínimo de la estabilización. Entrar corto.
*   **Reglas de Salida:**
    *   **Take Profit:** $2.0\times$ ATR(15min) de la entrada.
    *   **Stop Loss:** $1.0\times$ ATR(15min) de la entrada.
    *   **Gestión:** Mover stop a Break Even tras alcanzar $1.0\times$ ATR(15min) de beneficio.
    *   **Time Stop:** Cierre forzado a las 4 horas de la entrada o 19:00 NY.
*   **Parámetros:** Ventana de estabilización: 15 minutos, ATR(1m), ATR(15m).
*   **Timeframes:** M1 para estabilización, M15 para disparo y ATR.
*   **Ventana Horaria NY:** 07:30 - 15:00 NY.
*   **Filtros:** Spread máximo $\le 1.5$ pips. ATR(15m) $> 4$ pips.
*   **Riesgo de Overfitting:** **Medio**.
*   **Sensibilidad a Costos:** **Media-Alta**.
*   **Clasificación de Prioridad:** **Priority D (EXCLUDED)**.

---

### CARD ED09: ECB Rate Decision Post-Notices Drift [NEW]
*   **ID:** ED09
*   **Nombre:** ECB Rate Decision Post-Notices Drift
*   **Hipótesis:** Las conferencias de prensa de la presidenta del Banco Central Europeo (ECB) generan una deriva de precios direccional y persistente (Drift) que dura varias horas tras finalizar el comunicado oficial de tipos de interés.
*   **Reglas de Entrada:**
    *   **Long:** El día de decisión de tipos de la ECB, el par EURUSD rompe al alza durante la conferencia y el precio cierra en M30 con velas alcistas consecutivas por encima del nivel pre-anuncio. Entrar largo al inicio de la segunda hora de la conferencia.
    *   **Short:** El par rompe a la baja en la conferencia y cierra con velas bajistas M30 por debajo del nivel pre-anuncio. Entrar corto al inicio de la segunda hora de la conferencia.
*   **Reglas de Salida:**
    *   **Take Profit:** Fijo a 40 pips de la entrada.
    *   **Stop Loss:** Fijo a 25 pips de la entrada.
*   **Parámetros:** ECB Decision Day, Time offset: 60 minutos post conferencia.
*   **Timeframes:** M30.
*   **Ventana Horaria NY:** 08:30 - 12:30 NY (ventana de la ECB).
*   **Filtros:** Spread $\le 1.8$ pips.
*   **Riesgo de Overfitting:** **Bajo**. Fenómeno macro fundamental.
*   **Sensibilidad a Costos:** **Baja**.
*   **Clasificación de Prioridad:** **Priority D (EXCLUDED)**.

---

### CARD HY19: Hybrid Volatility-Filtered Trend Following
*   **ID:** HY19
*   **Nombre:** Hybrid Volatility-Filtered Trend Following (HVFTF)
*   **Hipótesis:** El uso del indicador SuperTrend en gráficos de 5 minutos permite el seguimiento dinámico de la tendencia local si se filtra de forma estricta por un umbral mínimo de volatilidad ATR para descartar los destructivos mercados laterales.
*   **Reglas de Entrada:**
    *   **Long:** SuperTrend M5 cambia a alcista (cierre por encima de la línea). Confirmar que el SuperTrend M15 también sea alcista y que el ATR(14) M5 sea $>0.0005$ (5 pips).
    *   **Short:** SuperTrend M5 cambia a bajista. Confirmar que SuperTrend M15 sea bajista y que el ATR(14) M5 sea $> 0.0005$.
*   **Reglas de Salida:**
    *   **Take Profit:** Fijo a $2.5\times$ la distancia del riesgo inicial.
    *   **Stop Loss:** Colocado en el valor actual de la línea de SuperTrend M5 en la entrada.
    *   **Gestión:** Mover stop a Break Even al alcanzar $1.2\times$ de beneficio. Trailing stop dinámico siguiendo la línea de SuperTrend M5 en cada barra.
    *   **Time Stop:** Cierre obligatorio a las 18:00 NY.
*   **Parámetros:** SuperTrend M5/M15 (10 ATR, 3 factor), ATR M5: 14 periodos.
*   **Timeframes:** M5 para disparo, M15 para filtro superior.
*   **Ventana Horaria NY:** 09:30 - 18:00 NY.
*   **Filtros:** Spread máximo $\le 2.0$ pips. ATR mínimo $\ge 5$ pips.
*   **Riesgo de Overfitting:** **Medio**. Múltiples filtros de indicadores.
*   **Sensibilidad a Costos:** **Media**.
*   **Clasificación de Prioridad:** **Priority C (MEDIA)**.

---

### CARD HY20: Hybrid M15 Trend + VWAP Mean Reversion
*   **ID:** HY20
*   **Nombre:** Hybrid M15 Trend + VWAP Mean Reversion
*   **Hipótesis:** Las desviaciones significativas del precio de 1 minuto respecto al VWAP intradía actúan como excesos que tienden a revertir velozmente a su media incluso en mercados con tendencia M15 definida.
*   **Reglas de Entrada:**
    *   **Long:** La tendencia en M15 es alcista (SMA10 > SMA30). El precio en M1 cierra por debajo de la línea de VWAP en $\ge 3$ pips. Confirmar vela M1 alcista.
    *   **Short:** Tendencia M15 bajista (SMA10 < SMA30). El precio en M1 cierra por encima del VWAP en $\ge 3$ pips. Confirmar vela M1 bajista.
*   **Reglas de Salida:**
    *   **Take Profit:** Retorno exacto al nivel de VWAP (TP1) y extensión de $+0.5\times$ ATR14 M15 para el $50\%$ restante (TP2).
    *   **Stop Loss:** Fijo a $1.5\times$ ATR14 M15.
    *   **Gestión:** Mover stop a BE al alcanzar el VWAP (TP1). Cierre por tiempo a los 30 minutos.
*   **Parámetros:** SMA10 M15, SMA30 M15, ATR M15: 14.
*   **Timeframes:** M15 para tendencia, M1 para disparo y VWAP.
*   **Ventana Horaria NY:** 09:00 - 17:00 NY.
*   **Filtros:** Spread máximo $\le 1.5$ pips. ATR M15 $\ge 5$ pips.
*   **Riesgo de Overfitting:** **Medio-Alto**.
*   **Sensibilidad a Costos:** **Media**.
*   **Clasificación de Prioridad:** **Priority C (DEFERRED)**.

---

### CARD HY06: Triple-Screen Volatility Compression Reversion [NEW]
*   **ID:** HY06
*   **Nombre:** Triple-Screen Volatility Compression Reversion
*   **Hipótesis:** Cuando las bandas de Bollinger en M1, M5 y M15 se comprimen simultáneamente (Triple Squeeze), el mercado está al borde de un desequilibrio extremo. Si el precio se estira rápidamente tras el squeeze, la primera reversión es de altísima fiabilidad.
*   **Reglas de Entrada:**
    *   **Long:** Ocurre un squeeze simultáneo (Bandwidth en el percentil $\le 20\%$ en M1, M5 y M15). El precio M1 cae velozmente por debajo de la banda inferior M15 en $\ge 5$ pips. Entrar largo en la primera vela M1 alcista.
    *   **Short:** Squeeze simultáneo. El precio M1 sube $\ge 5$ pips por encima de la banda superior M15. Entrar corto en la primera vela M1 bajista.
*   **Reglas de Salida:**
    *   **Take Profit:** Retorno a la banda media de Bollinger M15.
    *   **Stop Loss:** Colocado a $10$ pips.
*   **Parámetros:** Bollinger Bands (20, 2), Percentil squeeze: $20\%$.
*   **Timeframes:** M1, M5 y M15.
*   **Ventana Horaria NY:** 08:30 - 15:00 NY.
*   **Filtros:** Spread $\le 1.2$ pips.
*   **Riesgo de Overfitting:** **Alto**. Multi-timeframe squeeze requiere alta precisión en la alineación de barras.
*   **Sensibilidad a Costos:** **Alta**.
*   **Clasificación de Prioridad:** **Priority C (DEFERRED)**.

---

### CARD HY10: High-Frequency Bid-Ask Imbalance Fade [NEW]
*   **ID:** HY10
*   **Nombre:** High-Frequency Bid-Ask Imbalance Fade
*   **Hipótesis:** Los desequilibrios momentáneos en la cartera de órdenes (Bid-Ask Imbalance $>80\%$) seguidos de una micro-excursión del precio en gráficos de ticks tienden a ser arbitrados de forma inmediata por creadores de mercado.
*   **Reglas de Entrada:**
    *   **Long:** El Bid-Ask volume imbalance en el nivel 1 de la cartera de órdenes supera el $80\%$ del lado comprador, y el precio del tick baja $\ge 1.5$ pips. Entrar largo.
    *   **Short:** El volume imbalance supera el $80\%$ vendedor, y el precio del tick sube $\ge 1.5$ pips. Entrar corto.
*   **Reglas de Salida:**
    *   **Take Profit:** Retorno rápido de 2 pips.
    *   **Stop Loss:** Colocado a $1.5$ pips.
*   **Parámetros:** Imbalance ratio: 0.80, Tick change: 1.5 pips.
*   **Timeframes:** Ticks negociados (Order book data).
*   **Ventana Horaria NY:** 09:30 - 11:30 NY (alta liquidez).
*   **Filtros:** Spread máximo permitido $\le 0.5$ pips.
*   **Riesgo de Overfitting:** **Extremo**. Sensible a la microestructura y latencia.
*   **Sensibilidad a Costos:** **Extrema** (Inviable en brokers estándar).
*   **Clasificación de Prioridad:** **Priority D (EXCLUDED)**.
