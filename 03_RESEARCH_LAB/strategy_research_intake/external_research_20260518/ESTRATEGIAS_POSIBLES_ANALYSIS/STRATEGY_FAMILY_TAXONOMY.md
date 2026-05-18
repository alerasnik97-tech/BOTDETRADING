# STRATEGY FAMILY TAXONOMY — THE CORE ARCHITECTURE
**Date:** 2026-05-18
**Project:** Systematic Infrastructure Professionalization — Strategy Taxonomy
**Security Status:** READ-ONLY AUDIT & COMPILATION — NO CODE OR REPOSITORY MUTATION

---

## 1. Familias Taxonómicas Cuantitativas

Para evitar el sobrediseño y la redundancia teórica en el portafolio de trading algorítmico, clasificamos el universo de estrategias en **seis familias estructuradas**. Cada familia responde a una ineficiencia o desbalance microestructural de mercado diferente en el par EURUSD intradía.

---

## 2. Desglose Taxonómico por Familia

### FAMILIA 1: MEAN REVERSION (MR)
*   **Nombre:** Reversión a la Media Intradía.
*   **Lógica de Mercado:** Los flujos institucionales y desbalances temporales empujan el precio fuera de su zona de equilibrio estadístico temporal (VWAP diario, medias móviles largas). Sin catalizadores macro de alta relevancia que justifiquen una revaluación fundamental de la divisa, el precio tiende a regresar al valor promedio.
*   **Por qué podría existir Edge:** Las instituciones actúan como proveedoras de liquidez y arbitrajistas en EURUSD, limitando las excursiones excesivas y forzando al precio a retornar a sus anclajes medios cuando cesan los flujos unidireccionales de la apertura.
*   **Cuándo suele funcionar:** En mercados de rango (rango de fluctuación diaria promedio), sesiones de baja volatilidad y tardes tranquilas de NY (11:30 - 16:30 NY).
*   **Cuándo suele fallar:** En días de tendencia fuerte (Trend Days) impulsados por flujos sistémicos de capitales globales o noticias macro inesperadas.
*   **Datos Requeridos:** OHLCV en barras M5 + VWAP intradía (calculado sobre barras M1 con fallback TWAP).
*   **Timeframes Recomendados:** M5 para señales base, M1 para cálculo de VWAP y filtros Z-score.
*   **Horarios NY Recomendados:** 09:30 - 16:30 NY.
*   **Riesgo de Correlación con Manipulante:** **BAJO** (Opera el desvío dinámico del valor medio, no el barrido fractal).
*   **Riesgo de Overfitting:** **Bajo-Medio** (Mitigado utilizando desviaciones estándar robustas y Z-scores fijos).
*   **Dificultad de Programar:** **Baja-Media** (Cálculo acumulativo dinámico del VWAP diario).
*   **Compatibilidad FTMO:** **Alta** (Se definen stops fijos y targets realistas en el VWAP).
*   **Gate Mínimo Recomendado:** **Gate 1** (Prueba de señal en train).
*   **Motivo para Priorizar:** Excelente diversificador natural con Sharpe Ratio históricamente alto en pares altamente líquidos como EURUSD.

---

### FAMILIA 2: TREND PULLBACK (TP)
*   **Nombre:** Continuación de Tendencia en Pullbacks.
*   **Lógica de Mercado:** En tendencias sólidas, el dinero inteligente y los creadores de mercado no persiguen breakouts de precios extremos; esperan a que el par retroceda a zonas dinámicas de valor promedio (p. ej. EMA50 o VWAP) para re-acumular posiciones a favor de la tendencia principal con mejor coste medio.
*   **Por qué podría existir Edge:** El par EURUSD presenta estructuras tendenciales muy limpias en M15 durante periodos activos en Nueva York. Los soportes dinámicos atraen volumen secundario y reanudan la dirección original.
*   **Cuándo suele funcionar:** En mercados fuertemente direccionales tras noticias macroeconómicas o flujos marcados en la apertura.
*   **Cuándo suele fallar:** En mercados laterales de alta oscilación (Chop Markets) y falsas expansiones.
*   **Datos Requeridos:** M5 y M15 OHLCV estándar.
*   **Timeframes Recomendados:** M15 para definir el contexto de tendencia macro, M5 para el disparo en pullback local.
*   **Horarios NY Recomendados:** 08:00 - 17:00 NY.
*   **Riesgo de Correlación con Manipulante:** **BAJO** (Estrategia puramente direccional y de continuación).
*   **Riesgo de Overfitting:** **Medio** (Evitar el ajuste excesivo de las medias de tendencia).
*   **Dificultad de Programar:** **Media** (Lógica de sincronización multi-timeframe).
*   **Compatibilidad FTMO:** **Alta** (Stops colocados por debajo del mínimo local reciente reducen drásticamente el drawdown).
*   **Gate Mínimo Recomendado:** **Gate 2** (Requiere especificación anti-lookahead validada).
*   **Motivo para Priorizar:** Alinear el portafolio a favor de las tendencias institucionales intradía es el mejor amortiguador contra rachas de drawdowns coincidentes.

---

### FAMILIA 3: VOLATILITY EXPANSION (VE)
*   **Nombre:** Ruptura e Impulso de Volatilidad (Momentum Breakout).
*   **Lógica de Mercado:** El mercado se mueve de fases de contracción extrema (acumulación en rangos muy estrechos) a fases de expansión rápida de volatilidad. La ruptura de un nivel o canal técnico coincide con flujos unidireccionales muy fuertes.
*   **Por qué podría existir Edge:** El solapamiento de Londres y NY genera compresiones de volatilidad previas seguidas de expansiones de volumen institucional muy predecibles.
*   **Cuándo suele funcionar:** En periodos con catalizadores macro programados de alto impacto y aperturas de sesión líquidas.
*   **Cuándo suele fallar:** En falsos breakouts generados por tomas de liquidez minorista sin volumen institucional de soporte.
*   **Datos Requeridos:** M5/M15 OHLCV + Volumen relativo o Ticks negociados.
*   **Timeframes Recomendados:** M5 para definición de rango/canal y M15 para ATR y volumen relativo.
*   **Horarios NY Recomendados:** 07:00 - 14:00 NY.
*   **Riesgo de Correlación con Manipulante:** **BAJO** (Continuación de rango vs reversión de extremos).
*   **Riesgo de Overfitting:** **Alto** (Peligro crítico de sobreajustar los filtros de compresión/ancho de banda).
*   **Dificultad de Programar:** **Media** (Requiere cálculos robustos de percentiles históricos de volumen).
*   **Compatibilidad FTMO:** **Media-Alta** (Sensible al spread del broker en el momento exacto del breakout).
*   **Gate Mínimo Recomendado:** **Gate 2** (Requiere motor de percentiles rodantes backward-only).
*   **Motivo para Priorizar:** Captura movimientos explosivos y limpios con alta expectativa de R:R en el inicio de la sesión americana.

---

### FAMILIA 4: SESSION DYNAMICS & FAKEOUTS (SD)
*   **Nombre:** Dinámicas de Sesión y Falsas Rupturas Geográficas.
*   **Lógica de Mercado:** Los grandes operadores bancarios y algoritmos HF empujan intencionadamente el precio ligeramente por encima o debajo de los extremos de la sesión asiática o del Initial Balance para activar los stop-loss de los traders minoristas y recolectar liquidez rápida antes de revertir la cotización.
*   **Por qué podría existir Edge:** Existe un desbalance crónico de liquidez en los límites de sesiones que fuerza la reversión del precio con alta fiabilidad estadística.
*   **Cuándo suele funcionar:** En días de rango de consolidación general y mercados sin tendencia macro definida.
*   **Cuándo suele fallar:** En días de breakouts reales e impulsos direccionales muy limpios.
*   **Datos Requeridos:** OHLCV M5/M1 con timestamp preciso sincronizado con el broker.
*   **Timeframes Recomendados:** M5 para identificar el rango de sesión y M1 para la entrada precisa del fallo.
*   **Horarios NY Recomendados:** 07:00 - 11:30 NY.
*   **Riesgo de Correlación con Manipulante:** **EXTREMADAMENTE ALTO (HIGH_CORRELATION_RISK)**
*   **Riesgo de Overfitting:** **Medio-Alto** (Peligro de ajustar demasiado los buffers de excursión en pips).
*   **Dificultad de Programar:** **Alta** (Detección causal milimétrica de falsas rupturas en M1).
*   **Compatibilidad FTMO:** **Baja** (Drawdowns coincidentes con `Manipulante` destruirían el límite de pérdida diaria).
*   **Gate Mínimo Recomendado:** **Gate 3** (Requiere test de correlación y presupuesto de cartera consolidado).
*   **Motivo para Postergar:** **Diferidas de forma obligatoria**. Su solapamiento con `Manipulante` canibalizaría el rendimiento del portafolio.

---

### FAMILIA 5: SEASONAL & ESTACIONALES (SE)
*   **Nombre:** Patrones Estacionales y Horarios Intradía.
*   **Lógica de Mercado:** Los bancos, fondos de cobertura y traders institucionales tienen patrones y rutinas operativas fijas (almuerzos de Wall Street, cierres de libros semanales los viernes por la tarde, rollover diario de las 17:00 NY) que inyectan o retiran liquidez de forma sistemática.
*   **Por qué podría existir Edge:** La repetición recurrente del reloj biológico de los grandes mercados genera ineficiencias de reversión y consolidación predecibles.
*   **Cuándo suele funcionar:** En ventanas horarias hiper-específicas (p. ej. Viernes de 14:00 a 16:30 NY).
*   **Cuándo suele fallar:** En días festivos de baja actividad, cierres extraordinarios o shocks mundiales fuera del horario estándar.
*   **Datos Requeridos:** M15/M30 OHLCV estándar.
*   **Timeframes Recomendados:** M15 para disparos de precisión estacional.
*   **Horarios NY Recomendados:** Horarios fijos del calendario.
*   **Riesgo de Correlación con Manipulante:** **BAJO** (Gatillada por tiempo y reloj, no por acción fractal de precio).
*   **Riesgo de Overfitting:** **Medio-Alto** (Peligro de p-hacking si se buscan patrones horarios sin fundamento lógico).
*   **Dificultad de Programar:** **Baja** (Anclajes horarias simples).
*   **Compatibilidad FTMO:** **Alta** (Permite cerrar posiciones rápidamente antes de fines de semana o rollovers peligrosos).
*   **Gate Mínimo Recomendado:** **Gate 1** (Prueba de señal en train).
*   **Motivo para Priorizar:** Excelente diversificador de baja correlación que aprovecha la inacción horaria de Wall Street.

---

### FAMILIA 6: NEWS & EVENT-DRIVEN (ED)
*   **Nombre:** Trading Post-Noticia y Anomalías de Volatilidad Exógena.
*   **Lógica de Mercado:** La inyección repentina de datos macro de alta relevancia rompe el equilibrio temporal, induciendo a una sobrerreacción institucional masiva seguida de un reequilibrio paulatino del spread y del precio hacia su media justa.
*   **Por qué podría existir Edge:** Los creadores de mercado ensanchan artificialmente los spreads durante los primeros 5-10 minutos del anuncio para protegerse del flujo de órdenes desordenado. Al normalizarse la cotización, el mercado corrige las desviaciones irracionales del precio.
*   **Cuándo suele funcionar:** Tras anuncios macro programados en calendario (FOMC, NFP, ECB CPI).
*   **Cuándo suele fallar:** En eventos macro imprevistos, guerras, declaraciones políticas sorpresa o breakouts tendenciales permanentes.
*   **Datos Requeridos:** Feed de ticks de alta velocidad (BID/ASK) + Calendario económico automatizado.
*   **Timeframes Recomendados:** M1 para rastreo del rechazo y M5 para confirmación.
*   **Horarios NY Recomendados:** Variables según el calendario de noticias.
*   **Riesgo de Correlación con Manipulante:** **BAJO** (Trigger exógeno fundamentado en noticias).
*   **Riesgo de Overfitting:** **Alto** (Peligro en la selección retrospectiva de qué eventos son de alta volatilidad).
*   **Dificultad de Programar:** **Extrema** (Lógica de filtrado de noticias en tiempo real y spread dinámico).
*   **Compatibilidad FTMO:** **Baja** (El slippage y ensanchamiento de spreads en MT5 durante noticias destruye las reglas de drawdown).
*   **Gate Mínimo Recomendado:** **Gate 4** (Requiere feed de datos tick bid/ask real e infraestructura premium).
*   **Motivo para Postergar:** **Diferida a largo plazo**. La actual infraestructura local no cuenta con la velocidad necesaria para operar de forma segura durante noticias de alta fricción.
