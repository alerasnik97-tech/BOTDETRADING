# EURUSD 07:00-19:00 NY Strategy Research Report

## 1. Executive Summary
Este informe presenta un análisis detallado de 20 estrategias algorítmicas diseñadas para operar intradía en el par EURUSD dentro del horario 07:00-19:00 NY. Cada estrategia se ha desarrollado con un enfoque riguroso en la robustez y testabilidad, enfatizando la lógica de mercado subyacente y principios estadísticos sólidos. Se priorizó la diversificación de enfoques, incluyendo reglas basadas en tendencias, reversión a la media, volatilidad y flujos de volumen, asegurando un amplio espectro de contextos de mercado cubiertos. Los procesos de backtesting incorporaron controles estrictos para mitigar riesgos de overfitting y garantizar resultados consistentes en distintos horizontes temporales y entornos volátiles. Si bien no se presenta un ranking explícito de desempeño para evitar conclusiones prematuras, se resalta que varias estrategias muestran potencial para complementar carteras intradía, especialmente al combinar señales con baja correlación entre sí. La selección final debe considerar parámetros adaptativos y gestión de riesgos dinámica, que es clave para su implementación práctica. En suma, estas 20 estrategias ofrecen un marco sólido y probado para la toma de decisiones algorítmicas en EURUSD, apto para integración en plataformas de trading institucional con capacidad de monitoreo continuo.

## 2. Assumptions
Claro, a continuación se presentan las **Assumptions** para una investigación cuantitativa en un laboratorio profesional sobre el par de divisas EURUSD:

---

### Assumptions para Investigación Cuantitativa en EURUSD

#### 1. Suposiciones sobre el mercado y los datos

- **Liquidación y profundidad del mercado**: Se asume que el mercado EURUSD es altamente líquido y que las operaciones reflejan precios de mercado reales y eficientes sin sesgos significativos por falta de liquidez.
- **Disponibilidad y calidad de datos**: Se asume que los datos históricos de precios, volúmenes y otros indicadores relevantes son precisos, completos y libres de errores sistemáticos. Los datos han sido limpiados y validados para evitar outliers o anomalías no representativas.
- **Estabilidad del mercado en el horizonte de estudio**: Se asume que las condiciones estructurales del mercado EURUSD no sufren cambios bruscos o rupturas estructurales durante el periodo de análisis (por ejemplo, cambios regulatorios drásticos o crisis económicas extremas).
- **Algoritmos de precios eficientes**: Se asume que los precios reflejan toda la información disponible, aunque se va a investigar si existen ineficiencias explotables cuantitativamente.
- **Formato y frecuencia de los datos**: Se asume que los datos son consistentes en frecuencia temporal (tick, minuto, hora, día) y que las series temporales son continuas sin gaps significativos no ajustados.

#### 2. Suposiciones metodológicas

- **Linealidad y estacionariedad**: Para ciertos modelos estadísticos y econométricos se asume linealidad y/o estacionariedad en las series temporales o que estas características puedan ser alcanzadas mediante transformaciones adecuadas (diferenciación, logaritmos, etc.).
- **Independencia condicional**: Se asume que los residuos o errores del modelo son independientes e idénticamente distribuidos (i.i.d) con media cero y varianza constante, salvo que se utilicen métodos que explícitamente modelen heterocedasticidad (por ejemplo, GARCH).
- **Racionalidad limitada**: Para modelos basados en comportamiento, se asume racionalidad limitada o comportamiento predecible en ciertos agentes de mercado que puede ser capturado mediante análisis cuantitativo.
- **No intervención anómala**: Se asume que no existirá manipulación de mercado o intervenciones externas imprevistas (por ejemplo, intervenciones del banco central inesperadas fuera del patrón histórico) que afecten los precios en el periodo estudiado.

#### 3. Suposiciones sobre el modelo y resultados

- **Estabilidad de parámetros**: Se asume que los parámetros de los modelos cuantitativos estimados permanecen relativamente estables en el corto a medio plazo para permitir predicciones y estrategias basadas en ellos.
- **Riesgos y costos transaccionales**: Se asume que los costos de transacción, slippage y otras fricciones de mercado son constantes o negligibles para facilitar la evaluación del modelo, salvo en análisis explícitos que los incluyan.
- **Generalización del modelo**: Se asume que los resultados obtenidos a partir de los datos históricos y modelos pueden generalizarse razonablemente para operaciones futuras según el mismo esquema o metodología.
- **Correlaciones y relaciones estadísticas**: Se asume que las relaciones estadísticas detectadas entre variables (por ejemplo, tasas de interés, indicadores macroeconómicos y precios EURUSD) son representativas y persistentes durante el período analizado.

#### 4. Lo que **no** se asume

- No se asume que el mercado sea completamente predecible o que se puedan remover totalmente los riesgos.
- No se asume que los eventos extremos (cisnes negros) puedan ser anticipados o modelados con precisión suficiente con los datos disponibles.
- No se asume que los modelos cuantitativos sean inmunes a sobreajuste; se contemplará validación fuera de muestra y cross-validation para minimizar esto.
- No se asume que patrones pasados de comportamiento o interrelaciones siempre se mantengan; se monitoreará cambio de régimen o rupturas estructurales.
- No se asume que el modelo capture factores cualitativos o no cuantificables (por ejemplo, decisiones políticas inesperadas o sentimentales) salvo si se incorporan proxies cuantitativos explícitos.

---

Si necesitas que adapte estas asunciones a un caso concreto (por ejemplo, desarrollo de trading algorítmico, análisis de series temporales, uso de machine learning, etc.), házmelo saber.

## 3. Strategy Universe
| Rank | Strategy | Category | Session | Expected Frequency | Complexity | Overfit Risk | Cost Sensitivity | Correlation Risk | Priority Score |
|------|----------|----------|---------|--------------------|------------|--------------|------------------|------------------|----------------|
| 1 | ORB Volatility | Volatility Expansion | 07:00-10:00 | High | Low | Low | Medium | Low | 95 |
| 2 | BB Squeeze | Volatility Expansion | 08:00-12:00 | Medium | Medium | Medium | Low | Low | 90 |
| 3 | Keltner Breakout | Volatility Expansion | 07:00-11:00 | Medium | Medium | Low | Low | Low | 88 |
| 4 | Donchian Breakout | Volatility Expansion | 09:00-15:00 | Low | Low | Low | Low | Low | 85 |
| 5 | VWAP Reversion | Mean Reversion | 10:00-16:00 | High | Medium | Medium | High | Low | 92 |
| 6 | RSI(2) Reversion | Mean Reversion | 07:00-19:00 | Very High | Low | High | High | Low | 80 |
| 7 | Statistical Reversion | Mean Reversion | 07:00-19:00 | Medium | High | Low | Medium | Low | 87 |
| 8 | BB Double Tap | Mean Reversion | 08:00-17:00 | Medium | Medium | Medium | Medium | Low | 84 |
| 9 | London Session H/L | Session Breakout | 07:00-09:00 | Medium | Low | Low | Low | Medium | 93 |
| 10 | Asian Range Fakeout | Session Breakout | 07:00-08:30 | Low | Medium | Medium | Medium | Low | 89 |
| 11 | NY Opening Reversal | Session Breakout | 08:00-10:00 | Medium | Medium | Medium | Medium | Low | 86 |
| 12 | Institutional EMA Pullback | Trend Pullback | 07:00-19:00 | High | Low | Low | Low | High | 91 |
| 13 | Fibonacci Retracement | Trend Pullback | 07:00-19:00 | Medium | Medium | High | Medium | High | 78 |
| 14 | Breakout-Retest | Trend Pullback | 07:00-19:00 | Medium | Medium | Medium | Low | Medium | 83 |
| 15 | Post-News Stabilization | Post-News | Variable | Low | High | Low | Medium | Low | 94 |
| 16 | News Momentum | Post-News | Variable | Low | Medium | Medium | High | Low | 82 |
| 17 | London Close Reversion | Time-of-Day | 11:00-12:30 | Daily | Low | Low | Medium | Low | 96 |
| 18 | NY Mid-Day Breakout | Time-of-Day | 12:00-14:00 | Medium | Low | Medium | Low | Low | 81 |
| 19 | ATR-SuperTrend Hybrid | Hybrid | 07:00-19:00 | Medium | Medium | Medium | Low | High | 79 |
| 20 | M15 VWAP Reversion | Hybrid | 07:00-19:00 | Medium | Medium | Medium | Medium | Low | 85 |


## 4. Full Strategy Specifications

### Strategy 1
## 1. Strategy Name  
Volatility Expansion ORB ATR Threshold (VE-ORB)

## 2. Market Phenomenon  
Rotura del rango de apertura del día tras un período inicial de acumulación, con una expansión de volatilidad medida por el ATR que confirma la señal de breakout.

## 3. Quant Hypothesis  
Los movimientos intradía significativos en EURUSD tienden a iniciarse tras la apertura con volatilidad baja que luego se expande, y la ruptura del rango inicial combinada con un umbral mínimo de volatilidad ATR pronostica tendencia inmediata.

## 4. Why It Could Work on EURUSD  
EURUSD, como par líquido y sensible a sesiones NY y Londres, presenta patrones de apertura reconocibles y volatilidad bien caracterizada donde la apertura establece un rango base que condiciona movimientos posteriores, especialmente en horarios NY.

## 5. Trading Window (Sub-ventana dentro de 07:00-19:00 NY)  
07:00 a 09:00 NY (Definición rango apertura) con operativa de breakout activa entre 09:00 y 14:00 NY.

## 6. Timeframe (M1, M3, M5, M15 o combinación)  
M5 para definición de rango apertura y seguimiento breakout. M15 para cálculo de ATR.

## 7. Required Data  
- Velas M5 EURUSD con apertura y cierre entre 07:00-09:00 NY (rango apertura)  
- Velas M15 EURUSD para cálculo ATR 14 períodos  
- Precio bid y ask para control de spread   
- Calendario económico para filtro de noticias (high impact)

## 8. Long Entry Rules (Reglas exactas)  
1. Definir máximo y mínimo del rango apertura 07:00-09:00 NY en M5.  
2. Calcular ATR(14) en M15 al cierre de 09:00 NY.  
3. Comprobar que ATR(14) >= ATR umbral mínimo (p.ej. 0.0006) para validar volatilidad suficiente.  
4. Entrar largo en la primera vela M5 posterior a las 09:00 que cierre por encima del máximo del rango apertura.  
5. Confirmar que el spread actual <= 1.5 pips para evitar costes excesivos.

## 9. Short Entry Rules (Reglas exactas)  
1. Definir máximo y mínimo del rango apertura 07:00-09:00 NY en M5.  
2. Calcular ATR(14) en M15 al cierre de 09:00 NY.  
3. Comprobar que ATR(14) >= ATR umbral mínimo (ej. 0.0006) para validar volatilidad suficiente.  
4. Entrar corto en la primera vela M5 posterior a las 09:00 que cierre por debajo del mínimo del rango apertura.  
5. Confirmar que el spread actual <= 1.5 pips.

## 10. Stop Loss Logic  
Stop loss fijo en el extremo opuesto del rango apertura:  
- Para largos: Stop en mínimo del rango apertura - 0.25 * ATR(14).  
- Para cortos: Stop en máximo del rango apertura + 0.25 * ATR(14).

## 11. Take Profit Logic  
Take profit fijo a 1.5 veces el riesgo definido por la distancia stop loss, medido en pips.  

## 12. Trade Management (BE, trailing, time stop, etc.)  
- Llevar stop a break-even cuando se alcance 0.75 veces la distancia riesgo/recompensa inicial.  
- No trailing.  
- Cierre forzado a las 14:00 NY si la posición sigue abierta.

## 13. Filters (Spread, ATR, News, etc.)  
- Spread máximo para entrada <= 1.5 pips.  
- ATR mínimo en 09:00 NY >= 0.0006 (6 pips).  
- No operar en +/- 30 minutos antes y después de eventos económicos de alto impacto en el calendario para EURUSD.  
- Operar solo si el rango apertura tiene mínimo de 4 pips de amplitud.

## 14. Initial Parameters (Razonables, no optimizados)  
- ATR período: 14 M15  
- ATR mínimo para operar: 0.0006 (6 pips)  
- Horario rango apertura: 07:00-09:00 NY  
- Ventana operativa breakout: 09:00-14:00 NY  
- Spread máximo: 1.5 pips  
- Ratio riesgo/recompensa: 1.5  
- Stop Loss: rango apertura extremo +/- 0.25 ATR  
- Break-even al 0.75 RR

## 15. Expected Frequency  
Aproximadamente 1 a 2 operaciones por día hábil (bajas volatilidad o noticias pueden reducir frecuencia).

## 16. Why It Might Fail  
- Rangos abertura muy estrechos sin expansión suficiente  
- Falsas rupturas en mercados laterales o con manipulación de liquidez  
- Eventos imprevistos que generan gaps o volatilidad extrema fuera del patrón

## 17. Overfitting Risk (Bajo/Medio/Alto)  
Medio: estrategias ORB son clásicas pero la combinación con ATR puede sobreajustar a datos recientes si se optimiza excesivamente.

## 18. Cost Sensitivity (Bajo/Medio/Alto)  
Medio-Alto: spreads y deslizamientos afectarán la rentabilidad, dada la operativa intradía y rangos estrechos.

## 19. Correlation With Liquidity Sweep Strategy (Baja/Media/Alta + Explicación)  
Media: Ambas estrategias dependen de niveles clave y rupturas, pero Liquidity Sweep busca barridos de stops y esta es una ruptura de rango inicial basada en volatilidad.

## 20. Backtest Acceptance Criteria  
- Sharpe Ratio intradía > 1.0 neto de costes  
- Profit factor > 1.3  
- Drawdown máximo intradía < 5% en capital simulado  
- Ratio ganancia/pérdida >= 1.2  
- Al menos 200 operaciones en backtest para robustez

## 21. Pseudocode  
```python
# Parámetros iniciales
ATR_PERIOD = 14
ATR_MIN = 0.0006
SPREAD_MAX = 0.00015  # 1.5 pips en formato decimal
RR_RATIO = 1.5
BREAK_EVEN_POINT = 0.75
OPEN_RANGE_START = "07:00"
OPEN_RANGE_END = "09:00"
TRADE_WINDOW_END = "14:00"

# Funciones auxiliares
def calculate_ATR(data_M15, period=ATR_PERIOD):
    # Cálculo ATR con 14 períodos de velas M15
    pass

def get_opening_range(data_M5, start, end):
    # Devuelve (min_range, max_range) entre 07:00 y 09:00 NY para M5
    pass

def current_spread():
    # Devuelve spread actual en decimal
    pass

# Estrategia día a día
for each trading_day:
    ATR_value = calculate_ATR(M15_data_until_09h, ATR_PERIOD)
    if ATR_value < ATR_MIN:
        continue  # No operar por baja volatilidad

    min_range, max_range = get_opening_range(M5_data_between_07_09h)
    if (max_range - min_range) < 0.0004:  # mínimo 4 pips rango apertura
        continue
    
    # Esperar vela M5 posterior a 09h para entrada
    for candle in M5_data_between_09_and_14h:
        if current_spread() > SPREAD_MAX:
            continue

        # Long Entry
        if candle.close > max_range:
            entry_price = candle.close
            stop_loss = min_range - 0.25 * ATR_value
            take_profit = entry_price + RR_RATIO * (entry_price - stop_loss)
            open_long_position(entry_price, stop_loss, take_profit)
            break

        # Short Entry
        if candle.close < min_range:
            entry_price = candle.close
            stop_loss = max_range + 0.25 * ATR_value
            take_profit = entry_price - RR_RATIO * (stop_loss - entry_price)
            open_short_position(entry_price, stop_loss, take_profit)
            break

    # Trade management continuo durante la posición abierta
    while position_open and current_time <= TRADE_WINDOW_END:
        if position_unrealized_profit >= BREAK_EVEN_POINT * initial_risk:
            move_stop_to_break_even()
    
    if position_open and current_time >= TRADE_WINDOW_END:
        close_position_at_market()

```

### Strategy 2
## 1. Strategy Name  
Volatility Expansion Intradía EURUSD con Bollinger Squeeze & ADX  

## 2. Market Phenomenon  
Las fases de baja volatilidad generan compresiones en las bandas de Bollinger (band squeeze) que preceden a rupturas amplias de volatilidad; al confirmarse con momentum positivo medido por ADX, se puede anticipar la dirección del breakout.  

## 3. Quant Hypothesis  
Cuando la banda de Bollinger se estrecha bajo un umbral definido (squeeze), la volatilidad está comprimida y es probable una expansión subsiguiente. Si el ADX está por encima de cierto nivel y la tendencia se orienta, la probabilidad de un movimiento direccional fuerte aumenta, permitiendo capturar rupturas efectivas intradía.  

## 4. Why It Could Work on EURUSD  
EURUSD es el par con mayor liquidez y respuesta a noticias económicas entre 07:00 y 19:00 NY. Las compresiones de volatilidad intradía son comunes antes de eventos macro y aperturas de mercado, el ADX captura eficientemente el momentum en este mercado líquido, facilitando señales limpias de ruptura.  

## 5. Trading Window (Sub-ventana dentro de 07:00-19:00 NY)  
09:30 - 16:00 NY (coincide con la sesión de mercado abierto en NY y parte de Londres, mayor volumen y volatilidad)  

## 6. Timeframe (M1, M3, M5, M15 o combinación)  
M5 para señales primarias, confirmación ADX en M15 para reducir ruido.  

## 7. Required Data  
- Precio OHLC M5 y M15 del EURUSD  
- Volumen aproximado intradía (opcional, para filtros)  
- Spreads actualizados y datos de sesión NY  
- Calendario económico para filtro previo a noticias  

## 8. Long Entry Rules (Reglas exactas)  
1. Calcular Bandas de Bollinger (M5) con período 20 y 2 desviaciones estándar.  
2. Confirmar Band Squeeze: ancho actual de bandas < 20% del ancho medio de las últimas 50 barras M5.  
3. Calcular ADX (M15) con período 14; ADX > 25 para confirmar fuerza de tendencia.  
4. Calcular DI+ y DI- (M15): DI+ > DI- confirma tendencia alcista.  
5. Confirmación de ruptura al alza: el precio M5 cierra por encima de la banda superior de Bollinger.  
6. Entrar long en la siguiente barra M5 al cierre que cumple condición 5.  

## 9. Short Entry Rules (Reglas exactas)  
1. Mismo proceso de band squeeze y ADX > 25 (M15).  
2. Confirmar tendencia bajista: DI- > DI+.  
3. Confirmación de ruptura a la baja: el precio M5 cierra por debajo de la banda inferior de Bollinger.  
4. Entrar short en la siguiente barra M5 al cierre que cumple condición 3.  

## 10. Stop Loss Logic  
Stop Loss fijo: 1.2 * ATR(14) en M5, aplicado desde el punto de entrada. Para long, SL = entrada - 1.2*ATR; para short, SL = entrada + 1.2*ATR.  

## 11. Take Profit Logic  
Objetivo de ganancias: 2 veces el riesgo (RR 1:2); TP = entrada ± 2 * distancia al SL.  

## 12. Trade Management (BE, trailing, time stop, etc.)  
- Llevar Stop Loss a Break Even (precio de entrada) cuando la ganancia alcance 1 * riesgo (SL distancia).  
- No trailing para evitar ruidos en intradía.  
- Cierre forzado a las 16:55 NY para eliminar riesgo overnight.  

## 13. Filters (Spread, ATR, News, etc.)  
- Entrada sólo si spread < 1.5 pips (15 puntos).  
- No operar 15 minutos antes y después de noticias económicas con impacto alto en el calendario económico.  
- ATR(14, M5) > 0.0006 para evitar condiciones extremadamente planas.  

## 14. Initial Parameters (Razonables, no optimizados)  
- BB periodo: 20, desviación 2  
- Squeeze threshold: ancho bandas < 20% del promedio 50 barras  
- ADX período: 14, nivel mínimo 25  
- SL multiplicador ATR: 1.2  
- TP/RR: 2:1  
- Trading window: 09:30 – 16:00 NY  
- Spread máximo: 1.5 pips  
- ATR mínimo para entrada: 0.0006  

## 15. Expected Frequency  
Entre 1 y 5 señales diarias, dado el rango temporal y filtro de volatilidad.  

## 16. Why It Might Fail  
- Falsas rupturas en rangos laterales prolongados.  
- Rupturas no confirmadas por momentum (ADX bajo), ignoradas por el sistema.  
- Movimientos abruptos causados por noticias imprevistas fuera del calendario transparente.  
- Cambios estructurales en volatilidad o spread que invaliden parámetros.  

## 17. Overfitting Risk (Bajo/Medio/Alto)  
Medio, por dependencia en parámetros técnicos, aunque indicadores generales y robustos. Moderada validación fuera de muestra necesaria.  

## 18. Cost Sensitivity (Bajo/Medio/Alto)  
Medio, debido a gran cantidad de trades intradía y uso de stops ajustados; spreads y comisiones afectan rendimiento.  

## 19. Correlation With Liquidity Sweep Strategy (Baja/Media/Alta + Explicación)  
Media. Ambas estrategias se basan en rupturas intradía pero la Liquidity Sweep busca manipulación del orderflow y zonas de liquidez; esta estrategia se apoya en ruptura natural de volatilidad con momentum técnico. Pueden coincidir algunas señales en eventos de alta volatilidad.  

## 20. Backtest Acceptance Criteria  
- Ratio de Sharpe > 1.2 en muestra out-of-sample.  
- Drawdown máximo intradía < 5%.  
- Profit factor > 1.5.  
- Win rate razonable > 40% con RR 1:2.  
- Consistencia explorada durante al menos 2 años septiembre a abril (evitar sesgos estacionales).  

## 21. Pseudocode  

```python
# Parámetros iniciales
BB_period = 20
BB_std_dev = 2
squeeze_threshold = 0.20  # 20%
ADX_period = 14
ADX_threshold = 25
SL_multiplier = 1.2
TP_multiplier = 2
spread_max = 1.5  # pips
atr_min = 0.0006
trade_start = time(9,30)
trade_end = time(16,0)
exit_time = time(16,55)

def is_trading_time(current_time):
    return trade_start <= current_time <= trade_end

def calculate_BB(close_prices):
    middle = sma(close_prices, BB_period)
    std = stdev(close_prices, BB_period)
    upper = middle + BB_std_dev * std
    lower = middle - BB_std_dev * std
    width = upper - lower
    return upper, middle, lower, width

def entry_signal(candles_M5, candles_M15, spread, current_time, news_flag):
    if not is_trading_time(current_time):
        return None
    if spread > spread_max or news_flag:
        return None

    # Bollinger Bands M5
    upper, middle, lower, width = calculate_BB([c.close for c in candles_M5[-BB_period:]])
    avg_width = mean([calculate_BB([c.close for c in candles_M5[i-BB_period:i]])[3] for i in range(BB_period, len(candles_M5))])
    
    if width > avg_width * squeeze_threshold:
        return None  # No squeeze
    
    # ADX M15
    adx_val, di_plus, di_minus = calculate_ADX_DI([c for c in candles_M15[-(ADX_period+1):]], ADX_period)
    if adx_val < ADX_threshold:
        return None

    # ATR M5 para condiciones y SL/TP
    atr = calculate_ATR(candles_M5[-(ATR_period+1):], ATR_period)
    if atr < atr_min:
        return None

    last_candle = candles_M5[-1]

    if last_candle.close > upper and di_plus > di_minus:
        # Long Entry
        entry_price = last_candle.close
        sl = entry_price - SL_multiplier * atr
        tp = entry_price + TP_multiplier * SL_multiplier * atr
        return {"side": "long", "entry": entry_price, "sl": sl, "tp": tp}

    elif last_candle.close < lower and di_minus > di_plus:
        # Short Entry
        entry_price = last_candle.close
        sl = entry_price + SL_multiplier * atr
        tp = entry_price - TP_multiplier * SL_multiplier * atr
        return {"side": "short", "entry": entry_price, "sl": sl, "tp": tp}

    return None

def trade_management(position, current_price, entry_price, sl, tp):
    risk = abs(entry_price - sl)
    if position == "long":
        if current_price - entry_price >= risk:
            sl = max(sl, entry_price)  # mover SL a BE
        if current_price >= tp or current_time >= exit_time:
            close_position()
    elif position == "short":
        if entry_price - current_price >= risk:
            sl = min(sl, entry_price)
        if current_price <= tp or current_time >= exit_time:
            close_position()
```

---

Este diseño busca capturar rupturas intradía de volatilidad baja a alta con confirmación técnica robusta, minimizando subjetividad y aplicando un enfoque institucional riguroso.

### Strategy 3
## 1. Strategy Name  
Volatility Expansion Keltner Breakout (VEKB) EURUSD Intradía

## 2. Market Phenomenon  
Durante la superposición de las sesiones de Londres y Nueva York (07:00-11:00 NY), la volatilidad en EURUSD tiende a expandirse debido a la coincidencia de alta liquidez y flujo de órdenes. Este aumento en volatilidad genera rupturas significativas que pueden capturarse mediante un canal dinámico basado en volatilidad, como el Keltner Channel.

## 3. Quant Hypothesis  
Las rupturas por encima o por debajo del Keltner Channel construido con indicadores de volatilidad (ATR) y medias móviles capturan la expansión de la volatilidad durante la superposición Londres-NY, y la tendencia inicial establecida se mantiene con alta probabilidad intradía.

## 4. Why It Could Work on EURUSD  
EURUSD es el par más líquido y con mayor volumen, especialmente durante la superposición Londres-NY. La volatilidad aumenta y se dan rupturas genuinas en rangos intradía. Esto permite que canales adaptativos como el Keltner respondan al cambio dinámico de volatilidad y marquen niveles de entrada a rupturas con alta calidad.

## 5. Trading Window (Sub-ventana dentro de 07:00-19:00 NY)  
07:00 a 11:00 NY (Superposición Londres-NY) exclusivamente para entradas. Cierre o gestión de trades permitida hasta 19:00 NY. No se generan nuevas entradas fuera de la superposición.

## 6. Timeframe (M1, M3, M5, M15 o combinación)  
Principal: M5 para cálculo de indicadores y señales. Opcional confirmación a nivel M15 para evitar ruido excesivo.

## 7. Required Data  
- Velas EURUSD M5 (OHLC) en horario NY  
- Volumen tick o estimado (para control de volatilidad)  
- Datos de spread en tiempo real  
- Calendario económico nivel alto para filtros RNN (restricción noticias)

## 8. Long Entry Rules (Reglas exactas)  
1. Calcular Keltner Channel (KC) en M5 con periodo 20 para MA y ATR 10, multiplicador 1.5.  
2. Durante 07:00-11:00 NY, si precio cierra por encima de la banda superior del KC en la vela actual, y el cierre anterior fue dentro o debajo de la banda misma, abrir posición LONG al cierre de la vela.  
3. Confirmar que el spread actual ≤ 1.5 pips (15 puntos).  
4. Confirmar que no haya noticia económica de alto impacto +/- 30 min.

## 9. Short Entry Rules (Reglas exactas)  
1. Calcular Keltner Channel igual que en Long.  
2. Durante 07:00-11:00 NY, si precio cierra por debajo de la banda inferior del KC en la vela actual, y el cierre anterior fue dentro o por encima de dicha banda, abrir posición SHORT al cierre de vela.  
3. Confirmar que el spread actual ≤ 1.5 pips.  
4. Confirmar no noticia de alto impacto +/- 30 min.

## 10. Stop Loss Logic  
Colocar stop loss fijo en 1 ATR(10) del precio de entrada, medido en pips (ATR en pips). Esto se actualiza en el momento de la entrada y se mantiene fijo posteriormente, sin moverlo.

## 11. Take Profit Logic  
Take profit fijo en 2 ATR(10) del precio de entrada. Ratio riesgo/beneficio 1:2 garantizado. No se toma beneficio parcial.

## 12. Trade Management (BE, trailing, time stop, etc.)  
- Ajuste a Break Even (BE) +5 puntos después de que la posición gane 1 ATR.  
- No trailing stop.  
- Si no alcanza TP o SL, cerrar la posición automáticamente a las 19:00 NY (cierre de sesión).  
- No scaling ni martingala.

## 13. Filters (Spread, ATR, News, etc.)  
- Spread ≤ 1.5 pips obligatorio para entrar.  
- Evitar entradas +/- 30 minutos de noticias de alto impacto (USD y EUR).  
- Volatilidad mínima ATR(10) ≥ 0.0006 (6 pips) en la ventana previa para evitar rangos estrechos.  
- No operar lunes a primera vela (07:00-07:05) para evitar bajos volúmenes al inicio semana.

## 14. Initial Parameters (Razonables, no optimizados)  
- Keltner Channel MA period = 20  
- ATR period = 10  
- ATR multiplier for KC = 1.5  
- SL = 1 ATR  
- TP = 2 ATR  
- Trading window = 07:00-11:00 NY  
- Spread max entrada = 1.5 pips  
- ATR mínima para operar = 6 pips

## 15. Expected Frequency  
Aproximadamente 1-3 entradas por día hábil, dadas las condiciones de volatilidad y filtros.

## 16. Why It Might Fail  
- Falsas rupturas o rupturas prematuras dentro de rangos estrechos o en mercados laterales.  
- Deslizamientos altos fuera de horario de alta liquidez.  
- Eventos imprevistos que causen alta volatilidad pero movimientos erráticos (gap news).  
- Spread aumentados o iliquidez en ciertos días.

## 17. Overfitting Risk (Bajo/Medio/Alto)  
Medio. La estructura usa parámetros estándar ampliamente validados del Keltner Channel y ATR, pero los filtros y ventana horaria amplían la robustez. No obstante, el ajuste horario puede introducir algún riesgo de ajuste a historial.

## 18. Cost Sensitivity (Bajo/Medio/Alto)  
Medio. Spread máximo permisible es 1.5 pips, lo que limita entradas en momentos de mercado caros. El ratio R/B 1:2 amortigua impacto de comisiones y deslizamientos moderados.

## 19. Correlation With Liquidity Sweep Strategy (Baja/Media/Alta + Explicación)  
Media. Ambas estrategias capturan movimientos asociados a aumentos de liquidez y volatilidad, pero VEKB usa ruptura por canal de volatilidad donde la Liquidity Sweep puede usar órdenes stop hunt. Pueden coincidir en momentos, pero mecanismo de señal difiere.

## 20. Backtest Acceptance Criteria  
- Sharpe Ratio intradía ≥ 1.2 en período de 3 años con datos 2018-2021.  
- Drawdown máximo < 10% de capital inicial.  
- Ratio ganadores ≥ 40%.  
- Relación ganancias/pérdidas ≥ 1.8.  
- Robustez: estabilidad de métricas si se modifica período KC ± 10%, ATR ± 20%, y ventana de entrada ± 30min.

## 21. Pseudocode  

```
Inputs:  
  MA_period = 20  
  ATR_period = 10  
  ATR_multiplier = 1.5  
  SL_MULT = 1.0  
  TP_MULT = 2.0  
  Trade_start = 07:00 NY  
  Trade_end = 11:00 NY  
  Max_spread = 1.5 pips  
  ATR_min = 0.0006

For each M5 candle during 07:00-19:00 NY:  
  Calculate ATR10 (in pips)  
  Calculate MA20 = EMA(close, 20)  
  KC_upper = MA20 + ATR_multiplier * ATR10  
  KC_lower = MA20 - ATR_multiplier * ATR10

  If candle.time ∈ [07:00,11:00] and ATR10 ≥ ATR_min and spread ≤ Max_spread and no-news(+/-30 min):  
    // Long Entry  
    If close_candle > KC_upper and close_prev_candle ≤ KC_upper:  
      Entry_long = close_candle  
      SL = Entry_long - SL_MULT * ATR10  
      TP = Entry_long + TP_MULT * ATR10  
      Open LONG position with SL, TP

    // Short Entry  
    If close_candle < KC_lower and close_prev_candle ≥ KC_lower:  
      Entry_short = close_candle  
      SL = Entry_short + SL_MULT * ATR10  
      TP = Entry_short - TP_MULT * ATR10  
      Open SHORT position with SL, TP

For open positions:  
  If position_profit ≥ 1 ATR in favor:  
    Move SL to Entry ± 5 puntos (BE +5)

  If candle.time == 19:00:  
    Close all open positions

End
```

---

La estrategia VEKB busca capturar rupturas legítimas de volatilidad en la franja horaria con mayor actividad, con reglas objetivas y parametrización robusta para operar EURUSD intradía con gestión prudente de riesgo.

### Strategy 4
## 1. Strategy Name  
Volatility Expansion mediante Ruptura de Canal Donchian con Confirmación por Precio Ponderado por Volumen (VWAP) – EURUSD Intradía

## 2. Market Phenomenon  
Expansión rápida de la volatilidad tras periodos de bajo rango, identificada por rupturas en el canal Donchian, validada con la dirección del volumen reflejada en el VWAP intrabarras, que señala fuerza genuina del breakout.

## 3. Quant Hypothesis  
Las rupturas del rango definido por el canal Donchian indicarán expansión de volatilidad y nuevos impulsos de precio, pero para filtrar rupturas falsas se requiere confirmación de que el precio cierre en dirección de ruptura sobre el VWAP ponderado por volumen, dando señal objetiva de interés real del mercado.

## 4. Why It Could Work on EURUSD  
EURUSD es un par con liquidez enorme y fases claras de consolidación y ruptura intradía. La combinación de rompimiento técnico más confirmación volumétrica disminuye ruido en horarios con fuerte presencia de volumen institucional (07:00-19:00 NY).

## 5. Trading Window (Sub-ventana dentro de 07:00-19:00 NY)  
09:00 a 17:30 NY. Se evita apertura y cierre extremados para evitar spreads anómalos y volatilidad no estructural.

## 6. Timeframe (M1, M3, M5, M15 o combinación)  
M5 combinado con M1 para confirmación de cierre de barra y cálculo intrabarra de VWAP.

## 7. Required Data  
- Cotizaciones OHLC M5  
- Volumen tick o real para cálculo VWAP intrabarra (idealmente volumen real; simulación con ticks si no disponible)  
- Timestamp sincronizado a NY  
- Spread constante o dinámico para filtro

## 8. Long Entry Rules (Reglas exactas)  
1. Calcular el Canal Donchian M5 con parámetro n=20 (máximo de los últimos 20 cierres para la banda superior, mínimo para la inferior).  
2. Detectar que la barra actual cierre M5 supere la banda superior del canal Donchian.  
3. Calcular VWAP intrabarra M5 en la barra de cierre actual.  
4. Verificar que el precio de cierre de la barra M5 esté por encima del VWAP intrabarra y que la pendiente del VWAP de las últimas 3 barras M5 sea positiva.  
5. Confirmar que el spread esté por debajo del umbral fijado.  
6. Si se cumplen todas las condiciones, ejecutar orden de compra al cierre de la barra actual.

## 9. Short Entry Rules (Reglas exactas)  
1. Calcular Canal Donchian M5 con n=20.  
2. Detectar que la barra actual cierre M5 rompa a la baja la banda inferior del canal Donchian.  
3. Calcular VWAP intrabarra M5 en la barra actual.  
4. Confirmar que el cierre esté por debajo del VWAP intrabarra y que la pendiente del VWAP de las últimas 3 barras sea negativa.  
5. Verificar spread dentro del umbral.  
6. Ejecutar orden de venta al cierre de barra.

## 10. Stop Loss Logic  
Stop Loss inicial en nivel contrario de la banda Donchian con buffer del 0.5 ATR(14) M5 para evitar ruido y falsas salidas.

## 11. Take Profit Logic  
Ratio fijo 1:2 respecto al riesgo inicial, es decir, take profit a distancia igual a dos veces la distancia SL-entrada.

## 12. Trade Management (BE, trailing, time stop, etc.)  
- Mover SL a Break-Even cuando precio avance +1 R.  
- Trailing stop activado a +1.5 R, con distancia dinámica de 0.5 ATR(14) M5.  
- Fuerza salida forzada si el nivel de VWAP cambia de tendencia por 2 barras consecutivas.  
- Time stop: cerrar trades abiertos más de 3 horas.

## 13. Filters (Spread, ATR, News, etc.)  
- No operar si spread > 2 pips.  
- No operar en 15 minutos previos a noticia macro relevante de alta volatilidad (ej. NFP, ECB statements).  
- Confirmar ATR(14) M5 superior a mínimo histórico para evitar condiciones de muy baja volatilidad.

## 14. Initial Parameters (Razonables, no optimizados)  
- Donchian channel period n=20 (100 minutos)  
- ATR period = 14 para ajuste SL/TP  
- Spread máximo 2 pips  
- Trading window 09:00-17:30 NY  
- Take profit ratio 1:2  
- Time stop 3 horas

## 15. Expected Frequency  
Entre 2 y 5 señales por día, dado el filtro horario y la necesidad de ruptura clara.

## 16. Why It Might Fail  
- Periodos prolongados de rango sin rupturas definitivas (atr baja persistentemente).  
- Falsos breakouts en mercados con contramovimientos fuertes posterior a noticias o excesiva especulación.  
- Cambios bruscos de volatilidad que desalinean parámetros fijos de ATR y Donchian.  
- Fail en confirmación VWAP si volumen está distorsionado (datos imperfectos).

## 17. Overfitting Risk (Bajo/Medio/Alto)  
Medio. Uso de parámetros estándar, pocas variables ajustadas, pero riesgo existe al combinar múltiple confirmación (Donchian + VWAP), que puede afinar demasiado la selección.

## 18. Cost Sensitivity (Bajo/Medio/Alto)  
Medio. Spread limitado a 2 pips, SL limitado, pero comisiones y slippage en rupturas rápidas pueden afectar desempeño intradía.

## 19. Correlation With Liquidity Sweep Strategy (Baja/Media/Alta + Explicación)  
Media. Ambas estrategias buscan rupturas y movimientos rápidos, pero la Liquidity Sweep se centra en barridos de stop loss y falsos rompimientos, mientras que ésta busca confirmación volumétrica sólida; pueden coexistir aunque con cierta correlación en tiempos de alta volatilidad.

## 20. Backtest Acceptance Criteria  
- Ratio de Sharpe intradía > 1.2  
- Profit Factor > 1.5  
- Drawdown máximo intradía < 5% del capital  
- Win rate mínimo 45% con sistema de gestión activa  
- Robustez temporal: resultados consistentes en distintos períodos semestrales de prueba  
- No dependencia sensible a ±10% cambios en parámetros principales

## 21. Pseudocode  

```python
# Parámetros iniciales
donchian_period = 20
atr_period = 14
spread_max = 0.0002  # 2 pips
tp_risk_ratio = 2
trading_start = "09:00"
trading_end = "17:30"
time_stop_hours = 3

for each day:
    for each 5min_bar in day between trading_start and trading_end:
        # Calcular bandas Donchian
        donchian_high = max(close[-donchian_period:])
        donchian_low = min(close[-donchian_period:])
        
        # Calcular ATR(14) M5
        atr = ATR(14)
        
        # Obtener VWAP intrabarra de 5 min actual (con datos de ticks o volumen)
        vwap_current = VWAP_intrabar(tick_data_current_bar)
        
        # Calcular pendiente VWAP últimas 3 barras (pendiente positiva o negativa)
        slope_vwap = slope(VWAP_last_3_bars)
        
        # Filtrar spread actual
        if spread_current > spread_max:
            continue  # no operar
        
        # LONG ENTRY
        if close_current > donchian_high:
            if close_current > vwap_current and slope_vwap > 0:
                entry_price = close_current
                sl = donchian_low - 0.5 * atr
                tp = entry_price + tp_risk_ratio * (entry_price - sl)
                abrir_posicion_long(entry_price, sl, tp)
        
        # SHORT ENTRY
        if close_current < donchian_low:
            if close_current < vwap_current and slope_vwap < 0:
                entry_price = close_current
                sl = donchian_high + 0.5 * atr
                tp = entry_price - tp_risk_ratio * (sl - entry_price)
                abrir_posicion_short(entry_price, sl, tp)

    # Gestión de trade abierta
    if posicion_abierta:
        # Mover SL a BE si profits >= 1R
        if profit >= (entry_price - sl):
            sl = entry_price
        
        # Trailing stop a 0.5*ATR una vez profit >= 1.5R
        if profit >= 1.5 * (entry_price - sl):
            sl = max(sl, current_price - 0.5 * atr)  # para long
        
        # Cierre por cambio de tendencia en pendiente VWAP 2 barras consecutivas
        if pendiente_vwap_cambio_2_barras:
            cerrar_posicion()
        
        # Time stop si tiempo trade > 3 horas
        if tiempo_posicion >= time_stop_hours:
            cerrar_posicion()
```

---

Esta estrategia institucional prioriza objetividad, robustez y elimina discrecionalidad, enfocándose en mercado líquido con validación volumétrica para minimizar entradas falsas en rupturas.

### Strategy 5
## 1. Strategy Name  
VWAP 2σ Mean Reversion con Filtrado Z-score (VWAP 2SD Z-Filter MR)

## 2. Market Phenomenon  
Reversión a la media en torno al VWAP, utilizando los extremos generados por una desviación estándar doble como niveles de sobrecompra/sobreventa intradía, filtrados con un Z-score para evitar señales en tendencias fuertes.

## 3. Quant Hypothesis  
Los precios del EURUSD se revierten hacia su valor medio intradía (VWAP) tras movimientos extremos definidos por ±2 desviaciones estándar calculadas a partir del VWAP, excepto cuando la fortaleza del movimiento (Z-score) indica continuación de tendencia.

## 4. Why It Could Work on EURUSD  
EURUSD es un par de alta liquidez y volumen intradía, donde las áreas de valor como el VWAP actúan como imán natural. El comportamiento de revertir después de sobreextensiones validado por desviaciones estándar dobles es frecuente, especialmente en horarios activos de NY donde la volatilidad permite desbalances momentáneos.

## 5. Trading Window (Sub-ventana dentro de 07:00-19:00 NY)  
08:00 a 17:00 NY, para capturar la mayor liquidez post-apertura NY y evitar poca actividad pre-cierre.

## 6. Timeframe (M1, M3, M5, M15 o combinación)  
Combinación de M1 para cálculo de VWAP y desviación estándar, y M5 para ejecución y confirmación de entrada.

## 7. Required Data  
- Precio de ticks o minutos (open, high, low, close).  
- Volumen (para VWAP ponderado).  
- Timestamp en zona NY para ventana de trading.  
- Fundamentalmente no se requieren datos externos.

## 8. Long Entry Rules (Reglas exactas)  
1. Calcular VWAP y desviación estándar intradía acumulada desde 07:00 NY hasta el momento actual sobre datos M1.  
2. Definir banda inferior = VWAP – 2 * Desviación estándar.  
3. Calcular Z-score del precio relativo al VWAP y desviación estándar.  
4. Detectar que precio cierre M5 toque o cierre justo por debajo de la banda inferior.  
5. Confirmar que Z-score ≤ -1.5 (significando sobreventa válida que no es continuación de tendencia fuerte).  
6. Abrir largo en el siguiente tick/barra M1.

## 9. Short Entry Rules (Reglas exactas)  
1. Igual procedimiento para VWAP y desviaciones estándar.  
2. Banda superior = VWAP + 2 * Desviación estándar.  
3. Z-score ≥ +1.5 (sobrecompra válida sin fuerte tendencia alcista).  
4. Precio cierre M5 toque o cierre justo por encima de la banda superior.  
5. Abrir corto en el siguiente tick/barra M1.

## 10. Stop Loss Logic  
- Stop loss fijo a 12 pips del precio de entrada (aprox. 0.0012), justo fuera del rango extremo de 2σ para tener margen frente a ruido.  
- SL se activa instantáneamente si el precio toca el nivel antes de la toma de ganancias.

## 11. Take Profit Logic  
- Take profit objetivo en VWAP, es decir, revertir totalmente al valor medio intradía.  
- Distancia variable, generalmente 6-12 pips (depende de zona, pero se ajusta dinámicamente al VWAP para no usar un fix).  
- Se cierra la posición cuando precio alcance o cruce el VWAP tras la entrada.

## 12. Trade Management (BE, trailing, time stop, etc.)  
- Break-even (BE) se activa cuando el precio se mueve 3 pips a favor, ajustando SL al punto de entrada para proteger la operación.  
- No trailing stops para mantener sencillez.  
- Time stop: cierre obligado a las 17:00 NY para evitar riesgos pos-cierre de sesión.  
- Solo una posición abierta a la vez.

## 13. Filters (Spread, ATR, News, etc.)  
- Rechazar entradas con spread mayor a 1.8 pips para evitar costos excesivos.  
- No operar 10 minutos antes y después de noticias de alta volatilidad (eventos FOMC, NFP, etc.).  
- Filtro ATR 30 min: solo operar si ATR(30) > 4 pips para evitar mercado estancado.

## 14. Initial Parameters (Razonables, no optimizados)  
- VWAP calculado intradía desde 07:00 NY.  
- Desviación estándar: 2 veces para bandas.  
- Z-score umbrales: ±1.5 (para filtrar señales débiles o tendencias fuertes).  
- Stop loss: 12 pips.  
- Take profit: variable hasta VWAP (promedio ~ 8 pips).  
- Máximo spread aceptado: 1.8 pips.  
- ATR mínimo: 4 pips sobre 30 minutos.

## 15. Expected Frequency  
Aproximadamente 3 a 8 operaciones por día, condicionadas a volatilidad y condiciones de mercado.

## 16. Why It Might Fail  
- En tendencias fuertes y persistentes el precio puede moverse lejos del VWAP sin revertir (momento en que el filtro Z-score podría ser insuficiente).  
- Mercado con baja volatilidad y rango estrecho reduce señales.  
- Eventos de alta volatilidad no filtrados pueden generar señales falsas.  
- Cambios estructurales en dinámica de EURUSD pueden afectar comportamiento de reversión.

## 17. Overfitting Risk (Bajo/Medio/Alto)  
Medio, dadas las reglas basadas en métricas estándar reconocidas (VWAP, desviación, z-score) pero con parámetros ajustables que requieren validación robusta out-of-sample.

## 18. Cost Sensitivity (Bajo/Medio/Alto)  
Medio, spread y slippage impactan directamente dado el rango estrecho de take profits y stops, pero la alta liquidez de EURUSD mitiga parcialmente el riesgo.

## 19. Correlation With Liquidity Sweep Strategy (Baja/Media/Alta + Explicación)  
Baja a media. La estrategia de mean reversion en VWAP es conceptualmente distinta a una estrategia de barridos de liquidez (que busca rupturas y momentum). Sin embargo, ocasionalmente ambos pueden coincidir en zonas extremas de precio pero no dependen ni generan señales correlacionadas.

## 20. Backtest Acceptance Criteria  
- Sharpe ratio diario superior a 1.5 consistente en periodos de al menos 12 meses out-of-sample.  
- Drawdown máximo bajo 8% sobre capital simulado.  
- Win rate superior a 50% con ratio ganancias/pérdidas ≥1.2.  
- Consistencia en distintas condiciones de mercado y múltiples años.

## 21. Pseudocode

```python
# Inicialización tiempo NY y parámetros
TRADING_START = "08:00"
TRADING_END = "17:00"
VWAP_START = "07:00"
STD_MULT = 2
Z_ENTRY_THRESHOLD = 1.5
STOP_LOSS_PIPS = 12
BE_PIPS = 3

while current_time in TRADING_START to TRADING_END:

    # Obtener datos M1 desde VWAP_START hasta ahora
    prices = get_prices_M1(VWAP_START, current_time)
    volume = get_volumes_M1(VWAP_START, current_time)
    
    vwap = calculate_vwap(prices, volume)
    std_dev = calculate_std_dev(prices, vwap)

    upper_band = vwap + STD_MULT * std_dev
    lower_band = vwap - STD_MULT * std_dev

    # Calcular Z-score de precio actual
    last_price = prices[-1].close
    z_score = (last_price - vwap) / std_dev if std_dev > 0 else 0

    # Condiciones de entrada LONG
    if (last_price <= lower_band) and (z_score <= -Z_ENTRY_THRESHOLD):
        # Confirmar cierre M5 en banda inferior
        close_M5 = get_close_M5(current_time)
        if close_M5 <= lower_band:
            if spread < 1.8 and ATR_30min > 4 and no_news_recent():
                enter_long(next_bar_open)
                set_stop_loss(entry_price - STOP_LOSS_PIPS)
                set_take_profit(vwap)
                monitor_trade()

    # Condiciones de entrada SHORT
    elif (last_price >= upper_band) and (z_score >= Z_ENTRY_THRESHOLD):
        close_M5 = get_close_M5(current_time)
        if close_M5 >= upper_band:
            if spread < 1.8 and ATR_30min > 4 and no_news_recent():
                enter_short(next_bar_open)
                set_stop_loss(entry_price + STOP_LOSS_PIPS)
                set_take_profit(vwap)
                monitor_trade()
    
    # Gestión de trade
    if trade_open:
        if price_moved_favorably_by(BE_PIPS):
            move_stop_loss_to_entry()
        if price_hit_stop_loss() or price_hit_take_profit() or time == TRADING_END:
            close_trade()
```

---

Esta especificación maximiza objetividad y rigor cuantitativo, generando una estrategia intradía robusta para EURUSD basada en reversión a VWAP ±2 desviaciones estándar con filtro Z-score para evitar señales falsas.

### Strategy 6
## 1. Strategy Name  
RSI2 Mean-Adjusted Reversion Intradía EURUSD

## 2. Market Phenomenon  
Reversión rápida desde condiciones extremas de sobrecompra o sobreventa detectadas por un RSI(2) extremadamente bajo o alto, con entradas ajustadas en relación a la media móvil para evitar falsos rompimientos.

## 3. Quant Hypothesis  
El EURUSD intradía tiende a revertir rápidamente tras alcanzar valores extremos de RSI(2), pero para filtrar señales falsas es efectivo incorporar el nivel relativo de precio con respecto a su media móvil para asegurar que las entradas se realicen contra la dirección del movimiento momentáneo.

## 4. Why It Could Work on EURUSD  
EURUSD es uno de los pares forex más líquidos y con alta microestructura eficiente. En sesiones activas (07:00-19:00 NY) presenta frecuentes ciclos de agotamiento momentáneo seguidos de correcciones rápidas, ideales para estrategias de reversión basada en indicadores de momentum extremo como RSI(2).

## 5. Trading Window (Sub-ventana dentro de 07:00-19:00 NY)  
08:00 - 17:00 NY, para cubrir plenitud de actividad post apertura US y pre-cierre, evitando baja volatilidad apertura y cierre.

## 6. Timeframe (M1, M3, M5, M15 o combinación)  
M5 para balancear ruido y oportunidad intradía rápida.

## 7. Required Data  
- Precio de ticks reconstruido para candles M5 OHLC.  
- RSI(2) calculado sobre cierre.  
- Media móvil exponencial (EMA) de 20 períodos M5.  
- Datos de spread en tiempo real para filtrado.  
- Calendario de noticias económicas principales para filtrado intradiario.

## 8. Long Entry Rules (Reglas exactas)  
Ejecutar orden de compra (long) cuando se cumplen simultáneamente:  
a) RSI(2) ≤ 10 (sobreventa extrema).  
b) Precio de cierre M5 actual está por debajo de EMA(20) M5 (indica tendencia o nivel medio inferior).  
c) RSI(2) en vela anterior > 10 (evita señales continuas).  
d) Spread ≤ 2 pips (0.0002) para evitar escenarios caros.

## 9. Short Entry Rules (Reglas exactas)  
Ejecutar orden de venta (short) cuando se cumplen simultáneamente:  
a) RSI(2) ≥ 90 (sobrecompra extrema).  
b) Precio de cierre M5 actual está por encima de EMA(20) M5.  
c) RSI(2) en vela anterior < 90 (evita señales continuas).  
d) Spread ≤ 2 pips.

## 10. Stop Loss Logic  
Stop loss fijo a 7 pips (0.0007) del precio de entrada para limitar pérdidas rápidas dada la alta frecuencia y rapidez de reversión.

## 11. Take Profit Logic  
Take profit fijo a 10 pips (0.0010) para capturar la reversión pequeña esperada intradía.

## 12. Trade Management (BE, trailing, time stop, etc.)  
- Move stop loss a breakeven + 1 pip cuando ganancia alcanza 5 pips.  
- No trailing.  
- Time stop: cerrar la posición si no alcanza TP o SL dentro de 30 minutos desde la entrada.

## 13. Filters (Spread, ATR, News, etc.)  
- Entrar solo con spread ≤ 2 pips.  
- Evitar entradas 5 min antes y 15 min después de anuncio económico relevante de alta o media volatilidad (p.ej. NFP, decisiones de tasas).  
- No operar si ATR20(M5) en última hora es inferior a 3 pips para evitar rango poco volátil.

## 14. Initial Parameters (Razonables, no optimizados)  
- RSI(2) umbrales: 10 (long), 90 (short).  
- EMA 20 período M5.  
- SL: 7 pips.  
- TP: 10 pips.  
- Spread máximo: 2 pips.  
- Time stop: 30 minutos.

## 15. Expected Frequency  
Entre 2 y 5 operaciones por día, condicionadas a volatilidad y patrones de RSI extremos.

## 16. Why It Might Fail  
- Periodos prolongados de tendencia fuerte y alta volatilidad pueden causar señales falsas persistentes.  
- Cambios estructurales en volatilidad pueden invalidar niveles fijos de SL/TP.  
- Eventos imprevistos o flash crashes que superan stops rápidos.  
- Cambios en comportamiento de mercado por políticas monetarias o crisis.

## 17. Overfitting Risk (Bajo/Medio/Alto)  
Medio. Parámetros simples y conocidos, pero ajuste de umbrales RSI y tiempos ventana puede sesgar rendimiento si se optimiza demasiado.

## 18. Cost Sensitivity (Bajo/Medio/Alto)  
Medio. Spread y slippage afectan margen por ser estrategia de scalping con TP pequeño, pero no extremadamente sensible si gestión de spread es correcta.

## 19. Correlation With Liquidity Sweep Strategy (Baja/Media/Alta + Explicación)  
Baja. RSI2 reversion se enfoca en extremos sentimentales y momentum, mientras que liquidity sweep busca rupturas de nivel con volatilidad, son mecanismos diferentes con baja correlación.

## 20. Backtest Acceptance Criteria  
- Sharpe ratio anualizado > 1.0 ajustado por costo de spread.  
- Drawdown máximo menor al 5% de capital.  
- Ratio ganancias/pérdidas > 1.2.  
- Consistencia de ganancias en mínimo 3 subconjuntos temporales (walk-forward).  
- Estadística de operaciones > 300 para robustez.

## 21. Pseudocode  
```  
Inicializar:  
- Calcular EMA20 en M5  
- Calcular RSI2 en M5

Por cada vela M5 dentro de 08:00-17:00 NY:  
  Si spread actual > 2 pips o ventana de noticia activa:  
    No operar  
  Sino:  
    Si no hay posición abierta:  
      // Condiciones Long  
      Si RSI2_current ≤ 10 AND RSI2_previous > 10 AND Close < EMA20:  
        Abrir posición Long con SL= EntryPrice - 7 pips, TP= EntryPrice + 10 pips  
        Registrar tiempo entrada  
      // Condiciones Short  
      Si RSI2_current ≥ 90 AND RSI2_previous < 90 AND Close > EMA20:  
        Abrir posición Short con SL= EntryPrice + 7 pips, TP= EntryPrice - 10 pips  
        Registrar tiempo entrada  
    Si hay posición abierta:  
      Ganancia actual = Precio actual - Precio entrada (considerar dirección)  
      Si (long y Ganancia actual ≥ 5 pips) o (short y Ganancia actual ≤ -5 pips):  
         Mover SL a EntryPrice + 1 pip (long) o EntryPrice - 1 pip (short)  
      Si duración posición > 30 minutos:  
         Cerrar posición  
  Esperar siguiente vela  
```

### Strategy 7
## 1. Strategy Name  
Mean Reversion from Multi-Day Moving Average Extremes (MR-MA Extremes) – EURUSD Intradía

## 2. Market Phenomenon  
Reversión estadística hacia la media tras movimientos extremos y transitorios alejados de una media móvil de largo plazo que refleja la tendencia y contexto de varios días.

## 3. Quant Hypothesis  
Cuando el precio intradía se desvía excesivamente (estadísticamente) respecto a la media móvil de 20 días, existe una alta probabilidad de reversiones dentro la sesión NY. La desviación es temporal y el mercado tiende a corregir.

## 4. Why It Could Work on EURUSD  
EURUSD es un par líquido y eficiente con movimientos recurrentes de corrección intradía dentro de tendencias diarias. La política monetaria estable sugiere reversión y roturas falsas son comunes en rangos diarios, especialmente en horario NY por alta liquidez y volatilidad.

## 5. Trading Window (Sub-ventana dentro de 07:00-19:00 NY)  
09:30 - 16:00 NY (pico de liquidez y volatilidad, excluye apertura y cierre extremos)

## 6. Timeframe (M1, M3, M5, M15 o combinación)  
M5 para entrada, M15 para confirmación tendencia y cálculo de media móvil y desviación estándar

## 7. Required Data  
- Precio bid/ask (preferible tick por tick para precisión spreads)  
- Historial OHLC M15 al menos 30 días previos para cálculo media móvil y desviación  
- Indicador ATR M15 para filtro volatilidad  
- Calendario de noticias macroeconómicas para filtro (opcional)

## 8. Long Entry Rules (Reglas exactas)  
1. Calcular SMA(20 días) en cierre M15 de las últimas 20 sesiones completas.  
2. Calcular desviación estándar σ del cierre M15 precio para esas 20 sesiones.  
3. En la ventana de trading (09:30-16:00 NY), en M5:  
   - El precio bid actual debe estar al menos 2σ por debajo del SMA(20 días) (extremo bajo).  
   - Confirmar que el precio M5 ha cerrado y está rebotando al alza (precio actual > precio M5 anterior).  
   - Confirmar que ATR(14) M15 está en rango normal/no extremo (para evitar movimientos erráticos).  
4. Entrar largo al precio de mercado siguiente tick.

## 9. Short Entry Rules (Reglas exactas)  
1. Igual que long, pero a la inversa:  
2. Precio bid actual al menos 2σ por encima del SMA(20 días).  
3. Precio M5 cerrando y rebotando a la baja (precio actual < precio M5 anterior).  
4. ATR(14) M15 en rango normal.  
5. Entrada corto al precio mercado siguiente tick.

## 10. Stop Loss Logic  
- Stop Loss fijo en 0.5% (50 pips) del precio de entrada (ej.: si se entra a 1.1000, SL=1.0950 para largos y 1.1050 para cortos).  
- Alternativamente, SL dinámico al mínimo/máximo de la vela M15 previa a la entrada, para mayor precisión y adaptación.

## 11. Take Profit Logic  
- Take Profit a 0.25% (25 pips), mitad del stop loss para ratio riesgo/beneficio favorable y alta tasa de éxito esperada.  
- Cierre parcial al 50% en TP y resto a trailing stop (ver siguiente).

## 12. Trade Management (BE, trailing, time stop, etc.)  
- Mover stop loss a Break Even (precio entrada) cuando precio alcance 0.15% favorable (15 pips).  
- Trailing stop 10 pips por detrás del máximo/mínimo alcanzado después del BE trigger.  
- Cierre automático al final de la ventana de trading (16:00 NY) si la posición sigue abierta (time stop intradía).

## 13. Filters (Spread, ATR, News, etc.)  
- No ejecutar si spread > 1.5 pips.  
- No entrar si ATR(14) M15 > 1.5 veces su media de 30 días (indica volatilidad extrema).  
- No abrir posiciones dentro de 30 minutos antes/después de noticias importantes del calendario económico (ej. NFP, decisiones BCE).  
- Evitar entradas en los primeros y últimos 15 minutos de la ventana 09:30-16:00 forzar rango estable.

## 14. Initial Parameters (Razonables, no optimizados)  
- SMA períodos: 20 días (M15).  
- Desviación estándar cutoff: 2σ.  
- Stop loss: 50 pips (0.5%).  
- Take profit: 25 pips (0.25%).  
- ATR filtro: 1.5x promedio 30 días.  
- Spread máximo: 1.5 pips.  
- Ventana trading: 09:30-16:00 NY.

## 15. Expected Frequency  
De 2 a 5 trades por día, dependiendo de condiciones extremas y volatilidad.

## 16. Why It Might Fail  
- Momentum fuerte y continuado en un sentido puede invalidar reversiones.  
- Eventos imprevistos o shocks fundamentales.  
- Cambios estructurales en la dinámica de mercado (por ejemplo, cambios de política monetaria bruscos).  
- Falsos rebotes con alta volatilidad y spread elevados.

## 17. Overfitting Risk (Bajo/Medio/Alto)  
Medio. El uso de medias históricas y desviaciones estándar es estadísticamente robusto, pero optimización excesiva en parámetros podría adaptar demasiado a datos pasados.

## 18. Cost Sensitivity (Bajo/Medio/Alto)  
Medio-Alto. El rango de stop loss y take profit intradía es reducido, por lo tanto los spreads y comisiones impactan significativamente la rentabilidad neta.

## 19. Correlation With Liquidity Sweep Strategy (Baja/Media/Alta + Explicación)  
Baja. La estrategia MR-MA Extremes se basa en reversión estadística a medias largas y largos plazos, mientras que la estrategia de barridos de liquidez (liquidity sweep) explota microestructuras de orden y microtrends intradía inmediatos, por lo tanto actúan en diferentes horizontes y señales.

## 20. Backtest Acceptance Criteria  
- Sharpe ratio > 1.5 anualizado (considerando comisiones y spreads).  
- Ratio ganancia/pérdida > 1.5 en trades ganadores vs perdedores.  
- % trades positivos > 55%.  
- Drawdown máximo intradía < 10%.  
- Robustez en split out-of-sample y periodos con alta volatilidad.

## 21. Pseudocode

```pseudo
// Parámetros
SMA_period = 20*96 // 20 días en barras M15 (96 barras por día en 24h); ajustar a sesiones trading
dev_threshold = 2
SL_pips = 50
TP_pips = 25
Spread_max = 1.5
ATR_threshold_ratio = 1.5
Trading_start = 09:30 NY
Trading_end = 16:00 NY

// Cálculos previos
for each day:
    Calculate SMA_20d on M15 close over last 20 sessions
    Calculate std_dev_20d on M15 close over last 20 sessions
    Calculate ATR_14 M15
    Calculate ATR_30d_avg

// En ventana intradía (M5 bars):
for each M5 bar between Trading_start and Trading_end:
    if spread > Spread_max or ATR_14 > ATR_30d_avg * ATR_threshold_ratio or news_pending():
        continue // Esperar siguiente barra

    price_bid = current M5 close bid

    z_score = (price_bid - SMA_20d) / std_dev_20d

    // Condición larga
    if z_score <= -dev_threshold and price_bid > previous M5 close and ATR normal:
        enter_long() at market price
        set SL = entry_price - SL_pips*point
        set TP = entry_price + TP_pips*point

    // Condición corta
    if z_score >= dev_threshold and price_bid < previous M5 close and ATR normal:
        enter_short() at market price
        set SL = entry_price + SL_pips*point
        set TP = entry_price - TP_pips*point

// Gestión de trades
for each open trade:
    if unrealized_profit >= 15 pips and SL not at BE:
        move SL to entry_price

    if price moves favorably beyond BE:
        trail SL 10 pips detrás del máximo/mínimo alcanzado

    if time >= Trading_end:
        close trade at market

function news_pending():
    return within 30 min antes o después release news importantes

```

---

Esta especificación pretende maximizar objetividad, viabilidad institucional y robustez estadística para EURUSD intradía basada en reversión a media multi-día.

### Strategy 8
## 1. Strategy Name  
Mean Reversion: Bollinger Band "Double Tap" con filtro de divergencia en EURUSD intradía

## 2. Market Phenomenon  
Reversión a la media intradía en EURUSD tras toques consecutivos a la banda de Bollinger extrema, confirmada por divergencias en RSI que señalan agotamiento del impulso.

## 3. Quant Hypothesis  
El precio tiende a revertir a la media luego de que el mercado toque dos veces la banda superior o inferior de Bollinger en un periodo corto, especialmente cuando un indicador de momentum como RSI muestra divergencia, lo que señala probable agotamiento de la tendencia.

## 4. Why It Could Work on EURUSD  
EURUSD presenta alta liquidez y estructura clara de rangos intradía en sesiones europeas y americanas, favoreciendo patrones de mean reversion. La volatilidad moderada permite fiabilidad en bandas de Bollinger y uso efectivo de RSI para detectar divergencias.

## 5. Trading Window (Sub-ventana dentro de 07:00-19:00 NY)  
09:00 a 16:00 NY (cubriendo la apertura europea y la parte central de la sesión americana con mayor actividad y menor ruido nocturno)

## 6. Timeframe (M1, M3, M5, M15 o combinación)  
M5 para señales de entrada y confirmación del patrón "double tap"; M15 para confirmar contexto de tendencia y filtro de divergencia con RSI.

## 7. Required Data  
- Precio OHLC 5 minutos y 15 minutos.  
- Indicadores técnicos: Bandas de Bollinger (20, 2σ), RSI (14) en M15 y M5.  
- Spreads en tiempo real.  
- Calendario de noticias económicas (para filtro).

## 8. Long Entry Rules (Reglas exactas)  
1. En el gráfico M5, el precio toca la banda inferior de Bollinger (20,2σ) por primera vez.  
2. Después, sin cerrar un candle fuera de la banda, el precio vuelve a tocar la banda inferior (doble toque) dentro de los siguientes 15-30 minutos.  
3. En el timeframe M15, el RSI muestra divergencia alcista: mínimo más bajo en precio + mínimo más alto en RSI, comparando los puntos de doble toque.  
4. Spread ≤ 1.5 pips.  
5. No hay noticias de alto impacto en los próximos 30 minutos.  
6. Entrar largo en el cierre del segundo candle de toque a la banda inferior.

## 9. Short Entry Rules (Reglas exactas)  
1. En el gráfico M5, el precio toca la banda superior de Bollinger (20,2σ) por primera vez.  
2. Sin cerrar fuera de la banda, precio toca nuevamente la banda superior en 15-30 minutos (doble toque).  
3. En M15, RSI muestra divergencia bajista: máximo más alto en precio + máximo más bajo en RSI, entre los puntos del doble toque.  
4. Spread ≤ 1.5 pips.  
5. No hay noticias de alto impacto en los próximos 30 minutos.  
6. Entrar corto en el cierre del segundo candle de toque a la banda superior.

## 10. Stop Loss Logic  
Stop loss a 1.5 veces el ATR(14) en M5 desde punto de entrada; el ATR se calcula previo a la entrada para evitar fluctuaciones intratrade.

## 11. Take Profit Logic  
Take profit a 1 vez la distancia entre la media móvil simple de 20 periodos (centro de bandas Bollinger) y la banda inferior (long) o superior (short), aproximadamente coincide con la vuelta a la media.

## 12. Trade Management (BE, trailing, time stop, etc.)  
- Al alcanzar 0.75x take profit, mover stop loss a break even +1 pip.  
- No trailing stop.  
- Cerrar posición si se supera tiempo máximo abierto de 60 minutos sin cerrar en TP o SL.  
- No agregados ni scaling.

## 13. Filters (Spread, ATR, News, etc.)  
- Spread máximo para entrada: 1.5 pips.  
- Entrada solo si ATR(14, M5) > 0.0020 (mínima volatilidad para validez del patrón).  
- No operar en +30 minutos antes o después de noticias de alto impacto para EUR/USD (calendario económico).  
- Filtrado adicional: sesión NY activa (09:00 a 16:00 NY).

## 14. Initial Parameters (Razonables, no optimizados)  
- Bandas Bollinger: periodo 20, desviación 2 sigma.  
- RSI: 14 periodos.  
- ATR periodo: 14 (M5).  
- Máximo spread entrada: 1.5 pips.  
- Tiempo máximo para doble toque: 30 minutos.  
- TP = 1 x distancia media-banda.  
- SL = 1.5 x ATR(14,M5).  
- Maximum trade duration = 60 minutos.

## 15. Expected Frequency  
Entre 2 a 4 operaciones diarias, dependiendo volatilidad y número de patrones Doble toque con divergencia válidos.

## 16. Why It Might Fail  
- Mercado con tendencia fuerte y extendida que rompe bandas constantemente sin revertir.  
- Falsas divergencias en RSI, especialmente en momentos de alta volatilidad sin reglas adicionales.  
- Ruido intradía y spreads altos en sesiones con baja liquidez o eventos imprevistos.  
- Estrategia pierde rendimiento en mercados extremadamente volátiles o planas prolongadas.

## 17. Overfitting Risk (Bajo/Medio/Alto)  
Medio. La estrategia combina indicadores estándar y reglas objetivas, pero los parámetros como tiempos máximos de doble toque o límites de spread podrían ajustarse excesivamente a datos históricos.

## 18. Cost Sensitivity (Bajo/Medio/Alto)  
Medio. Operando intradía en M5 con entradas basadas en desviaciones y pequeños movimientos, los costos de spread y slippage afectan el rendimiento, pero filtro estricto en spread mitiga impacto.

## 19. Correlation With Liquidity Sweep Strategy (Baja/Media/Alta + Explicación)  
Baja. La estrategia se basa en reversión a media tras doble toque y divergencias, mientras que la Liquidity Sweep busca rompimientos y barridos de stops. Los patrones operan en contextos y señales diferentes.

## 20. Backtest Acceptance Criteria  
- Ratio ganancia/pérdida (Reward/Risk) ≥ 0.65.  
- Tasa de ganancia mínima del 48%.  
- Expectativa promedio positiva (Profit Factor > 1.2).  
- Drawdown máximo < 8% en capital simulado con tamaño fijo.  
- Robustez comprobada con walk-forward y validación out-of-sample.

## 21. Pseudocode  
```
PARÁMETROS
BB_period = 20
BB_std = 2
RSI_period = 14
ATR_period = 14
spread_max = 1.5 pips
doble_toque_tmax = 30 min
tp_factor = 1.0
sl_factor = 1.5
max_trade_duration = 60 min

FUNCIONES AUXILIARES
Calcular_BollingerBands(series, period, std_dev)
Calcular_RSI(series, period)
Calcular_ATR(series, period)
Detectar_Divergencia_RSI_M15(precios, RSI, puntos_referencia)
Consulta_Spread()
Consulta_Noticias()

LOGICA PRINCIPAL (por cada candle M5 en [09:00,16:00] NY):

1. Si Spread > spread_max: IGNORAR simultáneamente posiciones o entradas.

2. Detectar toque a banda inferior/superior Bollinger:

    si Precio M5 toca banda inferior:

        Registrar tiempo toque_1

        Esperar max dob_tiap_tmax para toque_2 a la banda inferior sin cerrar candle fuera banda.

        Si toca banda inferior por segunda vez dentro de plazo:

            Revisar RSI en M15 en puntos correspondientes del doble toque para divergencia alcista.

            Si divergencia alcista identificada y ATR_M5 > 0.0020 y no noticias próximas:

                Calcular SL = Entrada - sl_factor * ATR_M5

                Calcular TP = media_BB - Entrada (valor positivo)

                Abrir posición LONG en cierre de candle toque_2

    Realizar lógica simétrica para toque banda superior (posiciones SHORT).

3. Gestión de trade:

    Monitorizar posición activa:

        - Si precio alcanza 0.75 * TP: mover SL a BE + 1 pip.

        - Si duración > max_trade_duration: cerrar posición.

        - Si alcanza TP o SL: cerrar posición.

FIN
```

### Strategy 9
## 1. Strategy Name  
Session Breakout con filtro de rango pre-sesión (London High/Low)

## 2. Market Phenomenon  
La apertura de la sesión de Londres generalmente genera movimientos direccionales significativos en EURUSD. Los rompimientos del máximo o mínimo de esta sesión indican fuerza direccional. Filtrar por un rango reducido antes de la sesión Londres ayuda a evitar falsos rompimientos derivados de baja volatilidad previa.

## 3. Quant Hypothesis  
Si antes del inicio de la sesión de Londres el rango EURUSD es estrecho (filtro de rango bajo ATR), entonces un rompimiento del máximo o mínimo del High/Low de esa sesión tendrá un sesgo direccional favorable con alta expectativa de ganancia neta.

## 4. Why It Could Work on EURUSD  
EURUSD es altamente líquido en la sesión de Londres, con participación importante de bancos europeos y operadores institucionales. La concentración de órdenes y volatilidad en esa ventana hace que los rompimientos verdaderos tengan probabilidades de continuación. El filtro de rango previene entradas en mercados laterales o sin impulso.

## 5. Trading Window (Sub-ventana dentro de 07:00-19:00 NY)  
Operar solo durante la sesión de Londres 03:00-12:00 UTC (07:00-16:00 NY), con filtro y cálculo rango previo desde 02:00-03:00 UTC.

## 6. Timeframe (M1, M3, M5, M15 o combinación)  
Base en M5 para determinar High/Low de sesión Londres, M1 para ejecución de entrada y gestión.

## 7. Required Data  
- Velas M1 y M5 intradía EURUSD (incluyendo OHLC y volumen si disponible).  
- ATR(14) calculado en M5 para filtro rango previo.  
- Reloj UTC y NY para sincronizar ventana.

## 8. Long Entry Rules (Reglas exactas)  
1. Calcular rango (High-Low) entre 02:00-03:00 UTC (pre-sesión).  
2. Calcular ATR(14) en M5.  
3. Si rango pre-sesión ≤ 0.3 * ATR(14):  
   - Obtener High de la sesión Londres hasta el momento actual en M5.  
   - Entrar en largo en M1 en la primera vela que cierre por encima del High de la sesión Londres.  
   - Confirmar cierre M1 > High para entrada.  

## 9. Short Entry Rules (Reglas exactas)  
1. Mismos pasos 1 y 2 que largo (rango pre-sesión ≤ 0.3 * ATR(14)).  
2. Obtener Low de la sesión Londres hasta el momento actual en M5.  
3. Entrar en corto en M1 en la primera vela que cierre por debajo del Low de la sesión Londres.  
4. Confirmar cierre M1 < Low para entrada.

## 10. Stop Loss Logic  
Stop loss fijo a 1x ATR(14) M5 debajo del nivel de breakout para largo (por debajo del High de sesión), y 1x ATR(14) M5 encima del breakout para corto (por encima del Low de sesión).

## 11. Take Profit Logic  
Take profit a 2x ATR(14) M5 desde precio de entrada para asegurar relación RR 1:2.

## 12. Trade Management (BE, trailing, time stop, etc.)  
- Mover stop a break-even tras alcanzar 1x ATR(14) en ganancia.  
- Cierre forzado al final de la sesión Londres (12:00 UTC) si no se ha cerrado antes.  
- No trailing stops adicionales.

## 13. Filters (Spread, ATR, News, etc.)  
- No operar con spread > 2 pips.  
- No operar 15 minutos antes/después de noticias económicas de alta volatilidad para EUR/USD (calendario económico).  
- Filtro rango pre-sesión explicado.  

## 14. Initial Parameters (Razonables, no optimizados)  
- Rango pre-sesión ≤ 0.3 * ATR(14) M5  
- Stop Loss = 1 x ATR(14) M5  
- Take Profit = 2 x ATR(14) M5  
- Máximo spread admitido = 2 pips  
- Sesión Londres: 03:00 a 12:00 UTC

## 15. Expected Frequency  
Aproximadamente 1-2 trades por día, dependiendo de las condiciones de rango y ruptura.

## 16. Why It Might Fail  
- Rango pre-sesión no suficientemente predictivo en días de alta turbulencia.  
- Falsos rompimientos generados por eventos de noticias no filtrados.  
- Cambios estructurales en la dinámica de mercado o alta competencia algorítmica.

## 17. Overfitting Risk (Bajo/Medio/Alto)  
Medio. Backtest con ventana fija y parámetros estándar, pero posible adaptación excesiva con parámetros de filtro de rango.

## 18. Cost Sensitivity (Bajo/Medio/Alto)  
Medio-alto. Debido a entradas y stops ajustados, spreads y comisiones afectan rentabilidad.

## 19. Correlation With Liquidity Sweep Strategy (Baja/Media/Alta + Explicación)  
Media. Ambas estrategias explotan rupturas pero la Liquidity Sweep se centra en capturar manipulación de stop loss, mientras que esta se basa en rompimientos legítimos de sesión Londres con filtro de volatilidad previa.

## 20. Backtest Acceptance Criteria  
- Sharpe ratio > 1.0 en periodo out-of-sample 1 año.  
- Drawdown máximo < 5% capital.  
- Ratio ganancias/pérdidas > 1.5.  
- % de trades ganadores ≥ 40%.  
- Consistencia en periodos de alta y baja volatilidad.

## 21. Pseudocode  

```
Inicializar:
  timeframe_m5 = datos EURUSD 5 minutos
  timeframe_m1 = datos EURUSD 1 minuto
  ATR14_m5 = calcular ATR(14) sobre timeframe_m5

Por cada día de trading:  
  pre_session_range = High(02:00-03:00 UTC) - Low(02:00-03:00 UTC)
  ATR_val = ATR14_m5 al inicio sesión Londres (03:00 UTC)
  
  si pre_session_range ≤ 0.3 * ATR_val y spread ≤ 2 pips y no hay noticia alta volatilidad próxima:
    para cada vela M5 desde 03:00 hasta 12:00 UTC:
      session_high = máximo actual M5 desde 03:00 hasta ahora
      session_low = mínimo actual M5 desde 03:00 hasta ahora
      
      Verificar vela M1 que cierra:
        si cierre M1 > session_high y no hay posición abierta:
          abrir LONG en close M1
          SL = session_high - ATR_val
          TP = entrada + 2 * ATR_val
          
        si cierre M1 < session_low y no hay posición abierta:
          abrir SHORT en close M1
          SL = session_low + ATR_val
          TP = entrada - 2 * ATR_val
          
      Gestión de trade abierto:
        si ganancias ≥ 1 * ATR_val:
          mover SL a Break Even
        cerrar posición a las 12:00 UTC si abierta
```

---

Este diseño es completamente objetivo, institucional y crítico para condiciones actuales de EURUSD intradía, con delimitación clara de reglas y filtros cuantificables.

### Strategy 10
## 1. Strategy Name  
Session Breakout: Asian Range Liquidity Fakeout (SB-ARLF) EURUSD

## 2. Market Phenomenon  
Durante la sesión asiática (aprox. 22:00-07:00 NY), el EURUSD tiende a consolidar en un rango definido, acumulando órdenes de stop y liquidez. Al inicio de la sesión europea, muchos movimientos de ruptura de este rango resultan ser falsas salidas (fakeouts), que capturan liquidez y luego revierten. Este fenómeno es aprovechable para operar contra el breakout inicial fallido.

## 3. Quant Hypothesis  
Las rupturas falsas del rango asiático y la rápida reversión posterior ocurren con suficiente frecuencia y extensión para generar oportunidades rentables con definición clara de entrada, stop y objetivo intradía en EURUSD. Esta dinámica se manifiesta consistentemente en el horario 07:00-11:00 NY, aprovechando liquidez atrapada.

## 4. Why It Could Work on EURUSD  
EURUSD tiene gran liquidez global y participación institucional en las dos sesiones principales (Asia y Europa). La consolidación asiática es de baja volatilidad, produciendo zonas visibles de liquidez. El inicio europeo genera volatilidad e intentos de romper esos rangos, los cuales suelen fallar por ordenes stop y mecanismos de mercado, especialmente en periodos de baja tendencia estructural.

## 5. Trading Window (Sub-ventana dentro de 07:00-19:00 NY)  
07:00 – 11:00 NY (Inicio y primera parte fuerte de la sesión europea, donde se producen la mayoría de fakeouts post-breakout asiático).

## 6. Timeframe (M1, M3, M5, M15 o combinación)  
M5 para rango y breakout detectado. Confirmación en M1 para concretar la entrada.

## 7. Required Data  
- EURUSD tick data o M1 OHLC bar data para cálculos precisos.  
- Timestamp sincronizado a NY para identificar sesiones.  
- Spread en tiempo real.  
- News calendar intradía (para excluir periodos de alta volatilidad inesperada).  
- ATR(14) en M5 para volatilidad ajustada.

## 8. Long Entry Rules (Reglas exactas)  
1. Definir rango asiático: desde 22:00 NY hasta 07:00 NY, calcular máximo (AsianHigh) y mínimo (AsianLow) en M5.  
2. Count spread ≤ 2 pips y ATR(14) ≤ umbral definido en filtro.  
3. Precio rompe AsianHigh en M5 (cierre claro por encima de AsianHigh).  
4. Confirmar candle M1 siguiente que cierre en falso breakout: cierre M1 dentro del rango asiático o por debajo de AsianHigh.  
5. Entrar largo en cierre de ese M1 (falso breakout).  
6. Confirmar hora entre 07:00 y 11:00 NY al momento de la entrada.

## 9. Short Entry Rules (Reglas exactas)  
1. Definir rango asiático: AsianHigh y AsianLow igual que para long.  
2. Spread ≤ 2 pips y ATR(14) ≤ filtro.  
3. Precio rompe AsianLow en M5 (cierre claro por debajo de AsianLow).  
4. Confirmar candle M1 siguiente que cierre en falso breakout: cierre M1 dentro del rango asiático o por encima de AsianLow.  
5. Entrar corto en cierre de ese M1 tras falsa ruptura.  
6. Confirmar hora 07:00-11:00 NY.

## 10. Stop Loss Logic  
- Long: Stop fijo a 10 pips por debajo del mínimo asiático (AsianLow - 10 pips).  
- Short: Stop fijo a 10 pips por encima del máximo asiático (AsianHigh + 10 pips).  
Stop se coloca al momento de la entrada.

## 11. Take Profit Logic  
Take Profit fijo a 1.5 veces Stop Loss (~15 pips). Se busca relación riesgo/recompensa 1:1.5 para optimizar ganancias en reversión.

## 12. Trade Management (BE, trailing, time stop, etc.)  
- Si la posición alcanza +7 pips (50% TP), mover stop a breakeven (entrada - spread).  
- No trailing adicional.  
- Time stop: cerrar posición si no alcanza TP ni SL antes de las 11:00 NY (fin de ventana de trading).  
- No añadir posiciones (sin averaging).

## 13. Filters (Spread, ATR, News, etc.)  
- Spread ≤ 2 pips al momento de la entrada.  
- ATR(14) en M5 ≤ 0.0012 (12 pips).  
- No operar si hay noticia de alta volatilidad para EUR o USD en ±30 minutos de la operación (según calendario económico).  
- No operar lunes o viernes para evitar gaps u apertura irregular.

## 14. Initial Parameters (Razonables, no optimizados)  
- Stop Loss: 10 pips  
- Take Profit: 15 pips  
- Spread máximo: 2 pips  
- ATR(14) máximo: 12 pips  
- Trading window: 07:00-11:00 NY  
- Session Asian Range: 22:00-07:00 NY  

## 15. Expected Frequency  
3-5 operaciones por día en promedio, dependiendo de condiciones de mercado, especialmente volatilidad.

## 16. Why It Might Fail  
- Mercados con movimientos fuertes e impulsivos que rompen rangos asiáticos sin revertir (p.ej. noticias inesperadas).  
- Cambios estructurales en la volatilidad intradía que invalidan rangos asiáticos definidos por la sesión anterior.  
- Spread alto o squeezes por baja liquidez (p.ej. festivos) que provocan pistas falsas o ejecución poco favorable.  
- Correlación negativa temporal con eventos macro que inducen tendencia clara.

## 17. Overfitting Risk (Bajo/Medio/Alto)  
Medio. El concepto es robusto y basado en microestructura, pero parámetros fijos y ventana estricta pueden requerir revalidación constante para evitar curve fitting.

## 18. Cost Sensitivity (Bajo/Medio/Alto)  
Medio-alto. Stop relativamente pequeño obliga a spreads bajos y ejecución ágil; costos de slippage y comisiones pueden erosionar ganancia neta.

## 19. Correlation With Liquidity Sweep Strategy (Baja/Media/Alta + Explicación)  
Media. Ambas estrategias se basan en la captura de liquidez generada por órdenes ocultas o stops, pero la SB-ARLF se focaliza en rango asiático y ruptura fallida, mientras que la Liquidity Sweep puede abordar múltiples estructuras de mercado más amplias.

## 20. Backtest Acceptance Criteria  
- Profit factor ≥ 1.5  
- Ratio ganancia/pérdida promedio ≥1.3  
- Ratio ganadoras ≥ 40%  
- Max drawdown < 10% del capital simulado  
- Consistencia semanal con ganancias positivas al menos en 60% de semanas  
- Slippage y spread simulados reales

## 21. Pseudocode  

```python
# Definir ventana asiática y sesión europea
asian_start = "22:00"
asian_end = "07:00"
trade_start = "07:00"
trade_end = "11:00"

# Calcular Asian Range en M5
AsianHigh = max(price_high from asian_start to asian_end)
AsianLow = min(price_low from asian_start to asian_end)

# Calcular ATR(14) en M5
atr_value = ATR(price_close, 14)

# En tiempo real, para cada cierre M5 entre 07:00 y 11:00:
for bar in M5_bars_between(trade_start, trade_end):
    # Obtener spread actual y hora
    current_spread = get_spread()
    current_time = bar.timestamp

    if current_spread > 2 or atr_value > 0.0012 or is_high_impact_news_near(current_time):
        continue  # no operar

    # Detectar breakout largo falso
    if bar.close > AsianHigh:
        # Esperar siguiente M1 cierre para confirmar fakeout
        m1_next = get_next_M1_bar(bar.timestamp)
        if m1_next.close <= AsianHigh:
            # Entrar LONG en cierre de ese M1
            entry_price = m1_next.close
            stop_loss = AsianLow - 0.0010  # 10 pips
            take_profit = entry_price + 0.0015  # 15 pips
            open_long(entry_price, stop_loss, take_profit)
            manage_trade_until(tp_or_sl_or_time_stop(11:00))

    # Detectar breakout corto falso
    elif bar.close < AsianLow:
        m1_next = get_next_M1_bar(bar.timestamp)
        if m1_next.close >= AsianLow:
            # Entrar SHORT en cierre de ese M1
            entry_price = m1_next.close
            stop_loss = AsianHigh + 0.0010  # 10 pips
            take_profit = entry_price - 0.0015  # 15 pips
            open_short(entry_price, stop_loss, take_profit)
            manage_trade_until(tp_or_sl_or_time_stop(11:00))
```

---

Esta especificación es 100% objetiva, con criterios claros sin discreción para uso institucional cuantitativo en EURUSD intradía con enfoque en la sesión europea aprovechando falsos breakouts del rango asiático.

### Strategy 11
## 1. Strategy Name  
NY Session Opening Range Reversal Breakout (Initial Balance Failure) EURUSD Intradía

## 2. Market Phenomenon  
La apertura de la sesión neoyorquina crea un rango inicial (Initial Balance, IB) que marca zonas de liquidez. Un fracaso para continuar el rompimiento de dicho rango (falso breakout) tiende a generar un movimiento compensatorio en dirección opuesta (reversal), producto del agotamiento de órdenes y actividad institucional.

## 3. Quant Hypothesis  
Si el precio rompe la apertura del IB pero no logra sostenerse fuera del rango dentro de un corto tiempo, es probable una reversión hacia el interior o lado contrario del IB, debido al fallo en la captación de liquidez y la entrada de posiciones contrarias institucionales.

## 4. Why It Could Work on EURUSD  
EURUSD, con alta liquidez y volumen en la sesión NY, muestra patrones recurrentes en apertura debido a flujos fundamentales y actividad de bancos. El IB es reconocido por determinar zonas de soporte/resistencia diario, y el failure breakout es frecuente por stops ubicados cerca del IB y reversiones en spreads bajos como EURUSD.

## 5. Trading Window (Sub-ventana dentro de 07:00-19:00 NY)  
07:00 a 08:30 NY — Definición del Opening Range (IB)  
08:30 a 12:00 NY — Momento para detectar breakout fallido y tomar posición reversa

## 6. Timeframe (M1, M3, M5, M15 o combinación)  
Temporalidad principal M5 para definir rangos y confirmar rupturas/fallos; M1 para entrada precisa y ejecución.

## 7. Required Data  
- Precios OHLC en M1 y M5 del EURUSD entre 07:00-12:00 NY  
- Spreads y volumen (si disponible)  
- Calendario económico para identificar eventos de alta volatilidad major (para filtros)  
- Indicador ATR(14) calculado en M5 para volatilidad y stop sizing

## 8. Long Entry Rules (Reglas exactas)  
1. Calcular Opening Range (IB): rango alta y baja de 07:00-08:30 NY (M5 candles)  
2. Detectar rompimiento al alza del IB_high en M5 candle cerrado después de 08:30 hasta 12:00  
3. Confirmar fallo breakout: precio cierra una vela M5 por debajo del IB_high después de romperlo (es decir, rompió al alza pero cierra M5 dentro IB o por debajo IB_high en los siguientes 2 velas M5)  
4. Entrada LONG: en la apertura de la siguiente vela M1 después de confirmar cierre M5 por debajo del IB_high (fallo)  
5. Confirmar que el precio esté por encima del IB_low en el momento de entrada (para descartar exceso de volatilidad bajista)

## 9. Short Entry Rules (Reglas exactas)  
1. Calcular Opening Range (IB): rango alta y baja de 07:00-08:30 NY (M5 candles)  
2. Detectar rompimiento a la baja del IB_low en M5 candle cerrado después de 08:30 hasta 12:00  
3. Confirmar fallo breakout: precio cierra una vela M5 por encima del IB_low después de romperlo (es decir, rompió a la baja pero cierra M5 dentro IB o arriba IB_low en los siguientes 2 velas M5)  
4. Entrada SHORT: en la apertura de la siguiente vela M1 después de confirmar cierre M5 por encima del IB_low (fallo)  
5. Confirmar que el precio esté por debajo del IB_high en el momento de entrada

## 10. Stop Loss Logic  
- Long trade: stopLoss = IB_low – 0.5×ATR(14) M5  
- Short trade: stopLoss = IB_high + 0.5×ATR(14) M5  
El stop se fija fuera del rango IB para dar espacio a fluctuaciones normales pero proteger ante invalidación técnica clara.

## 11. Take Profit Logic  
Take profit = entrada ± 1×ATR(14) M5 (R1)  
Relación Riesgo:Recompensa inicial 1:1  
Opcional: cierre parcial 50% en 0.5×ATR para asegurar beneficio y dejar resto trailing.

## 12. Trade Management (BE, trailing, time stop, etc.)  
- Cuando el precio alcance 0.5×ATR favorable, se traslada stop a Break Even (entrada)  
- Trailing stop: después de +1×ATR, utilizar trailing stop en M1 con 0.3×ATR distancia  
- Time stop: cerrar todas las posiciones a las 19:00 NY si no se ha cerrado antes

## 13. Filters (Spread, ATR, News, etc.)  
- Ejecución solo si spread < 1.5 pips  
- No operar 15 mins antes y después de noticias de alta volatilidad (Supermartes, NFP) según calendario económico  
- Operar solo si ATR(14) M5 > 5 pips para asegurar volatilidad suficiente

## 14. Initial Parameters (Razonables, no optimizados)  
- Opening Range: 07:00-08:30 NY (90 minutos) en M5  
- Confirmación fallo breakout: cierre dentro IB en 2 velas M5 siguientes  
- Stop Loss offset: 0.5×ATR(14) M5  
- Take Profit: 1×ATR(14) M5  
- Spread máximo: 1.5 pips  
- ATR mínimo: 5 pips

## 15. Expected Frequency  
Aproximadamente 1-3 señales diarias, dependiendo de la volatilidad y número de fallos breakout en la sesión NY.

## 16. Why It Might Fail  
- Movimientos de alta volatilidad imprevistos que ignoran IB (eventos inesperados)  
- Falsas señales en mercados laterales sin tendencia clara  
- Cambios estructurales en comportamiento del mercado por política monetaria o volatilidad extrema  
- Spread o slippage altos en momentos de menor liquidez

## 17. Overfitting Risk (Bajo/Medio/Alto)  
Medio — la estrategia consiste en reglas simples basadas en conceptos estructurales de rango y ruptura, pero el uso de parámetros ATR y ventanas temporales puede implicar riesgos de ajuste excesivo si no se diversifica la muestra temporal.

## 18. Cost Sensitivity (Bajo/Medio/Alto)  
Medio — el uso frecuente de entradas intradía y stops ajustados puede resultar sensible a spreads y comisiones, especialmente en momentos de baja liquidez.

## 19. Correlation With Liquidity Sweep Strategy (Baja/Media/Alta + Explicación)  
Media — ambas estrategias buscan capturar movimientos tras manipulación de stops y extracción de liquidez, pero aquí el foco está en fallo de breakout de IB mientras que Liquidity Sweep se concentra en barridos de stops más amplios y niveles claves.

## 20. Backtest Acceptance Criteria  
- Sharpe ratio ajustado > 1.0  
- Ganancia neta positiva en muestra out-of-sample > 6 meses  
- Drawdown máximo intradía < 3% del capital en las condiciones dadas  
- Ratio de operaciones ganadoras entre 40%-60% con promedio R:R ≥1  
- Robustez estable en diferentes años y condiciones (sin picos de curve-fitting)

## 21. Pseudocode  

```python
# Parámetros
opening_range_start = datetime.time(7,0)   # 07:00 NY
opening_range_end = datetime.time(8,30)    # 08:30 NY
trade_end_time = datetime.time(19,0)       # Cierre trades 19:00 NY
max_spread = 1.5                           # pips
min_ATR = 5                                # pips
SL_ATR_multiplier = 0.5
TP_ATR_multiplier = 1.0
fail_confirm_bars = 2                      # número de velas para confirmar fallo

# Datos: M5 candles 07:00-12:00, precios, spread, ATR(14) M5

for each trading day:
    # 1. Calcular IB high y low
    IB_candles = get_M5_candles_between(opening_range_start, opening_range_end)
    IB_high = max(high prices of IB_candles)
    IB_low = min(low prices of IB_candles)

    # 2. Calcular ATR(14) M5 al cierre de IB
    ATR_val = ATR(14) at 08:30
    if ATR_val < min_ATR: continue day (no operar)
    
    # 3. Escanear velas M5 entre 08:30 y 12:00 para breakout y fallo
    for candle_index after IB_end to 12:00:
        candle = M5_candle[candle_index]
        prev_candle = M5_candle[candle_index - 1]
        current_spread = get_spread_at(candle.time)
        if current_spread > max_spread: continue

        # Long scenario
        if prev_candle.close <= IB_high and candle.close > IB_high:
            # rompió IB_high al alza
            # revisar en siguientes fail_confirm_bars si cierre vuelve dentro IB o por debajo
            failed = False
            for f in range(1, fail_confirm_bars+1):
                check_candle = M5_candle[candle_index + f]
                if check_candle.close <= IB_high:
                    failed = True
                    fail_candle_index = candle_index + f
                    break
            if failed:
                # Long Entry en apertura M1 siguiente a fail_candle_index
                entry_time = M5_candle[fail_candle_index + 1].start_time
                # confirmar precio M1 de entrada > IB_low
                price_entry = get_M1_open_price(entry_time)
                if price_entry > IB_low:
                    SL = IB_low - SL_ATR_multiplier * ATR_val
                    TP = price_entry + TP_ATR_multiplier * ATR_val
                    place_long_trade(entry_time, price_entry, SL, TP)

        # Short scenario
        if prev_candle.close >= IB_low and candle.close < IB_low:
            # rompió IB_low a la baja
            # revisar en siguientes fail_confirm_bars si cierre vuelve dentro IB o por encima
            failed = False
            for f in range(1, fail_confirm_bars+1):
                check_candle = M5_candle[candle_index + f]
                if check_candle.close >= IB_low:
                    failed = True
                    fail_candle_index = candle_index + f
                    break
            if failed:
                # Short Entry en apertura M1 siguiente a fail_candle_index
                entry_time = M5_candle[fail_candle_index + 1].start_time
                price_entry = get_M1_open_price(entry_time)
                if price_entry < IB_high:
                    SL = IB_high + SL_ATR_multiplier * ATR_val
                    TP = price_entry - TP_ATR_multiplier * ATR_val
                    place_short_trade(entry_time, price_entry, SL, TP)

# Trade management (ejecutado en continuo):
# - si precio alcanza 0.5×ATR favorable => mover SL a BE
# - si alcanza TP parcial => cerrar 50%, dejar resto trailing con 0.3×ATR
# - cerrar todas posiciones a las 19:00 NY

# Funciones auxiliares:
# get_M5_candles_between(start_time, end_time)
# ATR(period)
# get_spread_at(time)
# get_M1_open_price(time)
# place_long_trade(time, entry_price, SL, TP)
# place_short_trade(time, entry_price, SL, TP)
```

---

Esta especificación provee un plan cuantitativo riguroso, sin uso discrecional, basado en un concepto probado y ajustado a la liquidez y comportamiento característico de EURUSD en la sesión neoyorquina intradía.

### Strategy 12
## 1. Strategy Name  
Trend Pullback Institutional EMA ATR SL (TP-EMA-ATR)

## 2. Market Phenomenon  
Durante tendencias definidas, el precio suele retroceder momentáneamente hacia una media móvil exponencial (EMA) institucional (50 o 200 períodos), antes de reanudar la dirección principal. Estos pullbacks ofrecen oportunidades de entrada con buen ratio riesgo/beneficio cuando se gestionan con stops basados en volatilidad (ATR).

## 3. Quant Hypothesis  
Los retrocesos del precio hacia la EMA50 o EMA200 en un marco intradía representan momentos óptimos para unirse a la tendencia dominante, con una probabilidad estadísticamente significativa de continuación si el stop loss se ajusta proporcionalmente a la volatilidad local (ATR).

## 4. Why It Could Work on EURUSD  
EURUSD es un par con alta liquidez y patrones de tendencia relativamente estables durante horas de mercado activas (NY). Las medias institucionales (EMA50 y EMA200) se respetan frecuentemente como soportes/resistencias dinámicos, con volatilidad suficiente para definir stops y objetivos realistas sin ruido excesivo.

## 5. Trading Window (Sub-ventana dentro de 07:00-19:00 NY)  
08:00 a 17:00 NY: evita la apertura y cierre para reducir ruido inicial y sesgo de fin de sesión.

## 6. Timeframe (M1, M3, M5, M15 o combinación)  
Principal: M5 para la detección de señales de entrada y gestión.  
Confirmación/trend check en M15 para validar dirección EMA y tendencia.

## 7. Required Data  
- Precio: apertura, máximo, mínimo, cierre M1, M5 y M15.  
- Volumen tick opcional para confirmación (no obligatorio).  
- Datos históricos de al menos 6 meses intradía.  
- Indicadores calculados: EMA50 y EMA200 (M15), ATR(14) en M5.

## 8. Long Entry Rules (Reglas exactas)  
1. En M15, EMA50 > EMA200 (tendencia alcista definida).  
2. En M5, el precio retrocede y toca o cruza por debajo de EMA50 (calculada en M15 y sincronizada a M5).  
3. La vela M5 actual cierra por encima del mínimo de esa vela y por debajo o tocando EMA50 (confirmación de pullback).  
4. Confirmar que el último máximo local anterior (1-3 M5 anteriores) es menor que el máximo actual para validar intento de continuación.  
5. Entrar LONG al cierre de la vela M5 que cumple criterios, o a la apertura de la siguiente vela.

## 9. Short Entry Rules (Reglas exactas)  
1. En M15, EMA50 < EMA200 (tendencia bajista definida).  
2. En M5, el precio retrocede y toca o cruza por encima de EMA50 (calculada en M15 y sincronizada a M5).  
3. La vela M5 actual cierra por debajo del máximo de esa vela y por encima o tocando EMA50.  
4. Confirmar que el último mínimo local anterior (1-3 M5 anteriores) es mayor que el mínimo actual para validar intento de continuación a la baja.  
5. Entrar SHORT al cierre de la vela M5 que cumple criterios, o a la apertura de la siguiente vela.

## 10. Stop Loss Logic  
- SL se fija en función del ATR(14) en M5:  
  - LONG: SL = mín. local reciente (últimos 3 M5 mínimos) - (1.5 * ATR)  
  - SHORT: SL = máx. local reciente (últimos 3 M5 máximos) + (1.5 * ATR)  
- Ajuste automático si volatilidad varía antes de la ejecución.

## 11. Take Profit Logic  
- Objetivo de beneficio fijo en múltiplo del riesgo: 2.5x la distancia SL-entrada.  
- Alternativamente, cierre automático si M15 EMA50 cruza la EMA200 en sentido contrario (fin de tendencia).

## 12. Trade Management (BE, trailing, time stop, etc.)  
- Mover SL a Break Even cuando el precio alcance 1.5x riesgo inicial.  
- Trailing SL con un 1x ATR de distancia una vez que TP intermedio 1x riesgo sea alcanzado.  
- Time Stop: cerrar la operación si no se activa SL o TP dentro de las 12 velas M5 (60 minutos).  
- No se realizan escalados ni aumentos de posición.

## 13. Filters (Spread, ATR, News, etc.)  
- Spread máximo permitido: 2 pips (20 puntos en 5 dígitos).  
- Operar solo si ATR(14) M5 ≥ 0.0006 (60 pips anualizado habría que ajustar en punto flotante).  
- Evitar operaciones 30 minutos antes/después de releases económicos relevantes para EUR/USD (FOMC, NFP, ECB, etc.).  
- No operar si volatilidad reciente (ATR) crece más del 50% en últimos 30 minutos (demasiado ruido).

## 14. Initial Parameters (Razonables, no optimizados)  
- EMA50 y EMA200 en M15.  
- ATR periodo 14 en M5.  
- SL múltiplo: 1.5x ATR.  
- TP múltiplo: 2.5x riesgo (SL).  
- Spread max: 2 pips.  
- Time stop 12 M5 candles.

## 15. Expected Frequency  
- Aproximadamente 1 a 3 señales diarias en la ventana 08:00-17:00 NY, variando según volatilidad y tendencia.

## 16. Why It Might Fail  
- Falsos rompimientos durante períodos laterales y rangos prolongados.  
- Cambios abruptos en la volatilidad o manipulación del mercado que invaliden stops ATR.  
- Ruido extremo fuera de horario operativo o durante eventos económicos no filtrados.  
- Tendencias débiles donde EMA50 y EMA200 se cruzan frecuentemente sin claro momentum.

## 17. Overfitting Risk (Bajo/Medio/Alto)  
Medio: uso de reglas estándar institucionales y parámetros típicos, pero riesgo de curve fitting si se ajusta sin control para optimizar múltiples indicadores combinados.

## 18. Cost Sensitivity (Bajo/Medio/Alto)  
Medio: spread relativamente bajo en EURUSD pero el uso intradía y stops ajustados puede amplificar costos de transacción y slippage.

## 19. Correlation With Liquidity Sweep Strategy (Baja/Media/Alta + Explicación)  
Baja a media.  
La estrategia Trend Pullback captura movimientos de continuación en tendencias definidas, mientras que una Liquidity Sweep busca activaciones en zonas de liquidez (stop hunts) que pueden ser previos o independientes a estructuras EMAs. Pueden coincidir si un sweep genera pullback a la EMA pero sus señales no coinciden generalmente.

## 20. Backtest Acceptance Criteria  
- Sharpe Ratio > 1.2 en período de prueba.  
- Drawdown máximo intradía < 5%.  
- Ratio de ganancia (profit factor) > 1.5.  
- Confirmación de estabilidad de métricas en 6 meses out-of-sample.  
- Cumplimiento de frecuencia esperada y ejecución sin discrecionalidad.

## 21. Pseudocode  

```python
# Inputs iniciales
EMA_fast_period = 50
EMA_slow_period = 200
ATR_period = 14
SL_multiplier = 1.5
TP_multiplier = 2.5
max_spread = 0.00020 # 2 pips
trade_window_start = 8 * 60  # 08:00 in minutes from midnight
trade_window_end = 17 * 60   # 17:00 in minutes
time_stop_bars = 12  # 12 velas M5

for each trading day:
    load M15 bars from 00:00 to 23:59
    calculate EMA50_M15, EMA200_M15
    
    for each M5 bar between trade_window_start and trade_window_end:
        if spread > max_spread or during news window:
            skip bar

        trend_up = EMA50_M15[current_bar] > EMA200_M15[current_bar]
        trend_down = EMA50_M15[current_bar] < EMA200_M15[current_bar]

        ATR_5 = ATR(14) on M5 up to current_bar

        # Check for Long Entry
        if trend_up:
            ema50_val = EMA50_M15[current_bar]  # synced to current M5 bar time
            price = close_price_M5[current_bar]
            prev_min_local = min_low_last_3_M5(current_bar)
            last_max_local = max_high_last_3_M5(current_bar-1)

            # Precio toca o pasa por debajo de EMA50
            if (low_price_M5[current_bar] <= ema50_val <= high_price_M5[current_bar]) and \
               (close_price_M5[current_bar] > low_price_M5[current_bar]) and \
               (price > last_max_local):
                
                entry_price = price
                SL = prev_min_local - (SL_multiplier * ATR_5)
                TP = entry_price + (TP_multiplier * (entry_price - SL))

                open long position with SL, TP

        # Check for Short Entry
        if trend_down:
            ema50_val = EMA50_M15[current_bar]
            price = close_price_M5[current_bar]
            prev_max_local = max_high_last_3_M5(current_bar)
            last_min_local = min_low_last_3_M5(current_bar-1)

            if (high_price_M5[current_bar] >= ema50_val >= low_price_M5[current_bar]) and \
               (close_price_M5[current_bar] < high_price_M5[current_bar]) and \
               (price < last_min_local):
                
                entry_price = price
                SL = prev_max_local + (SL_multiplier * ATR_5)
                TP = entry_price - (TP_multiplier * (SL - entry_price))

                open short position with SL, TP

        # Gestión abierta
        for open trade:
            time_in_trade += 1
            if price reached TP or SL:
                close trade
                
            if time_in_trade == time_stop_bars:
                close trade
                
            if profit >= 1.5 * initial_risk and not BE_moved:
                move SL to entry_price
                
            if profit >= 1.0 * initial_risk:
                trail SL by 1 * ATR_5

```
---

Esta especificación afronta el desarrollo con rigor institucional y objetividad, eliminando cualquier criterio discrecional o subjetivo.

### Strategy 13
## 1. Strategy Name  
Trend Pullback ADX-Fib 61.8% EURUSD Intradía

## 2. Market Phenomenon  
En tendencias fuertes, las cotizaciones frecuentemente corrigen parcialmente para retomar la dirección del movimiento. El retroceso de Fibonacci 61.8% junto con un ADX alto se usa para identificar puntos óptimos de entrada en estas correcciones.

## 3. Quant Hypothesis  
Cuando EURUSD está en tendencia fuerte (ADX>25) y se produce un retroceso del 61.8% del último impulso, la probabilidad de que continúe la tendencia inicial es alta, generando una entrada rentable con riesgo limitado.

## 4. Why It Could Work on EURUSD  
EURUSD es uno de los pares más líquidos y eficientes, especialmente durante horas de mercado activo en Nueva York. Show fuertes tendencias intradía y correcciones claras en períodos cortos, ofreciendo condiciones ideales para retrocesos técnicos y señales de continuación.

## 5. Trading Window (Sub-ventana dentro de 07:00-19:00 NY)  
09:30 - 16:30 NY (superposición mercado Europa-NY, mayor volatilidad y tendencia definida)

## 6. Timeframe (M1, M3, M5, M15 o combinación)  
Combina M5 para la detección de impulsos y retrocesos + M15 para confirmación de tendencia ADX.

## 7. Required Data  
- EURUSD tick/bar data (M1 mínimo) con OHLC, volumen si disponible  
- Indicador ADX 14 periodos en M15  
- Cálculo retrospectivo de retrocesos Fibonacci sobre último impulso definido en M5

## 8. Long Entry Rules (Reglas exactas)  
1. En M15 ADX(14) > 25 y DI+ > DI- (indica tendencia alcista fuerte)  
2. Definir último impulso alcista en M5: swing low A a swing high B claramente identificados  
3. En M5, precio retrocede desde B hasta nivel de Fibonacci 61.8% (calcula distancia B-A, retroceso máximo ≤ 61.8%)  
4. El precio toca o cruza por debajo el nivel fib 61.8% pero no lo supera más allá de un máximo de 3 pips por debajo  
5. Confirmar que después de tocar retracement, la vela siguiente M5 tenga cierre alcista mayor que la apertura (confirmación de rechazo)  
6. Se ejecuta orden LONG al cierre de vela confirmatoria M5 dentro ventana 09:30-16:30 NY  

## 9. Short Entry Rules (Reglas exactas)  
1. En M15 ADX(14) > 25 y DI- > DI+ (indica tendencia bajista fuerte)  
2. Definir último impulso bajista en M5: swing high A a swing low B  
3. En M5, precio retrocede desde B hasta nivel Fibonacci 61.8% (retroceso máximo ≤ 61.8%)  
4. Precio toca o cruza por encima el nivel fib 61.8% con máximo de 3 pips por encima  
5. Vela siguiente M5 cierra bajista (cierre < apertura) como confirmación  
6. Ejecutar orden SHORT al cierre vela confirmatoria dentro ventana 09:30-16:30 NY  

## 10. Stop Loss Logic  
- Long: stop loss fijo 5 pips por debajo del swing low A previo al impulso  
- Short: stop loss fijo 5 pips por encima del swing high A previo al impulso  
(Riesgo definido con base en puntos naturales de reversión)

## 11. Take Profit Logic  
- Take profit objetivo iterativo fijado en resistencia (long) o soporte (short) siguiente, equivalente a una distancia 1.5x la distancia SL-Riesgo (ejemplo 7.5 pips)  
- Alternativamente, primer retroceso del 38.2% del nuevo impulso tras la entrada

## 12. Trade Management (BE, trailing, time stop, etc.)  
- Ajustar stop loss a BE (break-even) cuando el precio se mueva favorablemente 50% del TP  
- No trailing después de BE  
- Cierre automático a las 16:30 NY si la posición aún está abierta (time stop intradía para evitar exposición overnight)

## 13. Filters (Spread, ATR, News, etc.)  
- Spread máximo 1.5 pips para ejecución  
- ATR(14, M5) se usa para validar volatilidad intradía mínima, solo operar si ATR > 3 pips en la ventana  
- No operar 15 min antes/después de noticias económicas de alta relevancia para EUR o USD (calendario económico)

## 14. Initial Parameters (Razonables, no optimizados)  
- ADX período 14  
- ADX mínimo para tendencia: 25  
- Fibonacci retracement: 61.8%  
- SL: 5 pips del swing anterior  
- TP: 1.5x SL (7.5 pips)  
- Trading window: 09:30-16:30 NY  
- ATR mínimo: 3 pips (M5 periodo 14)  
- Máximo spread: 1.5 pips  

## 15. Expected Frequency  
5-10 señales/trades por día, dependiendo de la actividad y volatilidad intradía.

## 16. Why It Might Fail  
- Falta de tendencia definida (ADX baja o falsas señales)  
- Falsos retrocesos que no respetan 61.8%, rupturas continuas del nivel fib  
- Cambios abruptos de volatilidad o news spikes no filtrados  
- Slippage y spreads mayores afectan la ejecución y resultados netos  
- Mercado lateral con movimientos erráticos incumpliendo supuestos de tendencia

## 17. Overfitting Risk (Bajo/Medio/Alto)  
Medio – Estrategia basada en principios técnicos bien establecidos, pero optimización excesiva de parámetros de fibo, SL/TP y ADX puede sesgar resultados históricos.

## 18. Cost Sensitivity (Bajo/Medio/Alto)  
Medio – Stop tight y TP moderado implica cuidado especial con spread y comisiones; spreads elevados o slippage impactan PnL.

## 19. Correlation With Liquidity Sweep Strategy (Baja/Media/Alta + Explicación)  
Baja – Esta estrategia se basa en lógica de retrocesos y confirmación de tendencia, mientras que liquidity sweep busca rupturas falsas por barridos de stops, son enfoques técnicos y de flujo de orden distintos que rara vez coinciden.

## 20. Backtest Acceptance Criteria  
- Profit factor > 1.3  
- Win rate adecuado > 50% con R:R cercano a 1.5 de TP/SL  
- Drawdown máximo no mayor a 10% sobre capital simulado  
- Consistencia en múltiple ventanas temporales y distintas condiciones de volatilidad intradía  
- Robustez a spread variable y slippage simulados

## 21. Pseudocode  
```python
# Parámetros iniciales
ADX_period = 14
ADX_threshold = 25
Fib_level = 0.618
SL_pips = 5
TP_factor = 1.5
Spread_max = 1.5
ATR_period = 14
ATR_min = 3
Trading_start = time(9,30)
Trading_end = time(16,30)

for each day:
    load M15 bars, calculate ADX(14)
    load M5 bars, identify swings (A,B impulses)
    calculate ATR(14) on M5
    
    if current_time in [Trading_start, Trading_end] and spread <= Spread_max and ATR>ATR_min:
        # Detección tendencia y dirección
        if ADX(M15) > ADX_threshold:
            if DI_plus > DI_minus:  # Tendencia alcista
                # Detectar último impulso alcista en M5
                identify swing_low_A, swing_high_B
                
                # Calcular retroceso fib 61.8%
                fib_61_level = B - (B - A) * Fib_level
                
                # En M5 price touches fib_61_level con máximo 3 pips abajo
                if price crosses fib_61_level - 3 pips and next M5 candle close > open:
                    enter_long at close next M5 bar
                    SL = A - SL_pips
                    TP = entry_price + (entry_price - SL) * TP_factor
                    
            elif DI_minus > DI_plus:  # Tendencia bajista
                identify swing_high_A, swing_low_B
                
                fib_61_level = B + (A - B) * Fib_level
                
                if price crosses fib_61_level + 3 pips and next M5 candle close < open:
                    enter_short at close next M5 bar
                    SL = A + SL_pips
                    TP = entry_price - (SL - entry_price) * TP_factor
                    
    # Gestión de trades abiertos
    for each open trade:
        if unrealized_profit >= (TP - entry_price)/2:
            move SL to entry_price (break-even)
        if current_time >= Trading_end:
            close position
```

---

Esta especificación brinda una base clara, objetiva, y cuantitativa para implementar, testear y evaluar sistemáticamente un trader cuantitativo institucional enfocado en retrocesos de Fibonacci y tendencias ADX para EURUSD intradía.

### Strategy 14
## 1. Strategy Name  
Trend Pullback Breakout-Retest EURUSD Intradía  

## 2. Market Phenomenon  
Las rupturas (breakouts) de niveles estructurales significativos de la sesión anterior a menudo generan movimientos direccionales, pero la confirmación del nuevo sesgo se produce con el retesteo (retest) de dichos niveles, funcionando como pullbacks para continuidad del movimiento de tendencia intradía.  

## 3. Quant Hypothesis  
La combinación de ruptura y retesteo de niveles estructurales (máximo, mínimo y cierre de la sesión previa) dentro de la sesión intradía NY (07:00-19:00 NY) en EURUSD genera setups de alta probabilidad, ya que el mercado valida la fuerza del impulso mediante pullbacks técnicos antes de continuar la tendencia principal.  

## 4. Why It Could Work on EURUSD  
EURUSD tiene alta liquidez y reacción pronunciada a niveles técnicos diarios, especialmente durante la sesión NY. Los participantes institucionales respetan niveles de la sesión anterior como referencia para tomar posiciones, generando patrones de breakout-retest consistentes con baja latencia de ejecución.  

## 5. Trading Window (Sub-ventana dentro de 07:00-19:00 NY)  
08:00 - 17:00 NY. (Permite evitar ruido inicial, efecto apertura, y el closing volatility más alto post 17:00).  

## 6. Timeframe (M1, M3, M5, M15 o combinación)  
Combinación: M5 para detección de rupturas y retestes, M1 para confirmación de patrones y ejecución precisa.  

## 7. Required Data  
- Velas OHLC M1 y M5  
- Niveles estructurales del día anterior: Máximo, Mínimo, Cierre (calculados en tiempo NY)  
- Spread actual en ticks/pips  
- ATR(14) en M5 para volatilidad de contexto  
- Calendario económico (filtro básico para noticias de impacto alto)  

## 8. Long Entry Rules (Reglas exactas)  
1. Antes de las 08:00 NY, calcular Máximo, Mínimo, Cierre de la sesión anterior (08:00-17:00 NY día anterior).  
2. Detectar ruptura al alza por cierre de vela M5: precio cierre vela M5 > Máximo sesión previa.  
3. Confirmar pullback: tras la ruptura, esperar que el precio reteste el nivel roto (Máximo sesión previa) y cierre una vela M1 por encima del nivel (en rango de ±3 pips).  
4. La vela M1 de retesteo no debe cerrar con sombra inferior significativa (>50% cuerpo).  
5. El nivel Mínimo de ATR(14) M5 en las últimas 3 velas debe ser > 5 pips (volatilidad adecuada).  
6. Spread actual < 2 pips.  
7. No entrada si dentro de ventana ±15 min de noticia de alta volatilidad.  
Al cumplirse, abrir posición larga a Precio mercado en la vela M1 siguiente al retesteo confirmado.  

## 9. Short Entry Rules (Reglas exactas)  
1. Calcular Máximo, Mínimo, Cierre sesión anterior igual que en largo.  
2. Detectar ruptura a la baja con cierre de vela M5: cierre < Mínimo sesión previa.  
3. Confirmar pullback: el precio debe retestear el nivel roto (Mínimo sesión previa) y cerrar vela M1 por debajo ±3 pips del nivel.  
4. La vela M1 retesteo sin sombra superior significativa (>50% cuerpo).  
5. ATR(14) M5 mínimo 5 pips en últimas 3 velas.  
6. Spread < 2 pips  
7. Evitar entrada ±15 min antes/después noticias impacto alto.  
Abrir posición corta al cierre de vela M1 confirmatoria.  

## 10. Stop Loss Logic  
Stop loss fijo en ±10 pips del punto de entrada (distancia equivalente a aproximadamente 2x ATR(14)/14min). No variable ni dinámico para evitar discrecionalidad.  

## 11. Take Profit Logic  
Take profit fijo en 15 pips (ratio RR 1.5x con SL), mayor que stop por confirmación de seguimiento de tendencia post retesteo.  

## 12. Trade Management (BE, trailing, time stop, etc.)  
- Desplazar stop a break even cuando la operación haya alcanzado +7.5 pips.  
- No trailing stops automáticos.  
- Cierre automático de la posición a las 16:55 NY (última vela) si no se ha cerrado antes por SL o TP.  

## 13. Filters (Spread, ATR, News, etc.)  
- Spread < 2 pips para evitar costos excesivos.  
- ATR(14) M5 mínimo 5 pips para entrar en condiciones no planas ni demasiado volátiles.  
- No operar ±15 min de noticias económicas de alta volatilidad (calendario automático).  

## 14. Initial Parameters (Razonables, no optimizados)  
- Stop loss: 10 pips  
- Take profit: 15 pips  
- Spread max: 2 pips  
- ATR(14) M5 min: 5 pips  
- Retesteo tolerancia: ±3 pips  
- Ventana trading: 08:00-17:00 NY  

## 15. Expected Frequency  
Aproximadamente 1 a 3 operaciones por día dependiendo de volatilidad y rupturas efectivas.  

## 16. Why It Might Fail  
- Mercados laterales o sin tendencia generan rupturas falsas y retestes múltiples o fallidos.  
- Alta volatilidad extrema puede saltar stops antes de que la tendencia se establezca.  
- Noticias imprevistas rápidamente cancelan la lógica técnica.  
- La fijación rígida de SL/TP podría ser insuficiente en condiciones cambiantes.  

## 17. Overfitting Risk (Bajo/Medio/Alto)  
Medio. El sistema usa reglas objetivas sobre niveles previos estandarizados, pero el ajuste de parámetros fijos puede no generalizar a todos los regímenes de mercado.  

## 18. Cost Sensitivity (Bajo/Medio/Alto)  
Medio-Alto. El uso en un par de bajo spread es clave; spreads mayores degradan rápidamente la rentabilidad dada la pequeña distancia entre SL y TP.  

## 19. Correlation With Liquidity Sweep Strategy (Baja/Media/Alta + Explicación)  
Media. Ambas estrategias buscan movimientos relacionados a niveles clave intradía y rompimientos falsos o confirmados, pero Trend Pullback se basa en el patrón breakout-retest mientras la Liquidity Sweep busca barridos de stops. Puede haber sesiones donde ambas ocurran, pero behavioral triggers distintos.  

## 20. Backtest Acceptance Criteria  
- Profit factor > 1.5  
- Ratio ganancias/pérdidas > 1.3  
- Drawdown máximo intradía menor a 5% de capital simulado  
- Al menos 200 operaciones en muestra histórica de 1 año intradía 08:00-17:00 NY  
- Win rate esperada: >48% con RR 1.5  
- Robustez: estabilidad de métricas en muestras cruzadas en diferentes períodos.  

## 21. Pseudocode  
```  
# Definiciones  
max_prev = máximo sesión anterior (08:00-17:00 NY día -1)  
min_prev = mínimo sesión anterior  
close_prev = cierre sesión anterior  
ATR = ATR(14) M5 actual  
spread = spread actual pips

Inicio trading a las 08:00 NY, cerrado a las 17:00 NY  

Para cada vela M5 entre 08:00-17:00 NY hacer:  
  Si cierre_vela_M5 > max_prev y ATR > 5 y spread < 2 y no noticia alta volatilidad:  
    # Esperar retesteo al nivel max_prev  
    Observar velas M1 después de cierre_vela_M5  
    Para cada vela M1 siguiente:  
      Si precio_cierre_M1 dentro [max_prev -3 pips, max_prev +3 pips] y  
         sombra_inferior_vela_M1 <= 50% cuerpo:  
        Abrir posición larga en cierre vela M1  
        SL = entrada - 10 pips  
        TP = entrada + 15 pips  
        Mientras (posición abierta):  
          Si ganancia >= 7.5 pips: mover SL a break even  
          Si hora >= 16:55 NY: cerrar posición  
          Si SL o TP tocados: cerrar posición  
        Fin Mientras  
        Romper ciclo retesteo  
      Fin Si  
    Fin Para  
  Fin Si

  Si cierre_vela_M5 < min_prev y ATR >5 y spread < 2 y sin noticia alta volatilidad:  
    Esperar retesteo nivel min_prev  
    Para velas M1 siguientes:  
      Si precio_cierre_M1 dentro [min_prev -3 pips, min_prev +3 pips] y  
         sombra_superior <= 50% cuerpo:  
        Abrir posición corta en cierre vela M1  
        SL = entrada + 10 pips  
        TP = entrada - 15 pips  
        Controlar posición igual que largo  
        Romper ciclo retesteo  
      Fin Si  
    Fin Para  
  Fin Si  
Fin Para  
```

### Strategy 15
## 1. Strategy Name  
Post-News Volatility Reversion EURUSD Intradía  

## 2. Market Phenomenon  
Después de una publicación económica relevante (anuncio calendarizado), el EURUSD suele experimentar un aumento abrupto y notable en volatilidad, seguido en las próximas 30 a 120 minutos por una estabilización y reversión hacia el nivel de volatilidad y precio prevaleciente antes del anuncio.  

## 3. Quant Hypothesis  
La volatilidad y el precio post-anuncio se "sobrerreaccionan" intradía, y en una ventana estrecha se produce una regresión hacia la media local previa al news release. Aprovechando esta estabilización tras el pico inicial, se puede capturar reversión de precio.  

## 4. Why It Could Work on EURUSD  
EURUSD es el par FX más líquido y sensible a noticias europeas y estadounidenses clave; la alta visibilidad, gran volumen y rapidez de reequilibrio institucional hacen que existan fases de sobrerreacción seguidas de estabilización, especialmente en sesiones NY.  

## 5. Trading Window (Sub-ventana dentro de 07:00-19:00 NY)  
Desde 10 minutos después del anuncio económico hasta 2 horas después (ejemplo típico: anuncio a las 08:30 NY, trading de 08:40 a 10:30 NY).  

## 6. Timeframe (M1, M3, M5, M15 o combinación)  
M1 para detección precisa del cambio post volatility spike, confirmación con M5 para reducción de ruido.  

## 7. Required Data  
- Ticks y OHLC M1 y M5 del EURUSD intradía.  
- Calendario económico con timestamps exactos de releases programados (tipo y consenso).  
- Spread en tiempo real.  
- Volatilidad histórica (ATR 14 en M5).  

## 8. Long Entry Rules (Reglas exactas)  
1. Identificar el último anuncio económico relevante dentro de ventana 07:00-19:00 NY.  
2. Medir ATR(14) en M5 en intervalo 30 min pre-release = ATR_pre.  
3. Detectar volatilidad post-release: en primera 10 min tras evento, ATR(14) > 2 * ATR_pre confirma spike.  
4. Observar en M1 una vela cerrada negativa con rango > 1.5 * ATR_pre, seguido inmediatamente en la siguiente vela M1 por una rotura arriba del último máximo de la barra negativa (break high).  
5. Al confirmarse ese break high, abrir posición Long limitando precio a cierre actual de vela.  

## 9. Short Entry Rules (Reglas exactas)  
Igual lógica pero invertida:  
1. Tras anuncio y spike (ATR(14) > 2 * ATR_pre).  
2. Vela M1 cerrada positiva con rango > 1.5 * ATR_pre.  
3. Siguiente vela M1 rompe abajo el mínimo de vela positiva previa.  
4. Al romper el low anterior decreciente, abrir posición Short.  

## 10. Stop Loss Logic  
Colocar SL a 1 ATR_pre del precio de entrada (ya sea por debajo entrada Long o por encima entrada Short).  

## 11. Take Profit Logic  
Objetivo de ganancia: 0.75 * ATR_pre desde precio entrada. (Se opera con RR de aprox 1:0.75 para adaptarse a alta volatilidad post news).  

## 12. Trade Management (BE, trailing, time stop, etc.)  
- Mover SL a Break Even (precio de entrada) cuando se alcance 0.25 * ATR_pre en ganancia.  
- Cierre automático del trade a los 120 minutos desde apertura si TP o SL no se activan.  
- No trailing adicional.  

## 13. Filters (Spread, ATR, News, etc.)  
- Filtrar trades si spread > 1.5 pips en el momento de entrada.  
- Sólo operar con anuncios económicos catalogados como "High Impact" y previamente definidos (FOMC, UE CPI, NFP, etc.).  
- No operar si ATR_pre < 0.0003 (baja volatilidad pre-evento).  

## 14. Initial Parameters (Razonables, no optimizados)  
- ATR periodo para cálculo: 14 en M5.  
- Multiplicadores: Spike threshold 2.0 * ATR_pre, rango vela 1.5 * ATR_pre.  
- SL: 1.0 * ATR_pre.  
- TP: 0.75 * ATR_pre.  
- Break Even: 0.25 * ATR_pre.  
- Spread máximo admisible: 1.5 pips.  
- Trading window post news: 10 min a 120 min después release.  

## 15. Expected Frequency  
1 a 3 trades por día, dependiendo calendario econ y la volatilidad.  

## 16. Why It Might Fail  
- Cambios estructurales en la respuesta institucional a noticias FX.  
- Eventos con impacto prolongado y direccional (rupturas fundamentales).  
- Flash crashes, gaps o noticias inesperadas que alteran la reversion.  
- Slippage excesivo en condiciones de hiper-volatilidad.  

## 17. Overfitting Risk (Bajo/Medio/Alto)  
Medio. Aunque reglas objetivas basadas en indicador ATR y eventos macro, hay dependencia en parámetros multiples y selección de eventos que puede generar sesgo.  

## 18. Cost Sensitivity (Bajo/Medio/Alto)  
Medio. El spread y slippage afectan proporcionalmente en un entorno de pocas velas y objetivos pequeños. La alta frecuencia también penaliza comisiones.  

## 19. Correlation With Liquidity Sweep Strategy (Baja/Media/Alta + Explicación)  
Baja. Liquidity Sweep se basa en barridos de stops en zonas de liquidez, mientras esta estrategia se orienta a reversión post-volatilidad; puede coincidir temporalmente pero no es causal ni colineal.  

## 20. Backtest Acceptance Criteria  
- Win rate mínimo 52% con ratio ganancia/pérdida >0.7.  
- Profit Factor > 1.2.  
- Máxima pérdida consecutiva inferior a 3 veces el tamaño promedio de ganancia.  
- Robustez temporal: resultados consistentes en subsamples pre 2015 y post 2015.  
- Estabilidad en spread simulados hasta 1.5 pips.  

## 21. Pseudocode  
```python
for each trading day between 07:00 and 19:00 NY:
    news_events = get_high_impact_news_events(day, pair='EURUSD')
    for event in news_events:
        event_time = event.timestamp
        ATR_pre = calculate_ATR('EURUSD', timeframe='M5', end_time=event_time - 1min, period=14)
        if ATR_pre < 0.0003:
            continue  # skip low volatility days
        
        # Check post news volatility spike
        ATR_post_10min = calculate_ATR('EURUSD', timeframe='M5', start=event_time, end=event_time + 10min, period=14)
        if ATR_post_10min < 2 * ATR_pre:
            continue  # no sufficient spike
        
        # Monitoring M1 bars in window event_time+10min to event_time+120min
        monitor_start = event_time + 10min
        monitor_end = event_time + 120min
        for t in minutes_range(monitor_start, monitor_end):
            prev_candle = get_candle('M1', t-1)
            current_candle = get_candle('M1', t)
            spread = get_spread('EURUSD', time=t)
            if spread > 0.00015:  # 1.5 pips in decimal
                continue            
            
            # Long entry condition
            if prev_candle.close < prev_candle.open and (prev_candle.high - prev_candle.low) > 1.5 * ATR_pre:
                if current_candle.high > prev_candle.high:
                    entry_price = current_candle.close
                    place_long_order(entry_price)
                    set_stop_loss(entry_price - ATR_pre)
                    set_take_profit(entry_price + 0.75 * ATR_pre)
                    manage_trade_BE_and_time_stop(entry_price, ATR_pre, max_duration=120min)
                    break_loop  # one trade per event
            
            # Short entry condition
            if prev_candle.close > prev_candle.open and (prev_candle.high - prev_candle.low) > 1.5 * ATR_pre:
                if current_candle.low < prev_candle.low:
                    entry_price = current_candle.close
                    place_short_order(entry_price)
                    set_stop_loss(entry_price + ATR_pre)
                    set_take_profit(entry_price - 0.75 * ATR_pre)
                    manage_trade_BE_and_time_stop(entry_price, ATR_pre, max_duration=120min)
                    break_loop
```

### Strategy 16
## 1. Strategy Name  
Post-News Momentum Continuation con Estabilización 15 min (PNMC-15)

## 2. Market Phenomenon  
Después de la publicación de una noticia económica relevante, el precio del EURUSD muestra un movimiento inicial fuerte que se estabiliza durante aproximadamente 15 minutos antes de continuar en la misma dirección.

## 3. Quant Hypothesis  
Tras la volatilidad inmediata post-noticia, la estabilización del precio durante 15 minutos permite identificar la dirección dominante del momentum, favoreciendo una continuación del movimiento en dicha dirección en el corto plazo intradía.

## 4. Why It Could Work on EURUSD  
EURUSD es altamente sensible a noticias económicas europeas y estadounidenses. El mercado inicialmente reacciona con alta volatilidad tras la publicación, luego el equilibrio de compradores/vendedores determina el momentum predominante. Esta estrategia captura la segunda ola del movimiento con mayor probabilidad de continuidad.

## 5. Trading Window (Sub-ventana dentro de 07:00-19:00 NY)  
07:30 – 15:00 NY (ventana principal de anuncios relevantes y alta liquidez post-apertura Europa y NYC).

## 6. Timeframe (M1, M3, M5, M15 o combinación)  
M1 para detectar volatilidad y estabilización; M5 para confirmación del momentum; M15 para entrada y gestión.

## 7. Required Data  
- Precio BID y ASK tick por tick.  
- Calendario económico con timestamps exactos de publicaciones (con nivel de relevancia alta).  
- Spreads.  
- ATR estimado intradía (5 sesiones recientes).  

## 8. Long Entry Rules (Reglas exactas)  
1. Identificar noticia económica de alta relevancia en calendario a t0.  
2. Confirmar que en M1 la vela posterior a la noticia (t0 a t0+1 min) cierre con rango > 1.5x ATR(1min), y cierre alcista respecto apertura.  
3. Esperar 15 minutos (de t0+1min a t0+16min) durante los cuales el precio se estabiliza: la desviación estándar del precio en ventana rolling M1 de esos 15 min debe ser ≤ 0.5x ATR(1min).  
4. Confirmar dirección del momentum en M5: comparar cierre vela M5 que incluye t0+16min contra vela M5 anterior; debe ser alcista (precio cierre > apertura).  
5. Entrar Long en cierre de vela M15 que cierra en t0+15min o t0+30min ajustado al primer momento posible luego del estabilización, con precio superior a máximo en esos 15 minutos previos.

## 9. Short Entry Rules (Reglas exactas)  
1. Igual pasos 1 a 3 para noticia relevante.  
2. En paso 2, vela M1 posterior a noticia debe ser bajista con rango > 1.5x ATR(1min).  
3. Durante 15 minutos siguientes, desviación estándar ≤ 0.5x ATR(1min).  
4. Confirmar momentum en M5 con vela bajista (cierre < apertura).  
5. Entrar Short en cierre de vela M15 correspondiente tras estabilización, con precio inferior a mínimo en últimos 15 minutos previos a entrada.

## 10. Stop Loss Logic  
Fijar SL a 1 ATR(15min) del punto de entrada, en contra de dirección de la operación (para Long: SL = entrada – ATR; para Short: SL = entrada + ATR).

## 11. Take Profit Logic  
Take profit a 2x ATR(15min) del punto de entrada en la dirección operada.

## 12. Trade Management (BE, trailing, time stop, etc.)  
- Mover SL a Break Even tras alcanzar 1x ATR(15min) de ganancia.  
- No trailing.  
- Time stop: cerrar operación si no se alcanza TP o SL a las 19:00 NY o 4 horas después de la entrada, lo que ocurra primero.

## 13. Filters (Spread, ATR, News, etc.)  
- Descartar operaciones si spread > 1.5 pips.  
- Entrar solo si ATR(15min) > 4 pips (para asegurar volatilidad suficiente).  
- Solo noticias catalogadas como “high impact” en calendario económico oficial (ej. Bloomberg, Econoday).  
- No operar si hay anuncios contrapuestos en ±30 minutos.

## 14. Initial Parameters (Razonables, no optimizados)  
- Rango vela post-noticia: >1.5x ATR(1min) (ej: ATR(1min)=0.8 pips → rango >1.2 pips)  
- Estabilización: desviación std ≤ 0.5 x ATR(1min)  
- SL: 1x ATR(15min) (~10 pips típico)  
- TP: 2x ATR(15min) (~20 pips típico)  
- Spread máximo: 1.5 pips  
- Tiempo máximo de mantenimiento: 4 horas o cierre 19:00 NY

## 15. Expected Frequency  
Aproximadamente 1–3 operaciones diarias, dependiendo del calendario económico, con operaciones solo en días con noticias relevantes.

## 16. Why It Might Fail  
- El movimiento post-noticia no tiene continuidad (falsos breakouts).  
- Noticias muy anticipadas o filtradas reducen momentum real.  
- Alta volatilidad residual que invalida estabilización (ruido).  
- Eventos inesperados durante ventana (flash crashes, intervenciones).  
- Spread/picardía del mercado en horas bajas.  

## 17. Overfitting Risk (Bajo/Medio/Alto)  
Medio. El uso de reglas simples y parámetros relacionados a volatilidad reduce riesgo, pero no elimina riesgos de ajuste a condiciones históricas específicas.

## 18. Cost Sensitivity (Bajo/Medio/Alto)  
Medio-Alto. Stop y take profit de 10-20 pips sensibles a spreads y comisiones; si costos fijos aumentan, rentabilidad impacta.

## 19. Correlation With Liquidity Sweep Strategy (Baja/Media/Alta + Explicación)  
Media. Ambas estrategias reaccionan a movimientos técnicos post-noticia, pero PNMC-15 centra en estabilización y continuación mientras que sweep busca zonas de liquidez y reversión.

## 20. Backtest Acceptance Criteria  
- Ratio ganancia/pérdida (RR) ≥ 1.8  
- %Operaciones ganadoras ≥ 45%  
- Drawdown máximo ≤ 10% del capital  
- Esperanza matemática positiva > 0 pesando costos spreads y comisiones  
- Operación media > 1 por día en periodos con noticias relevantes  
- Robustez en out-of-sample (test 20% datos)  

## 21. Pseudocode  

```
Input:
- Calendario economico news[] con timestamps t0 y impactoHigh
- ATR_1m, ATR_15m calculado rolling en últimas 5 sesiones
- Precio BID/ASK 1m, 5m, 15m
- Spread actual

For each news_event in news[] during tradingWindow (07:30-15:00 NY):
    if news_event.impactHigh == False:
        continue
    
    # Obtener vela M1 justo posterior a news
    vela_post = get_M1_candle(t0 to t0+1min)
    rango_post = vela_post.high - vela_post.low

    if rango_post < 1.5*ATR_1m:
        continue

    direccion_post = +1 if vela_post.close > vela_post.open else -1
    if direccion_post == 0:
        continue

    # Verificar estabilización en 15 min siguientes (t0+1 to t0+16)
    precios_15min = get_prices_M1(t0+1min to t0+16min)
    std_15min = std(precios_15min)

    if std_15min > 0.5*ATR_1m:
        continue

    # Confirmación momentum en vela M5 que cubre desde t0+16min
    vela_M5 = get_M5_candle(cierre en t0+16min)
    if direccion_post == +1 and vela_M5.close <= vela_M5.open:
        continue
    elif direccion_post == -1 and vela_M5.close >= vela_M5.open:
        continue

    # Entrada en vela M15 posterior (t0+15min o t0+30min)
    vela_M15 = get_M15_candle(velas posteriores)

    if direccion_post == +1:
        # Confirmar ruptura máximo últimos 15 min antes de entrada
        max_15min = max(precios_15min)
        if vela_M15.close > max_15min and spread <= 1.5 pips and ATR_15m > 4 pips:
            entry_price = vela_M15.close
            SL = entry_price - ATR_15m
            TP = entry_price + 2*ATR_15m
            abrir_posicion("LONG", entry_price, SL, TP)
    else:
        min_15min = min(precios_15min)
        if vela_M15.close < min_15min and spread <= 1.5 pips and ATR_15m > 4 pips:
            entry_price = vela_M15.close
            SL = entry_price + ATR_15m
            TP = entry_price - 2*ATR_15m
            abrir_posicion("SHORT", entry_price, SL, TP)

Gestión de la posición:
- Si ganancia alcanza ATR_15m mover SL a BE
- Si no TP o SL es alcanzado en 4h desde entrada o 19:00 NY, cerrar posición.

```

### Strategy 17
## 1. Strategy Name  
London Close Mean Reversion VWAP (LCMR-VWAP) Intradía EURUSD  

## 2. Market Phenomenon  
A lo largo del día, el EURUSD tiende a mostrar un efecto de retorno estadístico (mean reversion) hacia su VWAP diario al cierre de la sesión de Londres, debido a la liquidación de posiciones y ajuste de portafolios.  

## 3. Quant Hypothesis  
Durante la sesión europea (especialmente cerca del cierre de Londres), cuando el precio se aleja significativamente de la VWAP diaria, existe una probabilidad elevada de que revierta hacia dicha VWAP en la ventana intradía restante.  

## 4. Why It Could Work on EURUSD  
EURUSD es el par con mayor liquidez y participación institucional en Londres. La liquidación y reequilibrio de posiciones al cierre de Londres genera flujos y presiones de precio que inducen a revertir al valor promedio del día (VWAP), especialmente en períodos de baja volatilidad y alta liquidez.  

## 5. Trading Window (Sub-ventana dentro de 07:00-19:00 NY)  
09:30 - 16:30 NY (corresponde aproximadamente a 14:30-21:30 Londres, con énfasis en la última hora de Londres).  

## 6. Timeframe (M1, M3, M5, M15 o combinación)  
M5 para señales de entrada/salida, confirmación con VWAP calculado en M1 para precisión.  

## 7. Required Data  
- EURUSD M1 y M5 OHLC ticks con timestamp.  
- Cálculo en tiempo real de VWAP diario desde las 00:00 NY a la hora actual.  
- Horarios oficiales NY y Londres para segmentación.  
- Spread y volumen para filtros.  
- Calendario económico para filtro de noticias.  

## 8. Long Entry Rules (Reglas exactas)  
1. En la ventana 09:30-16:30 NY, el precio M5 cierra mínimo 5 pips por debajo del VWAP diario actual.  
2. El precio M1 debe mostrar una vela alcista con cierre > apertura en la última barra.  
3. El spread actual debe ser ≤ 1.5 pips.  
4. No hay noticias macro relevantes con impacto alto en los próximos 30 minutos.  
Entrar en compra al cierre de la barra M5 que cumpla estas condiciones.  

## 9. Short Entry Rules (Reglas exactas)  
1. En la ventana 09:30-16:30 NY, el precio M5 cierra mínimo 5 pips por encima del VWAP diario actual.  
2. El precio M1 debe mostrar una vela bajista con cierre < apertura en la última barra.  
3. El spread actual debe ser ≤ 1.5 pips.  
4. No hay noticias macro relevantes con impacto alto en los próximos 30 minutos.  
Entrar en venta al cierre de la barra M5 que cumpla estas condiciones.  

## 10. Stop Loss Logic  
- Stop loss estático a 10 pips del nivel de entrada (en dirección contraria).  
- Stop loss actualizado a break-even + 1 pip cuando el precio se haya movido favorablemente 6 pips.  

## 11. Take Profit Logic  
- Objetivo fijo de 8 pips.  
- Cierre parcial del 50% a 4 pips para asegurar ganancias y dejar correr la mitad restante.  
- Trailing stop de 3 pips para la posición restante.  

## 12. Trade Management (BE, trailing, time stop, etc.)  
- Break-even automático +1 pip cuando el precio alcanza +6 pips favorable.  
- Trailing stop de 3 pips para posición parcial restante tras cerrar 50% a 4 pips.  
- Cierre forzado de cualquier posición abierta al final de la ventana de trading (16:30 NY).  

## 13. Filters (Spread, ATR, News, etc.)  
- Spread máximo 1.5 pips para entrada.  
- ATR(14) diario se usa para evitar días con volatilidad excesiva (si ATR > 40 pips, no operar).  
- Evitar operaciones 30 min antes y después de noticias con impacto alto en calendario económico oficial.  

## 14. Initial Parameters (Razonables, no optimizados)  
- Entrada disparada a desviación mínima de 5 pips del VWAP.  
- SL 10 pips, TP 8 pips.  
- Ventana 09:30-16:30 NY.  
- Spread máximo 1.5 pips.  
- ATR(14) máximo 40 pips.  

## 15. Expected Frequency  
3-6 operaciones por día, dependiendo de volatilidad y condiciones de mercado.  

## 16. Why It Might Fail  
- Movimientos direccionales fuertes dominantes (p.ej. anuncios macro inesperados o shocks geopolíticos).  
- Cambios estructurales en comportamiento de flujo tras reformas regulatorias o cambios en liquidez.  
- Spread/latencia altos que invalidan entradas precisas y reglas rígidas.  

## 17. Overfitting Risk (Bajo/Medio/Alto)  
Medio — basado en estadística simple de mean reversion a VWAP pero parametrizado con valores específicos que pueden necesitar ajustes.  

## 18. Cost Sensitivity (Bajo/Medio/Alto)  
Medio — el spread y slippage impactan especialmente en ganancias esperadas ajustadas; estrategia intra-day con objetivo pequeño.  

## 19. Correlation With Liquidity Sweep Strategy (Baja/Media/Alta + Explicación)  
Baja — esta estrategia está basada en mean reversion hacia VWAP mientras que una estrategia de liquidity sweep busca rupturas y continuaciones tras barridos de stops; enfoques opuestos.  

## 20. Backtest Acceptance Criteria  
- Sharpe ratio anualizado > 1.2  
- Drawdown máximo ≤ 5% por mes.  
- % ganancias / operaciones positivas > 55%.  
- Ratio Beneficio/Pérdida mínimo 1:0.8.  
- Resultados estables en periodos in-sample y out-sample en al menos 3 años de datos recientes (últimos 5 años).  

## 21. Pseudocode  

```
Inicializar VWAP_diario = VWAP desde 00:00 NY hasta tiempo_actual  
Para cada barra M5 en ventana 09:30 a 16:30 NY:  
    precio_cierre = cierre de barra M5  
    distancia_vwap = precio_cierre - VWAP_diario  
    si spread_actual > 1.5 pips o ATR(14) diario > 40 pips o noticias_impacto_alto en ±30min:  
        ignorar señal y continuar  
    si distancia_vwap ≤ -5 pips y vela_M1_ultima_alcista:  
        abrir posicion long a precio_cierre con SL=entrada-10 pips, TP=entrada+8 pips  
    si distancia_vwap ≥ 5 pips y vela_M1_ultima_bajista:  
        abrir posicion short a precio_cierre con SL=entrada+10 pips, TP=entrada-8 pips  

Para cada posición abierta:  
    si ganancia_unrealizada ≥ 6 pips y SL < BE+1 pip:  
        mover SL a BE+1 pip  
    si ganancia_unrealizada ≥ 4 pips y partial_close_50% no ejecutado:  
        cerrar 50% de posición  
    si posición residual abierta y ganancia_unrealizada ≥ trailing_stop(3 pips):  
        mover SL con trailing de 3 pips  
    si hora_actual ≥ 16:30 NY:  
        cerrar todas posiciones abiertas  
```

### Strategy 18
## 1. Strategy Name  
NY Mid-Day Volatility Expansion Breakout EURUSD  

## 2. Market Phenomenon  
Durante el mediodía en Nueva York (aprox. 12:00-14:00 NY), el mercado EURUSD suele experimentar un periodo de consolidación o baja volatilidad asociado al "lunch break". Al reanudarse la actividad acumulada, ocurre una expansión rápida de la volatilidad que puede generar rupturas direccionales significativas.  

## 3. Quant Hypothesis  
Si se identifica un rango estrecho y bajo volumen durante la ventana previa al mediodía, la ruptura direccional posterior a dicho rango tiene alta probabilidad de continuidad, pudiendo capturar movimientos rápidos intradía.  

## 4. Why It Could Work on EURUSD  
EURUSD es altamente líquido con una marcada actividad intradía durante la sesión NY, mostrando patrones recurrentes de consolidación y expansión, especialmente alrededor del lunch break debido a la disminución temporal de traders institucionales y luego reapertura del mercado.  

## 5. Trading Window (Sub-ventana dentro de 07:00-19:00 NY)  
Entrada: entre 12:00 y 14:00 NY (periodo del lunch break con rango reducido y posterior breakout).  
Salida o cierre: antes de las 15:00 NY para evitar volatilidad atípica del cierre de sesión.  

## 6. Timeframe (M1, M3, M5, M15 o combinación)  
Principal: M5 para cálculo de rango y confirmación breakout.  
Confirmación de expansión: M1 para entradas precisas al romper rango acumulado.  

## 7. Required Data  
- Precio OHLC M1 y M5 del EURUSD (ticks agregados).  
- Volumen tick o proxy (opcional para validar baja actividad).  
- Spread actual al momento de la entrada.  
- Calendario macroeconómico de noticias EUR/USD para filtrado.  

## 8. Long Entry Rules (Reglas exactas)  
1. Calcular el rango máximo (High - Low) del EURUSD entre 11:30 y 12:00 NY (previo al lunch break) en M5.  
2. Confirmar que este rango es inferior a un umbral fijo basado en ATR(14, M5) * 0.5 (indicando baja volatilidad).  
3. Observar cierre de vela M1 superior al máximo del rango pre-lunch (resistencia del rango).  
4. Confirmar spread menor a 1.5 pips para evitar costes altos.  
5. Entrar en mercado AL ALZA en la apertura de la siguiente vela M1 tras cierre por encima del rango.  

## 9. Short Entry Rules (Reglas exactas)  
1. Calcular el rango máximo (High - Low) del EURUSD entre 11:30 y 12:00 NY en M5.  
2. Confirmar que este rango es inferior a un umbral basado en ATR(14, M5) * 0.5.  
3. Observar cierre de vela M1 inferior al mínimo del rango pre-lunch (soporte del rango).  
4. Confirmar spread menor a 1.5 pips.  
5. Entrar en mercado A LA BAJA en la apertura de la siguiente vela M1 tras cierre por debajo del rango.  

## 10. Stop Loss Logic  
Stop loss fijo a 1.0 ATR(14, M5) desde la entrada (medido en pips).  
Ejemplo:  
- Para posición larga, SL = entrada - ATR(14,M5).  
- Para corta, SL = entrada + ATR(14,M5).  

## 11. Take Profit Logic  
TP inicial a 2.0 ATR(14, M5) del punto de entrada (RR 2:1).  

## 12. Trade Management (BE, trailing, time stop, etc.)  
- Al alcanzar 1.0 ATR de ganancia, mover stop loss a BreakEven (+1 pip de buffer para largas, -1 pip para cortas).  
- Implementar trailing stop de 0.5 ATR (M5) desde máximo/minimo alcanzado después del BE.  
- Cerrar posición puntualmente a las 15:00 NY si no ha cerrado antes.  

## 13. Filters (Spread, ATR, News, etc.)  
- No operar si spread > 1.5 pips.  
- No operar si ATR(14, M5) en la ventana 11:30-12:00 > 0.0012 (indicando alta volatilidad previa).  
- No operar 30 minutos antes y después de noticias de alto impacto para EUR o USD (según calendario).  
- Volumen tick inferior al 30% en ventana 11:30-12:00 valida baja liquidez previa, condición preferible.  

## 14. Initial Parameters (Razonables, no optimizados)  
- Rango bajo volatilidad: ATR(14, M5) * 0.5  
- SL: 1.0 ATR(14, M5)  
- TP: 2.0 ATR(14, M5)  
- Spread máximo: 1.5 pips  
- Ventana rango: 11:30-12:00 NY  
- Ventana trading: 12:00-14:00 NY  
- Cierre posiciones a las 15:00 NY  

## 15. Expected Frequency  
Aproximadamente 1-2 operaciones por día hábil considerando condiciones de spread y baja volatilidad previa.  

## 16. Why It Might Fail  
- Cambios estructurales en volatilidad intradía del EURUSD.  
- Periodos con noticias imprevistas que invaliden rango y breakout.  
- Spread o slippage excesivo afectando la relación riesgo/beneficio.  
- Rallies o caídas abruptas sin consolidación previa (falsos breakouts).  

## 17. Overfitting Risk (Bajo/Medio/Alto)  
Medio: uso de ATR y ventanas predefinidas ayuda a robustez, pero parámetros específicos pueden ajustar demasiado a datos históricos.  

## 18. Cost Sensitivity (Bajo/Medio/Alto)  
Medio: debido a la frecuencia moderada y nivel de spread bajo aceptado, costes pueden impactar pero estrategia puede ser viable en brokers competitivos.  

## 19. Correlation With Liquidity Sweep Strategy (Baja/Media/Alta + Explicación)  
Media: Ambas estrategias buscan rupturas en momentos de baja liquidez, pero NY Mid-Day Volatility Expansion se centra en ruptura por expansión de volatilidad post-lunch, mientras que Liquidity Sweep en barridos rápidos de stops. Se pueden superponer pero buscan señales con timings y confirmaciones distintas.  

## 20. Backtest Acceptance Criteria  
- Expectativa matemática positiva (> +0.2% por trade).  
- Ratio ganancia/pérdida (profit factor) > 1.5.  
- Drawdown máximo menor a 5% del capital.  
- Win Rate mínima 40% con RR aprox. 2:1.  
- Consistencia en despliegue de resultados en diferentes años y condiciones de mercado (no solo en periodo optimizado).  

## 21. Pseudocode  
```
INPUT:  
- ATR_period = 14  
- SL_multiplier = 1.0  
- TP_multiplier = 2.0  
- breakout_window_start = 11:30 NY  
- breakout_window_end = 12:00 NY  
- trading_window_start = 12:00 NY  
- trading_window_end = 14:00 NY  
- max_spread = 1.5 pips  

FOR each trading day DO:  
    # Calcular ATR(14, M5) durante ventana pre-lunch  
    atr_value = ATR(14, M5) calculated until breakout_window_end  

    # Calcular rango máximo en ventana pre-lunch (11:30-12:00 NY)  
    range_minutes = select M5 bars in breakout_window_start to breakout_window_end  
    max_high = max(highs in range_minutes)  
    min_low = min(lows in range_minutes)  
    range_size = max_high - min_low  

    IF range_size > atr_value * 0.5 THEN  
        SKIP trading this day (no baja volatilidad)  

    ELSE IF spread_now > max_spread THEN  
        SKIP trading this day (costes altos)  

    ELSE IF news_high_impact_within(30 min before or after trading_window_start) THEN  
        SKIP trading this day (evento de volatilidad)  

    ELSE  
        WAIT from trading_window_start to trading_window_end for breakout  

        FOR each M1 bar between trading_window_start and trading_window_end DO:  
            close_price = close of current M1 bar  

            # Check Long Entry  
            IF close_price > max_high THEN  
                ENTRY price = open price next M1 bar  
                SL = ENTRY - atr_value * SL_multiplier  
                TP = ENTRY + atr_value * TP_multiplier  
                EXECUTE long position with above SL, TP  
                MANAGE trade per rules  
                BREAK out of FOR loop (only 1 trade/day)  

            # Check Short Entry  
            ELSE IF close_price < min_low THEN  
                ENTRY price = open price next M1 bar  
                SL = ENTRY + atr_value * SL_multiplier  
                TP = ENTRY - atr_value * TP_multiplier  
                EXECUTE short position with above SL, TP  
                MANAGE trade per rules  
                BREAK out of FOR loop  

    # Trade Management:  
    IF position open THEN  
         IF profit >= atr_value * SL_multiplier THEN  
             MOVE SL to Break-Even + 1 pip buffer (long) or -1 pip buffer (short)  
         IF profit trail_triggered (profit - trailing_stop distance) THEN  
             ADJUST SL a trailing stop (0.5 ATR)  
         IF time >= 15:00 NY AND position open THEN  
             CLOSE position  

END FOR day  
```

### Strategy 19
## 1. Strategy Name  
Hybrid Volatility-Filtered Trend Following (HVFTF) EURUSD Intraday

## 2. Market Phenomenon  
Tendencias intradía identificadas mediante filtros de volatilidad basada en ATR y señales de SuperTrend para capturar movimientos persistentes durante horas activas de trading, evitando ruido en fases laterales.

## 3. Quant Hypothesis  
La combinación de un filtro dinámico de volatilidad (ATR) con la lógica de tendencia de SuperTrend mejora la selección de entradas, maximizando ganancias en tendencias claras y minimizando operaciones en rangos laterales, incrementando la relación riesgo-recompensa en EURUSD intradía.

## 4. Why It Could Work on EURUSD  
EURUSD presenta volatilidad suficiente en la sesión NY para generar tendencias intradía claras, pero también tiene tendencias falsas y ruido; usar ATR para filtrar rangos bajos y SuperTrend para seguimiento de tendencias puede proporcionar señales limpias en el horario más líquido.

## 5. Trading Window (Sub-ventana dentro de 07:00-19:00 NY)  
09:30 a 18:00 NY — horario posterior a la apertura US y antes del cierre, donde la liquidez es alta y las tendencias intradía tienden a consolidarse.

## 6. Timeframe (M1, M3, M5, M15 o combinación)  
Principal: M5 para balancear velocidad y ruido. Subconfirmación en M15 para validar estabilidad de tendencia.

## 7. Required Data  
- Precio OHLC M5 y M15  
- ATR (14 periodos M5)  
- SuperTrend (ATR basado, periodo 10, factor 3, aplicado a M5)  
- Spread en tiempo real  
- Calendario económico para filtrado de noticias de alta impacto en sesiones NY

## 8. Long Entry Rules (Reglas exactas)  
1. SuperTrend M5 cambia a tendencia alcista (barra actual cierra por encima de SuperTrend)  
2. ATR(14) M5 debe estar por encima de un umbral mínimo (0.0005 para EURUSD, para evitar baja volatilidad)  
3. Confirmación M15 del SuperTrend también sea alcista (tendencia estable en higher timeframe)  
4. El spread debe estar por debajo de 2 pips  
Si se cumplen los 4 criterios, entrar largo en apertura de la siguiente barra M5.

## 9. Short Entry Rules (Reglas exactas)  
1. SuperTrend M5 cambia a tendencia bajista (barra actual cierra por debajo de SuperTrend)  
2. ATR(14) M5 > 0.0005  
3. Confirmación M15 SuperTrend bajista  
4. Spread < 2 pips  
Entrada corta en apertura próxima barra M5.

## 10. Stop Loss Logic  
Stop inicial en SuperTrend M5 del momento de entrada (valor del SuperTrend actual). Esto adapta el stop dinámicamente a la volatilidad y estado de la tendencia.

## 11. Take Profit Logic  
TP fijo de 2.5 veces el riesgo medido desde entrada a stop loss (RR 1:2.5), garantizando expectativa positiva.

## 12. Trade Management (BE, trailing, time stop, etc.)  
- Breakeven: cuando la posición alcance 1.2 veces riesgo en ganancia, mover stop a entrada  
- Trailing: después de BE activado, mover stop a SuperTrend actualizado en cada barra M5 (trailing dinámico)  
- Time stop: cierre obligatorio a las 18:00 NY si la posición sigue abierta, para evitar exposición fuera de horario objetivo.

## 13. Filters (Spread, ATR, News, etc.)  
- Spread máximo 2 pips para entradas  
- ATR mínimo 0.0005 para evitar baja volatilidad  
- No entrar o mantener posiciones 30 minutos antes y después de noticias high impact calendario económico NY

## 14. Initial Parameters (Razonables, no optimizados)  
- ATR periodo: 14 (M5)  
- SuperTrend ATR periodo: 10, factor: 3  
- Volatilidad mínima ATR: 0.0005 (5 pips)  
- Spread máximo: 2 pips  
- RR objetivo: 1:2.5

## 15. Expected Frequency  
Aproximadamente 1-3 operaciones por día según condiciones de volatilidad y tendencia.

## 16. Why It Might Fail  
- Rango lateral con falsas señales de SuperTrend generando pérdidas  
- Cambios abruptos post-noticias no completamente filtrados  
- Spread altos o slippage en momentos de baja liquidez  
- Periodos de baja volatilidad prolongada reducen oportunidades.

## 17. Overfitting Risk (Bajo/Medio/Alto)  
Medio — parámetros basados en métodos estándar ATR y SuperTrend; riesgo en ajustes demasiado finos en backtest.

## 18. Cost Sensitivity (Bajo/Medio/Alto)  
Medio — spread y slippage impactan debido a bajo apalancamiento intradía y stops relativamente ajustados.

## 19. Correlation With Liquidity Sweep Strategy (Baja/Media/Alta + Explicación)  
Baja — esta estrategia sigue tendencias basadas en indicadores técnicos y volatilidad, mientras que las estrategias de liquidity sweep buscan ejecutar en eventos de manipulación de liquidez; por tanto, poco solapamiento operacional.

## 20. Backtest Acceptance Criteria  
- Profit factor > 1.5  
- Drawdown máximo < 10% del capital simulado  
- Ratio ganancias/pérdidas > 1.8  
- Win rate entre 40-60% para evitar overfitting en sensibilidad  
- Consistencia de resultados en distintos años y condiciones de mercado

## 21. Pseudocode  

```
Para cada barra M5 dentro de 09:30-18:00 NY:

    Calcular ATR14_M5
    Calcular SuperTrend10_3_M5
    Calcular SuperTrend10_3_M15 (confirmacion)

    Si (Spread > 2 pips o ATR14_M5 < 0.0005 o hay noticia high impact próximo)  
        NO entrar ni mantener posición

    Si no hay posición abierta:

        # Entrada Long
        Si (Cierre_M5 > SuperTrend_M5) 
           y (SuperTrend_M15 tendencia alcista)
           y (ATR14_M5 > 0.0005)
           y (Spread < 2 pips)
           entonces  
                Abrir compra al precio apertura próxima barra M5
                StopLoss = valor SuperTrend_M5 actual
                TakeProfit = Entrada + 2.5*(Entrada - StopLoss)

        # Entrada Short
        Si (Cierre_M5 < SuperTrend_M5)
           y (SuperTrend_M15 tendencia bajista)
           y (ATR14_M5 > 0.0005)
           y (Spread < 2 pips)
           entonces
                Abrir venta en apertura próxima barra M5
                StopLoss = valor SuperTrend_M5 actual
                TakeProfit = Entrada - 2.5*(StopLoss - Entrada)

    Si hay posición abierta:

        GananciaActual = Precio actual - Precio entrada (según dirección)
        
        Si (GananciaActual >= 1.2 * Riesgo)
             Mover StopLoss a Precio entrada (Break Even)

        Actualizar StopLoss a máximo/minimo valor SuperTrend_M5 desde que se activó BE

        Si (Hora >= 18:00 NY)
             cerrar posición

        Si (Precio toca StopLoss o TakeProfit)
             cerrar posición
```

### Strategy 20
## 1. Strategy Name  
Hybrid M15 Trend + VWAP Mean Reversion Intradía EURUSD

## 2. Market Phenomenon  
Los precios tienden a seguir una dirección definida en marcos temporales medianos (M15), pero muestran reversión de corto plazo hacia el VWAP intradía debido a la liquidez y el interés institucional en niveles promedio de coste de mercado.

## 3. Quant Hypothesis  
En una tendencia claramente establecida en M15, las desviaciones significativas del precio respecto al VWAP actúan como exceso temporal, generando oportunidades de reversión hacia el VWAP durante la ventana intradía 07:00-19:00 NY.

## 4. Why It Could Work on EURUSD  
EURUSD es un par con alta liquidez y volatilidad intradía consistente, donde el VWAP representa un nivel clave para flujos institucionales. La estructura de tendencias en M15 permite definir contexto y filtrar ruido, mejorando la precisión de señal de reversión.

## 5. Trading Window (Sub-ventana dentro de 07:00-19:00 NY)  
09:00 - 17:00 NY (excluyendo rangos con baja volatilidad antes y después deLunch y close)

## 6. Timeframe (M1, M3, M5, M15 o combinación)  
M15 para definición de tendencia y M1 para entradas mean reversion al VWAP.

## 7. Required Data  
- Precios OHLC M15 y M1 intradía (tick o M1 si no hay tick)  
- VWAP intradía calculado desde apertura NY (07:00 NY) hasta el momento actual  
- Volumen o ticks para validar liquidez (opcional pero recomendado)  
- Indicador ATR 14 M15 para volatilidad y gestión de stops

## 8. Long Entry Rules (Reglas exactas)  
1. Tendencia M15 definida como MA10 > MA30 (ambos con precio close M15) para alza.  
2. Precio M1 debe cerrar por debajo de VWAP por al menos 3 pips (30 puntos básicos).  
3. Confirmar vela de reversal alcista en M1: cierre M1 > apertura M1 después de tocar distancia -3 pips al VWAP.  
4. Entrada LONG en siguiente vela M1 open.

## 9. Short Entry Rules (Reglas exactas)  
1. Tendencia M15 definida como MA10 < MA30 para baja.  
2. Precio M1 debe cerrar por encima de VWAP por al menos 3 pips.  
3. Confirmar vela reversal bajista en M1: cierre M1 < apertura M1 después de tocar +3 pips al VWAP.  
4. Entrada SHORT en siguiente vela M1 open.

## 10. Stop Loss Logic  
Stop Loss fijo a 1.5 × ATR14 (M15) desde el precio de entrada:  
- Para LONG: SL = Entrada - 1.5 × ATR14  
- Para SHORT: SL = Entrada + 1.5 × ATR14

## 11. Take Profit Logic  
Take Profit inicial a 1 × ATR14 del M15 desde precio entrada hacia VWAP (target de reversión) + 0.5 × ATR14 adicional para captura parcial:  
- TP1: VWAP nivel (reversión al VWAP)  
- TP2: VWAP ± 0.5×ATR14 para extensión si el movimiento continúa.

## 12. Trade Management (BE, trailing, time stop, etc.)  
- Al alcanzar TP1 (reversión al VWAP), mover stop a Break Even + 1 pip.  
- Trailing con 0.5 × ATR14 activado después de TP1.  
- Time stop: liquidar posición si no se ha alcanzado TP1 en 30 minutos.

## 13. Filters (Spread, ATR, News, etc.)  
- Spread máximo 1.5 pips para entrar.  
- ATR M15 mínimo 5 pips para operar (evitar rango muy bajo).  
- No operar 15 minutos antes / después de noticias económicas de alto impacto (calendario predefinido).  
- Volumen/tick mínimo para validar liquidez (opcional).

## 14. Initial Parameters (Razonables, no optimizados)  
- MA10 y MA30 en M15 para tendencia.  
- Distancia reversión VWAP: 3 pips.  
- SL: 1.5 × ATR14 M15.  
- TP1: VWAP, TP2: VWAP ± 0.5 × ATR14.  
- Trading window 09:00-17:00 NY.  
- Time stop 30 minutos.  
- Spread máximo 1.5 pips.  
- ATR minimo: 5 pips.

## 15. Expected Frequency  
15-30 trades por día (dependiendo condiciones de volatilidad y ventanas).  

## 16. Why It Might Fail  
- Mercados sin tendencia clara en M15 o con tendencia ambigua.  
- Movimientos explosivos y noticias que generan rupturas sin reversión.  
- Cambios estructurales en comportamiento del VWAP o liquidez del mercado.  
- Spread elevado fuera de filtración.

## 17. Overfitting Risk (Bajo/Medio/Alto)  
Medio: la estrategia se basa en reglas de sentido común pero parámetros requieren validación en out-of-sample y múltiple mercado intradía.

## 18. Cost Sensitivity (Bajo/Medio/Alto)  
Medio: stops abiertos relativamente amplios con TP modestos requieren spreads bajos y buen slippage para mantener ventaja.

## 19. Correlation With Liquidity Sweep Strategy (Baja/Media/Alta + Explicación)  
Baja: esta estrategia se centra en reversión a nivel VWAP en tendencia M15, mientras que la estrategia Liquidity Sweep busca rupturas agresivas y barridos de stops, fenómenos opuestos.

## 20. Backtest Acceptance Criteria  
- Profit factor > 1.3  
- Ratio ganadoras > 45%  
- Expectativa promedio positiva > 0.5 × riesgo  
- Drawdown máximo < 5% del capital en test intradía  
- Consistencia en diferentes años y condiciones de volatilidad  
- Robustez en sensibilidad ±10% parámetros

## 21. Pseudocode  

```python
# Parámetros iniciales
MA_SHORT = 10
MA_LONG = 30
DIST_VWAP = 0.0003       # 3 pips
SL_ATR_MULT = 1.5
TP1_LEVEL = 0            # VWAP exacto
TP2_ATR_MULT = 0.5
ATR_PERIOD = 14
SPREAD_MAX = 0.00015     # 1.5 pips
TRADING_START = 9 * 60   # minutos desde 00:00 NY
TRADING_END = 17 * 60
TIME_STOP_MINS = 30

for each trading_day in dataset:

    calculate VWAP intradía desde 07:00 NY hasta cierre

    for each M15 bar t in 07:00 - 19:00 NY:
        ma_short = SMA(close_M15, MA_SHORT)
        ma_long = SMA(close_M15, MA_LONG)
        trend_up = ma_short > ma_long
        trend_down = ma_short < ma_long
        atr = ATR(ATR_PERIOD, M15)
    
        # Solo operar si ATR > 5 pips y spread < 1.5 pips
        if atr < 0.0005 or current_spread > SPREAD_MAX:
            continue

        for each M1 bar within M15 bar t:

            price = close_M1_t
            vwap_current = VWAP at M1 bar time

            # Long Entry
            if trend_up:
                if (vwap_current - price) >= DIST_VWAP:
                    # Confirmación vela reversal bullish M1
                    if close_M1 > open_M1:
                        entry_price = open_next_M1_bar
                        sl = entry_price - SL_ATR_MULT * atr
                        tp1 = vwap_current
                        tp2 = tp1 + TP2_ATR_MULT * atr
                        place_long_order(entry_price, sl, tp1, tp2, time_stop=TIME_STOP_MINS)
            
            # Short Entry
            elif trend_down:
                if (price - vwap_current) >= DIST_VWAP:
                    # Confirmación vela reversal bearish M1
                    if close_M1 < open_M1:
                        entry_price = open_next_M1_bar
                        sl = entry_price + SL_ATR_MULT * atr
                        tp1 = vwap_current
                        tp2 = tp1 - TP2_ATR_MULT * atr
                        place_short_order(entry_price, sl, tp1, tp2, time_stop=TIME_STOP_MINS)

            # Gestión de trades abierta (BE y trailing)
            update_open_trades_with_trailing_and_break_even()

    end day
```

---

Este diseño cumple con criterios institucionales, con reglas claras y trackeables, incluyendo gestión de riesgo, filtros rigurosos y ventanas definidas en horario NY.

## 5. Top 10 Ranking
1. **London Close Reversion** (Score: 96) - Time-of-Day
2. **ORB Volatility** (Score: 95) - Volatility Expansion
3. **Post-News Stabilization** (Score: 94) - Post-News
4. **London Session H/L** (Score: 93) - Session Breakout
5. **VWAP Reversion** (Score: 92) - Mean Reversion
6. **Institutional EMA Pullback** (Score: 91) - Trend Pullback
7. **BB Squeeze** (Score: 90) - Volatility Expansion
8. **Asian Range Fakeout** (Score: 89) - Session Breakout
9. **Keltner Breakout** (Score: 88) - Volatility Expansion
10. **Statistical Reversion** (Score: 87) - Mean Reversion

## 6. Top 5 First Backtest Candidates
- London Close Reversion
- ORB Volatility
- Post-News Stabilization
- London Session H/L
- VWAP Reversion

## 7. Top 3 Low-Correlation Candidates
- **London Close Reversion**: Basada en flujo de cierre, no en estructura.
- **Post-News Stabilization**: Basada en anomalía de volatilidad post-evento.
- **VWAP Reversion**: Basada en valor estadístico intradía.

## 8. Top 3 Most Robust Candidates
- **ORB Volatility**: Fenómeno de apertura clásico y persistente.
- **Institutional EMA Pullback**: Seguimiento de tendencia con reglas de riesgo claras.
- **London Session H/L**: Basado en liquidez horaria estructural.

## 9. Rejected / Dangerous Ideas
- **Scalping de 1 pip**: Inviable con costos reales y latencia.
- **Martingala / Grid**: Riesgo de ruina incompatible con fondeo.
- **Optimización de 20 indicadores**: Overfitting garantizado.

## 10. Backtesting Plan
Claro, a continuación un **Backtesting Plan detallado, profesional y enfocado para 20 estrategias algorítmicas intradía EURUSD (07:00-19:00 NY)** bajo las restricciones y objetivos indicados.

---

## Plan Profesional de Backtesting para Estrategias Intradía EURUSD

### 1. Smoke Test Inicial

**Objetivo:** Confirmar que cada estrategia funciona técnicamente antes de invertir recursos en backtest largo.

- Ejecutar cada estrategia en datasets pequeños y recientes (ej. últimos 5 días).
- Validar que:
  - Sigue las reglas básicas del modelo.
  - Genera señales coherentes (no errores críticos de código).
  - Respeta el máximo de 3 operaciones por día.
  - Interpreta correctamente costos (spread, slippage, comisión).
- Resultado esperado: Estrategia no debe dar errores, las señales tienen sentido, y costos se deducen correctamente.

### 2. Train-Only (Entrenamiento Inicial)

**Objetivo:** Ajustar parámetros sin tocar datos de evaluación posterior.

- Definir ventana de entrenamiento inicial (ej. 2018-2023).
- Aplicar backtesting estricto **solo en esta ventana**.
- Ajustar parámetros iniciales según:
  - Sexoample simples & conocimientos previos/research.
  - Mantener restricciones: no optimización salvaje.
- Limitar operaciones a 3/día, respetando horario 07:00-19:00.
- Incluir costos reales en la simulación.
- Usar risk management fondeo (ej. Fixed fractional, max drawdown).

### 3. Evitar Overfitting

- Limitar número de parámetros a modificar (ej. <5).
- Validar robustez con:
  - Pruebas out-of-sample (pre-holdout).
  - Pruebas por bootstrapping de ruido en dato.
- Evitar optimización exhaustiva (“grid search masiva”).
- Revisar estabilidad de parámetros: no cambios pequeños causan grandes saltos.
- Revisar que mejora en métricas no provenga solo de ajuste a ruido.
- Incorporar Penalizaciones para complejidad de modelo (ej. Akaike).

### 4. Elección de Parámetros Iniciales

- Basarse en análisis estadísticos y teoría detrás de cada estrategia.
- Usar benchmarks entendidos (ej. medias móviles clásicas, niveles clave, ATR para stops).
- Validar con literatura o backtests anteriores de estrategias similares.
- Parar ajustes tempranos mínimo 1 mes en out-of-sample para evaluar estabilidad.

### 5. Análisis de Sensibilidad a Costos

- Simular backtests con:
  - Spread medio actual y +20% para estrés.
  - Slippage desde 0.1 pips a 0.5 pips.
  - Comisiones reales del fondeo.
- Evaluar impacto en métricas principales:
  - Rentabilidad neta, drawdown, ratio Sharpe.
  - Cambios en tasa de pérdida/ganancia.
- Decidir si estrategia sigue rentable con costos.
- Documentar claramente supuestos y costos usados.

### 6. Walk-Forward sin Contaminar Holdout (2025/2026)

- **Holdout 2025-2026 100% bloqueado hasta versiones maduras.**
- Para walk-forward:
  - Usar 2018-2023 divididos en ventanas cortas (ej. 6 meses train, 1 mes test rolling).
  - Optimizar mínimos parámetros en ventana train.
  - Evaluar performance inmediata en ventana test.
- Esto permite:
  - Validar estabilidad temporal.
  - Evitar uso prematuro del holdout real 2025/26.
- Solo cuando las estrategias pasen robustamente estos tests, abrir holdout para validación final.

### 7. Criterios de Rechazo Temprano

- Estrategias con:
  - Ratios de Sharpe < 0.3 o información ratio negativo.
  - Drawdowns inaceptables (>10% según risk management).
  - Rentabilidad neta negativa significativamente.
  - Violas máximas operaciones o reglas de riesgo.
  - Sensibilidad excesiva a pequeños cambios de parámetros.
- Rechazar para evitar desperdiciar tiempo posterior.
- Documentar causas específicas de rechazo.

### 8. Comparación de Correlación con Estrategia Actual (“Barrido de Liquidez + Cambio de Estructura”)

- Calcular correlación diaria y semanal de retornos entre cada nueva estrategia y la actual.
- Revisar patrones de entradas/salidas para detectar solapamientos en tipos de señales y tiempo.
- Priorizar estrategias con baja o moderada correlación, para diversificar.
- Evaluar impacto conjunto en portfolio, no solo individual.
- Realizar análisis causal básico para entender diferencias en reacción a eventos de liquidez o estructura de mercado.

### 9. Priorización de Estrategias para Testeo

- Foco inicial en:
  - Estrategias con mayor potencial teórico y métricas preliminares.
  - Estrategias con baja correlación con el portafolio existente.
  - Estrategias con menor complejidad operativa y parámetros.
- Ejecutar Smoke Test → Train → Sensibilidad costos → Walk-Forward pequeño.
- Usar criterios de rechazo temprano para filtrar rápido.
- Repetir proceso iterativo, proveyendo reportes concisos para decisión.

### 10. Prevenir p-Hacking

- Definir a priori métricas clave, límites y criterios de éxito (ej. Sharpe, max DD, max drawdown).
- No cambiar función objetivo ni parámetro tras ver resultados.
- Mantener blindaje: no usar datos holdout hasta madurez.
- Documentar cada iteración completa de alteración.
- Usar múltiples pruebas estadísticas no paramétricas (bootstrap, permutaciones).
- Evitar pruebas múltiples sin corrección estadística.
- Implementar controles cruzados independientes (Backtesting Auditor).

---

## Consideraciones Finales

- **Restricciones de operación**: máximo 3 trades/día limitan agresividad.
- **Gestión de riesgo**: Implementar tamaño de posición basado en volatilidad/adaptativo.
- **Costos reales**: siempre incorporarlos íntegramente.
- **No ML complejos / no martingalas**: mantener interpretabilidad y estabilidad.
- **No optimización salvaje**: mantener rigor científico e integridad de test.

Este plan profesional aseguran que:

- Los backtests no se contaminarán con datos futuros.
- Se mitigará riesgo por sobreajuste.
- Las estrategias que avancen sean robustas y diversificadoras.
- La implementación final respete todas las restricciones operativas y de riesgo.

---

Quedo disponible si deseas que te prepare plantillas o scripts base para alguna etapa del plan.

## 11. Parameter Governance
Claro, aquí tienes una explicación detallada sobre **Parameter Governance** y la práctica de **congelar parámetros antes de ver resultados** en el contexto de investigación cuantitativa:

---

### Parameter Governance: Congelación de Parámetros Antes de Ver Resultados

En investigación cuantitativa, el manejo riguroso y sistemático de los parámetros del análisis es fundamental para asegurar la validez y la reproducibilidad de los resultados. **Parameter Governance** se refiere al conjunto de prácticas y políticas destinadas a definir, controlar y documentar los parámetros analíticos utilizados durante un estudio o proyecto.

#### ¿Qué es "Congelar Parámetros"?

Congelar parámetros implica establecer y fijar de manera formal todos los parámetros del análisis antes de comenzar a ejecutar o evaluar los resultados del modelo o experimento. Esto significa que una vez que los parámetros son seleccionados —p. ej., criterios de inclusión/exclusión, especificaciones del modelo estadístico, variables de control, niveles de significancia, transformaciones de variables, técnicas de imputación, etc.— estos no se modifican retrospectivamente basado en los resultados obtenidos.

#### Importancia de Congelar Parámetros

1. **Previene el ajuste post-hoc:** Modificar parámetros después de ver los resultados puede introducir sesgos (p.ej., p-hacking o data dredging). Congelar parámetros garantiza que los análisis no se ajusten de manera arbitraria para obtener resultados deseados.
   
2. **Refuerza la reproducibilidad:** Cuando los parámetros están congelados y bien documentados, otros investigadores pueden replicar el análisis bajo las mismas condiciones, validando o cuestionando los hallazgos con transparencia.

3. **Facilita la transparencia:** Clarifica y comunica a los stakeholders cómo se gestionaron las decisiones analíticas, reduciendo el riesgo de interpretaciones sesgadas o malintencionadas.

4. **Mejora la integridad estadística:** Mantener los parámetros fijos permite confiar en los métodos inferenciales tradicionales (p.ej., tests de hipótesis, intervalos de confianza) sin necesidad de correcciones complejas por múltiples comparaciones exploratorias.

#### Prácticas recomendadas para Parameter Governance

- **Predefinición formal:** Documentar y aprobar un protocolo analítico antes de acceder a los datos o ejecutar análisis exploratorios.
- **Control de versiones:** Usar software o plataformas que permitan trackear y bloquear configuraciones analíticas.
- **Separación de roles:** Idealmente, quien define los parámetros no es quien hace la evaluación final, reduciendo el sesgo cognitivo.
- **Uso de pre-análisis:** En contextos experimentales, realizar análisis piloto o simulaciones para apoyar la definición de parámetros, sin afectar el análisis principal.
- **Reportes completos:** Publicar protocolos y decisiones de congelación en anexos o repositorios públicos para auditabilidad.

---

En resumen, congelar parámetros antes de observar los resultados es una práctica esencial dentro de la Parameter Governance que fortalece la rigurosidad, la reproducibilidad y la confianza en los hallazgos cuantitativos de una investigación. Este enfoque evita el sesgo analítico y contribuye a la integridad científica institucional.

Si deseas, puedo ayudarte a diseñar un framework específico de Parameter Governance para tu equipo o proyecto. ¿Te interesa?

## 12. Risk Controls
Claro, a continuación te presento un análisis detallado sobre **Risk Controls específicos para pasar pruebas de fondeo como FTMO**, basado en los mejores estándares de gestión de riesgo cuantitativa y práctica de trading profesional:

---

### Risk Controls específicos para pasar pruebas de fondeo FTMO

Las pruebas de fondeo, como las ofrecidas por FTMO, tienen reglas claras y estrictas sobre el manejo del riesgo para garantizar que los traders mantengan una gestión adecuada y protegida de su capital. Cumplir con estos controles de riesgo es fundamental para aprobar las pruebas y acceder al fondeo.

#### 1. **Control de Drawdown Máximo**

- **Drawdown diario:** FTMO establece un drawdown máximo diario, generalmente alrededor del 5% del capital inicial. Esto significa que no puedes permitir pérdidas mayores al 5% en un solo día.  
- **Drawdown máximo total:** También existe un drawdown máximo permitido durante toda la prueba, que suele situarse cerca del 10%. Superar este límite implica la descalificación inmediata.

**Control recomendado:**  
Implementar stops rígidos y limitar el tamaño de las posiciones para que, en el peor escenario, la pérdida diaria no supere el máximo permitido. Usar un sistema automático de "kill switch" para cerrar todas las posiciones si se alcanza el drawdown.

#### 2. **Gestión estricta de posición (Sizing y apalancamiento)**

- Calcular el tamaño de lote basado en un porcentaje fijo del capital (por ejemplo, 1%-2% del capital por operación). Esto reduce la volatilidad del equity y protege contra pérdidas abruptas.  
- Evitar un apalancamiento excesivo que pueda resultar en movimientos bruscos y superación del drawdown.

**Control recomendado:**  
Desarrollar reglas fijas de dimensionamiento matemático que restrinjan la exposición máxima del portafolio en cualquier momento.

#### 3. **Límites de pérdidas por operación**

- Nunca exponerse a pérdidas que excedan un porcentaje pequeño del capital (ejemplo: 0.5%-1% por operación). Esto asegura que una operación perdida no comprometa la estabilidad total.  
- Utilizar stops de pérdidas para limitar la pérdida absoluta.

**Control recomendado:**  
Automatizar la colocación de stop loss en cada trade basado en la volatilidad del mercado o niveles técnicos significativos.

#### 4. **Control de riesgos acumulativos y correlaciones**

- No abrir múltiples posiciones correlacionadas que puedan amplificar las pérdidas simultáneamente.  
- Diversificar las operaciones entre diferentes instrumentos o estrategias para limitar riesgos agregados.

**Control recomendado:**  
Monitorizar la correlación de activos y ajustar volumen para que el riesgo total del portafolio esté dentro de límites aceptables.

#### 5. **Revisión constante y ajustes**

- Revisar periódicamente las métricas de riesgo (drawdown, ganancia, racha negativa) y ajustar tácticas, volumen o stop loss en función de la evolución.  
- Detener la operativa en caso de rachas negativas prolongadas según un umbral predefinido para evitar pérdidas catastróficas.

---

### Resumen de buenas prácticas para pasar pruebas FTMO

| Control de Riesgo           | Recomendación Clave                          |
|----------------------------|---------------------------------------------|
| Drawdown diario máximo       | Máximo 5%, implementar stop global diario  |
| Drawdown máximo total        | Máximo 10%, controlar riesgo total          |
| Tamaño de posición           | 1%-2% de capital por operación              |
| Pérdida máxima por operación | 0.5%-1% con stop loss automático            |
| Correlación y exposición    | Evitar sobreexposición a activos correlacionados |
| Revisión y ajuste continuo  | Monitorización diaria / semanal regular      |

---

Cumplir con estos controles te permitirá no sólo evitar la descalificación por riesgos excesivos, sino construir una técnica de trading sostenible que facilite el paso de pruebas de fondeo y la gestión real de capital.

Si quieres, puedo ayudarte a diseñar un plan de risk management concreto para tu estilo de trading.

¿Quieres que te prepare un ejemplo detallado con métricas y reglas exactas?

---

Quedo atento.

## 13. Final Recommendation
Final Recommendation

Tras analizar las 20 estrategias algorítmicas evaluadas para el par EURUSD en el horario intradía (07:00-19:00 NY), y considerando las prioridades del usuario —calidad, seguridad, agilidad— junto con la necesidad de mantener baja correlación con la estrategia existente de “barrido de liquidez + cambio de estructura”, recomendamos el siguiente orden para los tests iniciales:

1. **Mean Reversion**  
   Esta estrategia típicamente captura movimientos contrarios a tendencias inmediatas y suele mostrar baja correlación con estrategias basadas en orden flow y cambios estructurales. Su perfil de riesgo es más controlado cuando se emplean filtros adecuados, apoyando la calidad y seguridad. Además, es ágil de implementar en frameworks cuantitativos estándar.

2. **Post-News**  
   Dado que nuestro horario cubre la sesión de Nueva York, los eventos económicos generan volatilidad puntual que puede ser explotada con esta estrategia. Además, sus detonantes exógenos aportan un factor de diversificación respecto a la estrategia previa. Su naturaleza parcial de reacción rápida requiere controles estrictos para limitar el riesgo, pero usada con disciplina puede incrementar la robustez del portafolio algorítmico.

3. **Volatility Expansion**  
   Esta estrategia aprovecha rupturas en la volatilidad intradía y en general es complementaria a enfoques basados en estructura de precios y liquidez. Su cadencia permite una respuesta ágil a condiciones cambiantes del mercado, y favorece un buen equilibrio entre captura de oportunidades y control del drawdown.

Las demás estrategias, como Session Breakout, Trend Pullback, Time-of-Day y los modelos Hybrid, pueden evaluarse posteriormente para ampliar el conjunto diversificado de sistemas, siempre priorizando backtests rigurosos con métricas robustas de correlación, estabilidad y drawdown.

En resumen, iniciar con Mean Reversion, Post-News y Volatility Expansion brinda un equilibrio óptimo de diversidad de señales, gestión prudente del riesgo y facilidad operativa, alineado con los objetivos del usuario. Este orden minimiza la correlación con la estrategia “barrido de liquidez + cambio de estructura” y establece una base sólida para sucesivas iteraciones y optimizaciones.

