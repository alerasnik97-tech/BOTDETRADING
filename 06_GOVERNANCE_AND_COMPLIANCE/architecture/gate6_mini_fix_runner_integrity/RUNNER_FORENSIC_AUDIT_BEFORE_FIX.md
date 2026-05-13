# AUDITORÍA FORENSE DEL RUNNER (PRE-FIX)
**Archivo Auditado:** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/gate6_mini_runner.py`  
**Motor Auditado:** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v7_engine/engine.py`  
**Fecha de Corte:** 2026-05-13  

---

## 1. Mapeo Exhaustivo de Comportamientos (22 Puntos Forenses)

### 1. Generación de Señales
Las señales se extraen holísticamente aplicando el `First3MChochDetector` sobre el dataframe M3, filtrando previamente barridos fractales válidos en H1 (`H1FractalSweepDetector`) (*gate6_mini_runner.py*, líneas 131-140).

### 2. Definición de Variante V2_A
`V2_A_MARKET_CHOCH` consume el dataframe crudo de señales base intacto (*gate6_mini_runner.py*, línea 156), mapeando `entry_mode = "market"`.

### 3. Definición de Variante V2_B
`V2_B_STOP_CONFIRMATION` se declara con `entry_mode = "stop"` en el bucle principal mensual (*gate6_mini_runner.py*, línea 227).

### 4. Realidad Operativa de V2_B vs. V2_A
**CLONACIÓN CONFIRMADA:** Aunque la variable `entry_mode` adquiere el valor `"stop"`, esta jamás se transmite a `eng.execute_signal` (*gate6_mini_runner.py*, línea 243). El motor llama incondicionalmente a `next_bar_execute(side, signal_bar_close, ticks_after)` (*engine.py*, línea 131), la cual ejecuta a mercado al primer tick disponible en `execution.py` (líneas 18-44). Por lo tanto, **V2_B corría idéntica a V2_A**.

### 5. Definición y Disponibilidad de V2_C
`V2_C_LIMIT_FLOW` se omite de la lista de iteración activa `variants` (*gate6_mini_runner.py*, línea 172) y se documenta como **UNAVAILABLE** debido a la inexistencia física de series de desequilibrios de Order Flow volumétrico en el *Market Data Vault*, impidiendo su simulación sin fabricar datos falsos.

### 6. Definición de V2_D
`V2_D_LIMIT_REGIME_FILTER` aplica una máscara booleana temporal exigiendo que el rango ATR de 20 periodos en H1 supere la mitad de la mediana histórica global al momento del barrido fractal (*gate6_mini_runner.py*, líneas 149-169).

### 7. Construcción de Barras H1 y M3
Se sintetizan al vuelo cargando el archivo Parquet mensual de ticks en streaming y ejecutando `build_bars(ticks, timeframe)` (*gate6_mini_runner.py*, líneas 108-109).

### 8. Prevención de Lookahead Bias
Se impone causalidad estricta al acotar la búsqueda del fill al subconjunto `ticks_after = ticks.loc[choch_utc : slice_end]` y pasar el corte temporal exacto de la vela al `TestLeakageGuard` (*engine.py*, líneas 100-106).

### 9. Criterio de Selección del Fill
Se toma la primera fila indexada estrictamente posterior a la marca temporal de cierre de la señal ($T+1$) (*execution.py*, línea 26).

### 10. Cálculo de Distancia de Stop Loss (SL)
Se extrae de la propiedad estática pre-calculada en la detección del CHOCH (`sl_price`) (*gate6_mini_runner.py*, línea 232).

### 11. Cálculo de Take Profit (TP)
Se define aritméticamente proyectando la distancia de riesgo por el múltiplo `TP_R` ($2.1\text{ R}$) desde el precio exacto de fill con mechas/fricción incorporada (*gate6_mini_runner.py*, línea 249).

### 12. Activación de Break-Even (BE)
Se invoca nativamente pasando `be_trigger_r=1.4` a `close_position_with_costs` (*gate6_mini_runner.py*, línea 255), el cual mueve dinámicamente la barrera de corte al precio de entrada original + un offset intradiario mínimo de 0.5 pips al momento del cruce (*engine.py*, líneas 177-190).

### 13. Definición de Salida Forzada (Forced Exit)
El motor delega en el `ScheduleGuard` la intercepción de la hora límite (`16:00` NY), cortando en el bid/ask del tick concurrente (*engine.py*, líneas 172-175).

### 14. Ubicación de Truncamiento por `.head(3000)`
Se localiza en la construcción de la rebanada de simulación de cierre intradiario:  
`ticks_during = ticks.loc[fill.fill_time : pos_end].head(3000)` (*gate6_mini_runner.py*, línea 252).

### 15. Impacto Forense del Truncamiento por `.head(3000)`
**VULNERABILIDAD CRÍTICA:** Al limitar el tamaño a exactamente 3,000 ticks, en regímenes de ultra-alta frecuencia un horizonte de 8 horas intradiarias agota el cupo en los primeros minutos post-fill. En consecuencia, el bucle `for row in ticks_eval.itertuples()` (*engine.py*, línea 168) termina sin alcanzar SL/TP/Forced Exit y gatilla de forma silenciosa la regla final de salida EOM (*engine.py*, línea 211).

### 16. Identificación en el Ledger de Salidas EOM
Aparece tipificada unívocamente con el literal `"EOM"` en la propiedad `exit_reason` del objeto `TradeRecord` devuelto (*engine.py*, línea 211).

### 17. Distinción EOM Real vs. Artificial
El runner previo carecía de mecanismos para desglosar si el final de la secuencia correspondía al límite real del mes calendario en el archivo de datos o al agotamiento prematuro de la cota de `.head(3000)`.

### 18. Aplicación de Fricción Friccional (Slippage)
Se inyecta pasando la magnitud escalar a la configuración del `CostModel` (*gate6_mini_runner.py*, línea 189), el cual deduce la merma de pips de la ganancia bruta en la función `apply_costs_to_trade` (*engine.py*, línea 217).

### 19. Aplicación de Comisiones
El `CostModel` deduce el cargo fijo de comisiones institucionales (USD 5.0 Round-Turn) por contrato simulado, expresando su absorción equivalente sobre el PnL y la razón de rentabilidad neta (*engine.py*, líneas 222-226).

### 20. Mecánica de Actualización de Cuenta FTMO
Se ejecuta de manera causal invocando `self.ftmo.update_state(ts_utc, closed_pnl=net_pnl_usd)` al término físico de la operación (*engine.py*, líneas 266-270).

### 21. Atribución de Causas en Caída de $N$ Bajo Slippage
La caída drástica observada carece de segmentación; el runner anterior omitía registrar si la merma de posiciones se debía a:
1. Absorción del buffer de SL por el slippage de entrada/salida.
2. Quiebre irreversible previo del límite de pérdidas de la cuenta FTMO (`ftmo.blown`), inhabilitando futuros fills.
3. Insuficiencia de ticks válidos post-corte.

### 22. Tolerancia a Ausencia del Calendario de Noticias
Si el archivo `news_eurusd_am_fortress_v3.csv` faltaba de la ruta física, el bloque `try...except` capturaba el error silenciosamente, inicializaba `news_events = []` y permitía el flujo inalterado (*gate6_mini_runner.py*, líneas 54-71), infringiendo el estándar de seguridad *fail-close*.

---

## 2. Conclusiones y Severidad Global
La combinación de **clonación de lógica stop a mercado** en V2_B y **cierres artificiales silenciosos por EOM** en V2_D invalida parcialmente las distribuciones numéricas previas, justificando plenamente el rechazo de ChatGPT a emitir un veredicto definitivo sin antes reconstruir y reevaluar la sonda estructural.
