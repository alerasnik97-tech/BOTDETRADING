# TOP 5 HIPÓTESIS ESTRATÉGICAS FUTURAS (INSTITUTIONAL SHORTLIST)
**Criterio de Selección:** Proximidad a la discrecionalidad manual (3M), máxima objetividad de variables, bajo costo de I/O y supresión de minería de datos.

---

## 1. Sweep Quality + Displacement Gate (`HYP_001`)
*   **Lógica Causal:** Un barrido puramente técnico de un máximo o mínimo anterior no genera ventaja por sí solo (suele ser ruido intradiario). El verdadero *edge* institucional nace cuando la penetración del nivel tiene límites físicos acotados (trampa controlada) y es seguida inmediatamente por una expansión violenta del rango de las velas de reversión, confirmando la capitulación minorista y la entrada de volumen institucional.
*   **Por qué es Distinta de M3:** M3 evaluaba simples cruces booleanos del precio sobre el nivel estático y tomaba quiebres débiles en temporalidades lentas. Esta hipótesis inyecta filtros de momento continuo (ATR ratio) y opera nativamente en el disparador manual de **3M**.
*   **Variables Mínimas:** Penetración en pips ($1.5 \le \text{pips} \le 12.0$), ratio de ATR de las 3 velas posteriores vs. 3 previas ($> 1.5$), y quiebre estructural de 3M validado con cierre de vela.
*   **Cómo Medir Rápido:** Pre-filtrar la serie histórica identificando únicamente las barras donde el precio cruza los niveles de PDH/PDL o Asia, y computar la aserción de desplazamiento solo en esos instantes sin simular el resto del mes.
*   **Máximo Configs Recomendado:** **12 configuraciones** (3 umbrales de penetración $\times$ 2 ratios de ATR $\times$ 2 objetivos de Take Profit fijos).
*   **Criterio de Muerte Rápida:** $\text{PF}_{\text{val\_net}} < 1.05$ bajo la aplicación de $0.2$ pips de slippage asimétrico en las primeras 40 muestras.
*   **Criterio para Escalar:** $\text{PF}_{\text{val\_net}} \ge 1.15$ sostenido con un drawdown máximo inferior a $-4.0R$.
*   **Riesgo Principal:** Sobresensibilidad del cálculo del ATR intradiario ante saltos aislados de liquidez en aperturas de hora.

---

## 2. Post-News Liquidity Reversal (`HYP_002`)
*   **Lógica Causal:** Las publicaciones macroeconómicas de alto impacto (Tier-1) provocan un vacío de liquidez inmediato en el libro de órdenes. Los algoritmos de alta frecuencia barren incondicionalmente los *stop loss* visibles en los extremos intradiarios para llenar grandes bloques de órdenes antes de revertir el precio hacia la tendencia estructural media.
*   **Por qué es Distinta de M3:** M3 y sus predecesores implementaban un simple apagado del bot durante las noticias para evadir volatilidad. Esta hipótesis utiliza el evento del calendario curado como el *gatillo habilitador* exclusivo para buscar patrones de absorción.
*   **Variables Mínimas:** Marca de tiempo de evento Tier-1 en `news_eurusd_am_fortress_v3.csv`, ventana de retención post-noticia ($60 \le \text{segundos} \le 180$), y formación de un FVG de 3M opuesto al impulso inicial.
*   **Cómo Medir Rápido:** Iterar exclusivamente los bloques de Parquet correspondientes a las ventanas horarias exactas de las noticias Tier-1 registradas en el caché, ignorando el 98% de la serie inactiva.
*   **Máximo Configs Recomendado:** **8 configuraciones** (2 ventanas de espera post-shock $\times$ 2 umbrales de tamaño de FVG $\times$ 2 múltiplos de TP).
*   **Criterio de Muerte Rápida:** Destrucción de la equidad neta o cuenta FTMO teóricamente quemada en la partición de validación.
*   **Criterio para Escalar:** Retorno neto superior a $+15R$ en validación manteniendo un Win Rate $> 40\%$ bajo latencia estresada.
*   **Riesgo Principal:** Llenados teóricos irreales si el backtest asume ejecuciones instantáneas dentro del primer minuto de la noticia sin aplicar buffers de supresión.

---

## 3. London Sweep into NY Reversal (`HYP_003`)
*   **Lógica Causal:** La sesión de Londres establece la dirección inicial y a menudo manipula los extremos asiáticos para inducir un sesgo tendencioso. La apertura de Nueva York (08:00 NY) inyecta el mayor volumen interbancario del día, aprovechando la liquidez acumulada en el extremo de Londres para orquestar una reversión sostenida durante el *Killzone*.
*   **Por qué es Distinta de M3:** M3 no acoplaba la causalidad entre las sesiones de forma secuencial. Esta hipótesis condiciona la validez del setup a que Nueva York barra el extremo absoluto fijado por Londres e inmediatamente quiebre la estructura interna en 3M.
*   **Variables Mínimas:** Array estático de máximos/mínimos de Londres (03:00 a 07:00 NY), máscara temporal estricta de entrada (**08:00 a 11:00 NY**), y gatillo de CHoCH 3M.
*   **Cómo Medir Rápido:** Computar los extremos de Londres como metadatos pre-calculados y simular la lógica de órdenes exclusivamente dentro de la ventana de 3 horas de Nueva York.
*   **Máximo Configs Recomendado:** **6 configuraciones** (3 variantes de colocación de stop loss relativas al spread $\times$ 2 modelos de Take Profit).
*   **Criterio de Muerte Rápida:** Curva de capital plana o $\text{PF}_{\text{net}} < 1.05$ con $0.2$ pips de fricción tras 50 operaciones conjuntas.
*   **Criterio para Escalar:** Consistencia en la generación de señales ($N \ge 60$ por año) y factor de beneficio neto robusto superior a $1.20$.
*   **Riesgo Principal:** Falsa ruptura de continuidad si el mercado entra en un régimen tendencial fuerte donde Nueva York simplemente expande la dirección de Londres sin revertir.

---

## 4. Previous Day Liquidity Reclaim (`HYP_004`)
*   **Lógica Causal:** Los niveles de máximo y mínimo del día anterior (PDH/PDL) son las fronteras psicológicas más observadas por operadores minoristas. Un quiebre de estos niveles induce masivamente órdenes de compra/venta en ruptura. Si el precio fracasa en sostenerse por fuera del nivel y cierra físicamente dentro del rango, confirma una trampa institucional (*reclaim*) e inicia una búsqueda hacia la liquidez interna opuesta.
*   **Por qué es Distinta de M3:** M3 evaluaba el quiebre de estructura sin exigir la validación formal del *reclaim* (cierre de vela confirmando absorción) como filtro primario de entrada.
*   **Variables Mínimas:** Niveles estáticos de PDH/PDL del vector diario, condición booleana de cierre de vela 3M/5M regresando al rango, y presencia de un vacío FVG $\ge 2.0$ pips.
*   **Cómo Medir Rápido:** Extraer los niveles diarios del archivo de metadatos y evaluar rupturas intradiarias con un script vectorial simple antes de simular el detalle de ticks.
*   **Máximo Configs Recomendado:** **9 configuraciones** (3 umbrales de tamaño de FVG $\times$ 3 distancias teóricas te Take Profit).
*   **Criterio de Muerte Rápida:** El rendimiento global depende críticamente de 1 o 2 trades atípicos de gran recorrido, o el $\text{PF}_{\text{net}}$ cae por debajo de $1.0$ al deducir costos.
*   **Criterio para Escalar:** Distribución uniforme de retornos a lo largo de los 136 meses auditados y coincidencia exacta en verificación independiente.
*   **Riesgo Principal:** Entradas múltiples y falsos *reclaims* consecutivos en días de extrema consolidación lateral en torno al nivel diario.

---

## 5. Failed Breakout with Volatility Compression (`HYP_007`)
*   **Lógica Causal:** Las estrategias de reversión a la media maximizan su esperanza matemática cuando el mercado opera en regímenes de baja volatilidad estructural. Implementar un filtro de régimen que suprima señales en días de expansión tendencial fuerte garantiza que los barridos operados correspondan a rangos laterales propensos a rebotar de extremo a extremo.
*   **Por qué es Distinta de M3:** M3 carecía por completo de filtros de régimen de volatilidad macro. Operaba de manera idéntica en días de NFP expansivos que en lunes feriados bancarios.
*   **Variables Mínimas:** Rango verdadero diario previo ($\text{ATR}_{\text{daily}} < 50$ pips en EURUSD), compresión del ATR intradiario en la sesión asiática, y confirmación de barrido LTF.
*   **Cómo Medir Rápido:** Pre-calcular el vector de rangos diarios e ignorar por completo la simulación intradiaria en jornadas donde el rango previo superó el umbral crítico.
*   **Máximo Configs Recomendado:** **6 configuraciones** (2 umbrales de compresión de rango diario $\times$ 3 submúltiplos de Take Profit).
*   **Criterio de Muerte Rápida:** Inanición extrema de la muestra ($N_{\text{val}} < 30$ operaciones en 3 años) o nula capacidad de generar retornos superiores a la comisión de fondeo.
*   **Criterio para Escalar:** Superación incondicional de $\text{PF}_{\text{val\_net}} \ge 1.15$ con una curva de capital monótonamente ascendente en períodos de baja volatilidad estival.
*   **Riesgo Principal:** Sobreoptimización al seleccionar los umbrales históricos exactos de corte del ATR diario para forzar el filtrado perfecto de días perdedores.
