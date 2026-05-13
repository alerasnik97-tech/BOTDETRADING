# DESCOMPOSICIÓN CUANTITATIVA DE LA LÓGICA MANUAL DE TRADING
**Estrategia Madre:** “Barrido de Liquidez + Cambio de Estructura (CHoCH)”  
**Objetivo Forense:** Aislamiento de variables de microestructura para traducción 100% objetiva sin sesgo de look-ahead.

---

## 1. Nivel de Liquidez Barrido (Swept Liquidity Level)
*   **¿Ya fue modelada en M2/M3/M4?:** Sí, en M2 y M3 se implementaron barridos sobre máximos y mínimos de Asia, Londres, del día previo (PDH/PDL) y de la semana previa.
*   **¿Fue modelada bien o superficialmente?:** Superficialmente. Se tomaron niveles estáticos de rangos horarios fijos sin ponderar la acumulación real de volumen en dichos extremos.
*   **¿Qué dato objetivo haría falta?:** Mapeo de densidad del libro de órdenes o acumulación de ticks en el nivel (ej. perfiles de volumen de sesión o *swing points* fractales confirmados por al menos $N$ barras a izquierda y derecha).
*   **¿Qué riesgo de subjetividad tiene?:** Alto en manual, ya que el humano descarta visualmente niveles "obvios pero débiles" en favor de niveles "institucionales".
*   **¿Se puede programar sin mirar futuro?:** Sí, utilizando colas circulares causales para rastrear fractales estrictamente pasados a $T_0$.

---

## 2. Calidad del Nivel (Level Quality)
*   **¿Ya fue modelada en M2/M3/M4?:** No. Todos los extremos de sesión fueron tratados con idéntico peso probabilístico.
*   **¿Fue Calidad modelada bien o superficialmente?:** Nula.
*   **¿Qué dato objetivo haría falta?:** Mapeo del tiempo en que el nivel permaneció intacto (edad del nivel en barras) y la distancia en pips del rechazo previo más cercano para cuantificar la liquidez atrapada (stop orders).
*   **¿Qué riesgo de subjetividad tiene?:** Muy alto. El ojo humano califica un nivel como "limpio" o "sucio" basándose en la compresión previa.
*   **¿Se puede programar sin mirar futuro?:** Sí, midiendo la varianza de los precios máximos/mínimos locales en una ventana retrospectiva de $M$ períodos.

---

## 3. Dirección / Contexto HTF (HTF Bias / Context)
*   **¿Ya fue modelada en M2/M3/M4?:** Sí, en M3 se intentó usar la pendiente de medias móviles H1/H4 y la ubicación en zonas de Premium/Discount relativas al rango semanal/diario.
*   **¿Fue modelada bien o superficialmente?:** Superficialmente. El humano no solo mira un indicador, sino la intencionalidad del flujo de órdenes (ej. si el HTF acaba de hacer un barrido opuesto).
*   **¿Qué dato objetivo haría falta?:** Vector de estado direccional multi-temporal (ej. último quiebre estructural H1/H4 validado con cierre de cuerpo de vela).
*   **¿Qué riesgo de subjetividad tiene?:** Moderado. Definir el "rango de trabajo actual" en manual suele ajustarse a conveniencia del setup.
*   **¿Se puede programar sin mirar futuro?:** Sí, consumiendo exclusivamente los metadatos de las barras completas de H1/H4 en el instante $t - 1\text{ ms}$.

---

## 4. Condición de Sesión (Session Condition)
*   **¿Ya fue modelada en M2/M3/M4?:** Sí. Se establecieron máscaras horarias intradiarias.
*   **¿Fue modelada bien o superficialmente?:** En evolución. Los primeros bots operaron todo el día; en auditorías recientes se demostró que el humano restringe su operativa al **NY Open Killzone (08:00 a 11:00 NY)**.
*   **¿Qué dato objetivo haría falta?:** Ninguno extra, el reloj canónico en zona horaria `America/New_York` es suficiente.
*   **¿Qué riesgo de subjetividad tiene?:** Bajo. Es una regla dura.
*   **¿Se puede programar sin mirar futuro?:** Sí, es determinística.

---

## 5. Relación con Noticias (Relation to News)
*   **¿Ya fue modelada en M2/M3/M4?:** Sí, implementando bloqueos estáticos de tiempo (ej. $\pm 15$ minutos) en torno a eventos de alto impacto.
*   **¿Fue modelada bien o superficialmente?:** Superficialmente. El bot simplemente se apaga. El humano utiliza la noticia como el *catalizador* del barrido de liquidez para entrar en la reversión posterior al shock.
*   **¿Qué dato objetivo haría falta?:** Vector de eventos premium AM Fortress v3 sincronizado, midiendo la aceleración del tick en los primeros 180 segundos post-publicación.
*   **¿Qué riesgo de subjetividad tiene?:** Alto. Decidir cuándo el mercado "ya asimiló" el dato macroeconómico es altamente discrecional.
*   **¿Se puede programar sin mirar futuro?:** Sí, condicionando la habilitación de búsqueda de setups a que ocurra una noticia Tier-1 en los últimos $K$ minutos.

---

## 6. Tipo de Barrido (Sweep Type)
*   **¿Ya fue modelada en M2/M3/M4?:** Superficialmente en M3. Se definió como una penetración del precio máximo/mínimo.
*   **¿Fue modelada bien o superficialmente?:** Deficiente. El bot tomaba rupturas de 0.1 pips como "barridos válidos", introduciendo un ruido letal.
*   **¿Qué dato objetivo haría falta?:** Profundidad del barrido en pips (ej. exigiendo penetración mínima de $1.5$ pips y máxima de $12$ pips) y velocidad de absorción.
*   **¿Qué riesgo de subjetividad tiene?:** Moderado. El humano distingue visualmente un *raid* de liquidez rápido de una ruptura genuina con intención de continuación.
*   **¿Se puede programar sin mirar futuro?:** Sí, midiendo la excursión máxima del precio más allá del nivel durante la formación de la barra LTF.

---

## 7. Reclaim (Recuperación del Nivel)
*   **¿Ya fue modelada en M2/M3/M4?:** Parcialmente. Se exigía que la barra de barrido o la siguiente cerrara nuevamente por dentro del rango (cierre de cuerpo por debajo del PDH barrido).
*   **¿Fue modelada bien o superficialmente?:** Superficialmente. A menudo la recuperación ocurre varias barras más tarde en M3.
*   **¿Qué dato objetivo haría falta?:** Tiempo máximo permitido para el reclaim (ej. $\le 3$ barras M3) y confirmación de absorción de volumen en el extremo.
*   **¿Qué riesgo de subjetividad tiene?:** Bajo si se definen los límites de barras estrictamente.
*   **¿Se puede programar sin mirar futuro?:** Sí, evaluando la posición de los precios de cierre relativos al nivel estático barrido.

---

## 8. Desplazamiento Posterior (Subsequent Displacement)
*   **¿Ya fue modelada en M2/M3/M4?:** No de forma explícita como variable continua de momento.
*   **¿Fue modelada bien o superficialmente?:** Nula.
*   **¿Qué dato objetivo haría falta?:** Rango verdadero medio (ATR) de las barras de reversión superando en al menos $1.5\times$ el ATR de las barras previas, o la creación de un impulso ininterrumpido de $X$ pips en $Y$ minutos.
*   **¿Qué riesgo de subjetividad tiene?:** Alto. El humano califica el movimiento como "violento" o "lento" de forma intuitiva.
*   **¿Se puede programar sin mirar futuro?:** Sí, calculando la tasa de cambio (ROC) o la longitud de las velas LTF consecutivas.

---

## 9. Cambio de Estructura (Structural Shift / CHoCH)
*   **¿Ya fue modelada en M2/M3/M4?:** Sí, programando la ruptura del último *swing high* / *swing low* opuesto previo al extremo del barrido.
*   **¿Fue modelada bien o superficialmente?:** Superficialmente. En programación algorítmica previa, la identificación del *swing point* opuesto a menudo sufría de rezago o seleccionaba micro-picos internos irrelevantes.
*   **¿Qué dato objetivo haría falta?:** Definición estricta de *swing point* válido en 3M (ej. vela cuyo máximo es superior a las 2 velas anteriores y 2 posteriores) y exigencia de quiebre con *cierre de vela*, no solo con mecha.
*   **¿Qué riesgo de subjetividad tiene?:** Muy alto. Es la principal causa de divergencia entre el *backtest* manual y el bot.
*   **¿Se puede programar sin mirar futuro?:** Sí, manteniendo un árbol de estados de los últimos picos y valles confirmados causales.

---

## 10. FVG Real (Fair Value Gap Presence)
*   **¿Ya fue modelada en M2/M3/M4?:** Sí, en algunas variantes de M3 se programó como la no superposición entre el máximo de la vela 1 y el mínimo de la vela 3.
*   **¿Fue modelada bien o superficialmente?:** Superficialmente. No se midió el tamaño del FVG ni su ubicación relativa al impulso de ruptura.
*   **¿Qué dato objetivo haría falta?:** Tamaño mínimo en pips del vacío de liquidez (ej. $\ge 2.0$ pips en EURUSD) para filtrar ineficiencias milimétricas sin valor institucional.
*   **¿Qué riesgo de subjetividad tiene?:** Bajo. La fórmula matemática es exacta.
*   **¿Se puede programar sin mirar futuro?:** Sí.

---

## 11. Entrada (Entry Trigger)
*   **¿Ya fue modelada en M2/M3/M4?:** Sí. Órdenes a mercado al cierre de la vela de confirmación o limit orders en el borde del FVG.
*   **¿Fue modelada bien o superficialmente?:** Bien en cuanto a ejecución de la orden, pero carente de asimilación de fricción realista (slippage) en los reportes iniciales.
*   **¿Qué dato objetivo haría falta?:** Modelado estricto de deslizamiento asimétrico (0.2 pips obligatorios) simulando el cruce real del spread en el libro.
*   **¿Qué riesgo de subjetividad tiene?:** Ninguno en backtest algorítmico.
*   **¿Se puede programar sin mirar futuro?:** Sí, gatillando en la apertura de la barra $k + 1$.

---

## 12. Stop Loss (SL Placement)
*   **¿Ya fue modelada en M2/M3/M4?:** Sí. Colocación en el extremo absoluto del barrido (extremo estructural).
*   **¿Fue modelada bien o superficialmente?:** Bien. Es una regla dura de protección de capital.
*   **¿Qué dato objetivo haría falta?:** Inyección de un *buffer* o respiro mínimo (ej. $+0.5$ pips más allá del extremo) para absorber fluctuaciones aleatorias del spread interbancario.
*   **¿Qué riesgo de subjetividad tiene?:** Bajo.
*   **¿Se puede programar sin mirar futuro?:** Sí.

---

## 13. Take Profit (TP Target)
*   **¿Ya fue modelada en M2/M3/M4?:** Sí. Modelos de múltiplos fijos de riesgo ($1.4R$, $2.0R$) o apuntando a niveles estáticos opuestos.
*   **¿Fue modelada bien o superficialmente?:** Bien en baselines recientes (ej. TP fijo de $1.4R$ en Manipulante consolidado).
*   **¿Qué dato objetivo haría falta?:** Ninguno para múltiplos fijos. Para objetivos dinámicos, mapeo en tiempo real de la liquidez interna opuesta.
*   **¿Qué riesgo de subjetividad tiene?:** Nulo en múltiplos R fijos.
*   **¿Se puede programar sin mirar futuro?:** Sí.

---

## 14. Break-Even (BE Trigger)
*   **¿Ya fue modelada en M2/M3/M4?:** Sí. Activación al alcanzar umbrales fijos ($0.4R$, $0.5R$, o $1.25R$).
*   **¿Fue modelada bien o superficialmente?:** Bien implementada en el motor, pero ha demostrado ser un arma de doble filo al asfixiar prematuramente trades ganadores en entornos de alta compresión.
*   **¿Qué dato objetivo haría falta?:** Condicionamiento de BE a la creación de una nueva estructura a favor (ej. un nuevo *swing* superado) en lugar de un mero recorrido de pips.
*   **¿Qué riesgo de subjetividad tiene?:** Bajo.
*   **¿Se puede programar sin mirar futuro?:** Sí.

---

## 15. Invalidación (Setup Invalidation)
*   **¿Ya fue modelada en M2/M3/M4?:** Sí, caducidad por tiempo o si el precio cruza el nivel de SL antes de dar entrada en el FVG.
*   **¿Fue modelada bien o superficialmente?:** Superficialmente.
*   **¿Qué dato objetivo haría falta?:** Cancelación inmediata de la orden límite si el mercado alcanza el objetivo teórico de TP antes de retroceder a llenar la entrada (*mitigación fallida*).
*   **¿Qué riesgo de subjetividad tiene?:** Bajo si se codifica el árbol de estados.
*   **¿Se puede programar sin mirar futuro?:** Sí.

---

## 16. Cuándo NO Operar (When NOT to Trade)
*   **¿Ya fue modelada en M2/M3/M4?:** Sí, implementando cierres duros los viernes a las 16:55 NY y exclusión en ciertas franjas de noticias.
*   **¿Fue modelada bien o superficialmente?:** Superficialmente. Faltó excluir explícitamente el rollover diario completo (16:55 a 17:15 NY) en el flujo de señales nativo de los primeros pilotos.
*   **¿Qué dato objetivo haría falta?:** Filtro de compresión de volatilidad extrema (ej. rango del día anterior $< 40$ pips en EURUSD indica mercado inoperable por ausencia de desplazamiento).
*   **¿Qué riesgo de subjetividad tiene?:** Moderado en discrecional ("el mercado se siente lento").
*   **¿Se puede programar sin mirar futuro?:** Sí, estableciendo compuertas lógicas pre-sesión.
