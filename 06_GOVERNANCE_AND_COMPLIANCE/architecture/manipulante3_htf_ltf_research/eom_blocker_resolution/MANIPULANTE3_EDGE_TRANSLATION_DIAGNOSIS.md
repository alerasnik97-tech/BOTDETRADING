# MANIPULANTE 3.0 EDGE TRANSLATION DIAGNOSIS

Objetivo: separar la logica manual de la traduccion programatica actual.

La data manual no se copia literalmente. Se usa como evidencia de que la familia causal "barrido de liquidez + cambio de estructura" puede tener ventaja. La pregunta de esta fase es otra: si esta traduccion objetiva especifica de MANIPULANTE 3.0 conserva o no edge bajo constraints Data/News reales.

## 1. Que parte de la logica manual si se modela

MANIPULANTE 3.0 modela:
- Barrido de liquidez HTF/LTF.
- Confirmacion estructural posterior al barrido.
- Direccion filtrada por configuracion.
- Entrada objetiva.
- SL/TP/BE objetivos.
- Costos, bid/ask y slippage.
- Restricciones de noticias y rollover.

## 2. Que parte no se modela completamente

La traduccion actual no captura con suficiente granularidad:
- Calidad del barrido.
- Contexto narrativo del dia.
- Sesgo HTF mas fino.
- Calidad real del CHOCH.
- FVG realmente usable.
- Tiempo post-news y digestion posterior al evento.
- Intencion de desplazamiento.
- Liquidity target opuesto.
- Evitar zonas de rango muerto.
- Estructura de sesion.
- Filtro de volatilidad intradia.

## 3. Riesgo de interpretacion

Un RED limpio de MANIPULANTE 3.0 no significa que la logica manual este muerta. Significa que esta traduccion programable, con estos parametros, reglas y medicion, no supero el gate.

La logica manual puede fallar al traducirse porque el operador humano suele discriminar calidad contextual: si el sweep fue limpio o pobre, si el CHOCH tuvo displacement real, si el FVG tenia absorcion o solo ruido, si el dia estaba en rango muerto, o si el target opuesto era razonable. Esas decisiones no estan totalmente codificadas aqui.

## 4. Que no hacer

- No optimizar esta misma traduccion perdedora para que pase.
- No buscar por TEST.
- No relajar Data/News.
- No ocultar EOM artificial.
- No convertir una exclusion de datos en edge.
- No declarar listo para demo, fondeo o live.

## 5. Si MANIPULANTE 3.0 queda RED

El cierre correcto es: rechazo de esta traduccion objetiva actual, no rechazo definitivo de la familia manual. La siguiente linea debe ser edge-first, causal y con muerte rapida, no un sweep de cosmetica parametrica sobre la misma base.

