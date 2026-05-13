# NEXT EDGE HYPOTHESIS RECOMMENDATION

Maximo 3 hipotesis. No son aprobacion para sweep automatico; son candidatas para una nueva traduccion objetiva si MANIPULANTE 3.0 queda RED limpio.

## Hipotesis 1: Sweep quality + displacement gate

Logica causal:
El edge manual podria depender de que el barrido limpie un nivel relevante y luego aparezca desplazamiento real, no solo un cambio estructural nominal.

Datos necesarios:
- High/low intradia y niveles de liquidez previos.
- Medida de extension del sweep.
- Medida de rechazo posterior.
- Velas de displacement y rango relativo.
- FVG util posterior al desplazamiento.

Diferencia vs Manipulante 3.0:
No basta con barrido + confirmacion; exige calidad objetiva del sweep y displacement.

Riesgo de data mining:
Medio. El riesgo sube si se buscan umbrales finos. Mitigacion: pocos umbrales discretos precomprometidos y muerte rapida.

Como medir rapido:
Construir etiqueta binaria de sweep quality y comparar PF/N/WR contra la base sin tocar TEST para seleccion.

Criterio de muerte rapida:
Muere si VAL no supera PF_net 1.15 con N >= 40 y TEST no queda al menos PF_net 1.0 con N >= 50 bajo slippage 0.2.

## Hipotesis 2: Post-news liquidity shift only

Logica causal:
La ventaja podria aparecer solo cuando una noticia Tier-1 crea desplazamiento y luego el mercado barre liquidez contraria antes de continuar.

Datos necesarios:
- Calendario AM Fortress v3.
- Tiempo exacto desde noticia.
- Direccion del primer impulso post-news.
- Sweep posterior del lado opuesto.
- Confirmacion estructural posterior al sweep.

Diferencia vs Manipulante 3.0:
El evento de noticia deja de ser solo bloqueo; pasa a ser contexto causal controlado, con ventanas post-news predefinidas.

Riesgo de data mining:
Alto si se prueban muchas ventanas. Mitigacion: maximo 2 ventanas post-news precomprometidas y sin TEST selection.

Como medir rapido:
Medir solo dias con Tier-1 AM, comparar contra dias sin evento y exigir N minimo antes de lectura.

Criterio de muerte rapida:
Muere si N_val < 25 o si PF_val_net <= 1.10 bajo slippage 0.2.

## Hipotesis 3: Opposite liquidity target + dead-range filter

Logica causal:
El trade manual podria funcionar cuando hay un target opuesto claro y fallar en rangos muertos sin objetivo de liquidez limpio.

Datos necesarios:
- Liquidity pools opuestos: highs/lows de sesion, PDH/PDL, Asia high/low, NY open range.
- Distancia a target.
- Rango intradia y compresion previa.
- Time-of-day y estructura de sesion.

Diferencia vs Manipulante 3.0:
Agrega objetivo causal de salida/continuacion y evita zonas sin recorrido, sin cambiar retrospectivamente trades por resultado.

Riesgo de data mining:
Medio-alto. Mitigacion: pocos targets jerarquicos predefinidos y una sola definicion de rango muerto.

Como medir rapido:
Etiquetar cada setup como target claro / sin target antes de ver outcome, luego medir por grupos.

Criterio de muerte rapida:
Muere si el grupo target claro no mejora PF_val_net y expectancy frente a la base con N >= 40.

