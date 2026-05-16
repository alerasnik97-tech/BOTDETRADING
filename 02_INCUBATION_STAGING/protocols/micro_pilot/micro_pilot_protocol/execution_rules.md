# Reglas de Ejecución - Micro Piloto

> [!CAUTION]
> **ESTADO: NOT_ACTIVE_UNTIL_MICRO_PILOT_ALLOWED**

Este documento establece las fronteras de ejecución para evitar el "creep" de riesgo y la degradación del experimento.

## Definición de la Fase
El Micro Piloto es una **validación estadística**, no un despliegue operativo pleno. El objetivo principal es la **fidelidad**, no la rentabilidad.

## Reglas de Oro de Ejecución
1. **No Escalamiento:** El tamaño de posición es el mínimo técnico requerido para que la ejecución sea realista. No se escala por ningún motivo.
2. **Estrategia Congelada:** No se permite modificar un solo parámetro de la estrategia (SL, TP, filtros, horarios) durante el piloto.
3. **Línea Única:** Solo se opera la línea EURUSD CORE que ha superado el gate. Prohibido probar "variantes" en real.
4. **Dependencia del Gate:** Si el archivo `micro_pilot_gate` no está en `MICRO_PILOT_ALLOWED`, la ejecución real está físicamente prohibida.
5. **Prioridad Shadow:** La ejecución real debe ser un reflejo exacto de la señal Shadow. Si Shadow no opera, Real no opera.

## Cambios Estructurales
Cualquier cambio estructural en el core productivo (código del bot, lógica de entrada) invalida el Micro Piloto actual y obliga al retorno inmediato a la fase de **SHADOW_ONLY**.

## Meta del Piloto
Comprar evidencia real mínima (N=20 a N=30 trades en real) antes de considerar cualquier transición a la siguiente fase institucional.
