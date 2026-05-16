# Project Charter

## Purpose

Este repositorio existe para construir un laboratorio cuantitativo serio, local y auditable que permita desarrollar un bot de trading real, programable y no discrecional.

No es un generador de ideas infinitas. Es un sistema de investigacion disciplinado.

## Initial Economic Milestone

El primer hito economico del proyecto es llegar a una zona donde un bot sano, seguro y estable pueda generar al menos 20 USD.

Esto no significa:

- 20 USD por dia
- 20 USD garantizados
- 20 USD desde el dia 1

Si una linea no es robusta, segura y defendible, no sirve aunque "haga algo de plata".

## Project Philosophy

- Capital preservation before alpha.
- Seguridad frente a noticias antes de expansion de horario.
- Objetividad completa antes que narrativa.
- Pocas hipotesis, auditoria dura, implementacion profesional.
- Lo aprendido de lo que no funciona es parte central del activo del laboratorio.

## Methodological Priorities

1. Riesgo y seguridad operacional.
2. Consistencia estadistica.
3. Calidad metodologica.
4. Alpha.

El orden no es negociable.

## Risk Standard

Toda linea aceptable debe respetar como minimo:

- hard stop obligatorio desde el inicio
- no mantener pending orders dentro de kill zones de noticias
- forced flat pre-news cuando corresponda
- cooldown post-news
- fail-closed si la fuente de noticias no es operativa
- nada discrecional

## Quality Standard

Ninguna estrategia merece promocion solo por "ser la menos mala". Debe mostrar:

- PF defendible
- expectancy positiva
- drawdown tolerable
- robustez entre anios
- cantidad de trades suficiente
- sensibilidad razonable a parametros
- comportamiento consistente con y sin ruido operacional esperable

## Current Research Direction

Direccion principal vigente:

- precision PM / microestructura / alta calidad de entrada / tiempo de exposicion corto

Direccion secundaria vigente:

- ninguna activa

La investigacion AM 8:00 NY queda bloqueada hasta que News Fortress supere un estandar mas estricto de confiabilidad operacional.

## Non-Goals

Este proyecto no debe optimizar para:

- acumular nombres de strategies o waves
- salvar familias ya rechazadas con tuning defensivo
- vender edge marginal como exito
- relajar estandares para alcanzar rapido el hito de 20 USD

## Definition Of An Acceptable Bot

Un bot aceptable para real/fondeo no es "uno que gano algo". Es uno que:

- es 100% objetivo y reproducible
- sobrevive validacion seria y comparacion contra benchmarks
- no depende de una sola noticia o de un solo regimen
- no opera sin stop
- no tiene exposicion floja frente a eventos de alto impacto
- conserva capital de forma defendible antes de buscar expansion
