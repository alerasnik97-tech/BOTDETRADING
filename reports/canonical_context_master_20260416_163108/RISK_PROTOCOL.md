# Risk Protocol

## Prime Directive

El objetivo principal del sistema no es operar mas. Es evitar exposicion no auditada, especialmente alrededor de noticias de alto impacto.

Ninguna estrategia, runner o experimento puede violar esta prioridad.

## Mandatory Risk Controls

### Hard Stop From Origin

- Toda senal debe declarar un hard stop valido desde el origen.
- El motor rechaza senales sin stop valido.
- Solo se admiten stops estructuralmente definidos y verificables.

### Position And Exposure Limits

- Maximo una posicion abierta a la vez por defecto.
- Maximo de trades por dia definido por `EngineConfig`; una linea puede endurecerlo aun mas.
- Todo experimento debe tener horario explicito de entrada y forced flat por horario si aplica.

### Pending Orders And Signals

- No se permiten pending orders ni senales pendientes vivas dentro de ventanas bloqueadas por noticias.
- Si el protocolo de noticias exige cancelacion pre-news, se cancela antes del evento.

## News Fortress Operational Rules

### Fail-Closed Requirement

- Si `NewsConfig.enabled = True`, la fuente debe ser operativa.
- Si la fuente no es operativa, la corrida debe fallar; no puede continuar sin noticias como si nada.

### Base Fortress Defaults

El motor actual usa como defaults de `NewsConfig`:

- `pre_minutes = 30`
- `post_minutes = 60`
- `forced_exit_pre_news = True`
- `cancel_pending_pre_news = True`
- `pre_news_exit_minutes = 10`

Una linea puede endurecer estos valores, pero no relajarlos sin justificacion fuerte.

### Experiment-Specific Hardening

La linea de precision PM vigente endurece aun mas el bloqueo:

- `pre_minutes = 45`
- `post_minutes = 90`
- solo eventos `HIGH`
- solo eventos USD aprobados dentro del scope PM-safe

## Current News Reliability Classification

Veredicto actual del sistema de noticias:

- global: `SAFE ONLY UNDER NARROW CONDITIONS`
- PM-only narrow scope: `AUDITED AND SAFE FOR STRICT PM RESEARCH`
- AM / 8:00 NY: `NOT RELIABLE ENOUGH`

## PM-Safe Approved Scope

El scope PM-safe actual solo se considera valido para investigacion PM estricta y acotada. Incluye familias USD de horario fijo y validado:

- ISM manufacturing PMI
- ISM services PMI
- FOMC statement
- FOMC meeting minutes
- Federal Funds Rate
- FOMC press conference

Quedan excluidas:

- familias 08:30
- eventos EUR con riesgo de desalineacion DST
- cualquier sesion AM que dependa de cobertura mas amplia

## Rules For Sensitive Windows Like 8:00 NY

8:00 NY no se habilita por deseo ni por conveniencia estadistica. Solo podria habilitarse si:

- existe dataset AM aprobado con familias 08:30 relevantes cubiertas y auditadas
- la validacion exacta de timestamps en NY pasa de forma consistente
- runners, validacion y backtests fallan cerrado de punta a punta
- el kill de pending orders y forced flat pre-news esta verificado con evidencia
- el edge economico sobrevive esa proteccion, no solo el test tecnico

Hasta que eso ocurra:

- no se investiga AM serio
- no se habilita 8:00 NY

## Prohibited States

Estas situaciones quedan prohibidas:

- posiciones sin stop hard desde el inicio
- pending orders vivas cerca de noticias high impact
- continuar la corrida con news filter deshabilitado cuando se esperaba proteccion
- vender como edge una estrategia que solo sobrevive por falta de noticias cargadas
- promocionar una linea con muestra minima, holdout flojo o PF marginal

## Production And Funding Gate

Ningun bot puede considerarse apto para real/fondeo si no demuestra, como minimo:

- hard stop obligatorio
- protocolo de noticias operativo y auditado
- robustez multi-anual
- comportamiento OOS defendible
- ausencia de exposicion floja frente a high impact news

La meta economica de 20 USD no anula ninguna de estas exigencias.
