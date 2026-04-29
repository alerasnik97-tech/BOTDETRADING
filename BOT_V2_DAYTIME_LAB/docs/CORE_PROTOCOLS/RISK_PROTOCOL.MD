# Risk Protocol

## Prime Directive

El objetivo principal del sistema no es operar mas.
Es impedir exposicion no auditada, especialmente alrededor de noticias de alto impacto y ventanas de ejecucion degradada.

## Mandatory Risk Controls

### Hard Stop From Origin

- Toda senal debe declarar un hard stop valido desde el origen.
- El motor rechaza senales sin stop valido.
- No existe modo permitido sin stop.

### Fixed Research Risk

- Riesgo fijo de investigacion: `0.5%` por trade.
- No se permite maquillar edge con sizing o money management.

### Position And Exposure Limits

- Maximo una posicion abierta a la vez por defecto.
- Maximo de trades por dia definido por `EngineConfig`.
- Todo experimento debe tener horario explicito de entrada y forced flat por horario si aplica.

## News Fortress Operational Rules

### Fail-Closed Requirement

- Si `NewsConfig.enabled = True`, la fuente debe ser operativa.
- Si la fuente no es operativa, la corrida debe fallar.
- No se admite continuar como si la ausencia de noticias fuera un dataset valido.

### Base Fortress Defaults

El baseline actual de `NewsConfig` es:

- `pre_minutes = 30`
- `post_minutes = 60`
- `forced_exit_pre_news = True`
- `cancel_pending_pre_news = True`
- `pre_news_exit_minutes = 10`

Una linea puede endurecer estos valores.
No puede relajarlos sin justificacion fuerte.

### Guard Semantics

La arquitectura operativa separa de forma explicita:

- `entry_blocked`
- `pending_kill`
- `force_flat`
- `cooldown_blocked`

Y el forced flat usa la apertura de la primera barra cuyo open cae dentro de la kill zone.

### Pending Orders And Signals

- No se permiten pending orders ni pending signals vivas dentro de ventanas bloqueadas por noticias.
- Si el reloj entra en kill zone, la senal pendiente se cancela.

### Flat Before News

- Ninguna posicion debe permanecer viva al entrar en `force_flat`.
- Si el open de la barra bloqueada ya atraviesa SL o TP, el motor respeta primero ese gap.
- Si no hubo gap contra el contrato de riesgo, el motor ejecuta `news_fortress_kill` en la apertura de esa barra.

### Post-News Cooldown

- No se abren nuevas entradas durante `post_minutes`.
- Ese cooldown queda expresado de forma auditable.

### Late-Session Degradation Guard

- Desde `17:00 NY` el motor penaliza spread y slippage.
- `17:00-19:00 NY` no se trata como extension neutral del horario operativo.

## Current News Reliability Classification

- global: `SAFE UNDER FORTRESS CONDITIONS`
- PM-only narrow scope: `READY`
- AM / `08:00 NY`: `READY_FOR_STRICT_AM_RESEARCH`

## PM-Safe Approved Scope

El scope PM-safe aprobado hoy sigue siendo util para investigacion estricta PM y acotada.

Familias PM aprobadas:

- `ISM Manufacturing PMI`
- `ISM Services PMI`
- `FOMC Statement`
- `FOMC Meeting Minutes`
- `Federal Funds Rate`
- `FOMC Press Conference`

## AM Final Current Status

Existe un dataset local dedicado:

- `data/news_eurusd_am_fortress_v3.csv`

Su valor actual es de seguridad AM.
No es todavia una fuente global aprobada para abrir alpha en la manana.

Cobertura fuerte actual:

- `08:15`
- `08:30` (incluyendo `retail sales` y `unemployment claims`)
- `08:45`
- `09:45`
- `10:00`
- `14:00`
- `14:30`

Gaps criticos anteriormente abiertos:

- `retail sales m/m`: CERRADO via manifest auditado.
- `unemployment claims`: CERRADO via manifest auditado.

La ventana `08:00 NY` queda HABILITADA tras cumplir estas condiciones:
- existe coverage local aprobado para familias `08:15`, `08:30` y `10:00`.
- la validacion exacta de timestamps en NY pasa de forma consistente.
- runners, validacion y backtests fallan cerrado de punta a punta (verificado).
- el kill de pending orders y forced flat pre-news esta verificado con evidencia.
- el edge economico sobrevive esa proteccion (sujeto a investigacion alpha).

Estado actual:
- se investiga AM bajo reglas News Fortress v3.
- se habilita `08:00 NY` para Quantitative Research.
- no se trata `17:00-19:00 NY` como extension neutral del horario operativo

## Explicit Prohibitions

Quedan prohibidos:

- posiciones sin stop hard desde el inicio
- pending orders vivas cerca de high impact news
- continuar la corrida con fuente vetada o inconsistente
- usar el filtro de noticias como maquillaje estadistico
- vender como expansion segura un horario que todavia no tiene coverage AM completa

## Production And Funding Gate

Ningun bot puede considerarse apto para real o fondeo si no demuestra, como minimo:

- hard stop obligatorio
- protocolo de noticias operativo y auditado
- robustez multi-anual
- comportamiento OOS defendible
- ausencia de exposicion floja frente a high impact news

La meta economica de `20 USD` no anula ninguna de estas exigencias.
