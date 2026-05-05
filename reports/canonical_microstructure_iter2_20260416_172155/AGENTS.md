# AGENTS.md

## Canonical Workspace

El unico workspace canonico de este proyecto es:

`C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`

Todo trabajo futuro debe salir solo de este repo, sus archivos, sus datos y sus artefactos locales.

## Repo Mandate

- Trabajar solo localmente.
- No leer ni mezclar otros proyectos o carpetas.
- No usar nube.
- Trabajar solo sobre `main`.
- No hacer `push`, `pull`, `fetch`, `merge` ni `rebase`.
- No tocar Git salvo pedido explicito del usuario y con necesidad real.
- No abrir una nueva linea grande sin antes dejar evidencia y contexto persistido.

## Project Objective

Construir un laboratorio cuantitativo serio para desarrollar un bot de trading:

- 100% objetivo
- 100% programable
- 0% discrecional
- centrado en preservacion de capital
- blindado frente a noticias de alto impacto

La meta economica inicial es alcanzar un primer hito real de 20 USD como consecuencia natural de un edge sano, no como excusa para relajar estandares.

## Human Priority

La prioridad numero uno del proyecto es evitar de forma estructural cualquier repeticion del escenario de destruccion por noticias, ordenes vivas o riesgo no auditado.

Regla rectora:

`Risk-first. Capital preservation first. Alpha only after that.`

## Non-Negotiable Safety Rules

- Ninguna senal puede existir sin hard stop valido desde el origen.
- No se permiten posiciones sin stop ni pending orders vivas dentro de ventanas bloqueadas por noticias.
- Si News Fortress esta habilitado, la fuente debe ser operativa; no se admite modo fail-open.
- No se habilita una franja sensible solo porque "parece testeable"; primero debe ser segura.
- PF cercano a 1, edge marginal o "menos malo" no cuentan como victoria.
- Ninguna estrategia pasa a real/fondeo por marketing interno, intuicion o optimismo.

## Current Source Of Truth Hierarchy

Cuando existan contradicciones, manda esta jerarquia:

1. El comportamiento ejecutable del codigo actual en `research_lab/config.py`, `research_lab/engine.py`, `research_lab/news_filter.py`, `research_lab/validation.py` y runners activos.
2. Las auditorias recientes y sus resultados: `FORENSIC_META_AUDIT_MEMO.md`, `STRATEGIC_DECISION_MEMO.md`, `NEWS_FORTRESS_AUDIT.md`, `INITIAL_IMPLEMENTATION_RESULTS.md`.
3. Los artefactos recientes de `results/` y el bundle canonico `000_PARA_CHATGPT.zip`.
4. La documentacion historica previa, util como contexto pero no como verdad final si choca con evidencia nueva.

Nota importante:

- `task.md` y `walkthrough.md` fueron solicitados como entrada, pero no existen hoy como archivos reales dentro de este repo.

## Current Strategic Position

- Waves 1-5 ya limpiaron gran parte del ruido y de los falsos positivos.
- La familia EURUSD PM en M15 con reversion simple, continuation simple, HTF alignment y sweeps clasicos no dejo edge defendible.
- `zscore_mean_reversion_pm` queda solo como benchmark historico.
- News Fortress es globalmente seguro solo bajo condiciones estrechas.
- La unica direccion principal viva hoy es precision PM / microestructura / exposicion corta.
- La investigacion AM desde 8:00 NY sigue bloqueada hasta una auditoria mas fuerte del motor de noticias.

## How To Work Inside This Repo

- Leer primero `PROJECT_CHARTER.md`, `CURRENT_STATE_OF_LAB.md`, `RISK_PROTOCOL.md` y `RESEARCH_OPERATING_SYSTEM.md`.
- Mantener una sola direccion principal y, como maximo, una secundaria.
- Todo experimento nuevo debe ser 100% objetivo, auditable y comparable contra benchmark previo.
- Toda nueva linea debe declarar de forma explicita: contexto, entrada, salida, timeframe, ventana horaria, control de riesgo y politica de noticias.
- Despues de cada corrida material, actualizar el estado del laboratorio o dejar memo con veredicto honesto.

## What Must Not Happen

- No strategy spam.
- No rescates artificiales de familias ya muertas solo cambiando parametros.
- No usar el motor de noticias como maquillaje estadistico.
- No presentar una estrategia como "viva" si solo sobrevivio por muestra minima o por OOS fragil.
- No perder el contexto historico de lo que ya fracaso.


> **ESTE ARCHIVO ES UN SNAPSHOT HISTÓRICO.**
> La fuente viva y canónica está en: `BOT_V2_DAYTIME_LAB/docs/CORE_PROTOCOLS/AGENTS.MD`
