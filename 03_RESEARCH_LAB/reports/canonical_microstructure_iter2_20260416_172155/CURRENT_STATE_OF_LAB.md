# Current State Of Lab

## Authoritative Date

Estado consolidado y defendible al 2026-04-16, actualizado despues de la segunda iteracion seria sobre `pm_micro_reclaim_m3`.

## Evidence Base Used

Este estado se consolido contra:

- `000_PARA_CHATGPT.zip`
- `research_lab/STRATEGY_MASTER_MATRIX.md`
- `RESUMEN_EJECUTIVO.md`
- `AUDITORIA_SISTEMA_2026.md`
- `MAPA_ESTRATEGIAS.md`
- `CANONICAL_EXECUTION_CONTRACT.md`
- `STRATEGY_PROMOTION_POLICY.md`
- auditorias recientes y memorandos forenses
- artefactos de Waves 1-5 y estudios de ventanas
- resultados recientes de `audit_director`, `wave4`, `wave5`, `session_window_study_W1W2_CONSOLIDATED` y `pm_micro_reclaim_m3`
- codigo ejecutable en `engine.py`, `config.py`, `news_filter.py`, `validation.py`, `morning_challenge_runner.py`, `light_runner.py`

Archivos pedidos pero no presentes hoy en el repo:

- `task.md`
- `walkthrough.md`

## Firm Conclusions

- El laboratorio ya demostro que EURUSD PM en M15 es muy eficiente para las familias clasicas probadas.
- Waves 1-5 sirvieron para limpiar ilusiones, no para encontrar una linea productiva.
- El mejor residuo historico sigue siendo `zscore_mean_reversion_pm`, pero solo como benchmark, no como candidato serio.
- Los estudios de ventanas mejoraron control de dano, no edge.
- News Fortress quedo mas fuerte que antes, pero no es universalmente confiable para cualquier ventana.
- El pivot a precision ya existe en codigo y resultados, pero todavia no entrego edge defendible.

## Lines Considered Dead As Primary Research

Estas lineas deben darse por terminadas como direccion principal del laboratorio:

- `zscore_mean_reversion_pm` como candidato productivo
- `ict_fvg_liquidity_gap`
- la familia Wave 4: `h1_gated_zscore`, `h1_aligned_fvg`, `h1_trend_pullback_v2`
- la familia Wave 5: `london_sweep_reversion_pm`, `asia_sweep_reversion_pm`, `prev_day_extrema_sweep`
- nuevas variantes de PM M15 basadas en reversion simple, continuation simple, EMA alignment o sweep/fade clasico

## What Remains Alive Or In Quarantine

- `zscore_mean_reversion_pm`: vivo solo como benchmark historico de comparacion.
- `pm_micro_reclaim_m3`: linea cerrada como hipotesis viva. La segunda iteracion subio muestra pero destruyo PF, expectancy y consistencia.
- News Fortress PM-safe: operativo para investigacion PM estricta y acotada.
- Investigacion AM desde 8:00 NY: bloqueada.

## Current Benchmark

El benchmark practico sigue siendo `zscore_mean_reversion_pm` por ser la referencia historica menos mala del stack viejo.

Lectura honesta del audit consolidado mas reciente:

- PF aproximado 0.929
- expectancy negativa
- muestra suficiente para concluir que no hay edge defendible bajo ese marco

Eso alcanza para comparacion historica, no para promocion.

## Current Precision Prototype

`pm_micro_reclaim_m3` ya recibio una segunda iteracion seria.

Lectura honesta de esa iteracion:

- 37 trades totales en 2020-2025
- PF total aproximado 0.261
- expectancy R negativa
- 6 anios negativos
- 0 combinaciones pasan el serious gate

Conclusion:

- la frecuencia podia mejorarse
- pero al hacerlo el edge colapso
- el PF alto del prototipo previo era ruido por muestra miserable

## Contradictions Resolved

### 1. Documentacion historica vs evidencia reciente

`RESUMEN_EJECUTIVO.md`, `AUDITORIA_SISTEMA_2026.md` y `MAPA_ESTRATEGIAS.md` fueron utiles para auditar infraestructura y gaps historicos, pero hoy quedan superados en un punto clave: sugerian seguir hacia CHoCH + FVG / M5 como proximo desarrollo natural.

La evidencia posterior de Waves 1-5, auditorias duras y estudios de ventana indica algo mas fuerte:

- el problema no es solo "falta implementar CHoCH + FVG"
- el cuello de botella real es la combinacion EURUSD PM + M15 + familias logicas ya exprimidas

Por lo tanto, la conclusion defendible actual no es "seguir M15 con otra narrativa", sino salir de ese marco.

### 2. `STRATEGY_MASTER_MATRIX.md`

La matriz sigue siendo util como mapa historico, pero algunas etiquetas como `Ready` o `Repaired` no deben interpretarse como prueba actual de edge ni de promocion. El estado real lo mandan los resultados auditados mas recientes.

### 3. Seguridad de noticias

El repositorio historico contenia caminos que podian dar falsa sensacion de proteccion. El estado actual mas defendible es el posterior al hardening fail-closed y a la auditoria PM-safe, no el previo.

## Main Direction Today

No hay una linea promotable activa hoy dentro de esta familia.

La direccion defendible cambia de:

- "seguir iterando `pm_micro_reclaim_m3`"

a:

- "cerrar esta hipotesis y decidir en frio si merece investigarse otra familia microestructural realmente nueva"

No hay challenger secundario activo en este momento.

## Next Professional Step

Despues de dejar este contexto maestro firme, el siguiente paso correcto es:

1. usar la arquitectura de precision ya implementada como scaffold unico
2. no volver a iterar `pm_micro_reclaim_m3`
3. decidir si existe una hipotesis microestructural realmente distinta y auditabile, o si corresponde pivot estructural mas fuerte
4. mantener cerrado AM / 8:00 NY hasta nuevo veredicto serio de News Fortress
