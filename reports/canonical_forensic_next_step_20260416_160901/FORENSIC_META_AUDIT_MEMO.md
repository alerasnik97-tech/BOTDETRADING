# Forensic Meta Audit Memo

## Executive Verdict

Las Waves 1-5 limpiaron mucho ruido, pero no dejaron una linea productiva defendible en EURUSD PM bajo M15 y familias clasicas de reversion, continuation basica, alignment HTF, FVG o sweeps de liquidez.

## What Is Dead

- `zscore_mean_reversion_pm` sigue siendo solo "menos malo", no bueno. El consolidado OOS revisado arroja PF 0.929 y expectancy negativa, y los reruns mas recientes apenas rozan PF ~1.01 con retorno economico trivial.
- `ict_fvg_liquidity_gap` no sobrevivio la auditoria dura. El laboratorio la mantiene como baseline historico, no como edge.
- `h1_gated_zscore`, `h1_aligned_fvg`, `h1_trend_pullback_v2` y la familia Wave 4 no corrigieron el problema central: mejoran forma metodologica, no resultado economico.
- `london_sweep_reversion_pm`, `asia_sweep_reversion_pm` y `prev_day_extrema_sweep` quedan terminadas. Los resultados revisados son demasiado debiles para justificar mas iteracion incremental.

## What Is Only "Less Bad"

- El mejor residuo del stack viejo sigue siendo el z-score PM, pero como benchmark de comparacion, no como candidato real.
- El estudio de ventanas confirma que el mejor tramo no produjo edge serio; solo encontro una ventana menos destructiva.
- Se detecto y corrigio un defecto de reporting: `years_positive_oos` no se poblaba desde `report.py`. Eso podia distorsionar rankings, pero no cambia el veredicto central porque PF y expectancy siguen sin ser defendibles.

## Real Bottleneck

El cuello de botella no parece ser una sola variable aislada. La evidencia apunta a la combinacion:

- EURUSD altamente eficiente en PM.
- M15 demasiado grueso para capturar un edge pequeno sin inflar exposicion temporal.
- Familias logicas ya muy exprimidas: mean reversion simple, continuation simple, alignment EMA y sweep/fade basico.

La conclusion profesional no es "cambiar un parametro mas". La conclusion es cortar la linea PM-M15 clasica y, si se sigue, hacerlo con microestructura y tiempo de exposicion mucho mas corto.

## What I Would Not Trust For Production

- Cualquier estrategia actual de Waves 1-5.
- Cualquier baseline que hoy dependa de presentar un PF apenas encima de 1 con muestra pobre.
- Cualquier lectura optimista de AM hasta que el motor de noticias quede realmente confiable.
