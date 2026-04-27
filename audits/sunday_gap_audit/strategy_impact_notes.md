# Strategy Impact Notes: Sunday Gap Audit

## Estrategia Afectada: Candidato Shadow Actual
`tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_1_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m`

## Puntos de Contaminación Material
1. **Lunes Operativo (PDH/PDL):**
   - El algoritmo busca sweeps de PDH/PDL del "día anterior".
   - El lunes, el sistema toma PDH/PDL del **Domingo**.
   - Como el rango del domingo es minúsculo comparado con el del viernes, el algoritmo busca sweeps en niveles irrelevantes.
   - **Consecuencia:** Se pierden todos los trades genuinos de sweeps de niveles del Viernes que ocurren el Lunes.

2. **Asia Range (Lunes):**
   - El rango de Asia del lunes se calcula uniendo `prev_bars` (18:00+) y `curr_bars` (00:00-02:00).
   - Para el lunes, `prev_bars` son las del domingo.
   - Si faltan barras del domingo (gap de apertura), el rango de Asia queda mutilado.
   - **Consecuencia:** Sesgos en la detección de desviaciones de Asia el lunes temprano.

3. **Ejemplo Concreto:**
   - Si el lunes el precio barre el High del Viernes (PDH real), el sistema no lo detecta porque su PDH es el del Domingo (mucho más bajo).
   - El sistema podría dar una señal de "Sweep PDH" cuando en realidad solo barrió el máximo del domingo, lo cual no tiene relevancia institucional.

## Veredicto de Estrategia
El impacto es **MATERIAL**. El rendimiento de los lunes en el laboratorio institucional está contaminado y no refleja la realidad operativa de la estrategia. Esto podría estar inflando o desinflando artificialmente el Win Rate de los lunes.
