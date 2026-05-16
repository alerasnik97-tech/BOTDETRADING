# Weekly Impact Notes: Sunday Gap Audit

## Weekly Calculations Affected
1. **Previous Week High / Low:**
   - Si la semana termina el viernes pero hay barras el domingo, ¿se incluyen en la semana anterior o en la nueva?
   - La lógica actual de `baseline_truth_model.py` no tiene una función `compute_weekly_levels`.
   - Sin embargo, cualquier cálculo que use `frame.groupby(frame.index.isocalendar().week)` incluirá las barras del domingo en la semana correspondiente.
   - Dado que el domingo NY es técnicamente lunes UTC en su mayoría, a menudo cae en la misma semana que el lunes.
   - **Impacto:** Marginal para PWH/PWL si el domingo está dentro del rango del viernes, pero material si hay un gap de apertura fuerte que no se captura o se malinterpreta.

2. **Weekly VWAP:**
   - La pérdida de las primeras horas del domingo (17:00 a 22:00 UTC aprox) desplaza el punto de anclaje del VWAP semanal.
   - Si faltan las barras de apertura (17:00 NY), el VWAP semanal empieza con datos de las 22:00 UTC, perdiendo el volumen y precio de la apertura dominical.
   - **Impacto:** Medio. Distorsiona la pendiente y valor del VWAP semanal durante el lunes y martes.

3. **Weekly Roll:**
   - El laboratorio no detecta correctamente el "gap de apertura" del domingo si las barras están mutiladas.
   - Esto afecta la percepción de "fuerza" o "exhaustion" en la apertura semanal.

## Veredicto Weekly
La estructura de datos preparada tiene una **pérdida parcial de barras de apertura del domingo** (missing 17:00-21:00 UTC aprox en varios tramos). Esto degrada la calidad de los indicadores acumulativos semanales.
