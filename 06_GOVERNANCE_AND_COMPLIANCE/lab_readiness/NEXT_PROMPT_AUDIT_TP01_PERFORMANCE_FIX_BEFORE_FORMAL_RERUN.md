# NEXT PROMPT: AUDIT TP-01 PERFORMANCE FIX BEFORE FORMAL RERUN

Actuá como Claude Opus 4.7 Max en modo **Strategy Performance and Equivalence Auditor** senior, backtest safety engineer, no-lookahead auditor y quant code gatekeeper.

============================================================
OBJETIVO
============================================================

Auditar el fix de performance O(N²) de la estrategia `tp01_london_ny_momentum_pullback` y decidir si está 100% aprobada para relanzar la corrida formal train-only de 10 años (2015–2024).

NO correr backtest en esta fase.
NO strategy run.
NO optimization.
NO sweep.
NO validation.
NO holdout.
NO usar 2025/2026.
NO news.
NO high precision.
NO tocar engine.
NO tocar data cruda.

============================================================
PUNTOS DE AUDITORÍA REQUERIDOS
============================================================

1. **Revisión de Código del Fix**:
   - Analizar `03_RESEARCH_LAB/research_lab/strategies/tp01_london_ny_momentum_pullback.py`.
   - Confirmar la existencia de `_CACHE` e `_get_cached_indicators()`.
   - Verificar que los arrays precalculados se guardan en arrays de numpy de tipo float64.
   - Confirmar que el cálculo del percentile se realiza sobre el slice de numpy view sin dependencias de pandas ni dropna costosos.
   - Confirmar que el acceso a EMA (`ema_values[i - 1]`, `ema_prev[i - 2]`) y ATR (`atr_values[i]`) es estrictamente en tiempo constante $O(1)$.

2. **Revisión de No-Lookahead y Causalidad**:
   - Auditar minuciosamente los índices en `signal()`:
     - `i` para ATR actual.
     - `i - lookback : i` para la ventana previa de ATR (excluyendo estrictamente `i`).
     - `i - 1` para `ema_now` (EMA de precios de cierre hasta la barra previa).
     - `i - 2` para `ema_prev`.
   - Asegurar que no hay acceso a `i+1`, datos futuros o leaks.

3. **Revisión de la Suite de Test de Equivalencia**:
   - Inspeccionar `03_RESEARCH_LAB/research_lab/tests/test_tp01_performance_equivalence.py`.
   - Asegurar que el test de equivalencia compara el comportamiento bar-by-bar del optimizado contra una copia fidedigna del algoritmo O(N²) original.
   - Asegurar que el test de prevención de lookahead muta datos futuros y valida que la señal no cambie.
   - Asegurar que el test de cache verifica que se reuse el mismo objeto array (usando `assertIs`).
   - Asegurar que el test de invalidación de cache maneja correctamente dataframes de diferentes tamaños o contenidos.

4. **Resultados de las Pruebas Unitarias**:
   - Ejecutar la suite completa y confirmar que todos los tests pasan con éxito:
     ```powershell
     $env:PYTHONPATH="03_RESEARCH_LAB"
     python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_tp01_performance_equivalence.py" -v
     python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_priority_a_skeletons.py" -v
     ```
   - Confirmar que el runtime de Smoke es muy inferior a 1.0 segundo.

============================================================
CRITERIO DE SALIDA / PASO OPERATIVO
============================================================

Si todo es correcto, emitir la aprobación final:
**TP01_PERFORMANCE_FIX_APPROVED_READY_FOR_FORMAL_RERUN**

e indicar que el operador ya puede ejecutar el script `03_RESEARCH_LAB/scratch/formal_run_tp01.py` con el run ID definitivo para completar el dossier con datos reales de 10 años, lo cual ahora tomará escasos segundos gracias al fix quirúrgico.
