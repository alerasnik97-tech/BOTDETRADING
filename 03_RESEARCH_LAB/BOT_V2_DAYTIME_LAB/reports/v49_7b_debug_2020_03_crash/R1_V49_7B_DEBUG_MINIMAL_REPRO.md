# R1 V49.7B DEBUG ?" MINIMAL REPRO

Para reproducir el crash:
1. Usar el runner `v49_7b_full_scope_runner.py` con 800 configs.
2. Procesar mes 2020-03 (Parquet de ~5.7M ticks).
3. Observar el consumo de RAM tras cargar los ticks y generar la cachǸ de ventanas para todos los seales.
4. El proceso se detiene cuando el OS mata al proceso Python por OOM (Out of Memory) o el proceso queda "zombie".

**Solucin Validada**:
Usar el script `debug_v49_7b_2020_03_single_config.py` para verificar que la lgica es correcta para N pequeo. Para N=800, se requiere fragmentacin (batching).
