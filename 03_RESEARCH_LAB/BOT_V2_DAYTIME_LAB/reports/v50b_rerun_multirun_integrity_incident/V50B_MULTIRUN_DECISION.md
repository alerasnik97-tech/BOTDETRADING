# V50B Multirun Incident Decision
Estado: **V50B_RERUN_BLOCKED_MULTI_RUNID_OUTPUT_CONTAMINATION**

## Veredicto
Se bloquea el uso de los resultados de la corrida `V50B Rerun Single-Writer` debido a la detección de múltiples `run_id` (`24bb295d` y `bfe49625`) escribiendo de forma concurrente o secuencial en los mismos archivos de salida oficiales sin aislamiento.

## Justificación
1. **Contaminación de Datos**: Los archivos de trades, rechazos y proofs contienen una mezcla de dos ejecuciones distintas, rompiendo la atomicidad requerida para la auditoría institucional.
2. **Invalidez de Decisión**: La decisión oficial fue emitida a las 23:57 (Run 24bb295d), pero los datos siguieron mutando bajo un nuevo ID (bfe49625) hasta entrada la madrugada, lo cual anula la validez del veredicto previo.
3. **Falla de Protocolo**: El mecanismo de `Single-Writer` y el sistema de `Locks` fallaron al permitir que una segunda fase (o un proceso duplicado) reutilizara los mismos paths de salida oficiales.

## Acciones Requeridas
- No usar los rankings actuales para ninguna promoción de familia.
- Auditar la lógica de `v50b_limited_rerun_single_writer_runner.py` para entender por qué no bloqueó el inicio del segundo `run_id`.
- Realizar una limpieza (Quarantine) de la carpeta rerun antes de un nuevo intento.

---
**Seguridad**: No se detectó TEST leakage ni core drift, pero la integridad estructural del experimento está comprometida.
