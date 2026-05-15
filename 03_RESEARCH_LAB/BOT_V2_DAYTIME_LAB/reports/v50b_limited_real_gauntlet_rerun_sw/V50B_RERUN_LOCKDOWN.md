# V50B RERUN LOCKDOWN

**Fase**: v50b_limited_real_gauntlet_rerun_single_writer_safe
**Estado**: **ACTIVE_RE-EXECUTION**
**Fecha**: 2026-05-14

## Prohibiciones Estrictas
- **NO PARALLEL RUNNERS**: Prohibido iniciar mǭs de un proceso de investigación simultǭneamente.
- **NO TEST ACCESS**: El blindaje 2025-2026 estǭ activo. Cualquier intento de lectura serǭ abortado por el motor.
- **NO CORE DRIFT**: El motor `src/v7_engine` es inmutable.
- **NO PREVIOUS DATA**: Prohibido importar o mezclar resultados de la corrida `v50b_limited_real_gauntlet` (corrupta).
- **NO DUMMY DATA**: Prohibido el uso de `np.random` para trades o resultados. Toda evidencia debe ser fsica.

## Compromiso Single-Writer
Este Gauntlet opera bajo la poltica de escritor único: un solo PID, un solo Lock y escritura append-only para garantizar la integridad bit a bit.
