# V50B SINGLE-WRITER LOCKDOWN

**Estado**: **REMEDIATION_PHASE_ACTIVE**
**Fecha**: 2026-05-14

## Restricciones Proactivas
- **PROHIBIDO RERUN**: Queda prohibido lanzar el Gauntlet completo hasta que el Preflight de IO sea exitoso.
- **LOCK MANDATORY**: Todo runner futuro debe requerir la adquisición de un `run.lock` fsco.
- **NO DATA RESCUE**: Estǭ prohibido intentar rescatar o mezclar datos de la corrida corrupta. El rerun serǭ desde cero.
- **SINGLE WRITER**: Solo un proceso tiene permiso de escritura sobre los CSV de resultados consolidados.

## Objetivo de Integridad
Restaurar la confianza en los outputs del laboratorio eliminando la posibilidad tǸcnica de Race Conditions.
