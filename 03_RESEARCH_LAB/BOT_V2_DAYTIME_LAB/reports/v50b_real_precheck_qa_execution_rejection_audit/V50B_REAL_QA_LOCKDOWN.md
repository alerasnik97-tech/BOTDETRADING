# V50B REAL QA ?" LOCKDOWN

**Fecha**: 2026-05-14
**Objetivo**: Auditoría técnica del pre-check real y análisis de rechazos de ejecución.

## Restricciones Absolutas
- **NO FULL GAUNTLET**: No se autoriza el barrido masivo hasta resolver el gap de ejecución de F01, F06 y F08.
- **NO TEST**: Blindaje 2025-2026 activo.
- **NO CORE DRIFT**: Prohibido modificar el motor core.
- **AUDIT FOCUS**: El foco es la trazabilidad de señales hacia trades y la justificación de rechazos del motor.

## Compromiso
El laboratorio no aceptará un estado de "Listo" si no existe evidencia de que el motor puede procesar señales de todas las familias candidatas o, en su defecto, que los rechazos están debidamente justificados por reglas de negocio (Horario, Noticias, Throttling).
