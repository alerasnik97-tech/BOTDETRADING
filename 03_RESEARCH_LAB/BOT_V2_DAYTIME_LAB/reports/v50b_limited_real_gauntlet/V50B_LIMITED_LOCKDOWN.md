# V50B LIMITED LOCKDOWN

**Fecha**: 2026-05-14
**Objetivo**: Ejecución certificada del Gauntlet Real para F06, F08 y F12.

## Restricciones Absolutas
- **AISLAMIENTO TOTAL**: Prohibida la contaminación de estado entre configuraciones. Cada ejecución debe usar una instancia limpia del motor.
- **NO TEST**: Blindaje 2025-2026 activo. Bloqueo incondicional de cualquier acceso a datos fuera de 2020-2024.
- **NO CORE DRIFT**: Prohibido modificar `src/v7_engine` o `src/v6_utils`.
- **NO SYNTHETIC**: Prohibido el uso de generadores aleatorios o trades dummy. Toda evidencia debe ser física (ticks reales).
- **EXCLUSIÓN F01**: La familia F01 está prohibida en este Gauntlet por violar la política de ventana horaria NY.

## Compromiso
El laboratorio no reportará resultados que no puedan ser recalculados bit a bit desde los logs de trades y ticks originales.
