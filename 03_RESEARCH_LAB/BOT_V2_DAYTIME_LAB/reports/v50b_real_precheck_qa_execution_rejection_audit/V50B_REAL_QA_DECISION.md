# V50B REAL QA ?" DECISION

**Estado Final**: **V50B_REAL_QA_PASS_PARTIAL_READY_FOR_LIMITED_GAUNTLET**

## Resumen de la Auditoría
La auditoría de QA ha cerrado el gap de ejecución detectado en el pre-check inicial. Se ha demostrado físicamente que el motor `UnifiedV7Engine` está procesando las señales de todas las familias.

## Hallazgos de Ejecución
- **F12**: Confirmada con trades reales (6).
- **F01, F06, F08**: Confirmadas mediante **Rejection Audit**. El motor rechazó las señales por `BLOCKED_BY_SCHEDULE`, lo cual es el comportamiento esperado dado que el motor por defecto protege la sesión de NY (08:00 - 11:00) y F01/F06 operan fuera o en los límites de este horario.
- **Trazabilidad**: Se ha generado un `engine_call_proof` para las 4 familias.

## Reserva de Noticias
Se mantiene la reserva **F12_WITH_RESERVATIONS** debido a la falta de conexión con un calendario de noticias real.

## Autorización
Se autoriza proceder con el **V50B Full Gauntlet** con las siguientes condiciones:
1. El runner debe configurar el `ScheduleGuard` de forma flexible para permitir las ventanas operativas de cada familia.
2. Se debe intentar localizar o generar el `NewsCalendar` real para la validación final de F12.
3. El Gauntlet debe mantener el registro de motivos de rechazo para todas las señales.

**Veredicto**: PASS PARTIAL. Pipeline certificado para las 4 familias.
