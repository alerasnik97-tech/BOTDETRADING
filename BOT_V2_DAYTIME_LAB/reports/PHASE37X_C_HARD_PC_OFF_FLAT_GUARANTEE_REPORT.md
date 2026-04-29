# PHASE 37X-C HARD PC-OFF FLAT GUARANTEE REPORT

## 1. Lo más importante
Se ha implementado la política operativa final de **Garantía FLAT** antes de las **20:00 NY**. El sistema ha sido configurado para asegurar que ninguna posición de MANIPULANTE quede abierta al momento en que el usuario debe apagar su PC. La regla, ya auditada y aprobada en Phase 37X-A (+1.53R de impacto), se ejecuta ahora con mecanismos de redundancia (5 reintentos) y doble verificación (19:50 y 19:55 NY).

## 2. Veredicto final exacto
**HARD_PC_OFF_FLAT_GUARANTEE_READY**

## 3. Confirmación de Auditoría
**NO se re-auditó**. Se implementó la política operativa basada estrictamente en los resultados aprobados de la Phase 37X-A, los cuales demostraron que el cierre a las 19:45 NY no daña el edge estadístico de la estrategia.

## 4. Política final antes de apagar PC (NY)
- **16:30 NY**: Fin de ventana de entradas (No New Trades).
- **19:45 NY**: Inicio de **Cierre Forzado Obligatorio**.
- **19:45 - 19:49 NY**: Ventana de reintentos (hasta 5 intentos con `order_check`).
- **19:50 NY**: Primera verificación de estado FLAT.
- **19:55 NY**: Segunda verificación de estado FLAT.
- **20:00 NY**: Apagado automático del runner solo si se confirma `FLAT_CONFIRMED`.

## 5. ¿Puede quedar posición abierta después de 20:00?
**SÍ**, pero solo en caso de falla crítica externa (pérdida de conexión MT5/Broker). En tal caso, el sistema **NO se apagará**, emitirá una alerta roja en el panel de control y exigirá **MANUAL_CLOSE_REQUIRED** antes de que el usuario apague la PC.

## 6. Safe close retry
- **Lógica**: Hasta 5 intentos con esperas controladas.
- **Seguridad**: Se ejecuta `mt5.order_check` antes de cada envío para validar márgenes y conectividad.
- **Account Protection**: El módulo está bloqueado para cuentas Reales o Exness.

## 7. STATUS / heartbeat
- **STATUS Panel**: Ahora muestra explícitamente `SAFE_TO_TURN_OFF_PC` en verde cuando el deadline se cumple y la cuenta está flat.
- **Heartbeat**: Incluye los nuevos campos: `pc_off_deadline_ny`, `forced_close_attempts`, `last_close_attempt_status`, `flat_confirmed_1950`, `flat_confirmed_1955`, `safe_to_turn_off_pc` y `manual_intervention_required`.

## 8. Tests
- **Cantidad**: 10 casos de uso validados.
- **Resultado**: **PASS**.

## 9. Dry-run
- **Resultado**: Exitosa sincronización de gates.
- **Decisión**: `NO_TRADE` (News block por Crudo USD).
- **Order_sent**: False.

## 10. Seguridad
- **No Real / No Exness**: Validado por `account_gate`.
- **Inmutabilidad**: La lógica de MANIPULANTE (Phase 25) no ha sido modificada.

## 11. ZIP canónico
Actualizado con la infraestructura de cierre garantizado.

## 12. GitHub
Sincronizado en `main`.

## 13. Siguiente paso único
**Ejecución Operativa**: El sistema está listo para gestionar el cierre de hoy a las 19:45 NY. No se requieren más cambios técnicos.
