# PHASE 37X SAFE DAILY AUTO-SHUTDOWN REPORT

## 1. Objetivo
Implementar un ciclo de vida diario seguro para MANIPULANTE en FTMO Trial, garantizando que no queden posiciones abiertas al apagar la PC del usuario (20:00 NY).

## 2. Veredicto Final Exacto
**SAFE_DAILY_AUTO_SHUTDOWN_READY**

## 3. Política Horaria (NY)
- **Inicio Sesión**: 07:00 NY.
- **No New Trades**: 16:30 NY.
- **Manage Only**: 16:30 - 19:30 NY (si hay posición).
- **Forced Safe Close**: 19:45 NY (Cierre obligatorio si sigue abierta).
- **Verify Flat**: 19:50 NY.
- **Auto Shutdown**: 20:00 NY.
- **Viernes Hard Close**: 16:55 NY.

## 4. Protección de Posición Abierta
- **SL/TP Broker-side**: Siempre activos (regla inmutable).
- **BE/Gestión**: El bot mantiene el control hasta el cierre.
- **¿Puede apagarse?**: **NO**, el runner impedirá el auto-apagado si detecta una posición abierta y falló el cierre forzado.
- **A las 19:45 NY**: Se ejecuta `execute_safe_close` automáticamente en FTMO Demo.
- **A las 20:00 NY**: El runner se cierra solo si el estado es `FLAT`.

## 5. Runner Actualizado
- **Lifecycle integrado**: SÍ (`phase37x_session_lifecycle.py`).
- **Position State integrado**: SÍ (`phase37x_position_state.py`).
- **Safe Close integrado**: SÍ (`phase37x_safe_close.py`).

## 6. START / STOP / STATUS
- **Actualizados**: SÍ (con advertencias de ciclo de vida y posición).

## 7. Heartbeat
- **Actualizado**: SÍ.
- **Campos nuevos**: `session_state`, `position_state`, `critical_do_not_turn_off_pc`, `shutdown_allowed`.

## 8. Tests
- **Tests ejecutados**: Validación de estados horarios y detección de posiciones.
- **Resultado**: **PASS**.

## 9. Dry-run
- **Ejecutado**: SÍ.
- **Decisión**: `DRY_RUN_ALLOW_SIGNAL_READY` (en ventana SESSION_ACTIVE).
- **Order_sent**: False.

## 10. Seguridad
- **No Real**: SÍ.
- **No Exness**: SÍ.
- **No Estrategia Modificada**: SÍ.

## 11. ZIP Canónico
- **Actualizado**: SÍ.

## 12. GitHub
- **Push**: SÍ.

## 13. Siguiente Paso Único
**Monitorear el Cierre**: El usuario puede observar cómo el bot pasa a `MANAGE_ONLY` a las 16:30 NY y se apaga automáticamente a las 20:00 NY si está flat.
