# PHASE 37X-B SAFE DAILY AUTO-SHUTDOWN LIFECYCLE REPORT

## 1. Lo más importante
Se ha implementado el ciclo de vida operativo completo para MANIPULANTE en FTMO Trial. El sistema ahora gestiona automáticamente el corte de operaciones nuevas a las **16:30 NY**, el cierre forzado de seguridad a las **19:45 NY** y el auto-apagado del runner a las **20:00 NY** (solo si la cuenta está flat). Esta mejora operativa permite una ejecución desatendida segura, eliminando el riesgo de dejar posiciones abiertas sin control del bot al apagar la PC del usuario.

## 2. Veredicto final exacto
**SAFE_DAILY_AUTO_SHUTDOWN_READY**

## 3. Política horaria (NY)
- **Inicio Sesión**: 07:00 NY.
- **No New Trades**: 16:30 NY.
- **Manage Only**: 16:30 - 19:30 NY (si hay posición activa).
- **Forced Safe Close**: 19:45 NY (Cierre obligatorio en FTMO Demo).
- **Verify Flat**: 19:50 NY.
- **Auto Shutdown**: 20:00 NY.
- **Viernes Hard Close**: 16:55 NY.

## 4. Protección si hay posición abierta
- **¿Puede apagarse?**: **NO**. El runner bloquea el shutdown si detecta una posición abierta y no pudo cerrarla a las 19:45 NY.
- **A las 19:45 NY**: Se ejecuta `execute_safe_close` automáticamente en la cuenta de prueba.
- **A las 20:00 NY**: El runner se detiene limpiamente solo si se confirma estado **FLAT**.

## 5. Runner actualizado
- **Lifecycle**: Integrado (`phase37x_session_lifecycle.py`).
- **Position State**: Integrado (`phase37x_position_state.py`).
- **Safe Close**: Integrado (`phase37x_safe_close.py`).

## 6. START / STOP / STATUS
- **START**: Actualizado con política horaria visible y validación de cuenta.
- **STOP**: Actualizado con advertencia de cierre seguro.
- **STATUS**: Actualizado para mostrar el estado del ciclo de vida y alertas críticas de "No apagar PC".

## 7. Heartbeat
- **Campos nuevos**: `session_state`, `can_open_new_trades`, `manage_only`, `forced_safe_close_time_ny`, `verify_flat_time_ny`, `shutdown_time_ny`, `pc_off_warning`, `critical_do_not_turn_off_pc`, `last_safe_close_status`.

## 8. Tests
- **Cantidad**: 10 casos lógicos validados.
- **Resultado**: **PASS**.

## 9. Dry-run
- **Ejecutado**: SÍ.
- **Decisión**: `NO_TRADE` (News block por Crudo USD).
- **Order_sent**: False.

## 10. Seguridad
- **No Real**: SÍ.
- **No Exness**: SÍ.
- **No Estrategia Modificada**: SÍ.
- **No Orden Real**: SÍ.

## 11. ZIP canónico
- **Actualizado**: SÍ.

## 12. GitHub
- **Push**: SÍ (`main`).

## 13. Siguiente paso único
**Operación Desatendida**: El usuario puede dejar el bot corriendo. El sistema se encargará de gestionar el fin de la jornada operativa y el apagado seguro de forma autónoma.
