# PHASE 37X-C HARD PC-OFF DEADLINE FLAT GUARANTEE REPORT

## 1. Lo más importante
Se ha implementado una garantía de estado **FLAT** antes de las **20:00 NY**. El sistema ahora no solo intenta cerrar posiciones a las 19:45 NY, sino que reintenta el cierre hasta 5 veces si falla, y verifica el estado en dos ventanas críticas adicionales (19:50 y 19:55 NY). Si antes de las 20:00 NY el sistema no logra confirmar que la cuenta está flat, emite una alerta roja **CRITICAL_MANUAL_INTERVENTION_REQUIRED**, informando al usuario que NO es seguro apagar la PC sin intervención manual.

## 2. Veredicto final exacto
**HARD_PC_OFF_FLAT_GUARANTEE_READY**

## 3. Política final antes de apagar PC
- **19:45 NY (Forced Close)**: Cierre obligatorio con hasta 5 reintentos y `order_check`.
- **19:50 NY (Verify Flat 1)**: Primera verificación de seguridad.
- **19:55 NY (Verify Flat 2)**: Segunda verificación de seguridad.
- **20:00 NY (Shutdown)**: Apagado automático solo si se confirma `FLAT_CONFIRMED`.
- **Si NO está flat**: El sistema bloquea el shutdown, marca incidente crítico y recomienda cierre manual inmediato.

## 4. ¿Puede quedar posición abierta después de 20:00?
**NO** bajo condiciones normales de conexión. El sistema está diseñado para forzar el cierre. Solo podría quedar abierta si MT5 pierde conexión total con el broker o si el mercado está congelado. En ese caso, el bot **NO se apagará** y mostrará una advertencia visual persistente.

## 5. Safe close retry
- **Intentos**: 5 reintentos con backoff de 5 segundos.
- **Order Check**: Se realiza antes de cada envío para asegurar que la orden es válida.
- **Logs**: Cada intento queda registrado en `decisions.csv` y en el heartbeat.

## 6. STATUS / heartbeat
- **Dashboard**: El panel `.bat` ahora incluye un veredicto visual: `SAFE_TO_TURN_OFF_PC` (Verde), `NOT_SAFE_YET` (Amarillo) o `MANUAL_CLOSE_REQUIRED` (Rojo).
- **Campos nuevos**: `hard_flat_required_before_pc_off`, `pc_off_deadline_ny`, `forced_close_attempts`, `manual_intervention_required`.

## 7. Tests
- **Cantidad**: 10 casos de prueba lógicos validados.
- **Resultado**: **PASS**.

## 8. Dry-run
- **Decisión**: `NO_TRADE` (News block activo).
- **Order_sent**: False.

## 9. Seguridad
- **No Real / No Exness**: Confirmado por `account_gate`.
- **No Estrategia Modificada**: MANIPULANTE (Phase 25) permanece intacta.

## 10. ZIP canónico
Actualizado con toda la lógica de redundancia de cierre.

## 11. GitHub
Cambios sincronizados en `main`.

## 12. Siguiente paso único
**Monitoreo del Cierre**: Hoy a las 19:45 NY el bot ejecutará el primer cierre forzado real con la nueva lógica de reintentos si hay una posición activa.
