# REPORT: MANIPULANTE AUTO ALERTS START/STOP

**Fecha:** 2026-04-30 18:07:12-03:00
**Alcance:** Integración del loop de alertas de Telegram en el ciclo de vida operativo de MANIPULANTE.
**Estado Final:** AUTO_ALERTS_START_STOP_OK

## Archivos Inspeccionados
- `MANIPULANTE\START_MANIPULANTE.bat`
- `MANIPULANTE\STOP_MANIPULANTE.bat`
- `MANIPULANTE\STATUS_MANIPULANTE.bat`
- `BOT_V2_DAYTIME_LAB\src\phase45_run_alert_check.py`
- `BOT_V2_DAYTIME_LAB\src\phase37ze_quick_status_panel.py`

## Archivos Modificados
1. `BOT_V2_DAYTIME_LAB\src\phase45_run_alert_check.py`: Se agregó gestión de PID/Lock e idempotencia.
2. `BOT_V2_DAYTIME_LAB\src\phase37ze_quick_status_panel.py`: Integración de estado de alertas en el dashboard.
3. `MANIPULANTE\START_MANIPULANTE.bat`: Inicia alertas antes de lanzar el bot.
4. `MANIPULANTE\STOP_MANIPULANTE.bat`: Detiene alertas después de detener el bot.
5. `MANIPULANTE\16_OBSERVABILITY\alerts\START_ALERTS_LOOP_MANIPULANTE.bat`: Script auxiliar de inicio.
6. `MANIPULANTE\16_OBSERVABILITY\alerts\STOP_ALERTS_LOOP_MANIPULANTE.bat`: Script auxiliar de parada.
7. `MANIPULANTE\16_OBSERVABILITY\alerts\STATUS_ALERTS_LOOP_MANIPULANTE.bat`: Script auxiliar de estado.

## Backups Creados
- `MANIPULANTE\START_MANIPULANTE.bat.bak_auto_alerts_20260430_175900`
- `MANIPULANTE\STOP_MANIPULANTE.bat.bak_auto_alerts_20260430_175900`
- `MANIPULANTE\STATUS_MANIPULANTE.bat.bak_auto_alerts_20260430_175900`
- `BOT_V2_DAYTIME_LAB\src\phase45_run_alert_check.py.bak_auto_alerts_20260430_175900`
- `BOT_V2_DAYTIME_LAB\src\phase37ze_quick_status_panel.py.bak_auto_alerts_20260430_175900`

## Flujo Operativo Actualizado
- **START**: El usuario ejecuta `START_MANIPULANTE.bat`. El sistema activa el loop de alertas en segundo plano y luego lanza el runner del bot.
- **STOP**: El usuario ejecuta `STOP_MANIPULANTE.bat`. El sistema detiene el runner de forma segura y luego termina el proceso del loop de alertas.
- **STATUS**: `STATUS_MANIPULANTE.bat` muestra ahora el estado de `BOT RUNNER` y `TELEGRAM ALERTS` simultáneamente.

## Frecuencia y Spam
- **Intervalo del loop**: 60 segundos.
- **Deduplicación**: Se utiliza `phase45_alert_state.py` con cooldown de 10 min (1 min para Critical).
- **Eventos**: Telegram notifica solo al iniciar el loop y ante cambios de estado detectados por el motor de alertas. No hay spam cada 60s si el estado es persistente.

## Alertas Disponibles (Engine Phase45)
- **CRITICAL**:
  - `BOT_APAGADO_DURANTE_SESION`: Runner inactivo entre 08:00 y 18:00.
  - `REAL_OR_EXNESS_DETECTED`: Fallo de seguridad por cuenta no autorizada.
  - `MT5_DISCONNECTED`: Terminal sin conexión al servidor.
  - `HEARTBEAT_STALE`: Dashboard sin actualizar por > 5 min.
  - `ORDER_SEND_ERROR`: Error crítico en envío de orden.
- **INFO**:
  - `BLOQUEADO_NOTICIAS`: Operativa pausada por News Fortress.
  - `TRADE_TAKEN_DEMO`: Posición abierta detectada.
  - `SAFE_TO_TURN_OFF_PC_YES`: No hay riesgo, seguro apagar.
  - `ALERTS_LOOP_STARTED`: Notificación de inicio del sistema de monitoreo.

## Resultados de Pruebas
1. `RUN_ALERTS_ONCE_MANIPULANTE.bat`: OK (Detecta alertas una vez).
2. `START_ALERTS_LOOP_MANIPULANTE.bat`: OK (Inicia loop, crea PID/Lock).
3. `STATUS_ALERTS_LOOP_MANIPULANTE.bat`: OK (Reporta RUNNING/STOPPED).
4. `Idempotencia`: OK (No permite duplicar procesos si ya hay un PID activo).
5. `Integrated START/STOP`: OK (Sincronización completa bot+alertas).

## Seguridad
- NO se guardaron tokens.
- NO se imprimieron secretos.
- NO se modificó la estrategia ni parámetros de trading.
- NO se tocó MT5 ni se enviaron órdenes.

**Estado Final: AUTO_ALERTS_START_STOP_OK**

**Próximo paso recomendado:** Verificar la recepción de la alerta "ALERTS_LOOP_STARTED" en su Telegram al ejecutar START.
