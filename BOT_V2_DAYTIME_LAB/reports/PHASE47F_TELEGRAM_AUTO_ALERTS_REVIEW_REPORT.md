# PHASE47F - Telegram Auto Alerts Review Report

Generated at: 2026-04-30T21:59:00-03:00

## 1. Lo mas importante

El bloque Telegram/Auto Alerts no esta listo para commit selectivo.

La base compila y el sender lee variables de entorno nuevas, pero hay fallas operativas que impiden cerrar Phase47F:

- `STATUS_ALERTS_LOOP_MANIPULANTE.bat` falla porque importa `get_alerts_status` desde `phase37ze_quick_status_panel.py`, funcion que ya no existe en el estado actual despues de Phase47D.
- `STOP_ALERTS_LOOP_MANIPULANTE.bat` se basa en PID file y usa `taskkill /PID` sin validar que el PID siga siendo el loop de alertas del proyecto.
- `STOP_MANIPULANTE.bat` y `STATUS_MANIPULANTE.bat` no tienen integracion actual de alertas en los cambios commiteados despues de Phase47D.
- `phase45_telegram_sender.py` usa `parse_mode=HTML` por defecto y devuelve `response.json()` completo en `send-test`, lo que no es la superficie mas conservadora para diagnostico.

## 2. Veredicto final exacto

TELEGRAM_AUTO_ALERTS_REQUIRES_REPAIR

## 3. Archivos revisados

- `BOT_V2_DAYTIME_LAB/src/phase45_run_alert_check.py`
- `BOT_V2_DAYTIME_LAB/src/phase45_telegram_sender.py`
- `MANIPULANTE/16_OBSERVABILITY/alerts/README_ALERTS.md`
- `MANIPULANTE/16_OBSERVABILITY/alerts/alerts_config.example.json`
- `MANIPULANTE/START_MANIPULANTE.bat`
- `MANIPULANTE/16_OBSERVABILITY/alerts/START_ALERTS_LOOP_MANIPULANTE.bat`
- `MANIPULANTE/16_OBSERVABILITY/alerts/STATUS_ALERTS_LOOP_MANIPULANTE.bat`
- `MANIPULANTE/16_OBSERVABILITY/alerts/STOP_ALERTS_LOOP_MANIPULANTE.bat`
- `reports/MANIPULANTE_AUTO_ALERTS_START_STOP_REPORT.md`

Lectura minima adicional:

- `MANIPULANTE/STOP_MANIPULANTE.bat`
- `MANIPULANTE/STATUS_MANIPULANTE.bat`
- `MANIPULANTE/13_FTMO_TRIAL_AUTOMATION/STATUS_FTMO_TRIAL_AUTO.bat`
- `BOT_V2_DAYTIME_LAB/src/phase37ze_quick_status_panel.py`

## 4. Archivos modificados/commiteados

No hubo commit.

Solo se crearon reportes Phase47F:

- `BOT_V2_DAYTIME_LAB/reports/PHASE47F_TELEGRAM_AUTO_ALERTS_REVIEW_REPORT.md`
- `BOT_V2_DAYTIME_LAB/reports/PHASE47F_TELEGRAM_AUTO_ALERTS_REVIEW_REPORT.json`

## 5. Archivos excluidos

- Phase47A: excluido.
- LAB_STRATEGIES: excluido.
- Runtime/logs/data: excluido.
- Backups `.bak_*`: excluidos.
- Phase46 generated reports: excluidos.
- MT5/order-router/MQL5: excluidos.
- Unknowns: excluidos.

## 6. Validacion de secrets

Resultado: no se detecto token hardcodeado ni chat_id real hardcodeado en los candidatos revisados por patron.

Observaciones:

- `phase45_telegram_sender.py` lee `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, y opcionalmente `BOT_TELEGRAM_TOKEN`, `BOT_TELEGRAM_CHAT_ID`.
- El diagnostico muestra presencia y metadata enmascarada, no token completo.
- El `send-test` real no se ejecuto para evitar exponer respuesta de Telegram o generar mensaje innecesario; se ejecuto `--send-test --dry-run`.
- Hay que reparar la salida de `send-test` antes de commit para no imprimir `response.json()` completo.

## 7. Validacion START/STOP/STATUS

START:

- `START_MANIPULANTE.bat` agrega llamada a `START_ALERTS_LOOP_MANIPULANTE.bat`.
- No cambia estrategia ni parametros.

STOP:

- `STOP_MANIPULANTE.bat` actual no contiene integracion de `STOP_ALERTS_LOOP_MANIPULANTE.bat`.
- `STOP_ALERTS_LOOP_MANIPULANTE.bat` debe validar CommandLine antes de matar PID.

STATUS:

- `STATUS_MANIPULANTE.bat` actual no muestra estado de alertas.
- `STATUS_ALERTS_LOOP_MANIPULANTE.bat` falla por import inexistente.

## 8. Validacion loop/dedupe/spam

- Frecuencia declarada del loop: 60 segundos.
- `phase45_alert_state.py` tiene dedupe/cooldown persistente por `dedup_key`.
- Cooldown efectivo: 10 minutos general, 1 minuto para `CRITICAL`.
- `--once --dry-run` detecto 1 alerta y envio 0 mensajes.
- El mensaje `ALERTS_LOOP_STARTED` se dispara al iniciar loop y no esta deduplicado en el patch actual.
- No se valido start real porque puede enviar Telegram real y el bloque todavia falla en STATUS/STOP.

## 9. Pruebas ejecutadas

- `python -m py_compile BOT_V2_DAYTIME_LAB/src/phase45_run_alert_check.py`: PASS.
- `python -m py_compile BOT_V2_DAYTIME_LAB/src/phase45_telegram_sender.py`: PASS.
- `python BOT_V2_DAYTIME_LAB/src/phase45_telegram_sender.py --diag`: PASS, salida no incluida en este reporte.
- `python BOT_V2_DAYTIME_LAB/src/phase45_telegram_sender.py --send-test --dry-run`: PASS.
- `python BOT_V2_DAYTIME_LAB/src/phase45_run_alert_check.py --once --dry-run`: PASS, alertas detectadas 1, enviadas 0.
- `MANIPULANTE\16_OBSERVABILITY\alerts\STATUS_ALERTS_LOOP_MANIPULANTE.bat`: FAIL por import inexistente.
- `MANIPULANTE\16_OBSERVABILITY\alerts\STOP_ALERTS_LOOP_MANIPULANTE.bat`: ejecutado sin PID; no mato procesos, pero la logica requiere reparacion.
- `python BOT_V2_DAYTIME_LAB/src/phase46_ci_safety_check.py`: PASS con warnings.

## 10. Resultado de Phase46 local

`GITHUB_CI_READY_WITH_WARNINGS`, exit 0.

Los warnings son heuristicas existentes de keywords tipo `token`, `secret`, `api_key`; no se commitea nada en Phase47F.

## 11. Git commit/push

- Commit: no realizado.
- Push: no realizado.
- GitHub Actions: no aplica porque no hubo push.

## 12. Seguridad

- No se toco estrategia.
- No se abrio MT5.
- No se conecto MT5.
- No se enviaron ordenes.
- No se tocaron cuentas reales.
- No se toco Exness.
- No se tocaron TP/BE/BF.
- No se toco riesgo.
- No se tocaron horarios.
- No se uso `git add .`.
- No se uso reset hard.
- No se uso clean.

## 13. Siguiente paso unico

Autorizar una reparacion corta de Telegram/alerts que incluya:

1. `STATUS_ALERTS_LOOP_MANIPULANTE.bat` sin dependencia de `phase37ze_quick_status_panel.py`.
2. `STOP_ALERTS_LOOP_MANIPULANTE.bat` validando CommandLine antes de `taskkill`.
3. Integracion explicita de STOP/STATUS principales o ajustar el alcance para commitear solo helpers.
4. Sanitizar `phase45_telegram_sender.py` para no usar `parse_mode=HTML` por defecto ni imprimir `response.json()` completo.
