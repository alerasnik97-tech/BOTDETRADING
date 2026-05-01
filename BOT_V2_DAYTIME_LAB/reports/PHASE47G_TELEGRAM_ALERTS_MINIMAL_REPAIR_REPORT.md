# PHASE47G - Telegram Alerts Minimal Repair Report

Generated at: 2026-04-30T22:12:00-03:00

## 1. Lo mas importante

Se reparo el bloque minimo de Telegram alerts:

- `STATUS_ALERTS_LOOP_MANIPULANTE.bat` ya no depende de `phase37ze_quick_status_panel.py`.
- `STOP_ALERTS_LOOP_MANIPULANTE.bat` delega en una funcion Python que valida PID y command line antes de detener el proceso.
- `START_ALERTS_LOOP_MANIPULANTE.bat` valida estado previo y evita duplicados.
- `phase45_telegram_sender.py` no imprime token completo ni devuelve la respuesta completa de Telegram en el test.
- No se toco estrategia, MT5, ordenes, Phase47A ni LAB_STRATEGIES.

## 2. Veredicto final exacto

TELEGRAM_ALERTS_REPAIR_COMMITTED_OK

Nota: este reporte se crea antes del commit selectivo que lo contiene. El hash final queda informado en la respuesta operativa final.

## 3. Causa raiz de STATUS_ALERTS_LOOP

La causa raiz era una dependencia rota: `STATUS_ALERTS_LOOP_MANIPULANTE.bat` intentaba importar `get_alerts_status` desde `phase37ze_quick_status_panel.py`, pero esa funcion no existe en el estado actual posterior a Phase47D.

Reparacion aplicada:

- Se agrego `get_alerts_status()` en `phase45_run_alert_check.py`.
- El estado es read-only.
- Valida PID y command line.
- Lee heartbeat si existe.
- No inicia loop.
- No detiene procesos.
- No abre MT5.
- No conecta Telegram.
- No envia mensajes.

## 4. Reparacion de STOP_ALERTS_LOOP

`STOP_ALERTS_LOOP_MANIPULANTE.bat` ya no usa `taskkill` directo contra cualquier PID del archivo.

Ahora llama:

`python BOT_V2_DAYTIME_LAB/src/phase45_run_alert_check.py --stop-loop`

La funcion valida:

- PID existente.
- Command line contiene el proyecto.
- Command line contiene `phase45_run_alert_check.py`.
- Command line contiene `--loop`.

Si el PID no corresponde, devuelve `ALERTS_PID_OWNER_MISMATCH` y no mata nada.

## 5. Decision sobre START/STOP/STATUS_MANIPULANTE

- `MANIPULANTE/START_MANIPULANTE.bat`: entra al commit. Su cambio solo arranca el helper de alertas antes del bot.
- `MANIPULANTE/STOP_MANIPULANTE.bat`: entra al commit. Su cambio solo llama al stop seguro de alertas al final del STOP seguro ya reparado en Phase47D.
- `MANIPULANTE/STATUS_MANIPULANTE.bat`: no entra al commit. No tiene cambio local en esta fase; mostrar alertas dentro del panel principal requeriria tocar el panel tecnico fuera del alcance minimo de Phase47G.

## 6. Validacion de secrets

- No se detecto token hardcodeado.
- No se detecto chat_id real hardcodeado.
- No se commitea `.env`.
- No se commitea `alerts_config.local.json`.
- `phase45_telegram_sender.py` lee `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `BOT_TELEGRAM_TOKEN`, `BOT_TELEGRAM_CHAT_ID`.
- `--diag` no imprime token completo ni sufijo.
- `--send-test --dry-run` no envia Telegram real.
- Errores sanitizan token.

## 7. Validacion de spam/dedupe

- Frecuencia de loop: 60 segundos.
- Dedupe persistente: `AlertState`.
- Cooldown general: 10 minutos.
- Cooldown critical: 1 minuto.
- `ALERTS_LOOP_STARTED` no se envia por defecto.
- `START_ALERTS_LOOP_MANIPULANTE.bat` no manda test real ni `--once`; solo arranca el loop.
- En prueba dry-run: segundo START reporto `ALERTS_LOOP_ALREADY_RUNNING`.

## 8. Pruebas ejecutadas

- `python -m py_compile BOT_V2_DAYTIME_LAB/src/phase45_run_alert_check.py`: PASS.
- `python -m py_compile BOT_V2_DAYTIME_LAB/src/phase45_telegram_sender.py`: PASS.
- `python BOT_V2_DAYTIME_LAB/src/phase45_telegram_sender.py --diag`: PASS, sin token completo.
- `python BOT_V2_DAYTIME_LAB/src/phase45_telegram_sender.py --send-test --dry-run`: PASS.
- `python BOT_V2_DAYTIME_LAB/src/phase45_run_alert_check.py --once --dry-run`: PASS, detecto 1 alerta, envio 0.
- `MANIPULANTE\16_OBSERVABILITY\alerts\STATUS_ALERTS_LOOP_MANIPULANTE.bat`: PASS.
- `MANIPULANTE\16_OBSERVABILITY\alerts\START_ALERTS_LOOP_MANIPULANTE.bat` con `MANIPULANTE_ALERTS_DRY_RUN=1`: PASS.
- Segundo START en dry-run: PASS, `ALERTS_LOOP_ALREADY_RUNNING`.
- `MANIPULANTE\16_OBSERVABILITY\alerts\STOP_ALERTS_LOOP_MANIPULANTE.bat`: PASS.
- Estado final: `ALERTS_STOPPED`.
- Proceso post-test: sin loop `phase45_run_alert_check.py --loop` y sin `terminal64.exe`.
- `python BOT_V2_DAYTIME_LAB/src/phase46_ci_safety_check.py`: PASS con warnings.

## 9. Archivos commiteados

Commit selectivo planificado:

- `BOT_V2_DAYTIME_LAB/src/phase45_run_alert_check.py`
- `BOT_V2_DAYTIME_LAB/src/phase45_telegram_sender.py`
- `MANIPULANTE/16_OBSERVABILITY/alerts/README_ALERTS.md`
- `MANIPULANTE/16_OBSERVABILITY/alerts/alerts_config.example.json`
- `MANIPULANTE/16_OBSERVABILITY/alerts/START_ALERTS_LOOP_MANIPULANTE.bat`
- `MANIPULANTE/16_OBSERVABILITY/alerts/STATUS_ALERTS_LOOP_MANIPULANTE.bat`
- `MANIPULANTE/16_OBSERVABILITY/alerts/STOP_ALERTS_LOOP_MANIPULANTE.bat`
- `MANIPULANTE/START_MANIPULANTE.bat`
- `MANIPULANTE/STOP_MANIPULANTE.bat`
- `BOT_V2_DAYTIME_LAB/reports/PHASE47G_TELEGRAM_ALERTS_MINIMAL_REPAIR_REPORT.md`
- `BOT_V2_DAYTIME_LAB/reports/PHASE47G_TELEGRAM_ALERTS_MINIMAL_REPAIR_REPORT.json`

## 10. Archivos excluidos

- `MANIPULANTE/STATUS_MANIPULANTE.bat`: sin cambio en esta fase.
- `reports/MANIPULANTE_AUTO_ALERTS_START_STOP_REPORT.md`: excluido por estar desactualizado frente a Phase47G.
- Phase47A y `LAB_STRATEGIES/**`: excluidos.
- Runtime/logs/data: excluidos.
- `.env`, tokens, secrets, credentials: excluidos.
- Backups `.bak_*`: excluidos.
- ZIP canonico: excluido.
- MT5/order-router/MQL5: excluidos.
- Unknowns: excluidos.
- Phase46 generated reports: excluidos.

## 11. Resultado de Phase46 local

`GITHUB_CI_READY_WITH_WARNINGS`, exit 0.

Los warnings son heuristicas existentes por keywords tipo `token`, `secret`, `api_key`; no corresponden a un secreto expuesto por Phase47G.

## 12. Commit y push

Pendiente del commit selectivo posterior a este reporte.

## 13. GitHub Actions

Pendiente de verificar despues del push.

## 14. Seguridad

- No estrategia.
- No MT5.
- No ordenes.
- No real.
- No Exness.
- No TP/BE/BF.
- No riesgo.
- No horarios.
- No secrets.
- No `git add .`.
- No `git reset --hard`.
- No `git clean -fd`.
- No procesos externos matados.

## 15. Siguiente paso unico

Verificar el commit/push de Phase47G y GitHub Actions. Luego volver al ordenamiento del working tree antes de Phase47A.
