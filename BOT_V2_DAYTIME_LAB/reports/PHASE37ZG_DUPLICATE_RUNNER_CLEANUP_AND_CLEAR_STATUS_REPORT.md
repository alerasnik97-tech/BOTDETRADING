# PHASE37ZG DUPLICATE RUNNER CLEANUP AND CLEAR STATUS REPORT

## 1. Lo mas importante

Se limpiaron los runners duplicados antiguos y se inicio un unico runner limpio.

El panel ya no usa estados por color.

El estado final visible es:

`ESTADO GENERAL: OK - BOT ACTIVO`

Runner final unico:

`11248`

## 2. Veredicto final exacto

SINGLE_RUNNER_AND_CLEAR_STATUS_READY

## 3. Cuenta

- Cuenta: `FTMO-Demo / DEMO`
- Company: `FTMO Global Markets Ltd`
- Exness detectado: `NO`
- Real detectado: `NO`
- Estado cuenta: `FTMO_DEMO_TRIAL_CONFIRMED`

## 4. Operacion abierta si/no

`NO`

Evidencia:

- Position state: `FLAT`
- Positions: `[]`
- STATUS final: `OPERACION ABIERTA: NO`

## 5. Runners antes

Antes de la limpieza:

- `7028`
- `19068`

Ambos eran `python.exe` y el command line contenia:

`phase37_ftmo_trial_bot_runner.py`

## 6. Runners cerrados

Runners cerrados:

- `7028`
- `19068`

Condicion aplicada antes de cerrar:

- PID existe.
- Proceso es `python.exe` o `pythonw.exe`.
- Command line contiene `phase37_ftmo_trial_bot_runner.py`.

No se mato MT5.

## 7. Runner final unico

Despues de limpiar:

- Runner count: `0`
- `runner.lock` stale: eliminado.

Luego se inicio por:

`MANIPULANTE/START_MANIPULANTE.bat`

Runner final:

- PID: `11248`
- Proceso: `python.exe`
- Python: `C:\Users\alera\AppData\Local\Python\pythoncore-3.14-64\python.exe`
- Command line contiene `phase37_ftmo_trial_bot_runner.py`

## 8. STATUS final

STATUS final:

- `ESTADO GENERAL: OK - BOT ACTIVO`
- `CUENTA: FTMO-Demo / DEMO`
- `RUNNER: ACTIVO`
- `PID RUNNER: 11248`
- `MT5: ABIERTO`
- `NEWS: ALLOW`
- `ULTIMA DECISION: NO_TRADE`
- `OPERACION ABIERTA: NO`
- `SEGURO APAGAR PC: NO`

No quedo `DUPLICADO`.

## 9. Nuevas etiquetas de estado

Etiquetas finales:

- `OK - BOT ACTIVO`
- `BLOQUEADO - BOT ACTIVO PERO NO OPERA`
- `ERROR - BOT APAGADO`
- `PELIGRO - NO APAGAR PC`
- `DUPLICADO - LIMPIAR RUNNERS`

El panel mantiene compatibilidad con valores viejos:

- `VERDE` pasa a `OK - BOT ACTIVO`
- `AMARILLO` pasa a `BLOQUEADO - BOT ACTIVO PERO NO OPERA`
- `ROJO` pasa a `ERROR - BOT APAGADO`
- `CRITICO` pasa a `PELIGRO - NO APAGAR PC`
- `VIOLETA` pasa a `DUPLICADO - LIMPIAR RUNNERS`

## 10. Dry-run

Dry-run ejecutado:

`python BOT_V2_DAYTIME_LAB/src/phase37_ftmo_trial_bot_runner.py --ftmo-trial --dry-run --once --risk 0.005 --no-real --i-understand-demo-automation`

Resultado:

- `final_decision`: `DRY_RUN_NO_SIGNAL`
- `order_sent`: `false`
- `account_gate`: `FTMO_DEMO_TRIAL_CONFIRMED`
- `real_money_gate`: `REAL_BLOCKED`
- `position_state`: `FLAT`

## 11. Seguridad

- No se envio orden real.
- No se envio orden de prueba.
- No se toco Exness.
- No se cambio MANIPULANTE.
- No se cambio TP.
- No se cambio BE.
- No se cambio BF.
- No se mato MT5.
- No se borraron logs.
- No se agregaron secretos.
- No se uso `git add .`.
- No se hizo force push.

## 12. ZIP/Git

ZIP actualizado:

- Archivo: `000_PARA_CHATGPT.zip`
- `testzip`: `None`
- Duplicados internos: `0`
- Archivos Phase37ZG incluidos: `SI`

Git:

- Commit selectivo: `Phase37ZG cleanup duplicate runners and clear status labels`
- Push destino: `origin main`

## 13. Siguiente paso unico

Usar `STATUS_MANIPULANTE.bat` y confirmar que se mantiene:

`ESTADO GENERAL: OK - BOT ACTIVO`
