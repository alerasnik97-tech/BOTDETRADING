# PHASE37ZF CLEAN ASCII SPANISH UI FIX REPORT

## 1. Lo mas importante

Se reparo START y STATUS para Windows CMD con texto ASCII simple.

El panel ya no muestra emojis, acentos rotos, flechas ni ANSI color.

La deteccion de runner ahora usa procesos reales:

- Solo `python.exe` o `pythonw.exe`.
- Command line debe contener `phase37_ftmo_trial_bot_runner.py`.
- Excluye `phase37ze_quick_status_panel.py`.
- Excluye STATUS.
- Excluye PowerShell temporal y `Get-CimInstance`.

Estado actual detectado durante la reparacion:

- Runners reales vivos: `7028, 19068`.
- Estado correcto mostrado: `VIOLETA`.
- Motivo: hay dos runners previos al fix.
- START no inicio un tercer runner.

## 2. Veredicto final exacto

CLEAN_SPANISH_UI_READY

## 3. Errores corregidos

- Mojibake en BAT: bytes raros, guiones raros y texto roto.
- `chcp 65001` removido de la UI.
- Pipes PowerShell embebidos removidos del BAT de START.
- `for /f` fragil removido del BAT.
- Conteo defectuoso de runner reemplazado por deteccion real de PID.
- STATUS ya no puede contarse a si mismo como runner.
- PowerShell temporal ya no cuenta como runner.
- `PID Runner: 5` queda eliminado como caso valido.

## 4. START final

Archivo:

`MANIPULANTE/START_MANIPULANTE.bat`

Delegacion limpia a:

`MANIPULANTE/13_FTMO_TRIAL_AUTOMATION/START_FTMO_TRIAL_AUTO.bat`

Comportamiento:

- Si hay runner: muestra `ESTADO: BOT YA ESTA PRENDIDO`.
- No inicia otro bot.
- No crea duplicados.
- Si no hay runner: muestra `ESTADO: INICIANDO BOT`.
- Valida FTMO Demo antes de iniciar.
- Si FTMO Demo no esta confirmado: bloquea fail-closed.

## 5. STATUS final

Archivo:

`MANIPULANTE/STATUS_MANIPULANTE.bat`

Delegacion limpia a:

`MANIPULANTE/13_FTMO_TRIAL_AUTOMATION/STATUS_FTMO_TRIAL_AUTO.bat`

Panel:

- `ESTADO GENERAL`
- `CUENTA`
- `RUNNER`
- `PID RUNNER`
- `MT5`
- `NEWS`
- `ULTIMA DECISION`
- `OPERACION ABIERTA`
- `SEGURO APAGAR PC`
- `ULTIMA ACTUALIZACION`

Estados:

- `VERDE`
- `AMARILLO`
- `ROJO`
- `CRITICO`
- `VIOLETA`

## 6. Runner detection

Implementado en:

`BOT_V2_DAYTIME_LAB/src/phase37ze_quick_status_panel.py`

Tambien se endurecio:

`BOT_V2_DAYTIME_LAB/src/phase37_ftmo_trial_bot_runner.py`

Resultado:

- 0 runners validos = `RUNNER: APAGADO`, `ESTADO GENERAL: ROJO`.
- 1 runner valido = `RUNNER: ACTIVO`.
- Mas de 1 runner valido = `ESTADO GENERAL: VIOLETA`.

El lock del runner ahora revisa otros runners reales antes de aceptar inicio.

## 7. Tests

Pruebas ejecutadas:

- `python -m py_compile` sobre panel y runner: OK.
- Scan ASCII en 6 archivos reparados: OK, sin bytes mayores a 127.
- STATUS con runners activos reales: OK, detecto `7028, 19068`.
- STATUS con runner apagado por test logico: OK, `ROJO / APAGADO / ---`.
- STATUS con 1 runner por test logico: OK, `VERDE / ACTIVO / 1234`.
- STATUS con duplicados por test logico: OK, `VIOLETA / 1234, 5678`.
- START con runner activo real: OK, no inicio otro.
- START tocado 3 veces: OK, PIDs antes y despues iguales.
- Lock del runner con runners activos: OK, devuelve `False`.
- Dry-run once del runner nuevo: OK, `order_sent=false`.

Nota operativa:

Los dos runners vivos fueron iniciados antes del fix. Mientras sigan vivos pueden reescribir `quick_status.txt` con formato viejo. El runner nuevo ya escribe el formato Phase37ZF requerido.

## 8. Seguridad

- No se envio orden real.
- No se envio orden de prueba.
- No se toco Exness.
- No se cambio MANIPULANTE.
- No se cambio TP.
- No se cambio BE.
- No se cambio BF.
- No se borraron logs.
- No se agregaron secretos.
- No se abrio live real.

El preflight legacy del proyecto confirmo root correcto, pero marco faltantes legacy no relacionados con esta UI:

- `CURRENT_STATE_OF_LAB.md`
- `EURUSD_MANUAL_EDGE_FINAL_DECISION.md`
- `ZIP_PACKAGING_AUDIT.md`

La reparacion se mantuvo dentro del root canonico.

## 9. ZIP/Git

ZIP actualizado:

- Archivo: `000_PARA_CHATGPT.zip`.
- Entradas: `234`.
- `testzip`: `None`.
- Duplicados internos: `0`.

Archivos Phase37ZF dentro del zip:

- `MANIPULANTE/START_MANIPULANTE.bat`
- `MANIPULANTE/STATUS_MANIPULANTE.bat`
- `MANIPULANTE/13_FTMO_TRIAL_AUTOMATION/START_FTMO_TRIAL_AUTO.bat`
- `MANIPULANTE/13_FTMO_TRIAL_AUTOMATION/STATUS_FTMO_TRIAL_AUTO.bat`
- `BOT_V2_DAYTIME_LAB/src/phase37ze_quick_status_panel.py`
- `BOT_V2_DAYTIME_LAB/src/phase37_ftmo_trial_bot_runner.py`
- `BOT_V2_DAYTIME_LAB/reports/PHASE37ZF_CLEAN_ASCII_SPANISH_UI_FIX_REPORT.md`

Git:

- Commit selectivo: `Phase37ZF clean Spanish ASCII UI and runner detection fix`.
- Push destino: `origin main`.

## 10. Siguiente paso unico

Cerrar los dos runners previos al fix y volver a iniciar con `START_MANIPULANTE.bat` para quedar con un solo PID nuevo.
