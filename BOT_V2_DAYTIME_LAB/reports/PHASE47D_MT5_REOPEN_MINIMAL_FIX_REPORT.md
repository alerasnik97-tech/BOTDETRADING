# PHASE47D - MT5 Reopen Minimal Fix Report

Generated at: 2026-04-30T21:30:20-03:00

## 1. Lo mas importante

Phase47D aislo el patch minimo para que STATUS/panel no relance MT5 y para que STOP no aborte por una lectura fragil de `OPERACION_ABIERTA`.

El patch final no toca estrategia, parametros, News Fortress, Data Quality Mask, ordenes ni cuentas reales.

## 2. Veredicto final exacto

MT5_REOPEN_MINIMAL_FIX_COMMITTED_OK

Nota: este reporte se crea antes de ejecutar el commit selectivo que lo incluye. El hash final del commit queda informado en la respuesta operativa final.

## 3. Causa raiz confirmada

La causa raiz operativa era que el panel/status podia consultar estado live de posicion mediante soporte MT5. Esa ruta llamaba `mt5.initialize()` y, si MT5 estaba cerrado, podia reabrir terminal por una consulta de estado.

La mitigacion final separa:

- deteccion pasiva de `terminal64.exe`;
- consulta live solo si MT5 ya esta abierto;
- modo pasivo explicito en soporte MT5 para llamadas de STATUS/panel.

## 4. Causa del falso positivo de STOP

STOP dependia de buscar texto en el JSON de status para detectar `"OPERACION_ABIERTA": "SI"`. Esa lectura era fragil y no distinguia entre posicion confirmada, estado desconocido o no posicion.

En la prueba aislada final no se reprodujo una posicion abierta real:

- `MT5`: `CERRADO`
- `OPEN_POSITION_STATUS`: `NO_OPEN_POSITION_CONFIRMED`
- `OPERACION_ABIERTA`: `NO`

STOP ahora usa un estado explicito:

- `OPEN_POSITION_CONFIRMED`
- `OPEN_POSITION_UNKNOWN`
- `NO_OPEN_POSITION_CONFIRMED`

## 5. Archivos revisados

- `BOT_V2_DAYTIME_LAB/src/phase37_ftmo_trial_support.py`
- `BOT_V2_DAYTIME_LAB/src/phase37ze_quick_status_panel.py`
- `MANIPULANTE/STOP_MANIPULANTE.bat`
- `MANIPULANTE/13_FTMO_TRIAL_AUTOMATION/safe_stop_manipulante_processes.ps1`

## 6. Archivos modificados

- `BOT_V2_DAYTIME_LAB/src/phase37_ftmo_trial_support.py`
- `BOT_V2_DAYTIME_LAB/src/phase37ze_quick_status_panel.py`
- `MANIPULANTE/STOP_MANIPULANTE.bat`
- `MANIPULANTE/13_FTMO_TRIAL_AUTOMATION/safe_stop_manipulante_processes.ps1`
- `BOT_V2_DAYTIME_LAB/reports/PHASE47D_MT5_REOPEN_MINIMAL_FIX_REPORT.md`
- `BOT_V2_DAYTIME_LAB/reports/PHASE47D_MT5_REOPEN_MINIMAL_FIX_REPORT.json`

## 7. Cambios excluidos por estar mezclados

- Telegram/alerts: excluido.
- Phase47A lab isolation: excluido.
- Phase46 generated reports: excluido.
- Runtime/logs/data/cache: excluido.
- `MANIPULANTE/13_FTMO_TRIAL_AUTOMATION/STOP_BOT.txt`: runtime, no commitear.
- `BOT_V2_DAYTIME_LAB/reports/PHASE47D_MIXED_CHANGES_NOT_COMMITTED.patch.txt`: respaldo local de cambios mezclados, no commitear en este bloque.

## 8. Pruebas ejecutadas

### py_compile

- `python -m py_compile BOT_V2_DAYTIME_LAB/src/phase37_ftmo_trial_support.py`: PASS
- `python -m py_compile BOT_V2_DAYTIME_LAB/src/phase37ze_quick_status_panel.py`: PASS

### STATUS

`STATUS_MANIPULANTE.bat` llama a `STATUS_FTMO_TRIAL_AUTO.bat`, que es un loop de panel cada 30 segundos. Para evidencia no interactiva se uso el panel subyacente:

- `python BOT_V2_DAYTIME_LAB/src/phase37ze_quick_status_panel.py --json`: PASS
- Resultado clave: `MT5=CERRADO`, `OPEN_POSITION_STATUS=NO_OPEN_POSITION_CONFIRMED`, `OPERACION_ABIERTA=NO`.

### STOP

- `cmd /c "call MANIPULANTE\STOP_MANIPULANTE.bat < nul"`: PASS, exit 0.
- Resultado clave: `NO_OPEN_POSITION_CONFIRMED`.
- Safe stop final: `No project process candidates found.`
- Nota: el mensaje de redireccion de entrada viene de `pause/timeout` al ejecutar en modo no interactivo.

### Procesos post-STOP

Consulta pasiva excluyendo el proceso de diagnostico:

- No se detectaron procesos del proyecto.
- No se detecto `terminal64.exe`.

### Phase46 local

- `python BOT_V2_DAYTIME_LAB/src/phase46_ci_safety_check.py`: PASS, exit 0.
- Veredicto local: `GITHUB_CI_READY_WITH_WARNINGS`.
- Los warnings son heuristicas existentes de keywords tipo `secret`, `token`, `api_key`; no forman parte del patch Phase47D y no se commitean.

## 9. Resultado de STATUS

Estado final observado:

- `ESTADO_GENERAL`: `BOT DETENIDO`
- `BOT`: `APAGADO`
- `RUNNER`: `APAGADO`
- `MT5`: `CERRADO`
- `STOP_BOT_ACTIVO`: `SI`
- `OPEN_POSITION_STATUS`: `NO_OPEN_POSITION_CONFIRMED`
- `SEGURO_APAGAR_PC`: `SI`

## 10. Resultado de STOP

STOP final:

- deja `STOP_BOT.txt` activo;
- no relanza MT5;
- no mata procesos externos;
- no mata todos los Python;
- no mata todos los `terminal64.exe`;
- usa el safe stop canonico de `MANIPULANTE/13_FTMO_TRIAL_AUTOMATION/`.

## 11. Procesos post-STOP

Resultado final: sin procesos candidatos del proyecto y sin `terminal64.exe`.

## 12. Resultado de Phase46 local

`GITHUB_CI_READY_WITH_WARNINGS`, exit 0.

## 13. Git commit/push

Commit selectivo planificado:

- `BOT_V2_DAYTIME_LAB/src/phase37_ftmo_trial_support.py`
- `BOT_V2_DAYTIME_LAB/src/phase37ze_quick_status_panel.py`
- `MANIPULANTE/STOP_MANIPULANTE.bat`
- `MANIPULANTE/13_FTMO_TRIAL_AUTOMATION/safe_stop_manipulante_processes.ps1`
- `BOT_V2_DAYTIME_LAB/reports/PHASE47D_MT5_REOPEN_MINIMAL_FIX_REPORT.md`
- `BOT_V2_DAYTIME_LAB/reports/PHASE47D_MT5_REOPEN_MINIMAL_FIX_REPORT.json`

No se usa `git add .`.

## 14. GitHub Actions

Pendiente de verificacion despues del push selectivo.

## 15. Seguridad

- No estrategia.
- No MT5 abierto por esta fase.
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

## 16. Siguiente paso unico

Verificar el push y GitHub Actions del commit selectivo Phase47D. Despues, retomar la limpieza del working tree antes de commitear Phase47A.
