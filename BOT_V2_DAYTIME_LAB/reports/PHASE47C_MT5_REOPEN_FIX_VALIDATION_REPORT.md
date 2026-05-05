# PHASE47C MT5 REOPEN FIX VALIDATION REPORT

## 1. Lo mas importante

Se valido el bloque candidato de MT5 reopen de forma conservadora. El fix conceptual es correcto en dos puntos:

- `phase37_ftmo_trial_support.py` evita llamar `mt5.initialize()` si `terminal64.exe` no esta corriendo.
- `phase37ze_quick_status_panel.py` no consulta posiciones live si MT5 esta cerrado.

Pero el bloque no es seguro para commit todavia: `phase37ze_quick_status_panel.py` mezcla cambios de Telegram/status panel, `STOP_MANIPULANTE.bat` mezcla el fix MT5 con `STOP_ALERTS_LOOP`, y `STOP` llama un script nuevo bajo `MANIPULANTE/13_FTMO_TRIAL_AUTOMATION/` mientras el alcance solicitado mencionaba `MANIPULANTE/16_OBSERVABILITY/`.

## 2. Veredicto final exacto

`MT5_REOPEN_FIX_OUT_OF_SCOPE_CHANGES_DETECTED`

No se hizo commit ni push.

## 3. Causa raiz validada

La causa raiz probable es que una ruta de status/panel podia terminar llamando funciones que inicializan MT5. En MetaTrader5 Python, `mt5.initialize()` puede abrir o reconectar el terminal si se invoca cuando MT5 esta cerrado.

La mitigacion correcta es pasiva:

- consultar primero procesos `terminal64.exe` con `tasklist`;
- no llamar `mt5.initialize()` cuando MT5 no esta corriendo;
- no consultar posiciones live desde status si `MT5=CERRADO`;
- dejar `STOP_BOT.txt` activo hasta START explicito;
- cortar solo procesos del proyecto, no todos los Python ni todos los terminales.

## 4. Archivos revisados

- `BOT_V2_DAYTIME_LAB/src/phase37_ftmo_trial_support.py`
- `BOT_V2_DAYTIME_LAB/src/phase37ze_quick_status_panel.py`
- `MANIPULANTE/STOP_MANIPULANTE.bat`
- `MANIPULANTE/16_OBSERVABILITY/safe_stop_manipulante_processes.ps1` (no existe)
- `MANIPULANTE/13_FTMO_TRIAL_AUTOMATION/safe_stop_manipulante_processes.ps1` (existe, untracked, llamado por STOP)
- `BOT_V2_DAYTIME_LAB/reports/MANIPULANTE_MT5_REOPEN_INCIDENT_REPORT.md` (no existe)

## 5. Cambios confirmados como seguros

Seguro como idea tecnica, pero no commiteado:

- `phase37_ftmo_trial_support.py`: agrega guardia previa por `terminal64.exe` antes de `mt5.initialize()`.
- `phase37ze_quick_status_panel.py`: `build_status()` calcula `mt5_running = (mt5 == "ABIERTO")` y pasa esa condicion a `_live_position_open()`.

No seguro para commit como bloque:

- `phase37ze_quick_status_panel.py` tambien agrega estado de Telegram alerts, env vars de Telegram y cambios de render.
- `STOP_MANIPULANTE.bat` tambien llama `STOP_ALERTS_LOOP_MANIPULANTE.bat`.
- El script de limpieza real esta en una ruta distinta a la pedida para commit.

## 6. Pruebas ejecutadas

### py_compile

```text
python -m py_compile BOT_V2_DAYTIME_LAB/src/phase37_ftmo_trial_support.py
PHASE37_SUPPORT_PY_COMPILE_EXIT=0

python -m py_compile BOT_V2_DAYTIME_LAB/src/phase37ze_quick_status_panel.py
PHASE37ZE_PANEL_PY_COMPILE_EXIT=0
```

### Phase46 CI local

No ejecutado. Motivo: el commit se aborto antes por out-of-scope; ejecutar Phase46 reescribe reportes y agregaria ruido al working tree sin habilitar el commit.

### STATUS

No se ejecuto el BAT completo porque `STATUS_MANIPULANTE.bat` es un loop infinito de 30 segundos. Se ejecuto el panel subyacente una vez:

```text
python BOT_V2_DAYTIME_LAB/src/phase37ze_quick_status_panel.py --json
```

Resultado relevante:

```text
MT5: CERRADO
STOP_BOT_ACTIVO: SI
OPERACION_ABIERTA: NO
SEGURO_APAGAR_PC: SI
```

No abrio MT5.

### STOP

Se ejecuto:

```text
cmd /c "echo. | call MANIPULANTE\STOP_MANIPULANTE.bat"
```

Resultado:

```text
exit_code: 1
PELIGRO - HAY OPERACION ABIERTA
```

STOP no llego a la limpieza profunda. Ademas, por el `pause` del BAT y la ejecucion con pipe, CMD emitio salida corrupta posterior. No se repitio la ejecucion para no generar mas ruido.

### Procesos post-STOP

Consulta pasiva corregida:

```powershell
Get-CimInstance Win32_Process | Where-Object { ... }
```

Resultado: sin procesos coincidentes impresos para `BOT DE TRADING ultimo`, `MANIPULANTE` o `terminal64`.

## 7. Git

- Archivos agregados: ninguno.
- Commit: no creado.
- Push: no ejecutado.
- GitHub Actions: no verificado porque no hubo push.
- `git diff --cached --name-status`: vacio.

## 8. Archivos excluidos

No se agregaron ni commitearon:

- `LAB_STRATEGIES/`
- `PROJECT_ZONES_AND_BRANCHING_RULES.md`
- reportes Phase47A/Phase47B
- Telegram/alerts como bloque separado
- Phase46 generated reports
- runtime/logs/jsonl/db/csv
- `mt5_demo_executor_lab/mt5_order_router.py`
- MQL5 calendar
- configs live
- secrets
- `.env`
- backups `.bak`
- ZIP canonico
- data pesada
- unknowns

## 9. Seguridad

- No se cambio estrategia.
- No se abrio MT5.
- No se conecto a MT5 live.
- No se enviaron ordenes.
- No se cerraron ordenes.
- No se modificaron ordenes.
- No se toco real.
- No se toco Exness.
- No se cambio TP/BE/BF.
- No se cambio riesgo.
- No se cambiaron horarios.
- No se tocaron secrets.
- No se uso `git add .`.
- No se uso `git commit -a`.
- No se uso `git reset --hard`.
- No se uso `git clean`.

## 10. Siguiente paso unico

Aislar el fix MT5 reopen en un patch minimo antes de commitear: separar Telegram/status-panel del archivo `phase37ze_quick_status_panel.py`, decidir la ruta canonica del `safe_stop_manipulante_processes.ps1`, y corregir/validar `STOP_MANIPULANTE.bat` para que no aborte falsamente por operacion abierta.
