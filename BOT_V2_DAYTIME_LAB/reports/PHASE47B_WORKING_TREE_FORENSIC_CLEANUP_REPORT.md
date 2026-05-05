# PHASE47B WORKING TREE FORENSIC CLEANUP REPORT

## 1. Lo mas importante

El working tree no esta seguro para commit. Hay una mezcla de Phase47A, fixes operativos protegidos, alertas Telegram, reportes generados, runtime/logs/data y cambios desconocidos. No se hizo `git add`, commit, push, reset, clean ni stash.

## 2. Veredicto exacto

`WORKING_TREE_NOT_SAFE_TO_COMMIT`

Motivo: existen cambios protegidos y no relacionados en `MANIPULANTE/`, Phase37/45, MT5/live, runtime y miles de outputs/data. Phase47A esta armado, pero debe esperar.

## 3. Estado Git actual

- Branch: `main`
- Archivos tracked modificados/borrados: `30`
- Archivos untracked expandidos: `2338`
- Total expandido pre-reporte Phase47B: `2368`
- `git status --short` original tenia directorios compactados; `git ls-files -o --exclude-standard` expandio los untracked reales.

Ultimos commits:

```text
8d4c263 Fix Phase46 GitHub Actions CI failure
d311d1c Phase46 GitHub CI safety tests for Manipulante
ffde8ec Phase45B runner recovery and stale lock fix
b446885 Phase45 Manipulante free alerts Telegram email foundation
54b5dcf Phase44 Manipulante observability foundation
```

## 4. Clasificacion por categoria

```json
{
  "A_PHASE47A_LAB_ISOLATION": 12,
  "B_MT5_REOPEN_FIX_CANDIDATE": 4,
  "C_PHASE45_TELEGRAM_AUTO_ALERTS": 9,
  "D_PHASE46_CI_GENERATED_REPORTS": 4,
  "E_GENERATED_RUNTIME_DO_NOT_COMMIT": 2218,
  "F_SECRET_OR_LOCAL_CONFIG_DO_NOT_COMMIT": 1,
  "G_UNKNOWN_REQUIRES_MANUAL_REVIEW": 80,
  "H_PROTECTED_CHANGE_REQUIRES_DECISION": 40
}
```

### A) PHASE47A_LAB_ISOLATION

Cantidad: 12

Archivos:

- `BOT_V2_DAYTIME_LAB/reports/PHASE47A_LAB_ISOLATION_AND_MANIPULANTE_PROTECTION_REPORT.json`
- `BOT_V2_DAYTIME_LAB/reports/PHASE47A_LAB_ISOLATION_AND_MANIPULANTE_PROTECTION_REPORT.md`
- `BOT_V2_DAYTIME_LAB/src/phase47a_lab_isolation_guard.py`
- `LAB_STRATEGIES/EURUSD_DAYTIME/README_EURUSD_DAYTIME_LAB.md`
- `LAB_STRATEGIES/EURUSD_DAYTIME/_templates/strategy_config_template.json`
- `LAB_STRATEGIES/EURUSD_DAYTIME/_templates/strategy_research_template.md`
- `LAB_STRATEGIES/EURUSD_DAYTIME/correlation/.gitkeep`
- `LAB_STRATEGIES/EURUSD_DAYTIME/reports/.gitkeep`
- `LAB_STRATEGIES/EURUSD_DAYTIME/shared/.gitkeep`
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/.gitkeep`
- `LAB_STRATEGIES/README_LAB_STRATEGIES.md`
- `PROJECT_ZONES_AND_BRANCHING_RULES.md`

Estado: listos como bloque conceptual, pero bloqueados hasta resolver cambios protegidos previos.

### B) MT5_REOPEN_FIX_CANDIDATE

Cantidad: 4

Archivos candidatos:

- `BOT_V2_DAYTIME_LAB/src/phase37_ftmo_trial_support.py`
- `BOT_V2_DAYTIME_LAB/src/phase37ze_quick_status_panel.py`
- `MANIPULANTE/STOP_MANIPULANTE.bat`
- `MANIPULANTE/13_FTMO_TRIAL_AUTOMATION/safe_stop_manipulante_processes.ps1`

Evidencia leida: `phase37_ftmo_trial_support.py` evita inicializar MT5 si `terminal64.exe` no esta corriendo; `STOP_MANIPULANTE.bat` llama limpieza profunda; `safe_stop_manipulante_processes.ps1` mata procesos del proyecto. Requiere validacion manual antes de commit.

### C) PHASE45_TELEGRAM_AUTO_ALERTS

Cantidad: 9

Archivos candidatos:

- `BOT_V2_DAYTIME_LAB/src/phase45_run_alert_check.py`
- `BOT_V2_DAYTIME_LAB/src/phase45_telegram_sender.py`
- `MANIPULANTE/16_OBSERVABILITY/alerts/README_ALERTS.md`
- `MANIPULANTE/16_OBSERVABILITY/alerts/alerts_config.example.json`
- `MANIPULANTE/START_MANIPULANTE.bat`
- `MANIPULANTE/16_OBSERVABILITY/alerts/START_ALERTS_LOOP_MANIPULANTE.bat`
- `MANIPULANTE/16_OBSERVABILITY/alerts/STATUS_ALERTS_LOOP_MANIPULANTE.bat`
- `MANIPULANTE/16_OBSERVABILITY/alerts/STOP_ALERTS_LOOP_MANIPULANTE.bat`
- `reports/MANIPULANTE_AUTO_ALERTS_START_STOP_REPORT.md`

Evidencia leida: agrega lock/heartbeat al loop, variables de entorno nuevas para Telegram, scripts START/STOP/STATUS de alertas y wiring en `START_MANIPULANTE.bat`. No incluir runtime ni backups.

### D) PHASE46_CI_GENERATED_REPORTS

Cantidad: 4

Archivos:

- `BOT_V2_DAYTIME_LAB/reports/PHASE46_GITHUB_CI_SAFETY_TESTS_REPORT.json`
- `BOT_V2_DAYTIME_LAB/reports/PHASE46_GITHUB_CI_SAFETY_TESTS_REPORT.md`
- `BOT_V2_DAYTIME_LAB/reports/PHASE46_GITHUB_CI_SAFETY_TESTS_REPORT.json.bak_phase47a_lab_isolation_20260430_205314`
- `BOT_V2_DAYTIME_LAB/reports/PHASE46_GITHUB_CI_SAFETY_TESTS_REPORT.md.bak_phase47a_lab_isolation_20260430_205314`

Estado: generados por ejecuciones locales de CI/reporte. No commitear backups; los reportes solo se commitean si una fase CI lo pide explicitamente.

### E) GENERATED_RUNTIME_DO_NOT_COMMIT

Cantidad: 2218

Ejemplos:

- `BOT_V2_DAYTIME_LAB/outputs/final_project_structure_manipulante/zip_validation/final_structure_zip_entries.txt`
- `BOT_V2_DAYTIME_LAB/outputs/final_project_structure_manipulante/zip_validation/final_structure_zip_validation.json`
- `BOT_V2_DAYTIME_LAB/outputs/final_project_structure_manipulante/zip_validation/final_structure_zip_validation.md`
- `legacy_archive_2026/_audit_dest_extras.ps1`
- `BOT_V2_DAYTIME_LAB/src/BOT_V2_DAYTIME_LAB/outputs/phase35_final_real_readiness_audit/dry_run_order_simulation/phase35_dry_run_order_simulation.json`
- `BOT_V2_DAYTIME_LAB/src/BOT_V2_DAYTIME_LAB/outputs/phase35_final_real_readiness_audit/dry_run_order_simulation/phase35_dry_run_order_simulation.md`
- `BOT_V2_DAYTIME_LAB/src/BOT_V2_DAYTIME_LAB/outputs/phase35_final_real_readiness_audit/dry_run_order_simulation/phase35_dry_run_orders.csv`
- `BOT_V2_DAYTIME_LAB/src/BOT_V2_DAYTIME_LAB/outputs/phase35_final_real_readiness_audit/risk_lot_audit/phase35_lot_scenarios.csv`
- `BOT_V2_DAYTIME_LAB/src/BOT_V2_DAYTIME_LAB/outputs/phase35_final_real_readiness_audit/risk_lot_audit/phase35_risk_lot_audit.json`
- `BOT_V2_DAYTIME_LAB/src/BOT_V2_DAYTIME_LAB/outputs/phase35_final_real_readiness_audit/risk_lot_audit/phase35_risk_lot_audit.md`
- `BOT_V2_DAYTIME_LAB/src/phase37ze_quick_status_panel.py.bak_auto_alerts_20260430_175900`
- `BOT_V2_DAYTIME_LAB/src/phase45_run_alert_check.py.bak_auto_alerts_20260430_175900`

Incluye logs, heartbeats, runtime, caches, CSV, outputs, zipbak, backups y data/research artifacts.

### F) SECRET_OR_LOCAL_CONFIG_DO_NOT_COMMIT

Cantidad: 1

Ruta enmascarada en JSON. No se leyo contenido.

### G) UNKNOWN_REQUIRES_MANUAL_REVIEW

Cantidad: 80

Ejemplos:

- `BOT_V2_DAYTIME_LAB/ZIP_CONTENTS_MANIFEST.md`
- `ZIP_CONTENTS_MANIFEST.md`
- `reports/canonical_context_master_20260416_163108/AGENTS.md`
- `reports/canonical_context_master_20260416_163108/RESEARCH_OPERATING_SYSTEM.md`
- `reports/canonical_context_master_20260416_163108/RISK_PROTOCOL.md`
- `reports/canonical_microstructure_iter2_20260416_172155/AGENTS.md`
- `reports/canonical_microstructure_iter2_20260416_172155/RESEARCH_OPERATING_SYSTEM.md`
- `reports/canonical_microstructure_iter2_20260416_172155/RISK_PROTOCOL.md`
- `research_lab/README.md`
- `BOT_V2_DAYTIME_LAB/ZIP_UPLOAD_IDENTITY_MARKER.md`
- `BOT_V2_DAYTIME_LAB/configs/phase24_forward_demo_candidate_config.json`
- `BOT_V2_DAYTIME_LAB/configs/phase24_forward_demo_candidate_config_hash.txt`

### H) PROTECTED_CHANGE_REQUIRES_DECISION

Cantidad primaria: 40

Total con flag protegido, incluyendo candidatos B/C/E: 66

Ejemplos:

- `BOT_V2_DAYTIME_LAB/src/phase37_ftmo_trial_support.py`
- `BOT_V2_DAYTIME_LAB/src/phase37ze_quick_status_panel.py`
- `BOT_V2_DAYTIME_LAB/src/phase45_run_alert_check.py`
- `BOT_V2_DAYTIME_LAB/src/phase45_telegram_sender.py`
- `MANIPULANTE/04_OPERACION_DIARIA/MANIPULANTE_DAILY_RUNBOOK.md`
- `MANIPULANTE/04_OPERACION_DIARIA/MANIPULANTE_KILL_SWITCH.md`
- `MANIPULANTE/09_COMPLIANCE/MT5_LIVE_NEWS_ADAPTER/MANIPULANTE_CalendarBootstrapEA.mq5`
- `MANIPULANTE/12_MICRO_REAL_READINESS/MICRO_REAL_KILL_SWITCH.md`
- `MANIPULANTE/12_MICRO_REAL_READINESS/MICRO_REAL_POSITION_SIZE_POLICY.md`
- `MANIPULANTE/16_OBSERVABILITY/alerts/README_ALERTS.md`
- `MANIPULANTE/16_OBSERVABILITY/alerts/alerts_config.example.json`
- `MANIPULANTE/START_MANIPULANTE.bat`
- `MANIPULANTE/STATUS_TECNICO_MANIPULANTE.bat`
- `MANIPULANTE/STOP_MANIPULANTE.bat`
- `mt5_demo_executor_lab/mt5_order_router.py`
- `BOT_V2_DAYTIME_LAB/reports/PHASE36S_LIVE_NEWS_LOT_FEASIBILITY_REPORT.json`
- `BOT_V2_DAYTIME_LAB/reports/PHASE36S_LIVE_NEWS_LOT_FEASIBILITY_REPORT.md`
- `BOT_V2_DAYTIME_LAB/reports/PHASE36_LIVE_NEWS_MT5_DRYRUN_READINESS_REPORT.json`
- `BOT_V2_DAYTIME_LAB/reports/PHASE36_LIVE_NEWS_MT5_DRYRUN_READINESS_REPORT.md`
- `BOT_V2_DAYTIME_LAB/src/phase36_live_data_quality_gate.py`

## 5. Riesgos detectados

- Cambios en `MANIPULANTE/` mezclados con research/lab.
- Cambios en Phase37/45 y scripts START/STOP, que son superficie live/protegida.
- `mt5_demo_executor_lab/mt5_order_router.py` reescrito casi completo; aunque declara fail-closed, es superficie de ordenes y requiere revision manual estricta.
- `MANIPULANTE_CalendarBootstrapEA.mq5` modificado; implica MQL5/MT5 y no debe mezclarse con Phase47A.
- `MANIPULANTE/STATUS_TECNICO_MANIPULANTE.bat` aparece borrado.
- Miles de CSV/outputs/legacy/data no deben commitearse masivamente.
- Un archivo local/config sensible fue detectado y enmascarado.

## 6. Que NO se debe commitear

- Runtime, logs, heartbeats, locks, pid, decisions.csv y telemetry.
- `.bak_*`, `.zipbak`, backups temporales.
- Caches de live news y estados locales.
- Data pesada CSV y outputs de research.
- Config local o posible secret/local config.
- `mt5_demo_executor_lab/mt5_order_router.py` sin revision manual separada.
- Cambios MQL5/MT5 sin fase explicita.
- Phase47A mientras existan cambios protegidos previos.

## 7. Que podria commitearse en commits separados

Commit candidato 1, solo tras validacion: MT5 reopen/process-stop fix.

- `MANIPULANTE/13_FTMO_TRIAL_AUTOMATION/safe_stop_manipulante_processes.ps1`
- `BOT_V2_DAYTIME_LAB/src/phase37_ftmo_trial_support.py`
- `MANIPULANTE/STOP_MANIPULANTE.bat`
- `BOT_V2_DAYTIME_LAB/src/phase37ze_quick_status_panel.py`

Commit candidato 2, solo tras validacion: Telegram auto alerts.

- `MANIPULANTE/START_MANIPULANTE.bat`
- `BOT_V2_DAYTIME_LAB/src/phase45_telegram_sender.py`
- `reports/MANIPULANTE_AUTO_ALERTS_START_STOP_REPORT.md`
- `MANIPULANTE/16_OBSERVABILITY/alerts/STATUS_ALERTS_LOOP_MANIPULANTE.bat`
- `MANIPULANTE/16_OBSERVABILITY/alerts/START_ALERTS_LOOP_MANIPULANTE.bat`
- `MANIPULANTE/16_OBSERVABILITY/alerts/STOP_ALERTS_LOOP_MANIPULANTE.bat`
- `MANIPULANTE/16_OBSERVABILITY/alerts/alerts_config.example.json`
- `BOT_V2_DAYTIME_LAB/src/phase45_run_alert_check.py`
- `MANIPULANTE/16_OBSERVABILITY/alerts/README_ALERTS.md`

Commit candidato 3, solo cuando el arbol este controlado: Phase47A lab isolation.

- `BOT_V2_DAYTIME_LAB/reports/PHASE47A_LAB_ISOLATION_AND_MANIPULANTE_PROTECTION_REPORT.json`
- `BOT_V2_DAYTIME_LAB/reports/PHASE47A_LAB_ISOLATION_AND_MANIPULANTE_PROTECTION_REPORT.md`
- `BOT_V2_DAYTIME_LAB/src/phase47a_lab_isolation_guard.py`
- `LAB_STRATEGIES/EURUSD_DAYTIME/README_EURUSD_DAYTIME_LAB.md`
- `LAB_STRATEGIES/EURUSD_DAYTIME/_templates/strategy_config_template.json`
- `LAB_STRATEGIES/EURUSD_DAYTIME/_templates/strategy_research_template.md`
- `LAB_STRATEGIES/EURUSD_DAYTIME/correlation/.gitkeep`
- `LAB_STRATEGIES/EURUSD_DAYTIME/reports/.gitkeep`
- `LAB_STRATEGIES/EURUSD_DAYTIME/shared/.gitkeep`
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/.gitkeep`
- `LAB_STRATEGIES/README_LAB_STRATEGIES.md`
- `PROJECT_ZONES_AND_BRANCHING_RULES.md`

## 8. Secuencia recomendada

1. Revisar manualmente los cambios protegidos y decidir cuales pertenecen al fix MT5 reopen.
2. Si el fix esta validado, commitear solo su set minimo con `git add` selectivo.
3. Revisar Telegram auto alerts y commitear solo codigo/docs/scripts validados, excluyendo runtime/backups/local config.
4. Resolver o descartar con autorizacion explicita los cambios high-risk restantes.
5. Re-ejecutar `phase47a_lab_isolation_guard.py`.
6. Reci?n ahi commitear Phase47A como commit separado.

## 9. Gitignore sugerido, no aplicado

- `**/__pycache__/`
- `*.pyc`
- `*.pid`
- `*.lock`
- `*.sqlite`
- `*.db`
- `**/runtime/`
- `**/live_news_cache/`
- `**/alerts_loop.last_heartbeat.json`
- `**/heartbeat.json`
- `**/heartbeat.txt`
- `**/quick_status.txt`
- `**/decisions.csv`
- `*.zipbak`
- `*.bak_*`

## 10. Confirmaciones de seguridad

- No se cambio estrategia.
- No se abrio MT5.
- No se conecto a MT5.
- No se enviaron ordenes.
- No se cerraron ordenes.
- No se toco real.
- No se toco Exness.
- No se cambio TP/BE/BF.
- No se cambio riesgo.
- No se cambiaron horarios.
- No se leyeron contenidos de posibles secrets.
- No se ejecuto `git add`.
- No se ejecuto commit.
- No se ejecuto push.
- No se ejecuto reset.
- No se ejecuto clean.
- No se ejecuto stash.

## 11. Reporte JSON

`BOT_V2_DAYTIME_LAB/reports/PHASE47B_WORKING_TREE_FORENSIC_CLEANUP_REPORT.json` contiene el inventario clasificado por categoria. No embebe diffs ni contenidos de archivos.

## 12. Siguiente paso unico

Revisar primero el candidato `MT5_REOPEN_FIX_CANDIDATE` y decidir si se valida como commit separado minimo. No commitear Phase47A todavia.
