# PHASE47A LAB ISOLATION AND MANIPULANTE PROTECTION REPORT

## 1. Lo mas importante

Phase47A creo la estructura de laboratorio aislado para `EURUSD Daytime Lab`, la documentacion de zonas/branching, templates de research y un guard pasivo de proteccion.

El guard detecta cambios protegidos existentes en el working tree. Por eso esta fase no queda lista para commit/push seguro todavia.

## 2. Veredicto final exacto

`LAB_ISOLATION_REQUIRES_REPAIR`

Motivo: la capa Phase47A existe, pero `git status` ya contenia cambios protegidos antes de esta fase y el guard finaliza con `LAB_ISOLATION_GUARD_FAIL_PROTECTED_CHANGE`.

## 3. Estructura creada

```text
LAB_STRATEGIES/
  README_LAB_STRATEGIES.md
  EURUSD_DAYTIME/
    README_EURUSD_DAYTIME_LAB.md
    strategies/.gitkeep
    shared/.gitkeep
    reports/.gitkeep
    correlation/.gitkeep
    _templates/
      strategy_research_template.md
      strategy_config_template.json
```

## 4. Reglas de zonas

Zona roja:

- `MANIPULANTE/`
- START/STOP/STATUS de MANIPULANTE
- scripts live Phase37
- observability live Phase44
- Telegram live Phase45
- Phase46 CI Safety salvo extension pasiva aprobada
- News Fortress
- Data Quality Mask
- configs live, `.env`, tokens y secretos

Zona amarilla:

- `LAB_STRATEGIES/`
- research nuevo
- backtests futuros autorizados
- configs independientes
- reportes exploratorios
- correlacion no operativa

Zona verde:

- portfolio y promocion futura solamente despues de validacion individual
- comparacion contra MANIPULANTE
- correlacion menor a 0.5 antes de pensar en portfolio
- nunca ejecucion real automatica

## 5. Reglas de branching

- `main` queda como superficie protegida de MANIPULANTE.
- No desarrollar estrategias nuevas en `main`.
- Crear ramas `research/<nombre_estrategia>` para cada linea futura.
- Trabajar solo dentro de `LAB_STRATEGIES/`.
- No usar `git add .`.
- No usar `git commit -a`.
- No usar `git push --force`.
- Usar `git add` selectivo.
- PR/revision antes de merge.
- Phase46 CI verde antes de merge.

## 6. Guardrail creado

Ruta:

```text
BOT_V2_DAYTIME_LAB/src/phase47a_lab_isolation_guard.py
```

Valida:

- branch actual
- cambios detectados por `git status --porcelain`
- clasificacion `ALLOWED_LAB_CHANGE`
- clasificacion `ALLOWED_DOC_CHANGE`
- clasificacion `PROTECTED_CHANGE`
- clasificacion `UNKNOWN_CHANGE`
- exit 0 sin cambios protegidos
- exit 1 con cambios protegidos

Resultado de ejecucion:

```text
changes_detected: 2342
protected_changes: 50
verdict: LAB_ISOLATION_GUARD_FAIL_PROTECTED_CHANGE
exit_code: 1
```

## 7. Templates creados

- `LAB_STRATEGIES/EURUSD_DAYTIME/_templates/strategy_research_template.md`
- `LAB_STRATEGIES/EURUSD_DAYTIME/_templates/strategy_config_template.json`

El config template queda con `live_trading_allowed=false`, `paper_demo_allowed=false`, `real_trading_allowed=false`, `touches_manipulante=false`, `requires_news_fortress=true` y `requires_data_quality_mask=true`.

## 8. Archivos modificados

Archivos nuevos Phase47A:

- `PROJECT_ZONES_AND_BRANCHING_RULES.md`
- `LAB_STRATEGIES/README_LAB_STRATEGIES.md`
- `LAB_STRATEGIES/EURUSD_DAYTIME/README_EURUSD_DAYTIME_LAB.md`
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/.gitkeep`
- `LAB_STRATEGIES/EURUSD_DAYTIME/shared/.gitkeep`
- `LAB_STRATEGIES/EURUSD_DAYTIME/reports/.gitkeep`
- `LAB_STRATEGIES/EURUSD_DAYTIME/correlation/.gitkeep`
- `LAB_STRATEGIES/EURUSD_DAYTIME/_templates/strategy_research_template.md`
- `LAB_STRATEGIES/EURUSD_DAYTIME/_templates/strategy_config_template.json`
- `BOT_V2_DAYTIME_LAB/src/phase47a_lab_isolation_guard.py`
- `BOT_V2_DAYTIME_LAB/reports/PHASE47A_LAB_ISOLATION_AND_MANIPULANTE_PROTECTION_REPORT.md`
- `BOT_V2_DAYTIME_LAB/reports/PHASE47A_LAB_ISOLATION_AND_MANIPULANTE_PROTECTION_REPORT.json`

Archivos existentes cambiados intencionalmente por Phase47A:

- Ninguno como cambio final previsto.

Nota: `phase46_ci_safety_check.py` fue ejecutado y reescribio sus reportes, pero esos reportes fueron restaurados desde backups creados inmediatamente antes para no pisar evidencia previa.

## 9. Backups creados

- `BOT_V2_DAYTIME_LAB/reports/PHASE46_GITHUB_CI_SAFETY_TESTS_REPORT.md.bak_phase47a_lab_isolation_20260430_205314`
- `BOT_V2_DAYTIME_LAB/reports/PHASE46_GITHUB_CI_SAFETY_TESTS_REPORT.json.bak_phase47a_lab_isolation_20260430_205314`

## 10. Seguridad

- No se creo estrategia nueva.
- No se backtesteo.
- No se busco edge.
- No se abrio MT5.
- No se conecto a MT5.
- No se enviaron ordenes.
- No se cerraron ordenes.
- No se modificaron ordenes.
- No se toco real.
- No se toco Exness.
- No se cambio TP/BE/BF.
- No se cambio riesgo.
- No se cambiaron horarios.
- No se tocaron secrets ni `.env`.
- No se modifico ZIP canonico.

## 11. Confirmacion sobre MANIPULANTE

Phase47A no modifico `MANIPULANTE/`.

Advertencia bloqueante: el working tree ya contiene cambios en `MANIPULANTE/` y otros paths protegidos. Esos cambios no fueron creados por esta fase, pero impiden declarar el repositorio limpio o hacer commit/push seguro de Phase47A.

## 12. Confirmacion sobre MT5 y ordenes

No se abrio MT5, no se ejecutaron scripts MT5, no se conecto a brokers y no se tocaron ordenes.

Advertencia: el guard detecto cambios existentes en `mt5_demo_executor_lab/mt5_order_router.py`, por eso mantiene el fallo de proteccion.

## 13. Phase46 CI local

Comando ejecutado:

```powershell
python BOT_V2_DAYTIME_LAB/src/phase46_ci_safety_check.py
```

Resultado:

```text
exit_code: 0
FINAL VERDICT: GITHUB_CI_READY_WITH_WARNINGS
```

Warnings: heuristicas de palabras `secret`, `token` y `api_key` en reportes/outputs ya trackeados. No fueron creadas por Phase47A.

## 14. Git status

Branch:

```text
main
```

Estado:

```text
dirty
total_status_lines: 2342
```

Bloqueo:

- existen cambios protegidos previos
- existen muchos cambios no relacionados con Phase47A
- no corresponde `git add`
- no corresponde commit
- no corresponde push

## 15. Siguiente paso unico

Revisar y resolver el working tree protegido previo (`MANIPULANTE/`, Phase37/45, `mt5_demo_executor_lab/` y cambios no relacionados) antes de intentar commitear Phase47A.
