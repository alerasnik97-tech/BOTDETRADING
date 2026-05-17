# NEXT PROMPT: FIX FORMAL RUNNER PRE-EXECUTION BLOCKERS (GATED)

Actuá como Claude Opus 4.7 Max en modo **Institutional Research Infrastructure Engineer + Backtest Runner Repair Engineer + Python Safety Engineer + Metric Reconciliation Gatekeeper + Pre-Execution Quant Gatekeeper** senior.

============================================================
CONTEXTO
============================================================

La auditoría externa pre-ejecución
(`06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/FORMAL_RUNNER_PRE_EXECUTION_EXTERNAL_AUDIT_REPORT.md`,
commit `08747a3b`, branch `infra/formal-runner-cost-gates-20260517`) resolvió:

`FORMAL_RUNNER_BLOCKED_EXECUTE_SIGNATURE_RISK`

El runner oficial `research_lab.runners.formal_train_runner` es **seguro e import-safe**,
dry-run por defecto, scope train-only estricto, cost profiles correctos y monótonos,
reconciliation gate fail-closed, CLI fail-closed (8/8), static scan limpio, tests 97/97.
PERO el path `execute=True` está **roto** y **no escribe artefactos**. NO se autorizó
regenerar TP-01. MR-01 sigue **BLOQUEADO**.

============================================================
OBJETIVO
============================================================

Reparar SOLO el path `execute=True` del runner oficial y su capa de escritura de
dossier, dejarlo cubierto por tests con fakes (sin backtest real), y entregarlo a
una nueva auditoría externa. **NO** regenerar TP-01 en esta fase.

============================================================
BLOCKERS A CORREGIR (con evidencia exacta)
============================================================

**B1 — `run_backtest` (BLOCKER).**
`03_RESEARCH_LAB/research_lab/runners/formal_train_runner.py:328-331` llama
`run_backtest(strategy_module=, frame=, params=, engine_config=)` pero
`03_RESEARCH_LAB/research_lab/engine.py:597-609` exige además los posicionales
**sin default** `news_block: np.ndarray` y `news_filter_used: bool` (anteriores al `*,`).
→ `TypeError` en el primer profile. Fix: construir un `news_block` neutro
(sin news; p.ej. `np.zeros(len(bundle.frame), dtype=bool)` con el dtype/contrato
que el engine espera — verificar el uso interno real en `engine.py`) y pasar
`news_filter_used=False`. Prohibido activar news.

**B2 — `summarize_result` (BLOCKER).**
`formal_train_runner.py:332-334` llama con 5 posicionales
`(result.strategy_name, result.trades, result.equity_curve, result.params, False)`
pero `03_RESEARCH_LAB/research_lab/report.py:281-293` exige 7 mínimos:
faltan `initial_capital: float` y `selected_score: float | None` (sin default).
Fix: pasar `initial_capital` desde la config (capital inicial canónico, p.ej.
100000.0 — usar la fuente de verdad existente, NO un literal mágico nuevo) y
`selected_score=None`. Mantener el desempaquetado de 5 valores (el retorno ES
una 5-tupla `summary, trades_export, monthly, yearly, equity_export`).

**B3 — `load_backtest_data_bundle` data_dirs (RISK).**
`formal_train_runner.py:312-315` pasa `data_dirs=(req.data_path,)` (`tuple[str]`)
donde `03_RESEARCH_LAB/research_lab/data_loader.py:359-368` anota `list[Path]`.
Fix: normalizar a `[Path(req.data_path)]` y verificar el uso interno real en
`load_price_data` / `load_prepared_ohlcv` (str vs Path) antes de habilitar execute.

**B4 — Artifact-write gap (CO-BLOCKER).**
`formal_train_runner.py:288-346`: el branch execute arma manifest+recs en memoria
y retorna un dict; **no** persiste nada (`grep` no muestra `open`/`json.dump`/
`to_csv`/`mkdir`). Verificado empíricamente: el dry-run no creó ningún directorio.
Fix: tras `seal_run_only_if_reconciled`, escribir el dossier formal:
- `RUN_MANIFEST.json` (con branch/commit reales — ver W2),
- snapshot de cost/config por profile,
- `summary.json` por profile,
- `trades.csv` / `equity_curve.csv` por profile **bajo
  `heavy_output_dir(req.output_dir, profile)`** (es decir
  `…/reports/formal_train_only/<run>/local_outputs_do_not_commit/<profile>/`),
- tablas livianas bajo el run dir validado,
usando exclusivamente `validate_output_dir` + `heavy_output_dir` ya existentes.
Crear directorios con `mkdir(parents=True, exist_ok=True)`. NO escribir nada
fuera del área validada. NO ZIP. NO root. NO data vault.

**W2 — Manifest provenance.** `branch`/`commit` son placeholders
(`"(unset)"` / `"(caller-supplied)"`). El runner debe resolver branch/commit
reales (vía `git rev-parse`/`git branch --show-current` capturados por el caller
o helper) y NO sellar con placeholders.

**W3 (recomendado).** Incluir `reconcile_cost_profiles` en el gate de sellado:
pasar `profiles={...}` (cost_profile/execution_mode por carpeta) a
`reconcile_all` para que `COST_PROFILE_MISLABEL`/`DUPLICATE` formen parte del gate.

============================================================
TESTS OBLIGATORIOS (sin backtest real)
============================================================

Agregar a `03_RESEARCH_LAB/research_lab/tests/test_formal_train_runner_contract.py`
(o un test nuevo hermano) cubriendo el path execute con **fakes/monkeypatch**:

1. `execute=True` con `STRATEGY_REGISTRY`, `load_backtest_data_bundle`,
   `run_backtest`, `summarize_result` **fakeados** (devuelven DataFrames sintéticos
   limpios) → corre 3 profiles, sella, y **escribe** manifest+summary+trades+equity
   en las rutas esperadas (assert de archivos creados bajo
   `local_outputs_do_not_commit/<profile>`).
2. Si un profile devuelve un ledger contradictorio → `ReconciliationGateError`
   y **NO** se escribe sello.
3. Manifest contiene branch/commit **no placeholder**.
4. Loop ejecuta exactamente 3 profiles (base/conservative/stress).
5. Firmas: un test que llame `run_backtest`/`summarize_result` con los kwargs del
   runner contra firmas reales (o `inspect.signature`) y falle si faltan args.
6. Re-verde de las 6 suites: `test_formal_train_runner_contract.py`,
   `test_cost_profiles.py`, `test_metric_reconciliation.py`, `test_engine.py`,
   `test_engine_stop_entry.py`, `test_lab_preflight*.py`.

============================================================
REGLAS ABSOLUTAS
============================================================

NO main. NO force push. NO merge. NO rebase.
NO backtest real. NO strategy run real. NO TP-01. NO MR-01.
NO optimization. NO sweep. NO walk-forward. NO validation.
NO holdout. NO sealed_holdout. NO 2025/2026. NO F06. NO news. NO high precision.
NO tocar lógica de señal / engine PnL / cost routing (ya auditado y verde) salvo
lo mínimo para B1 (firma) sin alterar comportamiento existente.
NO modificar data. NO data vault. NO ZIP. NO git add . NO root files.
NO commitear `scratch/` ni `local_outputs_do_not_commit/`.
NO tocar los dirty preexistentes ajenos de `strategy_research_intake/`.
`execute=True` SOLO en tests con fakes; NUNCA contra dataset real en esta fase.

============================================================
PASOS
============================================================

1. Precheck git + procesos. Branch `fix/formal-runner-execute-path-20260517`
   desde `infra/formal-runner-cost-gates-20260517` (NO main).
2. Corregir B1, B2, B3, B4, W2 (y W3 recomendado) en `formal_train_runner.py`
   (capa de orquestación/IO únicamente; sin duplicar engine/strategy).
3. Agregar los tests con fakes (sección anterior). Sin backtest real.
4. Correr las 6 suites + dry-run + matriz CLI fail-closed (8/8). Falla →
   `BLOCKED_TEST_FAILURE`, no sellar.
5. Crear `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/FORMAL_RUNNER_EXECUTE_PATH_FIX_REPORT.md`
   (estado, diffs, B1–B4/W2/W3 cerrados, evidencia de tests, sin ejecución real).
6. Entregar a auditoría externa pre-ejecución v2 ANTES de regenerar TP-01.
7. Stage explícito: SOLO código del runner + tests + doc del fix. NO data/heavy/
   scratch/ZIP/root. NO git add . Commit + push del branch de fix.

============================================================
SALIDA
============================================================

- NO declarar runner listo para producción ni TP-01 regenerable: eso lo decide la
  auditoría externa posterior.
- NO declarar edge / rentable / champion / incubation / FTMO / demo / real.
- MR-01 permanece **BLOQUEADO** hasta que: (a) este fix pase auditoría externa v2,
  (b) TP-01 sea regenerado limpio y gate-green con el runner oficial, (c) TP-01 sea
  auditado externamente.
- El runner oficial sigue reemplazando `scratch/formal_run_tp01.py` (decisión de
  gobernanza vigente); esta fase solo lo vuelve ejecutable y verificable.
