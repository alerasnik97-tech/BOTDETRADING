# NEXT PROMPT: REGENERATE TP-01 TRAIN-ONLY WITH THE OFFICIAL RUNNER (FIRST REAL EXECUTION — GATED)

Actuá como Claude Opus 4.7 Max en modo **Institutional Quant Backtest Operator +
Metric/Cost Reconciliation Gatekeeper + Backtest Safety Engineer + Data-Leakage
Auditor + Quant Lab Release Gatekeeper** senior.

============================================================
CONTEXTO
============================================================

El runner oficial `research_lab.runners.formal_train_runner` está
**execution-ready y aprobado** por auditoría externa v2:

- `FORMAL_RUNNER_EXECUTE_PATH_FIX_EXTERNAL_AUDIT_V2_REPORT.md`
  → `FORMAL_RUNNER_EXECUTE_FIX_APPROVED_WITH_WARNINGS_FOR_TP01_REGENERATION`.

Branch del fix aprobado: `fix/formal-runner-execute-path-20260517`
Commit aprobado: `ba96de4934a66d3938874d725d5fc29800757f52`
B1/B2/B3/B4 PASS · W2/W3 PASS · 110/110 tests · dry-run safe · CLI fail-closed
8/8 · static scan clean · output policy pass · sin code/data modificados.

Esta es la **PRIMERA EJECUCIÓN REAL** del runner (todo lo previo fue fakes).
Warning heredado (W-c): el camino real engine↔loader↔report↔reconcile se ejerce
end-to-end por primera vez aquí; por eso el dossier resultante DEBE pasar
auditoría externa posterior antes de cualquier conclusión.

============================================================
OBJETIVO
============================================================

Regenerar el dossier formal **train-only 2015–2024** de **TP-01**
(`tp01_london_ny_momentum_pullback`) usando ÚNICAMENTE el runner oficial,
`execute=True` **una sola vez**, 3 cost profiles reales, reconciliation+cost
gate obligatorio.

============================================================
PRECONDICIONES DURAS
============================================================

1. SOLO `research_lab.runners.formal_train_runner`. PROHIBIDO `scratch/`.
2. UNA sola estrategia: `tp01_london_ny_momentum_pullback`.
3. `FormalRunRequest.execute=True` SOLO tras `preflight` OK.
4. Dataset EXACTO: `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared`.
5. Fechas EXACTAS: `2015-01-01` → `2024-12-31`.
6. Si preflight, git identity, reconciliation o seal fallan → ABORTAR
   (`BLOCKED_PREFLIGHT` / `BLOCKED_GIT_IDENTITY` / `BLOCKED_RECONCILIATION_GATE`);
   NO sellar, NO commitear outputs pesados.
7. NO modificar runner/engine/report/data_loader/strategies/tests para pasar gates.

============================================================
REGLAS ABSOLUTAS
============================================================

NO main. NO force push. NO merge. NO rebase. NO git add .
NO holdout. NO sealed_holdout. NO 2025/2026. NO F06. NO news. NO high precision.
NO optimization. NO sweep. NO validation. NO walk-forward.
NO segunda estrategia. NO MR-01. NO tocar lógica de señal TP-01. NO tocar data vault.
NO modificar el runner para saltear gates. NO ZIP. NO root files.
NO commitear `scratch/` ni `local_outputs_do_not_commit/`.
NO tocar dirty preexistente ajeno (`strategy_research_intake/`).
NO declarar edge / rentable / champion / incubation / FTMO / demo / real.

============================================================
PASOS
============================================================

1. Precheck git + procesos (sin backtest activo). Branch
   `research/tp01-regen-official-runner-execute-20260517` desde
   `fix/formal-runner-execute-path-20260517` (NO main).
2. Suite completa sin backtest (debe seguir 110/110):
   `test_formal_train_runner_execute_contract.py` · `test_formal_train_runner_contract.py`
   · `test_cost_profiles.py` · `test_metric_reconciliation.py` · `test_engine.py`
   · `test_engine_stop_entry.py` · `test_lab_preflight*.py`. Falla → `BLOCKED_TEST_FAILURE`.
3. `preflight()` para TP-01 (dry-run): 3 profiles reales base/conservative/stress,
   output bajo `…/reports/formal_train_only/tp01_london_ny_momentum_pullback/<RUN_ID>`,
   manifest con branch/commit reales (no placeholder), `reconciliation_required=true`.
4. `run_single_strategy_formal_train_only(execute=True)` UNA sola vez para TP-01,
   train-only 2015–2024, M1, profiles base/conservative/stress. Heavy artifacts →
   `<output_dir>/local_outputs_do_not_commit/<profile>/`.
5. **Gate obligatorio** (ya integrado en el runner): per-profile `reconcile_all`
   + cost-profile reconciliation; `ending_equity ≈ 100000 + Σ pnl_usd`;
   `stop_loss⇒loss`/`take_profit⇒win`; `drawdown_pct` no todo 0; PF/expectancy/
   total_return coherentes; cada profile self-report; base<conservative<stress.
   Si falla → NO sella, NO escribe dossier de éxito (verificado por la auditoría).
6. Confirmar artefactos escritos SOLO bajo el output validado:
   `manifests/RUN_MANIFEST.json`, `configs/<p>_ENGINE_CONFIG.json`,
   `profile_reports/<p>/summary.json`, `profile_reports/<p>/tables/{monthly,yearly}.csv`,
   `local_outputs_do_not_commit/<p>/{trades,equity_curve}.csv`. Cero ZIP, cero root.
7. Crear `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/TP01_POST_FIX_RECONCILIATION_REPORT.md`
   (estado, gate-green, comparación pre-fix vs post-fix por profile, métricas reales).
8. Entregar a **auditoría externa posterior del dossier** (protocolo v3) ANTES de
   cualquier conclusión. NO declarar nada sobre el edge.

============================================================
SALIDA ESPERADA / SEGURIDAD
============================================================

- Esperado: edge neto **negativo** (~ −9% additive base), maxDD real ~8–9%,
  NO +135%; conservative/stress monotónicamente peores que base.
- TP-01 sigue siendo **candidato a rechazo**; la reconciliación valida métricas,
  NO rehabilita la estrategia.
- NO declarar champion / rentable / incubation / FTMO / demo / real.
- **MR-01 permanece BLOQUEADO** hasta que TP-01 esté regenerado limpio,
  gate-green y auditado externamente.
- Stage explícito: SOLO el reporte liviano `TP01_POST_FIX_RECONCILIATION_REPORT.md`
  (+ próximo prompt). NO data/heavy/scratch/ZIP/root. NO git add . NO commitear
  `local_outputs_do_not_commit/`.
- Safety obligatoria en el reporte: backtest_run=YES (train-only, autorizado);
  optimization/sweep/validation/holdout/2025-26/news/high_precision=NO;
  data_modified=NO; force_push=NO; git_add_dot=NO; second_strategy=NO; MR01=BLOCKED.
