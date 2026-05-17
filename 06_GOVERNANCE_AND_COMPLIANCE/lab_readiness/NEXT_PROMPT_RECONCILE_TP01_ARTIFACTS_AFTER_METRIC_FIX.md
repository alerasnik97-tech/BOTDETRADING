# NEXT PROMPT: RECONCILE TP-01 ARTIFACTS AFTER METRIC FIX (GATED)

Actuá como Claude Opus 4.7 Max en modo **Institutional Quant Backtest Operator + Metric Reconciliation Auditor** senior, backtest safety engineer, cost-model auditor y quant gatekeeper.

============================================================
CONTEXTO
============================================================

Fix branch `fix/shared-metric-cost-integrity-20260517`
Report `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/TP01_METRIC_FIX_REPORT.md`
Estado: `TP01_METRIC_FIX_PARTIAL_OWNER_REVIEW_REQUIRED`.

Corregido y testeado (45/45 tests, sin regresión):
- D1 signo PnL short (`engine.directional_pnl_usd`, ambos exit paths).
- D2 inflación de equity (`cash += pnl_usd`, sin add-back de `risk_usd`).
- D3/D4/D5 (result/equity/drawdown/summary) resueltos como consecuencia de D1/D2.
- Gate de reconciliación `research_lab/metric_reconciliation.py` + tests.

NO corregido (owner review): D6 cost-profile differentiation (harness scratch
+ falta de tiers `conservative`/`stress_mode` en el engine).

============================================================
PRECONDICIÓN DURA (NO EJECUTAR SI NO SE CUMPLE)
============================================================

ANTES de cualquier re-run formal de 3 perfiles:
1. El owner debe decidir y documentar el modelo de costos:
   - definir tier real `conservative` (spread/slippage params) y, si aplica, `stress_mode`;
   - extender `SUPPORTED_COST_PROFILES` / `SUPPORTED_EXECUTION_MODES` en `config.py`;
   - corregir `scratch/formal_run_tp01.py` mapping (revisado, no auto-promover scratch);
   - cada `summary.json` debe auto-reportar su propio profile; `RUN_MANIFEST` solo perfiles realmente distintos.
2. Si el owner NO ha decidido el modelo de costos → ABORTAR
   `BLOCKED_COST_MODEL_OWNER_DECISION_REQUIRED` (puede reconciliarse SOLO el perfil `base`).

============================================================
REGLAS ABSOLUTAS
============================================================

NO main. NO force push. NO merge. NO rebase.
NO holdout. NO 2025/2026. NO news. NO high precision. NO F06.
NO optimization. NO sweep. NO validation. NO WFA.
NO tocar lógica de señal TP-01. NO tocar MR-01. NO tocar data vault.
NO commitear `local_outputs_do_not_commit/`. NO ZIP. NO git add . NO root files.
Dataset: SOLO `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared/`.
`sealed_holdout_2025_2026/` NUNCA se abre.

============================================================
PASOS
============================================================

1. Confirmar fix branch mergeada/disponible y `TP01_METRIC_FIX_REPORT.md` presente.
2. Correr la suite (sin backtest):
   `python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -p "test_metric_reconciliation.py" -v`
   `... -p "test_engine.py"` · `... -p "test_engine_stop_entry.py"` · `... -p "test_lab_preflight*.py"`
   Si algo falla → `BLOCKED_TEST_FAILURE`.
3. Re-ejecutar SOLO la corrida formal train-only 2015–2024 de TP-01 bajo el engine corregido
   (perfil `base` siempre; conservative/stress solo si la precondición de costos se cumplió).
   M1, normal_mode, dataset train-only. NO optimization/sweep/holdout.
4. **Gate obligatorio antes de sellar**: pasar `trades.csv` + `equity_curve.csv` + summary por
   `metric_reconciliation.reconcile_all`. Cero violaciones requerido. Verificar:
   `ending_equity ≈ 100000 + Σ pnl_usd`; `stop_loss⇒loss`/`take_profit⇒win`;
   `drawdown_pct` no todo 0; PF/expectancy/total_return coherentes; perfiles self-report.
5. Comparar el nuevo dossier vs el artefacto pre-fix (debe pasar de "suspect" a reconciliado);
   esperado: edge neto negativo (~ −9% additive), maxDD real ~8–9%, NO +135%.
6. Entregar a auditoría externa (protocolo v3) antes de cualquier conclusión de viabilidad.

============================================================
SALIDA
============================================================

- Crear `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/TP01_POST_FIX_RECONCILIATION_REPORT.md`
  (estado, evidencia gate-green, comparación pre/post, edge real de TP-01).
- NO declarar champion / rentable / incubation / FTMO/demo/real.
- TP-01 sigue siendo, defecto-independiente, candidato a rechazo (PF<1, expectancy<0,
  0 trades 2019–2024) — la reconciliación NO lo rehabilita; solo valida las métricas.
- MR-01 permanece **bloqueado** hasta que TP-01 esté regenerado limpio y gate-green
  y el modelo de costos esté decidido por el owner.
- Stage explícito: solo docs/código necesario + tests. NO data/heavy/ZIP/root. NO git add .
