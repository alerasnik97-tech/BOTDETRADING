# NEXT PROMPT: REGENERATE TP-01 WITH THE OFFICIAL RUNNER (GATED)

Actuá como Claude Opus 4.7 Max en modo **Institutional Quant Backtest Operator + Metric/Cost Reconciliation Gatekeeper** senior, backtest safety engineer, data-leakage auditor y quant lab release gatekeeper.

============================================================
CONTEXTO
============================================================

Ambos gates cerrados (infra + governance):
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/COST_MODEL_OWNER_DECISION_RESEARCH_ONLY.md`
  (`conservative` 1.20/1.30 ratificado research/train-only).
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/FORMAL_RUNNER_AND_COST_MODEL_GATES_REPORT.md`
  (`FORMAL_RUNNER_AND_COST_GATES_READY_FOR_TP01_REGENERATION`).

Runner oficial: `research_lab.runners.formal_train_runner` (reemplaza `scratch/`).
Tests acumulados: 97/97 verdes (runner 41, cost 11, recon 19, engine 17, stop_entry 3, preflight 6).

Cadena de fixes vigente (branches `fix/...`/`infra/...`, sin merge a main):
PnL signo · equity/drawdown/summary · cost routing real · reconciliation gate · runner oficial.

============================================================
OBJETIVO
============================================================

Regenerar el dossier formal **train-only 2015–2024** de TP-01 usando ÚNICAMENTE
el runner oficial, 3 cost profiles reales, reconciliation gate obligatorio.

============================================================
PRECONDICIONES DURAS
============================================================

1. Usar SOLO `research_lab.runners.formal_train_runner`. PROHIBIDO `scratch/formal_run_tp01.py`.
2. `FormalRunRequest.execute=True` SOLO tras `preflight` OK.
3. Si preflight o gate fallan → ABORTAR (`BLOCKED_PREFLIGHT` / `BLOCKED_RECONCILIATION_GATE`);
   NO sellar, NO commitear outputs pesados.
4. Branches de fix/infra disponibles por decisión del owner (NO mergear a main vos).

============================================================
REGLAS ABSOLUTAS
============================================================

NO main. NO force push. NO merge. NO rebase.
NO holdout. NO 2025/2026. NO news. NO high precision. NO F06.
NO optimization. NO sweep. NO validation. NO walk-forward.
NO tocar lógica de señal TP-01. NO tocar MR-01. NO tocar data vault.
NO modificar el runner para saltear gates. NO commitear `scratch/` ni
`local_outputs_do_not_commit/`. NO ZIP. NO git add . NO root files.
Dataset: SOLO `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared`.

============================================================
PASOS
============================================================

1. Precheck git + procesos (sin backtest activo). Branch
   `research/tp01-regen-official-runner-20260517` desde la cadena de fixes vigente.
2. Suite completa sin backtest:
   `test_formal_train_runner_contract.py` · `test_cost_profiles.py`
   · `test_metric_reconciliation.py` · `test_engine.py`
   · `test_engine_stop_entry.py` · `test_lab_preflight*.py`. Falla → `BLOCKED_TEST_FAILURE`.
3. `preflight()` para TP-01 (dry-run). Verificar 3 profiles reales, output bajo
   `…/reports/formal_train_only/…`, manifest sin duplicados, reconciliation_required true.
4. `run_single_strategy_formal_train_only(execute=True)` para TP-01, train-only
   2015–2024, M1, profiles base/conservative/stress. Heavy → `local_outputs_do_not_commit/`.
5. **Gate obligatorio**: `reconcile_all` (trades+equity+summary+profiles) ⇒ CERO
   violaciones; `ending_equity ≈ 100000 + Σ pnl_usd`; `stop_loss⇒loss`/`take_profit⇒win`;
   `drawdown_pct` no todo 0; PF/expectancy/total_return coherentes; cada summary
   self-report su profile; base<conservative<stress. Si falla → NO sellar.
6. Crear `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/TP01_POST_FIX_RECONCILIATION_REPORT.md`
   (estado, gate-green, comparación pre/post, edge real por profile).
7. Entregar a auditoría externa (protocolo v3) antes de cualquier conclusión.

============================================================
SALIDA
============================================================

- NO declarar champion / rentable / incubation / FTMO/demo/real.
- Esperado: edge neto negativo (~ −9% additive base), maxDD real ~8–9%, NO +135%;
  conservative/stress monotónicamente peores que base.
- TP-01 sigue siendo candidato a rechazo; la reconciliación valida métricas,
  NO rehabilita la estrategia.
- MR-01 permanece **bloqueado** hasta TP-01 regenerado limpio, gate-green y auditado.
- Stage explícito: solo reportes livianos. NO data/heavy/scratch/ZIP/root. NO git add .
