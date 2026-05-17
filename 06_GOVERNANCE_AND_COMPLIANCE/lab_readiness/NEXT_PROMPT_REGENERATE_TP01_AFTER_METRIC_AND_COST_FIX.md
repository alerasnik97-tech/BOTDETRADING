# NEXT PROMPT: REGENERATE TP-01 AFTER METRIC + COST FIX (DOUBLE-GATED)

Actuá como Claude Opus 4.7 Max en modo **Institutional Quant Backtest Operator + Metric/Cost Reconciliation Auditor** senior, backtest safety engineer y quant gatekeeper.

============================================================
CONTEXTO
============================================================

Cadena de fixes (todas en branches `fix/...`, sin merge a main):
- `fix/shared-metric-cost-integrity-20260517` — PnL signo + equity/drawdown/summary + gate. `TP01_METRIC_FIX_PARTIAL_OWNER_REVIEW_REQUIRED`.
- `fix/institutional-cost-profile-routing-20260517` — 3 cost profiles reales (base/conservative/stress), monotónicos, self-report, gate. `COST_PROFILE_PARTIAL_OWNER_REVIEW_REQUIRED`.

Reportes:
`06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/TP01_METRIC_FIX_REPORT.md`
`06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/COST_PROFILE_OWNER_DECISION_AND_ROUTING_FIX_REPORT.md`

Tests acumulados: 56/56 verdes (cost_profiles 11, metric_reconciliation 19, engine 17, stop_entry 3, preflight 6).

============================================================
PRECONDICIONES DURAS (NO EJECUTAR SI NO SE CUMPLEN)
============================================================

GATE 1 — Owner ratifica magnitudes `conservative`:
`conservative_spread_multiplier=1.20`, `conservative_slippage_multiplier=1.30`
(actualmente `OWNER_APPROVED_DEFAULTS_REQUIRED`). Si el owner no ratifica → ABORTAR
`BLOCKED_CONSERVATIVE_MULTIPLIERS_NOT_RATIFIED` (puede regenerarse SOLO `base` y `stress`,
que usan valores no inventados).

GATE 2 — Runner oficial committeable:
`scratch/formal_run_tp01.py` está git-ignored (`.gitignore:41 scratch/`) y NO puede ser
el mecanismo sellado. El owner debe proveer/aprobar un runner formal versionado (fuera de
`scratch/`) que consuma el ruteo corregido (`config.resolved_cost_profile`). Si no existe →
ABORTAR `BLOCKED_OFFICIAL_RUNNER_REQUIRED`.

GATE 3 — Branches de fix mergeadas/disponibles por decisión del owner (NO mergear vos).

============================================================
REGLAS ABSOLUTAS
============================================================

NO main. NO force push. NO merge. NO rebase.
NO holdout. NO 2025/2026. NO news. NO high precision. NO F06.
NO optimization. NO sweep. NO validation. NO WFA.
NO tocar lógica de señal TP-01. NO tocar MR-01. NO tocar data vault.
NO commitear `local_outputs_do_not_commit/` ni `scratch/`. NO ZIP. NO git add . NO root files.
Dataset: SOLO `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared/`.
`sealed_holdout_2025_2026/` NUNCA se abre.

============================================================
PASOS (POST-GATES)
============================================================

1. Confirmar GATE 1/2/3. Si falla cualquiera → ABORTAR con el código correspondiente.
2. Correr la suite sin backtest:
   `test_cost_profiles.py` · `test_metric_reconciliation.py` · `test_engine.py`
   · `test_engine_stop_entry.py` · `test_lab_preflight*.py`. Falla → `BLOCKED_TEST_FAILURE`.
3. Regenerar SOLO TP-01 train-only 2015–2024 con el runner oficial, 3 profiles reales
   (base/normal_mode, conservative/conservative_mode, stress/stress_mode). M1.
4. **Gate obligatorio antes de sellar**: `metric_reconciliation.reconcile_all` sobre
   trades+equity+summary+profiles ⇒ CERO violaciones. Verificar:
   `ending_equity ≈ 100000 + Σ pnl_usd`; `stop_loss⇒loss`/`take_profit⇒win`;
   `drawdown_pct` no todo 0; PF/expectancy/total_return coherentes;
   cada `summary.json` self-report su profile; base<conservative<stress en costo;
   `RUN_MANIFEST` sin perfiles duplicados.
5. Comparar vs artefacto pre-fix (debe pasar de suspect a reconciliado). Esperado: edge
   neto negativo (~ −9% additive base), maxDD real ~8–9%, NO +135%; conservative/stress
   peores que base monotónicamente.
6. Entregar a auditoría externa (protocolo v3) antes de cualquier conclusión.

============================================================
SALIDA
============================================================

- Crear `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/TP01_POST_FIX_RECONCILIATION_REPORT.md`
  (estado, evidencia gate-green, comparación pre/post, edge real, costo por profile).
- NO declarar champion / rentable / incubation / FTMO/demo/real.
- TP-01 sigue siendo candidato a rechazo (PF<1, expectancy<0, 0 trades 2019–2024);
  la reconciliación valida métricas, NO rehabilita la estrategia.
- MR-01 permanece **bloqueado** hasta TP-01 regenerado limpio y gate-green.
- Stage explícito: solo docs/código necesario + tests. NO data/heavy/scratch/ZIP/root. NO git add .
