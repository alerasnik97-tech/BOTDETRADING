# NEXT PROMPT: FORMAL TRAIN-ONLY BACKTEST — MR-01 (GATED / BLOCKED)

Actuá como Claude Opus 4.7 Max en modo **Institutional Quant Backtest Operator** senior,
backtest safety engineer y data-leakage auditor.

============================================================
ESTADO DE BLOQUEO (LEER PRIMERO — NO EJECUTAR)
============================================================

Este prompt está **BLOQUEADO**. NO ejecutar todavía.

Precondición dura: el fix de
`06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_FIX_TP01_METRIC_INCONSISTENCY.md`
debe estar **mergeado, reconciliado y reportado** en
`06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/TP01_METRIC_FIX_REPORT.md` con estado
`TP01_METRIC_FIX_DONE`, con el **gate de reconciliación ledger↔dossier** en verde.

Motivo: la auditoría externa (v3, tercera reproducción determinística sobre commit
`66e42063`) probó 4 defectos en la capa **compartida** de métricas/costos del engine
(equity desacoplada del ledger; inversión de signo en ~49% de trades; `summary.json`
auto-contradictorio; cost profiles mislabeled/duplicados — sólo 2 corridas reales,
conservative inexistente, stress en conservative_mode). Correr MR-01 sobre el harness
actual produciría un dossier con **corrupción idéntica**. NO ejecutar hasta levantar el gate.

============================================================
OBJETIVO (POST-GATE)
============================================================

Tras levantar el gate, ejecutar el backtest formal **train-only 2015–2024** de **MR-01**
(siguiente estrategia en cola tras el bloqueo de TP-01), en proceso fresco y aislado,
sobre 3 cost profiles **realmente diferenciados** (base/normal, conservative/conservative,
stress/stress).

============================================================
REGLAS ABSOLUTAS
============================================================

NO main. NO force push. NO merge. NO rebase.
NO holdout. NO 2025/2026. NO news. NO high precision (`price_source='bid'`).
NO optimization. NO sweep. NO validation. NO WFA. NO F06.
NO tocar engine salvo lo ya mergeado por el metric/cost-fix.
NO tocar data cruda / vault. NO commitear `local_outputs_do_not_commit/`. NO ZIP. NO git add .
Dataset: SOLO `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared/`.
`sealed_holdout_2025_2026/` NUNCA se abre.

============================================================
PASOS (POST-GATE)
============================================================

1. Verificar `TP01_METRIC_FIX_REPORT.md == TP01_METRIC_FIX_DONE` y gate de reconciliación verde.
   Si no → ABORTAR `BLOCKED_METRIC_FIX_NOT_DONE`.
2. Confirmar identidad/parámetros de MR-01 (config snapshot) antes de correr.
3. Ejecutar la corrida formal train-only 2015–2024 (M1, 3 profiles realmente distintos).
4. Generar dossier liviano (RUN_MANIFEST, CONFIG_SNAPSHOT, FORMAL_DOSSIER.md,
   tablas anual/mensual/cost/distribution, SHA256 manifest).
5. **Auto-reconciliación obligatoria** antes de sellar: recomputar desde `trades.csv`
   (`ΣR, PF, expectancy, ending_equity, max_dd`), confirmar invariantes de signo
   (`stop_loss⇒loss`, `take_profit⇒win`), que `equity` deriva del ledger, y que
   cada `summary.json` auto-reporta su propio `cost_profile`/`execution_mode` (unicidad real).
6. Auditar actividad anual (detectar inactividad/regime drift como en TP-01).

============================================================
SALIDA
============================================================

- NO declarar champion. NO declarar rentable. NO promover a incubation/FTMO/demo/real.
  Sólo producir el dossier formal + su reconciliación.
- Entregar a auditoría externa (mismo protocolo v3 que TP-01) antes de cualquier conclusión.
- Stage explícito sólo de reportes livianos. NO `git add .`. NO data pesada. NO ZIP.

============================================================
RECORDATORIO
============================================================

TP-01 quedó **BLOQUEADA por integridad de métricas** y, defecto-independiente, con
**edge neto negativo** (PF<1, expectancy −0.068R, mediana R −0.55, 3 años negativos /
1 positivo) y **0 trades 2019–2024**. Fuera de la primera ola. MR-01 no hereda ninguna
asunción de viabilidad de TP-01.
