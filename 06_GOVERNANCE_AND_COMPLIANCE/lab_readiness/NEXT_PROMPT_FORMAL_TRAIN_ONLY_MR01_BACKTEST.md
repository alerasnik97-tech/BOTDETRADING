# NEXT PROMPT: FORMAL TRAIN-ONLY BACKTEST — MR-01 (GATED)

Actuá como Claude Opus 4.7 Max en modo **Institutional Quant Backtest Operator** senior,
backtest safety engineer y data-leakage auditor.

============================================================
ESTADO DE BLOQUEO (LEER PRIMERO)
============================================================

Este prompt está **BLOQUEADO** y NO debe ejecutarse todavía.

Precondición dura: el fix de
`06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_FIX_TP01_METRIC_INCONSISTENCY.md`
debe estar **mergeado, reconciliado y reportado** en
`06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/TP01_METRIC_FIX_REPORT.md`
con estado `TP01_METRIC_FIX_DONE`.

Motivo: la auditoría externa
(`TP01_FORMAL_DOSSIER_EXTERNAL_AUDIT_REPORT.md`) probó que la capa de
métricas/equity es **compartida** y tiene dos defectos (equity desacoplada del
ledger; inversión de signo direccional en ~49% de trades). Correr MR-01 sobre el
harness actual produciría un dossier con **corrupción idéntica** y métricas no
confiables. NO ejecutar MR-01 hasta levantar el gate.

============================================================
OBJETIVO (POST-GATE)
============================================================

Una vez levantado el gate, ejecutar el backtest formal **train-only** 2015–2024
de la estrategia **MR-01** (la siguiente en cola tras el rechazo de TP-01),
en proceso fresco y aislado, sobre los 3 cost profiles (base, conservative, stress).

============================================================
REGLAS ABSOLUTAS
============================================================

NO holdout. NO 2025/2026. NO news. NO high precision (price_source bid).
NO optimization. NO sweep. NO validation. NO WFA.
NO tocar engine salvo lo ya mergeado por el metric-fix.
NO tocar data cruda ni el vault.
NO commitear `local_outputs_do_not_commit/` (heavy) ni ZIP ni root files.
Dataset: SOLO `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared/`.
`sealed_holdout_2025_2026/` NUNCA se abre.

============================================================
PASOS REQUERIDOS (POST-GATE)
============================================================

1. Verificar `TP01_METRIC_FIX_REPORT.md == TP01_METRIC_FIX_DONE` y que el
   gate de reconciliación (test ledger↔dossier) está verde. Si no, ABORTAR.
2. Confirmar identidad/parametrización de MR-01 antes de correr (config snapshot).
3. Ejecutar la corrida formal train-only 2015–2024 (M1, normal_mode, 3 profiles).
4. Generar dossier liviano: `RUN_MANIFEST.json`, `CONFIG_SNAPSHOT.json`,
   `*_FORMAL_DOSSIER.md`, tablas anuales/mensuales/cost/distribution + SHA256 manifest.
5. **Auto-reconciliación obligatoria**: recomputar desde `trades.csv`
   (`Σpnl_r`, `PF`, `expectancy`, `ending_equity`, `max_dd`) y confirmar que
   reconcilian con el dossier antes de sellar. Confirmar invariantes de signo
   (`stop_loss⇒loss`, `take_profit⇒win`) y que `equity` deriva del ledger.
6. Auditar actividad anual (detectar inactividad/regime drift como en TP-01).

============================================================
CRITERIO DE SALIDA
============================================================

- NO declarar champion. NO declarar rentable. NO promover a incubation.
  Sólo producir el dossier formal train-only + su reconciliación.
- Entregar el dossier a auditoría externa (mismo protocolo que TP-01) antes de
  cualquier conclusión de viabilidad.
- Stage explícito sólo de reportes livianos. NO `git add .`. NO data pesada.

============================================================
RECORDATORIO
============================================================

TP-01 quedó **RECHAZADA y fuera de la primera ola** (edge neto negativo en todo
view confiable + 0 trades 2019–2024 + dossier bloqueado por integridad de
métricas). MR-01 no hereda ninguna asunción de viabilidad de TP-01.
