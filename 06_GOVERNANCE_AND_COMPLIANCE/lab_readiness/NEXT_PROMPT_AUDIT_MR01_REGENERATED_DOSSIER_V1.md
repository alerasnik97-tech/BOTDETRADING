Actuá como Claude Opus 4.7 Max en modo:

1. External Institutional Dossier Auditor.
2. Metric Reconciliation Auditor.
3. Backtest Integrity Reviewer.
4. Git Surface Auditor.
5. Cost Profile / Execution Mode Auditor.
6. Data Leakage Prevention Officer.
7. Strategy Rejection Gatekeeper.

OBJETIVO
Realizar la auditoría externa formal del dossier de MR-01 regenerado y sellado de forma 100% limpia.

Estrategia:
mr01_anchor_elastic

Scope:
TRAIN-ONLY 2015-01-01 a 2024-12-31

Run ID a Auditar:
MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_163509

============================================================
REGLAS ABSOLUTAS
============================================================
NO main.
NO force push.
NO merge.
NO rebase.
NO git add .
NO holdout.
NO sealed_holdout.
NO 2025/2026.
NO validation.
NO optimization.
NO sweep.
NO F06.
NO news.
NO high precision.
NO segunda estrategia.
NO tocar engine.py o report.py.
NO tocar runner.
NO tocar data.

============================================================
TAREAS DE AUDITORÍA
============================================================
1. Cargar el manifiesto `RUN_MANIFEST.json` de la corrida y validar que contenga la rama `research/mr01-official-runner-run-20260517` y el commit `8aa811832b43b16047496917ddabb670f2640e6f`.
2. Verificar que los reportes livianos y de perfil de costo existan y muestren concordancia exacta en las métricas (Base PF = 0.7122, Cons PF = 0.6534, Stress PF = 0.6070).
3. Confirmar que no se hayan violado las reglas de reconciliación en los perfiles.
4. Confirmar que los archivos pesados de trades/equity se encuentren en `local_outputs_do_not_commit` y no estén rastreados por Git.
5. Auditar el reporte `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/MR01_POST_RUN_RECONCILIATION_REPORT.md`.
6. Realizar la auditoría temporal y la estabilidad de la estrategia (trades por año y el hallazgo de 0 trades en 2016-2024).
7. Emitir el dictamen final y el reporte formal de auditoría externa de MR-01.
