Actuá como Claude Opus 4.7 Max en modo:

1. Institutional Quant Backtest Operator.
2. Metric Reconciliation Gatekeeper.
3. Data-Leakage Prevention Officer.
4. Output Policy Officer.
5. Git Safety Officer.

OBJETIVO
Ejecutar la PRIMERA corrida formal y sellado de la estrategia MR-01 usando únicamente el runner oficial:

research_lab.runners.formal_train_runner

Estrategia:
mr01_anchor_elastic

Scope:
TRAIN-ONLY 2015-01-01 a 2024-12-31

Esta ejecución está autorizada por la auditoría externa v3 del dossier de TP-01:
DECISION = TP01_CLOSED_REJECTED_MR01_RELEASED_FOR_FORMAL_RUN

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
NO walk-forward.
NO F06.
NO news.
NO high precision.
NO segunda estrategia (ejecutar únicamente mr01_anchor_elastic).
NO tocar engine.py.
NO tocar report.py.
NO tocar data_loader.py.
NO tocar runner.
NO tocar strategy code.
NO tocar data vault.
NO modificar datos.
NO ZIP.
NO root files.
NO commitear local_outputs_do_not_commit.
NO commitear trades.csv.
NO commitear equity_curve.csv.
NO declarar edge.
NO declarar rentable.
NO declarar champion.
NO declarar FTMO/demo/real.

============================================================
BLOQUE 0 — PRECHECK Y CREACIÓN DE RAMA
============================================================
1. Verificar que no haya procesos activos de Python.
2. Crear la rama oficial de investigación para MR-01 a partir de la rama de auditoría actual:
   `git switch audit/tp01-regenerated-dossier-v3-20260517`
   `git switch -c research/mr01-official-runner-run-20260517`

============================================================
BLOQUE 1 — TESTS PREVIOS
============================================================
Correr la suite de 110/110 tests para asegurar estabilidad absoluta antes del run:
$env:PYTHONPATH="03_RESEARCH_LAB"
python -m unittest discover -s "03_RESEARCH_LAB\research_lab\tests" -v

============================================================
BLOQUE 2 — DRY-RUN OBLIGATORIO
============================================================
Correr el dry-run oficial sin --execute:
python -m research_lab.runners.formal_train_runner \
  --strategy mr01_anchor_elastic \
  --start 2015-01-01 \
  --end 2024-12-31 \
  --data-path "05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared" \
  --output-dir "03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_HHMMSS"

Confirmar preflight y parámetros en consola.

============================================================
BLOQUE 3 — EJECUCIÓN REAL MR-01
============================================================
Correr una sola vez agregando el flag `--execute`:
python -m research_lab.runners.formal_train_runner \
  --strategy mr01_anchor_elastic \
  --start 2015-01-01 \
  --end 2024-12-31 \
  --data-path "05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared" \
  --output-dir "03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/mr01_anchor_elastic/MR01_OFFICIAL_RUNNER_RUN_2015_2024_20260517_HHMMSS" \
  --execute

============================================================
BLOQUE 4 — VALIDAR ARTEFACTOS Y REPORTE
============================================================
1. Verificar que el manifiesto se selle con `sealed: True`.
2. Crear un reporte de reconciliación liviano y el reporte de lab readiness bajo `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/MR01_POST_RUN_RECONCILIATION_REPORT.md`.
3. Stagear explícitamente y de manera segura (con `git add` uno a uno) los reportes livianos y de configuración, dejando `local_outputs_do_not_commit` ignorado y libre de tracking.
4. Confirmar el commit y subir la rama a origin.
