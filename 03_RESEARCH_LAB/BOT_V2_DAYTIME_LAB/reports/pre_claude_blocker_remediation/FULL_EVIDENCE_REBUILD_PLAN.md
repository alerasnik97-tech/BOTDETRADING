# FULL EVIDENCE REBUILD PLAN

**Fecha:** 2026-05-15
**Estado:** PLAN ONLY — NO EJECUTAR sin aprobación explícita
**Precondición:** Freeze institucional vigente (`PRE_CLAUDE_FREEZE_NOTICE.md`)

---

## 1. Objective

Reconstruir evidencia institucional de F06 **sin contaminación, sin
leakage, totalmente reproducible y auditable desde GitHub**, partiendo de
cero. No se busca edge ni se certifica nada en este plan; se define la
arquitectura que la reconstrucción **debe** cumplir.

## 2. Non-Negotiable Rules

- NO 2025. NO 2026. NO validation. NO holdout.
- NO usar `V50B_RERUN_TRADES.csv` ni `V50B_RERUN_MASTER_RANKING.csv` viejos.
- NO leer desde carpetas cuarentenadas.
- NO source-of-truth dentro de un directorio cuarentenado.
- NO script generador ausente ni untracked.
- NO certificación con N=10 ni con muestras triviales.
- NO contar configs duplicadas como independientes.
- NO ZIP como entrega. NO raw/tick/parquet/lock en el branch de evidencia.
- NO force push. NO history rewrite destructivo. NO borrar evidencia previa.
- NO re-correr lo viejo "para arreglar"; se construye pipeline nuevo.

## 3. Required Architecture

Output path nuevo (timestamped, único, fuera de cuarentena):

```
03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/v50b_f06_evidence_rebuild_YYYYMMDD_HHMM/
```

Debe contener exactamente:

- `run_id` único (1 run = 1 id).
- `ledger/TRADES_<run_id>.csv` — un solo run_id en todo el archivo.
- `MANIFEST_<run_id>.json` (ver §4).
- `config/CONFIG_<run_id>.yaml` (parámetros explícitos, versionado).
- `COMMAND.txt` (línea exacta ejecutada).
- `ENVIRONMENT.txt` (python, libs, OS, semilla).
- `results/RANKING_<run_id>.csv` (sin columnas de validación en modo train).
- `COST_REPORT_<run_id>.md` (modelo de costos con spread real).
- `HASHES.txt` (sha256 de inputs, script, config, cada output).
- `ROW_COUNTS.txt`.
- `SAFETY_VERIFICATION.md`.
- Script generador **trackeado** y referenciado por hash.

## 4. Required Manifest

`MANIFEST_<run_id>.json` debe incluir:

- `run_id`
- `script_path` + `script_sha256`
- `config_sha256`
- `input_dataset_path` + `input_dataset_sha256` (dataset train-only, fuera de cuarentena)
- `date_scope` (rango explícito, train-only)
- `symbol` = EURUSD
- `families` evaluadas
- `exact_months` (lista explícita; justificación si no es continuo)
- `train_only: true`
- `validation_evaluated: false`
- `holdout_touched: false`
- `row_count_input`, `trade_count`, `rejected_count`
- `output_hashes` (mapa archivo→sha256)
- `git_commit_sha`, `git_branch`
- `generated_at`, `generator_pid`

Un manifest sin estos campos = inválido por construcción.

## 5. Required Guards (abort-on-violation, fail-closed)

- ABORT si el output dir ya existe.
- ABORT si el ledger contiene más de un `run_id`.
- ABORT si se generan columnas de validación en modo train-only.
- ABORT si aparece cualquier datetime de mercado 2025 o 2026.
- ABORT si cualquier path de input resuelve a una carpeta cuarentenada.
- ABORT si el script generador no está trackeado en git.
- ABORT si se detecta degeneración de configs (≤K tuplas únicas para N configs).
- ABORT si configs duplicadas producen outputs idénticos sin estar marcadas
  explícitamente como deduplicadas.
- ABORT si falta cualquier campo obligatorio del manifest o algún hash.

## 6. Required Tests (deben pasar antes de declarar evidencia)

- `test_single_run_id_ledger` — el ledger tiene exactamente 1 run_id.
- `test_no_validation_columns_in_train_output`.
- `test_no_quarantined_input_path`.
- `test_no_2025_2026_market_datetime`.
- `test_producing_script_is_tracked`.
- `test_config_uniqueness` (cada config declarada produce un resultado
  distinto o está marcada como duplicada a propósito).
- `test_manifest_hashes_match_outputs`.
- `test_cost_model_components` — spread + slippage + comisión round-turn
  presentes y parametrizados; STRESS = worst-case real.
- `test_sample_size_floor` — N por familia por encima de un piso
  estadístico definido (no N=10).

## 7. Required Future Steps (secuencia, una fase a la vez)

1. **Fase 1 — Forensic freeze** (este pack). ✅ en curso.
2. **Fase 2 — Pipeline rebuild** (script trackeado + guards + tests).
3. **Fase 3 — Clean F06 train-only rerun** (1 run_id, dataset limpio).
4. **Fase 4 — Cost hardening con spread real** (round-turn + news widening).
5. **Fase 5 — Monte Carlo train-only**.
6. **Fase 6 — Walk-forward train-only** (purgado + embargo).
7. **Fase 7 — Nueva auditoría Claude** sobre evidencia reconstruida.
8. **Fase 8 — Solo entonces**, diseñar (no ejecutar) V50C.

Ninguna fase avanza si la anterior no pasa sus tests y su auditoría.

## 8. Out of Scope (de este plan y de su futura ejecución)

- NO V50C. NO validation. NO holdout. NO 2025. NO 2026.
- NO demo. NO FTMO. NO real. NO nuevas familias. NO sweeps de optimización.

## 9. Definition of Done (Fase 1, este pack)

- Freeze notice publicado.
- Forensic report con hallazgos confirmados.
- Certificaciones previas superseded (sin borrar nada).
- Este plan publicado.
- Audit pack liviano en GitHub, auditable, sin data pesada.
