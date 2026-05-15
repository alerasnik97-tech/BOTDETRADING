> ---
> ## ADDENDUM 2026-05-15 -- PR4 warning closure
> `PR4_WARNING_CLOSURE_REPORT.md` supersedes the previous warning state.
> Current evidence: **78/78 tests OK**, `validate_config` PASS, `dry_run`
> `DRY_RUN_SCHEMA_VALIDATED`, `output_good` `READY_FOR_CLAUDE_AUDIT`, and all
> bad synthetic output fixtures `BLOCKED_GUARD_FAILED`. W1 is closed on
> complete synthetic fixtures, W2 is closed, W3 is closed with limits, and W4
> is closed with lightweight stdlib runtime validators. F06/F08/F12 remain
> **NOT CERTIFIED**. No strategy/backtest/validation/holdout/2025/2026 was run
> or touched.
> ---

> ---
> ## ℹ️ ADDENDUM 2026-05-15 — superseded by internal surgical audit
> Una auditoría interna posterior (`PR4_INTERNAL_SURGICAL_AUDIT.md`) encontró
> y corrigió un bug crítico **fail-OPEN** (`_mini_yaml_load`) más otros
> huecos. Cifras vigentes: **54/54 tests OK** (no 35/35). Estado:
> PR4_READY_WITH_WARNINGS. Este reporte se conserva como registro histórico
> de la Fase 2; la fuente de verdad de tests/hallazgos es el audit PR4.
> ---

# F06 EVIDENCE REBUILD FOUNDATION REPORT

**Fecha:** 2026-05-15
**Fase:** 2 — Foundation scaffold (NO strategy run)
**Branch:** `research/f06-evidence-rebuild-foundation-20260515`

---

## 1. Status

**FOUNDATION_READY_FOR_CLAUDE_AUDIT**

## 2. Executive Summary

Se construyó el esqueleto fail-closed que codifica como guards y tests cada
blocker confirmado en V50B. No se corrió estrategia, backtest, validation,
holdout, ni se tocó 2025/2026. 35/35 tests pasan con `unittest` (pytest no
disponible). El scaffold se auto-bloqueó correctamente (`dry_run` →
`BLOCKED_GUARD_FAILED`) porque el script generador aún no estaba trackeado:
prueba viva de comportamiento fail-closed. Tras el commit, el script queda
trackeado y `dry_run` pasa a `DRY_RUN_SCHEMA_VALIDATED` (ver §6).

## 3. Why This Exists

F06/F08/F12 = **NOT CERTIFIED**. La evidencia V50B fue invalidada
(`reports/pre_claude_blocker_remediation/`). Este trabajo **no certifica
nada**: solo crea la infraestructura que hace imposible repetir los errores.
El objetivo es que el laboratorio no pueda volver a mentirse.

## 4. Claude Blockers Addressed

| Blocker | Guard / Test / Doc | Status |
| :--- | :--- | :--- |
| Multi-RunID contamination | `check_single_run_id` + `test_single_run_id_guard` | ADDRESSED |
| N=10 certification | `check_sample_size_floor` (floor=100) + `test_sample_size_floor` | ADDRESSED |
| Ranking degeneracy | `check_config_uniqueness` + `test_config_uniqueness_guard` | ADDRESSED |
| Validation columns en train-only | `check_no_validation_columns` + `test_no_validation_columns_train_only` | ADDRESSED |
| Missing/untracked generator script | `check_script_tracked` + `test_script_tracked_guard` (dry_run se auto-bloqueó) | ADDRESSED |
| Quarantined Source of Truth | `check_no_quarantined_path` + `test_no_quarantined_inputs` | ADDRESSED |
| Cost model sin spread | `check_cost_model_components` + `test_cost_model_components` | ADDRESSED |
| Manifest incompleto | `manifest_schema.json` + `validate_manifest` + `test_manifest_schema`/`test_manifest_hashes` | ADDRESSED |
| Leakage 2025/2026 | `check_no_2025_2026` + `test_no_2025_2026_guard` | ADDRESSED |
| GitHub auditability | tracked scaffold + manifest git_commit/branch + PR draft | ADDRESSED |

## 5. Files Created

```
pipelines/f06_evidence_rebuild/
  README.md
  configs/F06_REBUILD_TRAIN_ONLY_TEMPLATE.yaml
  configs/SCHEMA.md
  schemas/manifest_schema.json
  schemas/ledger_schema.json
  schemas/ranking_schema.json
  schemas/cost_report_schema.json
  scripts/f06_rebuild_pipeline.py
  scripts/validate_rebuild_outputs.py
  tests/_loader.py
  tests/test_manifest_schema.py
  tests/test_single_run_id_guard.py
  tests/test_no_validation_columns_train_only.py
  tests/test_no_quarantined_inputs.py
  tests/test_no_2025_2026_guard.py
  tests/test_script_tracked_guard.py
  tests/test_config_uniqueness_guard.py
  tests/test_manifest_hashes.py
  tests/test_sample_size_floor.py
  tests/test_cost_model_components.py
  fixtures/synthetic_clean_ledger.csv
  fixtures/synthetic_bad_multi_runid_ledger.csv
  fixtures/synthetic_bad_validation_columns.csv
  fixtures/synthetic_bad_2025_rows.csv
  fixtures/synthetic_bad_duplicate_configs.csv
  fixtures/synthetic_cost_model_sample.csv
reports/f06_evidence_rebuild_foundation/
  BRANCH_STRATEGY_REPORT.md
  F06_EVIDENCE_REBUILD_FOUNDATION_REPORT.md
  DRYRUN_MANIFEST.json
```

Todos los fixtures son sintéticos (<10 KB). Cero raw/tick/parquet/zip/lock.

## 6. Tests Run

- **Comando:** `python -m unittest discover -s pipelines/f06_evidence_rebuild/tests -p "test_*.py"`
  (pytest NO instalado → método: `unittest` de stdlib, sin instalar dependencias)
- **Resultado:** `Ran 35 tests ... OK` (35 passed, 0 failed)

| Test | Result | Purpose |
| :--- | :--- | :--- |
| test_manifest_schema | PASS | Falla si falta cualquier campo o si un const inseguro |
| test_single_run_id_guard | PASS | Pasa clean, falla multi-runid, fail-closed si vacío |
| test_no_validation_columns_train_only | PASS | Falla con N_val/PF_val/.../combined_pass |
| test_no_quarantined_inputs | PASS | Falla con QUARANTINED/DO_NOT_USE/v50b_limited.../TRADES/RANKING |
| test_no_2025_2026_guard | PASS | Falla con cualquier 2025/2026 |
| test_script_tracked_guard | PASS | Falla si script no trackeado (simulado, sin git) |
| test_config_uniqueness_guard | PASS | Falla 50 configs→1 tupla sin flag deduplicated |
| test_manifest_hashes | PASS | sha256 determinista; rechaza output_hashes vacío |
| test_sample_size_floor | PASS | Falla N=10, pasa N>=100 |
| test_cost_model_components | PASS | Falla si falta spread/slippage/round-turn |

Demostración operativa (no corre estrategia):
- `validate_config` → **PASS**.
- `dry_run` (pre-commit) → **BLOCKED_GUARD_FAILED** (script aún no trackeado:
  fail-closed correcto).
- `dry_run` (post-commit, script trackeado) → **DRY_RUN_SCHEMA_VALIDATED**
  (ver `DRYRUN_MANIFEST.json` refrescado).

## 7. What Was NOT Done

- NO strategy run.
- NO backtest.
- NO validation.
- NO holdout.
- NO 2025.
- NO 2026.
- NO certification.
- NO sweeps / optimization / nuevas familias.
- NO uso de outputs viejos contaminados ni carpetas cuarentenadas.

## 8. Safety Verification

| Gate | Estado |
| :--- | :--- |
| test_touched | NO |
| validation_touched | NO |
| holdout_touched | NO |
| raw_data_mutated | NO |
| sweep_run | NO |
| optimization_run | NO |
| old_quarantined_outputs_used_as_input | NO |
| zip_used_as_primary_delivery | NO |
| active_research_process | NONE (precheck) |
| scope_escalation | NONE |

## 9. Decision

**READY_FOR_CLAUDE_NIGHT_AUDIT**

## 10. Next Step

**CLAUDE_NIGHT_AUDIT**

## 11. Copy-Paste Summary for ChatGPT

```
FASE 2 FOUNDATION COMPLETA — FOUNDATION_READY_FOR_CLAUDE_AUDIT.
Branch: research/f06-evidence-rebuild-foundation-20260515 (base = PR#3 freeze).
Scaffold fail-closed creado: pipeline + 4 schemas + config template +
validator + 6 fixtures sintéticos + 10 guard tests. 35/35 tests OK con
unittest (pytest no instalado). validate_config PASS. dry_run se auto-bloqueó
por script no trackeado (fail-closed correcto) y pasa a
DRY_RUN_SCHEMA_VALIDATED tras el commit. NO strategy/backtest/validation/
holdout/2025/2026. F06/F08/F12 siguen NOT CERTIFIED. Next: auditoría
nocturna Claude; solo si aprueba → FASE 3 Clean F06 Train-only Rerun.
```
