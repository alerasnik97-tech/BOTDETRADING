> ---
> ## ADDENDUM 2026-05-15 -- warning closure supersedes this warning state
> Follow-up hardening is recorded in `PR4_WARNING_CLOSURE_REPORT.md`.
> Current evidence is **78/78 tests OK**. W1 is closed on complete synthetic
> output fixtures, W2 is closed, W3 is closed with limits, and W4 is closed
> with lightweight stdlib runtime validators. F06/F08/F12 remain
> **NOT CERTIFIED** and Fase 3 remains blocked until external audit accepts PR4.
> ---

# PR4 INTERNAL SURGICAL AUDIT

**Fecha:** 2026-05-15
**PR:** #4 — Add F06 evidence rebuild foundation guards
**Branch:** `research/f06-evidence-rebuild-foundation-20260515`
**Base:** `research/pre-claude-blocker-remediation-20260515`
**Auditoría:** interna, pre-Claude, brutalmente honesta. Se buscaron fallos
ANTES de la auditoría nocturna y se corrigieron los críticos.

---

## 1. Status

**PR4_READY_WITH_WARNINGS**

Se encontró **1 bug crítico real (fail-OPEN)** y varios huecos. El crítico y
los huecos de bajo riesgo fueron corregidos. Quedan WARNINGS documentados
(no bloqueantes para la fase foundation/dry, sí obligatorios antes de FASE 3).

## 2. Executive Summary

La foundation NO estaba tan lista como reportaba la Fase 2. Hallazgo
principal: `_mini_yaml_load` (fallback usado si PyYAML no está instalado)
**mis-parseaba listas** (`families:` + `- F06` → `{"families":{"families":
[...]}}`). Como **PyYAML 6.0.3 SÍ está instalado**, ese código nunca se
ejecutó y los tests previos no lo cubrían: era un **fail-OPEN latente** —
si el entorno de auditoría no tuviera PyYAML, el guard de 2025/2026
iteraría claves de un dict en vez de fechas y **no detectaría un 2025**.
Corregido y ahora testeado directamente. Además se cerró el hueco de
"hash falso" en el manifest, se extrajo/testeó el guard de output_dir, se
endureció el schema y se agregaron 6 tests. 54/54 tests OK.

## 3. Git / Diff Review

| Check | Resultado |
| :--- | :--- |
| Branch correcto | `research/f06-evidence-rebuild-foundation-20260515` |
| HEAD (pre-fix) | `398162a8` |
| Diff vs base | 29 files, 1894 ins — **solo** `pipelines/f06_evidence_rebuild/**` y `reports/f06_evidence_rebuild_foundation/**` |
| Paths inesperados | NINGUNO (no BLOCKED_UNEXPECTED_DIFF) |
| Heavy / raw / tick / parquet / zip / lock | NINGUNO |
| Outputs viejos contaminados agregados | NINGUNO |
| Working tree | Solo untracked pre-existente ajeno al PR (no staged) |

**Git: CLEAN.**

## 4. Code Review Findings

| # | Check | Resultado |
| :-- | :--- | :--- |
| 1 | No corre backtest real | OK |
| 2 | dry_run no lee raw data | OK |
| 3 | No referencia carpetas quarantined como input | OK (tokens solo como BLOCKLIST) |
| 4 | No usa `V50B_RERUN_TRADES.csv` | OK (en blocklist) |
| 5 | No usa `MASTER_RANKING.csv` viejo | OK (en blocklist) |
| 6 | validation=true prohibido | OK |
| 7 | holdout=true prohibido | OK |
| 8 | 2025/2026 prohibido | OK (con WARNING W2) |
| 9 | Falla si output_dir existe | OK — **FIX**: extraído a `check_output_dir_absent` + test |
| 10 | Falla si script no trackeado | OK (demostrado: dry_run pre-commit = BLOCKED) |
| 11 | Falla si manifest incompleto | OK |
| 12 | Falla si falta hash | OK — **FIX**: ahora exige sha256-hex real |
| 13 | Falla si ledger multi run_id | OK a nivel guard (WARNING W1: no cableado en validate_outputs) |
| 14 | Falla si validation columns | OK a nivel guard (WARNING W1) |
| 15 | Falla si sample < floor | OK |
| 16 | Cost model exige spread/slippage/round-turn | OK |
| 17 | Sin bypass force/ignore_errors/allow_unsafe | OK — **FIX**: ahora test-enforced |
| 18 | No silencia errores críticos como warning | OK (dry_run imprime "warn" pero igual hard-falla vía manifest) |

### Bug crítico encontrado y corregido
**C1 — `_mini_yaml_load` fail-OPEN.** Mis-nesteaba `key:` + items `- `.
Bajo el fallback (sin PyYAML), `data_scope.exact_months` quedaba como dict
y `check_no_2025_2026` iteraba la clave `"exact_months"` en vez de las
fechas → un 2025 NO se detectaría. **Reescrito** (stack con
parent+key; convierte el child a lista al ver el primer `- `). **Testeado
directamente** por `test_config_parsing` (corre el fallback aunque PyYAML
esté instalado).

## 5. Schema Review Findings

- `manifest_schema.json`: consts de seguridad correctos
  (`train_only=true`, `validation_evaluated=false`, `holdout_touched=false`,
  `allow_2025/2026=false`, `input_is_quarantined_path=false`,
  `script_is_tracked=true`), `output_hashes` obligatorio + patrón sha256,
  `status` enum, `safety_flags`/`cost_model` con los 3 componentes.
  **FIX**: `exact_months` regex `[0-1][0-9]` (permitía mes 00/19) →
  `(0[1-9]|1[0-2])`.
- `ledger_schema.json` / `ranking_schema.json` / `cost_report_schema.json`:
  correctos como **contrato declarativo**.
- **Hallazgo honesto (WARNING W4):** los `.json` de schema **NO** se ejecutan
  contra nada en runtime (no hay dependencia `jsonschema` cableada). El
  enforcement real es `validate_manifest` (subset liviano, ahora con
  hex) + los `check_*`. Los schemas son spec/aspiracional salvo el manifest.

## 6. Test Quality Findings

- Asserts fuertes, fixtures sintéticos, casos positivos+negativos,
  fail-closed-on-empty. No pasan con el guard roto (verificado).
- **Huecos encontrados y cubiertos (6 tests nuevos):**
  `test_config_parsing` (cubre C1 y el fallback),
  `test_no_unsafe_override_flags` (scan de fuente),
  `test_manifest_status_allowed_values`,
  `test_output_dir_must_not_exist`,
  `test_old_v50b_paths_explicitly_forbidden`,
  `test_manifest_hash_hex_enforced`.
- Hueco residual (WARNING W1): no hay test de `validate_output_dir`
  inspeccionando un ledger/ranking real porque la función todavía no lo
  hace (ver Blockers).

## 7. Test Run Results

- **Comando:** `python -m unittest discover -s pipelines/f06_evidence_rebuild/tests -p "test_*.py"`
  (pytest NO instalado → `unittest` stdlib, sin instalar dependencias)
- **Total:** 54 · **passed:** 54 · **failed:** 0 · **warnings:** 0
  (pre-fix eran 35; +19 por los 6 tests nuevos)

## 8. Dry Run / Validate Config Results

- `validate_config` → **PASS**.
- `dry_run` → **DRY_RUN_SCHEMA_VALIDATED** (script trackeado post-commit).
- dry_run NO generó trades reales (verificado: sin archivos TRADES/RANKING/RERUN).
- dry_run NO generó ranking real.
- dry_run NO leyó data real (`input_dataset_path = DRY_RUN_NO_INPUT`).
- PyYAML 6.0.3 presente → parser activo es el real; fallback ahora también correcto y testeado.

## 9. Blockers Found

| Blocker | Severity | Fixed | Remaining risk |
| :--- | :--- | :--- | :--- |
| C1: `_mini_yaml_load` mis-nesta listas (fail-OPEN 2025/2026 sin PyYAML) | **CRÍTICA** | **SÍ** (reescrito + `test_config_parsing`) | Bajo: parser solo soporta el template fijo (sin anchors/flow) — aceptable, documentado |
| C2: manifest aceptaba hashes falsos/cortos | ALTA | **SÍ** (hex sha256 enforced + test) | Ninguno material |
| C3: guard `output_dir_must_not_exist` inline, sin test | MEDIA | **SÍ** (`check_output_dir_absent` + test) | Ninguno |
| C4: `manifest_schema` mes regex laxo (00/19) | BAJA | **SÍ** (regex corregido) | Ninguno |
| W1: `validate_output_dir` superficial — no abre ledger/ranking reales ni verifica que `output_hashes` matcheen archivos en disco | ALTA (para FASE 3) | **NO** | **Bloqueante antes de FASE 3**, no para foundation/dry (aún no hay outputs reales) |
| W2: `check_no_2025_2026` por substring (bypass por epoch / formatos sin año; falsos positivos en numéricos) | MEDIA | NO (documentado) | FASE 3 debe validar columnas datetime tipadas |
| W3: `check_config_uniqueness` heurístico (50%/single) | MEDIA | NO (documentado) | FASE 3 debe probar que cada config cambia el comportamiento, no solo la tupla |
| W4: schemas `.json` no enforced en runtime (salvo manifest liviano) | MEDIA | NO (documentado) | FASE 3 debería cablear validación de schema real |
| W5: `read_csv` carga todo en memoria | BAJA | NO | FASE 3 con ledger grande → streaming |
| W6: `check_script_tracked` depende del CWD para git | BAJA | NO (fail-closed seguro) | FASE 3 resolver repo root explícito |

## 10. Changes Made

- `scripts/f06_rebuild_pipeline.py`: reescrito `_mini_yaml_load`
  (fix C1); `validate_manifest` ahora exige sha256-hex en
  `script_sha256`/`config_sha256`/`output_hashes` (C2); extraído
  `check_output_dir_absent` y cableado en `cmd_dry_run` (C3).
- `schemas/manifest_schema.json`: regex `exact_months` endurecido (C4).
- `tests/`: +6 tests (config_parsing, no_unsafe_override_flags,
  manifest_status_allowed_values, output_dir_must_not_exist,
  old_v50b_paths_explicitly_forbidden, manifest_hash_hex_enforced).
- `reports/.../DRYRUN_MANIFEST.json`: regenerado con el validador endurecido.
- Nuevo: este `PR4_INTERNAL_SURGICAL_AUDIT.md`.

Ningún cambio corre estrategia/backtest ni toca datos.

## 11. Safety Verification

- strategy_run: NO
- backtest_run: NO
- validation_touched: NO
- holdout_touched: NO
- raw_data_mutated: NO
- old_quarantined_outputs_used: NO
- zip_used_as_primary_delivery: NO

## 12. Decision

**READY_FOR_CLAUDE_NIGHT_AUDIT** (con WARNINGS documentados; W1 es
bloqueante para FASE 3, NO para esta foundation).

## 13. Copy-Paste Summary for ChatGPT

```
PR#4 AUDITADO INTERNAMENTE → PR4_READY_WITH_WARNINGS.
Bug CRÍTICO encontrado: _mini_yaml_load mis-nesteaba listas → fail-OPEN del
guard 2025/2026 si PyYAML ausente (PyYAML 6.0.3 está instalado, por eso
nunca se ejecutó y no estaba testeado). CORREGIDO + test directo del
fallback. También corregido: manifest aceptaba hashes falsos (ahora exige
sha256-hex), guard output_dir extraído+testeado, regex mes endurecido.
+6 tests. 54/54 OK. validate_config PASS, dry_run DRY_RUN_SCHEMA_VALIDATED,
sin tocar datos. WARNINGS documentados (no bloquean foundation): W1
validate_output_dir superficial (BLOQUEANTE antes de FASE 3), W2 guard
2025/2026 por substring, W3 uniqueness heurístico, W4 schemas .json no
enforced en runtime. F06/F08/F12 siguen NOT CERTIFIED. Próximo:
auditoría nocturna Claude. NO avanzar a FASE 3.
```
