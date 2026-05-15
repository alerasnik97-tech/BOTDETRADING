# F06 EVIDENCE REBUILD — FAIL-CLOSED FOUNDATION

**Estado:** SCAFFOLD ONLY — NO ejecuta estrategia, NO corre backtest, NO
certifica nada. F06/F08/F12 = NOT CERTIFIED.

Este pipeline existe para que **el laboratorio no pueda volver a mentirse**.
Codifica como guards fail-closed los blockers que la auditoría extrema de
Claude detectó en V50B.

## Por qué existe

La evidencia V50B/F06 fue invalidada (ver
`reports/pre_claude_blocker_remediation/`). Antes de reconstruir nada, se
construye la infraestructura que hace **imposible** repetir:

| Blocker V50B | Guard que lo previene |
| :--- | :--- |
| Multi-RunID contamination | `check_single_run_id` |
| N=10 usado para certificar | `check_sample_size_floor` (floor=100) |
| Ranking degenerado (150 cfg / 6 tuplas) | `check_config_uniqueness` |
| Columnas validation en train-only | `check_no_validation_columns` |
| SoT dentro de carpeta cuarentenada | `check_no_quarantined_path` |
| Script generador ausente/untracked | `check_script_tracked` |
| Manifest incompleto (stub 4 líneas) | `check_manifest` + `manifest_schema.json` |
| Cost model sin spread | `check_cost_model_components` |
| Holdout/2025-2026 leakage | `check_no_2025_2026` |
| GitHub sin trazabilidad | manifest git_commit/branch + tracked script |

## Estructura

```
f06_evidence_rebuild/
├── README.md                  (este archivo)
├── configs/                   template train-only + schema explicado
├── scripts/                   pipeline scaffold (dry_run) + validator
├── schemas/                   manifest / ledger / ranking / cost_report
├── tests/                     guards fail-closed (fixtures sintéticos)
└── fixtures/                  CSV sintéticos (NO data real)
```

## Comandos seguros (no corren estrategia)

```
python scripts/f06_rebuild_pipeline.py validate_config --config <yaml>
python scripts/f06_rebuild_pipeline.py dry_run        --config <yaml>
python scripts/f06_rebuild_pipeline.py validate_outputs --output-dir <dir>
python scripts/validate_rebuild_outputs.py --output-dir <dir> --manifest <m> --config <c>
```

`dry_run` valida config + paths + guards y emite un manifest con status
`DRY_RUN_SCHEMA_VALIDATED` o `BLOCKED_GUARD_FAILED`. **Nunca** lee raw data
ni produce trades reales.

## Reglas absolutas (fail-closed)

El pipeline ABORTA (fail-closed) si: el output dir ya existe; un input
contiene tokens de cuarentena; la config permite 2025/2026; validation u
holdout habilitados; el script no está trackeado; falta manifest o hashes;
aparecen columnas de validación en modo train-only; el ledger tiene >1
run_id; el ranking es degenerado; o el sample size < floor.

## Fuera de alcance

NO V50C. NO validation. NO holdout. NO 2025/2026. NO sweeps. NO
optimización. NO nuevas familias. NO usar outputs viejos contaminados.
La FASE 3 (Clean F06 Train-only Rerun) solo se autoriza tras auditoría
Claude de esta base.
