# CONFIG SCHEMA — F06_REBUILD_TRAIN_ONLY_TEMPLATE.yaml

Cada campo existe para neutralizar un blocker confirmado por la auditoría
extrema. Todos son **fail-closed**: si el valor no es el institucionalmente
seguro, el pipeline ABORTA.

| Campo | Valor exigido | Por qué existe (blocker que previene) |
| :--- | :--- | :--- |
| `project` | `Trading BOT` | Identidad del proyecto en el manifest/trazabilidad. |
| `phase` | `F06_EVIDENCE_REBUILD` | Marca que esto es reconstrucción, no certificación. |
| `mode` | `TRAIN_ONLY` | Prohíbe cualquier evaluación de validation/holdout. |
| `symbol` | `EURUSD` | Foco único; evita scope creep a otros instrumentos. |
| `families` | `[F06]` | Una sola familia; evita multi-familia / selección post-hoc. |
| `session.timezone/start/end` | `America/New_York` `07:00`-`17:00` | Sesión institucional fija; evita ventanas ad-hoc. |
| `risk.max_trades_per_day` | `3` | Límite de riesgo duro del proyecto. |
| `data_scope.allow_2025` | `false` | Protege holdout 2025. Leakage = BLOCK. |
| `data_scope.allow_2026` | `false` | Protege holdout 2026. Leakage = BLOCK. |
| `data_scope.validation_enabled` | `false` | Previene columnas `*_val` en train-only (blocker confirmado). |
| `data_scope.holdout_enabled` | `false` | Holdout intocable. |
| `data_scope.exact_months` | 5 meses 2020-2024 | Scope explícito y reproducible; nada de 2025/2026. |
| `input_rules.forbid_quarantined_paths` | `true` | SoT no puede vivir/leerse en carpeta cuarentenada. |
| `input_rules.forbid_legacy_v50b_outputs` | `true` | Prohíbe reusar outputs viejos contaminados. |
| `input_rules.forbid_old_master_ranking` | `true` | `MASTER_RANKING.csv` viejo (degenerado) prohibido. |
| `input_rules.forbid_old_trades_csv` | `true` | `V50B_RERUN_TRADES.csv` (91.7% contaminado) prohibido. |
| `output_rules.output_dir_must_not_exist` | `true` | Evita sobrescribir / mezclar outputs (multi-writer). |
| `output_rules.single_run_id_only` | `true` | Un ledger = un run_id. Previene Multi-RunID. |
| `output_rules.no_validation_columns_in_train_only` | `true` | Bloquea atestación falsa train-only. |
| `output_rules.manifest_required` | `true` | Sin manifest completo no hay evidencia. |
| `output_rules.hashes_required` | `true` | Reproducibilidad: hashes de input/script/config/outputs. |
| `cost_model.require_real_spread_component` | `true` | Cost model anterior no modelaba spread. |
| `cost_model.require_slippage_component` | `true` | Slippage obligatorio. |
| `cost_model.require_round_turn_commission` | `true` | Comisión round-turn (no one-way) obligatoria. |
| `cost_model.components` | lista de los 3 | Lista explícita verificada por `check_cost_model_components`. |
| `sample_size.min_trades_per_family` | `100` | Prohíbe certificación con N=10. |
| `sample_size.min_trades_per_month_for_reporting` | `10` | Piso de granularidad mensual. |

## Notas

- Este archivo es **template**: define el contrato, no autoriza correr.
- `cost_model.components` es una extensión institucional (superset del
  template solicitado) para que el guard valide presencia explícita de los
  tres componentes; los `require_*` se mantienen como exige el estándar.
- Cualquier desviación de estos valores en un run real debe producir
  `BLOCKED_GUARD_FAILED`, nunca un warning silencioso.
