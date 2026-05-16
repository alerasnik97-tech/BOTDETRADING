# PARALLEL RESEARCH LAB FAILURE TRIAGE REPORT

## 1. Status
**TRAIN_ONLY_LAB_BLOCKED_BY_ENGINE_BASE_FAILURE**

## 2. Executive Summary
La auditoría quirúrgica de la suite `research_lab` (177 tests) revela que, si bien el laboratorio está cerca de la apertura, existen **fallos críticos en el motor base (OHLCV)** y en las **guardas de seguridad (Preflight)** que bloquean la autorización institucional. Los fallos se concentran en la precisión del tiempo de entrada/salida (offset de una barra) y en la validación de contratos de riesgo. Los módulos de noticias y alta precisión (Level 2/3) también presentan fallos, pero estos son **postergables** ya que no afectan el núcleo de la fase EURUSD Train-only.

## 3. Test Inventory
- **Total Tests:** 177
- **Passed:** 140
- **Failures:** 16
- **Errors:** 9
- **Skipped:** 12

## 4. Failure Classification Table

| test_file | test_name | failure_type | blocks_eurusd_train_only | fix_required_before_lab | recommended_fix |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `test_lab_preflight` | `test_assert_train_data_no_holdout_fails_leaky_data` | REAL_ENGINE_REGRESSION | **YES** | YES | Corregir lógica de detección de fechas >= 2025 en `lab_preflight.py`. |
| `test_engine` | `test_entry_time_uses_next_bar_open_time` | REAL_ENGINE_REGRESSION | **YES** | YES | Corregir offset de tiempo en el bucle principal de `run_backtest`. |
| `test_engine` | `test_enforce_hard_stop_rejects_signal_without_stop` | REAL_ENGINE_REGRESSION | **YES** | YES | Implementar validación estricta en `validate_signal_risk_contract`. |
| `test_engine` | `test_boundary_fill_on_19_label...` | REAL_ENGINE_REGRESSION | **YES** | YES | Ajustar `entry_open_index` para manejar límites de sesión. |
| `test_engine` | `test_final_close_applies_exit_costs` | REAL_ENGINE_REGRESSION | **YES** | YES | Asegurar que la última barra del dataset genere un registro de trade. |
| `test_engine` | `test_optional_max_hold_bars_exits...` | REAL_ENGINE_REGRESSION | **YES** | YES | Corregir contador de barras en la lógica de salida por tiempo. |
| `test_engine_stop_entry` | `test_invalid_stop_entry_direction...` | REAL_ENGINE_REGRESSION | **YES** | YES | Validar dirección vs precio en órdenes stop de entrada. |
| `test_level2_execution` | (Todos) | BLOCKS_HIGH_PRECISION_ONLY | NO | NO | Postergado hasta fase de alta precisión. |
| `test_level3_precision` | (Todos) | BLOCKS_HIGH_PRECISION_ONLY | NO | NO | Postergado hasta fase de alta precisión. |
| `test_am_news_builder` | (Todos) | BLOCKS_NEWS_ONLY | NO | NO | Postergado hasta fase de noticias. |

## 5. Engine Risk Analysis
El riesgo es **MEDIO-ALTO** para el núcleo OHLCV. 
- Los tests indican un **desplazamiento de una barra (15 min)** en las entradas y salidas registradas. Esto invalidaría cualquier métrica de performance real.
- La **guarda de seguridad Preflight** falló al no detectar datos de 2025 simulados como "leaky". Esto es un **BLOQUEADOR CRÍTICO** de seguridad.

## 6. News / Legacy Analysis
- Las noticias están correctamente **fail-closed**.
- Los fallos en `AMNewsBuilder` se deben a discrepancias en el contrato de datos antiguos y no afectan la carga de OHLCV puro.
- Se clasifican como `DEFERRABLE_NEWS_LEGACY`.

## 7. Must Fix Before Lab (Grupo A)
1. **Surgical Fix Preflight:** Asegurar que `assert_train_data_no_holdout` sea infalible.
2. **Surgical Fix Engine Timing:** Sincronizar el índice de entrada del backtest para que coincida con el "Open of next bar" real (T+1).
3. **Surgical Fix Engine Validation:** Activar los `ValueError` requeridos en `validate_signal_risk_contract`.

## 8. Deferred Modules
- **High Precision (L2/L3):** Postergado.
- **News Logic:** Postergado.
- **USDJPY:** Postergado.

## 9. Conclusion
La remediación de los 7 fallos críticos del Grupo A permitirá la apertura del laboratorio EURUSD Train-only bajo un marco de seguridad institucional.
