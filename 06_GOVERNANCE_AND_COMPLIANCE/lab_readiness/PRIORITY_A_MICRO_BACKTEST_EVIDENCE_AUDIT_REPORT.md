# PRIORITY A MICRO BACKTEST EVIDENCE AUDIT REPORT

## 1. Status
**MICRO_EVIDENCE_APPROVED_WITH_METADATA_WARNING**

## 2. Executive Summary
Este reporte de auditoría examina la evidencia física generada durante la primera micro-corrida train-only de las 4 estrategias Priority A (`mr01_anchor_elastic`, `mr02_vwap_stretch_reversion`, `tp01_london_ny_momentum_pullback` y `ve_orb_volatility_expansion`).
La corrida fue ejecutada bajo el ID de ejecución `EURUSD_PRIORITY_A_MICRO_TRAIN_ONLY_20260516_205000`.
El análisis concluye que las estrategias son técnicamente estables y seguras contra data leakage, pero emite un warning formal de metadatos debido a una inconsistencia de registro de ventana de datos (`data_range`) entre la ejecución real y las declaraciones contenidas en el manifiesto global y los snapshots de configuración.
La corrida técnica es declarada como **Aprobada con Advertencia**, permitiendo la transición al backtest formal 2015–2024 con directrices de reconciliación estrictas.

## 3. Commit Surface Audit
Se auditó el commit de micro-corrida `d26774a933ce5fbd136fdb5bbeac37d2c27acdc0` y el commit de documentación previo `6281385d227d79e2fe8a65dc066559687f056558`.
Se confirma que los commits contienen **únicamente** los siguientes archivos ligeros de reporte y control:
* `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/PRIORITY_A_MICRO_TRAIN_ONLY_BACKTEST_REPORT.md`
* `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_PRIORITY_A_FORMAL_TRAIN_ONLY_BACKTEST_PLAN.md`
* `RUN_MANIFEST.json`
* `configs/*_CONFIG_SNAPSHOT.json`
* `strategy_reports/*/summary.json`

Se verifica la **ausencia absoluta** de código fuente modificado del motor (`engine.py`), archivos pesados de ticks/OHLCV, CSVs de trades (`trades.csv` o `equity_curve.csv`), archivos Parquet, y archivos ZIP no autorizados.

## 4. Manifest Audit
El archivo `manifests/RUN_MANIFEST.json` fue inspeccionado:
- `run_id`: `EURUSD_PRIORITY_A_MICRO_TRAIN_ONLY_20260516_205000` (Correcto)
- `authorized_scope`: `EURUSD_PRIORITY_A_MICRO_TRAIN_ONLY` (Correcto)
- `data_path`: `["05_MARKET_DATA_VAULT\\eurusd_data\\prepared_train_2015_2024\\prepared"]` (Correcto)
- `holdout_used`: `false` (Correcto, blindaje garantizado)
- `optimization_run`: `false` (Correcto)
- `strategies_run`: Solo contiene las 4 Priority A autorizadas. No se detectan ghost strategies (`VE-01`, `SD-01`, `ED-01`).
- **Inconsistencia Identificada**: El manifiesto declara un campo `"data_range": ["2015-01-01", "2024-12-31"]` en su metadata, que representa el rango objetivo de entrenamiento, pero difiere del sub-periodo de ejecución real de alta fidelidad M1 (2019-2020).

## 5. Config Snapshot Audit
Se leyeron los archivos JSON en `configs/`:
- Parámetros de estrategias cargados desde `DEFAULT_PARAMS` sin optimizaciones de rejilla.
- News filter desactivado (`NewsConfig.enabled = false`), lo cual cumple con el protocolo fail-closed establecido.
- Alta precisión desactivada para evitar fugas metodológicas.
- `max_trades_per_day = 3` configurado en el motor de backtest para limitar la concentración intradía.
- **Inconsistencia Identificada**: Al igual que el manifiesto, los snapshots registran `"data_range": ["2015-01-01", "2024-12-31"]` en su metadata, representando el target de entrenamiento teórico, en contraste con el sub-periodo real ejecutado.

## 6. Strategy Summary Audit
Se leyeron los archivos `summary.json` correspondientes a cada estrategia:
- `news_filter_used`: `false` para todos.
- `insufficient_sample`: `true` para `mr01`, `mr02` y `ve_orb` (1 trade c/u).
- `selected_score`: `0.0` para todas.
- Ninguna métrica de performance fue utilizada para clasificar o seleccionar campeonas durante esta corrida.

## 7. Data Window Consistency
El análisis forense cruzado entre los archivos de salida, logs de la consola y la configuración del runner scratch revela lo siguiente:
- **Ejecución Real**: Las estrategias se corrieron sobre resolución nativa **M1** en el rango **2019-01-01 a 2020-12-31** (2 años de sub-periodo de entrenamiento, acumulando millones de filas en memoria y garantizando un pre-stress técnico realista).
- **Metadata Documental**: El runner grabó por defecto el rango de entrenamiento teórico `"data_range": ["2015-01-01", "2024-12-31"]` en los JSONs.
- **Veredicto**: Esta discrepancia es clasificada como **inconsistencia de metadatos benigna** (`MICRO_METADATA_WINDOW_WARNING`). No compromete la integridad del holdout ni representa leakage, ya que el periodo real ejecutado (2019-2020) está estrictamente contenido en el dataset de entrenamiento y excluye los años prohibidos (2025/2026). La advertencia queda registrada y se exige su reconciliación en la fase formal.

## 8. Leakage / Safety Audit
Un escaneo exhaustivo ripgrep en el directorio de salida arrojó los siguientes resultados:
- **Fuga de Holdout (2025/2026)**: Cero coincidencias. No hay acceso a los datos sellados.
- **Prohibited Dependencies**: Cero uso de noticias activas, buffers de Forex Factory, o precision level 2.
- **ZIP workflow**: Cero archivos ZIP nuevos en el workspace. Se respeta la regla de no contaminación de raíz.

## 9. Output Policy
Se confirma que los archivos CSV pesados generados por la corrida local (`trades.csv` y `equity_curve.csv` de ~6MB a ~8MB cada uno) han sido **excluidos estrictamente de Git** a través de las reglas actualizadas en `.gitignore`. Esto evita la degradación de rendimiento en la sincronización del repositorio.

## 10. Strategy Interpretability Notes
- **MR-01 (`mr01_anchor_elastic`)**: `execution-status: PASS`, `evidence_status: INSUFFICIENT_SAMPLE`. Se valida que la lógica carga e inicializa sin desbordamientos de arreglos.
- **MR-02 (`mr02_vwap_stretch_reversion`)**: `execution-status: PASS`, `evidence_status: INSUFFICIENT_SAMPLE`. Se confirma que los checks quirúrgicos de finitud previenen problemas matemáticos en VWAP.
- **TP-01 (`tp01_london_ny_momentum_pullback`)**: `execution-status: PASS`, `evidence_status: MICRO_SAMPLE_TECHNICALLY_USEFUL`. Muestra robusta de 139 operaciones procesadas correctamente. El profit factor resultante (1.20) no debe usarse para tomar decisiones de producción.
- **VE-ORB (`ve_orb_volatility_expansion`)**: `execution-status: PASS`, `evidence_status: INSUFFICIENT_SAMPLE`. El Profit Factor registrado de `Infinity` es declarado formalmente como **no interpretable** (`NON_INTERPRETABLE_SINGLE_TRADE`) al provenir de un único trade.

## 11. Decision
**APROBADO CON ADVERTENCIA DE METADATOS**. Se autoriza el avance a la fase de **Backtest Formal Train-Only (2015-2024)**. Los skeletons Priority A son seguros y robustos. La advertencia sobre la ventana de datos obliga a que el runner formal mapee con precisión absoluta los metadatos correspondientes al rango completo de 10 años.

## 12. Safety Verification
- micro_backtest_run: YES
- formal_backtest_run: NO
- optimization_run: NO
- sweep_run: NO
- validation_run: NO
- holdout_used: NO
- 2025_2026_used: NO
- news_used: NO
- high_precision_used: NO
- engine_modified_by_micro: NO
- data_modified: NO
- force_push: NO
- git_add_dot_used: NO

## 13. Copy-Paste Summary for ChatGPT
- Status: MICRO_EVIDENCE_APPROVED_WITH_METADATA_WARNING
- Cause: Mismatch between 2019-2020 M1 actual run vs 2015-2024 target range recorded in JSONs.
- Safety: No holdout leakage, no ZIPs, no core mutations.
- Strategy Edge: TP-01 shows 139 trades OK; MR-01, MR-02, VE-ORB verified for contract only.
- Recommendation: Proceed to formal single-strategy backtests 2015-2024.
