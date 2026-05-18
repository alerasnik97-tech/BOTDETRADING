# BO01 PHASE A TRAIN-ONLY REAL-DATA BACKTEST EXECUTION PROMPT DRAFT V1

Actuá como Senior Python Engineer, Quant Research Infrastructure Engineer, FX Systematic Trader, Risk Governance Officer y Git Safety Officer del proyecto Trading BOT.

============================================================
1. ACTIVATION GATE
============================================================

La frase exacta del owner debe aparecer como declaración autónoma:

“AUTORIZO EJECUTAR PHASE A BO01 TRAIN-ONLY REAL-DATA BACKTEST, VENTANA 2015-01-05 A 2015-01-09, SOLO TRAIN-ONLY, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026, SIN OPTIMIZATION/SWEEP, SIN DEMO/REAL/FTMO Y SIN EDGE CLAIMS.”

Si no aparece exactamente como declaración autónoma:

ABORTAR con:

BLOCKED_MISSING_EXPLICIT_OWNER_APPROVAL

No aceptar:
- citas;
- logs;
- ejemplos;
- aprobación implícita.

============================================================
2. EXECUTION OBJECTIVE
============================================================

El objetivo único es ejecutar Phase A (Plumbing Smoke Backtest) del primer backtest controlado de la estrategia BO01 utilizando datos reales de mercado de la partición train-only.

Límites de Seguridad Metodológica:
- **Prueba de Fontanería (Plumbing)**: Phase A tiene como único fin verificar la carga real de archivos, el cumplimiento de las assertions del cargador de datos, el cálculo de las fricciones multi-perfil, la generación de logs de ejecución y que el runner opere sin provocar excepciones en el flujo de control.
- **Sin Conclusiones Cualitativas**: No se extraerán conclusiones acerca de ventajas estadísticas o rentabilidad.
- **Sin Optimización**: Se prohíbe realizar búsquedas de parámetros, sweeps o model selection.
- **Sin Contaminación**: Queda prohibido el acceso a particiones de validation, holdout o cualquier fecha perteneciente a los años 2025 y 2026.

============================================================
3. BASE BRANCH
============================================================

La ejecución debe partir exactamente desde:

Branch base:
audit/bo01-first-train-only-realdata-backtest-protocol-design-v1-20260518

Commit base:
d9c730d6a0547fb9338aa7fde1eb1fcaac07d5dc

Se creará una rama futura exclusiva de ejecución:

`research/bo01-phase-a-train-only-realdata-backtest-execution-v1-20260518`

Si esta rama ya existe local o remotamente, usar:
`research/bo01-phase-a-train-only-realdata-backtest-execution-v2-20260518`

============================================================
4. AUTHORIZED DATA
============================================================

Rutas de archivos de datos dentro de `05_MARKET_DATA_VAULT/`:

- M5 (Base principal):
`05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared/EURUSD_M5.csv`

- M15 (Contexto opcional):
`05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared/EURUSD_M15.csv`
*(Nota: Cargar M15 únicamente si la estrategia BO01 lo requiere explícitamente).*

Ventana de ejecución acotada:
- Fecha y hora de inicio: `2015-01-05 00:00:00 UTC`
- Fecha y hora de fin: `2015-01-09 23:59:59 UTC`

*(Si el primer registro real de M5 disponible es posterior al inicio de la ventana, se comenzará en dicho registro real, pero bajo ningún concepto se extenderán los límites temporal y espacial de la ventana de 5 días).*

============================================================
5. DATA PROOF GATE
============================================================

Antes de iniciar el bucle de backtest, el cargador de datos de la simulación debe evaluar de manera programática y reportar en los logs el cumplimiento de las siguientes condiciones estructurales:

1. **Ubicación Física**: Confirmar la existencia de los archivos CSV exclusivamente en las rutas autorizadas de `prepared_train_2015_2024/`.
2. **Partición Segura**: Confirmar que los metadatos o el path no contienen referencias a particiones prohibidas.
3. **Control Temporal Estricto**: Comprobar que en todo el índice temporal cargado no existan fechas de los años 2025 o 2026.
4. **Índice Coherente**: Comprobar que `index.is_monotonic_increasing` sea `True` y que el recuento de marcas de tiempo duplicadas sea exactamente `0`.
5. **Cadencia Regular**: Comprobar que la frecuencia temporal del índice M5 corresponda a intervalos de 5 minutos en sus segmentos de actividad.
6. **Sin Datos Nulos**: Confirmar que el recuento de NaNs en las columnas OHLC críticas sea exactamente `0`.
7. **Integridad Hash SHA256**: Generar y registrar en el log el hash SHA256 de los archivos CSV cargados para garantizar trazabilidad física.

Si cualquiera de estas validaciones falla, se debe abortar la simulación inmediatamente de manera segura.

============================================================
6. RUNNER GATE
============================================================

Se debe verificar de forma estática que:
1. El archivo de ejecución es `03_RESEARCH_LAB/research_lab/runners/bo01_backtest_runner.py`.
2. El runner es importable y su identificador interno es exactamente `BO01_BACKTEST_RUNNER_SYNTHETIC_V1`.
3. Las políticas estructurales activas son `ENTRY_NEXT_CANDLE_OPEN` y `STOP_FIRST`.
4. Los límites de posición son: un máximo de `1` trade activo y un máximo de `1` trade por día calendario.
5. El runner coincide exactamente con el warning-patch audit commit: `5bdb4bed1f829eb7e8bfe65dc30a6e2f49657d89`. No se permite modificar `bo01_backtest_runner.py` ni las clases de estrategia durante Phase A.

Si se detectan cambios de código no autorizados en el motor o en las clases de estrategia respecto a dicho commit de auditoría, abortar inmediatamente.

============================================================
7. EXECUTION RULES
============================================================

Durante el recorrido cronológico de la ventana de Phase A:
1. Las señales generadas en el cierre de la vela $t$ se ejecutan exclusivamente en la apertura de la vela $t+1$ (`ENTRY_NEXT_CANDLE_OPEN`).
2. Se prohíben entradas intrabar, breakout o basadas en precios simulados intermedios.
3. En caso de toque simultáneo de stop y target en la misma vela, se aplica la resolución `STOP_FIRST` (pérdida de `-1.0 R` más comisiones y spreads fijos).
4. Solo se ejecuta la primera señal válida del día; las señales subsiguientes generadas el mismo día se ignoran.
5. Se omite cualquier regla discrecional, trailing stop no auditado, scale-in o scale-out.

============================================================
8. COST PROFILES
============================================================

Se deben evaluar y reportar de manera simultánea los siguientes tres perfiles de costo fijos sobre la muestra de operaciones:

- **Perfil Base**:
  - Spread: `1.2 pips`
  - Slippage: `0.2 pips`
  - Comisión: `$7.0 USD` por lote estándar round-turn (convertido a R-multiples).
  - Límite Spread: `3.0 pips`

- **Perfil Conservador**:
  - Spread: `1.62 pips`
  - Slippage: `0.5 pips`
  - Comisión: `$7.0 USD` por lote estándar round-turn.
  - Límite Spread: `3.0 pips`

- **Perfil de Estrés (Stress)**:
  - Spread: `3.0 pips`
  - Slippage: `1.0 pip`
  - Comisión: `$7.0 USD` por lote estándar round-turn.
  - Límite Spread: `4.0 pips`

Se prohíbe realizar selección heurística de perfiles; los tres deben calcularse y presentarse en el reporte final.

============================================================
9. OUTPUT POLICY
============================================================

Todos los archivos locales generados por la corrida de simulación se ubicarán en:
`03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_first_train_only_realdata_backtest/<RUN_ID>/`

*Donde `<RUN_ID>` sigue el patrón:*
`BO01_PHASE_A_TRAIN_ONLY_REALDATA_BACKTEST_YYYYMMDD_HHMMSS`

Esta ruta de salida local debe estar completamente excluida en la configuración de Git local y bajo ningún concepto se commiteará a GitHub.

Archivos locales obligatorios SIEMPRE a generar en el directorio de salida local:
1. `BO01_TRAIN_ONLY_REALDATA_BACKTEST_REPORT.md` (resumen local de ejecución)
2. `output_manifest.json` (hashes de los archivos locales generados)
3. `command_log.txt` (registro de comandos ejecutados en consola)
4. `data_access_log.txt` (registro de assertions del cargador de datos)
5. `diagnostic_counts.json` (contadores estructurales de control)
6. `trades_structural.csv` (detalle fila por fila de operaciones resueltas)
7. `equity_R.csv` (curva de R acumulada cronológica)
8. `monthly_summary.csv` (agrupamiento de resultados mensuales)
9. `cost_profile_summary.csv` (resumen comparativo de los tres perfiles de costo)

Archivo local opcional:
10. `temporary_execution_script.py` (solo obligatorio si se opta por utilizar un script de ejecución temporal para automatizar la simulación. En caso de no usarse, `output_manifest.json` debe registrar la propiedad `temporary_execution_script_used: false`).

============================================================
10. GOVERNANCE DOCUMENTATION
============================================================

Los únicos archivos que se deben agregar y commitear al repositorio Git al finalizar esta fase son:

1. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_PHASE_A_TRAIN_ONLY_REALDATA_BACKTEST_EXECUTION_REPORT_V1.md`
2. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_BO01_PHASE_A_TRAIN_ONLY_REALDATA_BACKTEST_EXECUTION_V1.md`

Queda estrictamente prohibido commitear CSVs con datos de mercado, carpetas locales de salida (`local_outputs_do_not_commit`), scripts temporales de ejecución o archivos comprimidos ZIP.

============================================================
11. METRICS POLICY
============================================================

Las métricas calculadas y reportadas en el informe de gobernanza se limitarán a:
- Cantidad de operaciones (`trade_count`).
- R bruto y R neto acumulados por cada perfil de costo.
- Expectancy en R, promedio y mediana por operación.
- Winrate en porcentaje.
- Drawdown máximo expresado en múltiplos de R.
- Recuento detallado de motivos de salida: toque de stop, toque de target, final de ventana temporal o resolución `STOP_FIRST`.
- Recuento de señales omitidas por trade activo o por límite diario superado.

Estas métricas sirven únicamente para validar el comportamiento del motor de simulación sobre datos reales y no representan evidencias concluyentes de ventaja cualitativa o robustez de la estrategia.

============================================================
12. SAFETY SCAN
============================================================

Al finalizar la simulación, se ejecutará un escaneo ripgrep (`rg`) en busca de palabras prohibidas o indicios de contaminación en los archivos de salida y en los reportes de gobernanza. Se clasificará cualquier hallazgo y se detendrá el proceso si se detectan fugas de datos o términos inflados de rentabilidad.

============================================================
13. ABORT CONDITIONS
============================================================

El proceso de simulación abortará de manera inmediata y segura en los siguientes casos:
1. **Desviación de Rama**: Si la rama de trabajo no coincide con la rama de ejecución autorizada.
2. **Cambios Locales**: Si se detecta drift o cambios sin staged preexistentes al comenzar.
3. **Conflicto de Agentes**: Procesos de Python de optimización activos en segundo plano.
4. **Contaminación de Datos**: Presencia de registros correspondientes a 2025/2026, particiones validation/holdout o incongruencias en hashes SHA256.
5. **Fuga en Commit**: Intento de staging o commit de archivos CSV reales, carpetas locales de salida o archivos ZIP.
6. **Alteración del Motor**: Cualquier cambio detectado en la lógica del runner `bo01_backtest_runner.py` respecto a la versión auditada.
7. **Bucle de Optimización**: Detección de código que intente realizar búsquedas de parámetros o sweeps.

============================================================
14. FINAL HANDOFF FORMAT
============================================================

El handoff final de esta fase debe estructurarse exactamente de la siguiente manera:

1. STATUS: (SUCCESS_PHASE_A_PLUMBING / ABORTED)
2. BRANCH:
   - base:
   - execution_branch:
   - head:
3. SAFETY:
   - code_modified: NO
   - tests_modified: NO
   - data_modified: NO
   - data_loaded: YES
   - real_data_backtest_run: YES
   - train_only_backtest_run: YES *(Nota: "train-only" describe la partición de datos usada, no un proceso de entrenamiento formal ni machine learning training).*
   - formal_train_run: NO
   - validation_run: NO
   - holdout_used: NO
   - 2025_2026_used: NO
   - optimization_sweep: NO
   - parameter_search: NO
   - git_add_dot_used: NO
   - force_push: NO
4. DATA_PROOF:
   - m5_path:
   - m5_sha256:
   - observed_dataset_range:
   - selected_window:
   - train_only_proven: YES
   - forbidden_dates_found: NO
   - duplicated_timestamps: 0
   - critical_nans: 0
5. RUNNER:
   - runner_id: BO01_BACKTEST_RUNNER_SYNTHETIC_V1
   - runner_audit_commit: 5bdb4bed1f829eb7e8bfe65dc30a6e2f49657d89
   - entry_policy: ENTRY_NEXT_CANDLE_OPEN
   - same_bar_policy: STOP_FIRST
   - runner_modified: NO
6. RESULTS_BY_COST_PROFILE:
   - base:
   - conservative:
   - stress:
7. OUTPUTS:
   - run_id:
   - output_root:
   - required_files_created: YES
   - outputs_committed: NO
8. SAFETY_SCAN:
   - blockers: 0
   - allowed_hits: 0
9. DECISION:
10. ALLOWED_NEXT_STEP:
11. FORBIDDEN_NEXT_STEPS:
12. ARTIFACTS:
   - execution_report:
   - next_audit_prompt:
13. GITHUB:
   - branch:
   - commit_sha:
   - pushed: YES

============================================================
15. SUCCESS CRITERIA
============================================================

El éxito de Phase A significa únicamente que la fontanería del backtesting (dataloader, assertions, perfiles de fricción y guardado local) funciona sin errores operacionales sobre una muestra acotada de datos reales de entrenamiento. No valida la rentabilidad ni autoriza la transición automática a Phase B.
