# NEXT PROMPT — AUDIT OF BO01 BACKTEST RUNNER WARNING PATCH

Actuá como auditor institucional destructivo read-only, Senior Python Reviewer, Quant Infrastructure Auditor, Data Policy Auditor y Git Safety Officer del proyecto Trading BOT.

============================================================
OBJETIVO
============================================================

Auditar destructivamente el patch menor aplicado al BO01 Backtest Runner.

El patch reporta:

- W-01 corregido: `validate_backtest_frame` ahora realiza un escaneo explícito de todo el índice temporal (`frame.index.year.isin([2025, 2026]).any()`).
- Test de fecha interna maliciosa 2025/2026 agregado.
- W-02 corregido: try-except en `run_bo01_backtest_on_frame` ahora captura tanto `ValueError` como `TypeError` para signals malformados non-dict, loggeándolos como `invalid_signal_count`.
- Test de señal malformada non-dict agregado.
- W-03 corregido: el contador `skipped_signals_active_position` ahora cuenta con precisión cada iteración omitida debido a la posición abierta.
- Test de incremento de barras omitidas agregado.
- W-04 corregido: `compute_cost_r` documentado y estructurado con constantes FX explícitas y test del cálculo de R de comisión agregado.
- Tests sintéticos: 25 passed, 0 failed.
- No datos reales.
- No backtest real.
- No train.
- No validation.
- No holdout.
- No 2025/2026 salvo tests negativos sintéticos.

============================================================
ACTIVATION GATE
============================================================

La frase exacta del owner debe aparecer como declaración autónoma:

“AUTORIZO AUDITORÍA EXTERNA READ-ONLY DEL PATCH DE WARNINGS DEL RUNNER BO01 SINTÉTICO, SIN CARGAR DATOS DE MERCADO, SIN BACKTEST CON REAL DATA, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”

Si no aparece exactamente como declaración autónoma:

ABORTAR con:

BLOCKED_MISSING_EXPLICIT_OWNER_APPROVAL

============================================================
CONTEXTO
============================================================

Repo local:

C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo

Branch a auditar:

research/bo01-backtest-runner-warning-patch-v1-20260518

Commit a auditar:

[Insertar commit SHA del patch]

Archivos autorizados para inspección:

1.
03_RESEARCH_LAB/research_lab/runners/bo01_backtest_runner.py

2.
03_RESEARCH_LAB/research_lab/tests/test_bo01_backtest_runner_contract.py

3.
03_RESEARCH_LAB/research_lab/tests/test_bo01_backtest_runner_execution.py

4.
03_RESEARCH_LAB/research_lab/tests/test_bo01_backtest_runner_safety.py

5.
06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_BACKTEST_RUNNER_WARNING_PATCH_REPORT_V1.md

NO tocar ningún archivo. La auditoría debe ser 100% READ-ONLY.

============================================================
REGLAS ABSOLUTAS DE AUDITORÍA
============================================================

PROHIBIDO:

- NO modificar código.
- NO escribir archivos de código.
- NO commitear cambios que alteren la lógica.
- NO cargar datos reales.
- NO ejecutar backtest con datos reales.
- NO optimizar.
- NO hacer sweeps.
- NO usar 2025/2026.
- NO acceder a validation/holdout.
- NO git add .

PERMITIDO:

- Ejecutar tests sintéticos de la suite (25 tests).
- Ejecutar análisis estático de código.
- Crear reporte de auditoría markdown.
- Crear prompt de decisión futuro para el owner.
- Confirmar estado de Git y commits.

============================================================
PASOS DE VERIFICACIÓN REQUERIDOS
============================================================

1. **Precheck Git**:
   Verificar que la rama activa es `research/bo01-backtest-runner-warning-patch-v1-20260518` y no hay staged changes ni worktree drift.

2. **Diff Scope Check**:
   Verificar mediante `git diff` contra la base `audit/bo01-backtest-runner-synthetic-v1-20260518` que solo se modificaron/crearon los archivos whitelisted.

3. **Date Guard Verification (W-01)**:
   Inspeccionar `validate_backtest_frame` y confirmar que escanea todo el índice temporal para 2025/2026 y que el test correspondiente pasa.

4. **Exception Handling Verification (W-02)**:
   Confirmar que `run_bo01_backtest_on_frame` captura `TypeError` y que el test correspondiente de señal non-dict pasa.

5. **Counter Verification (W-03)**:
   Confirmar que `skipped_signals_active_position` aumenta y que el test correspondiente pasa.

6. **Commission Assumptions (W-04)**:
   Confirmar las constantes FX y la documentación del cálculo de R de comisión.

7. **Test Suite Execution**:
   Ejecutar los 25 tests sintéticos y confirmar que todos pasan (25 passed, 0 failed).

8. **Static Safety Scan**:
   Escanear el código en busca de palabras prohibidas y clasificar los hits.

9. **Git Output Check**:
   Confirmar que no se escribieron archivos de datos, CSV locales o ZIPs prohibidos.

============================================================
FORMATO DE REPORTING DE AUDITORÍA
============================================================

El handoff final debe contener:

1. STATUS: BO01_BACKTEST_RUNNER_WARNING_PATCH_AUDIT_PASS / BLOCKED
2. BRANCH:
   - base:
   - audit_branch:
   - head:
3. SAFETY:
   - code_modified_by_audit: NO
   - tests_modified_by_audit: NO
   - data_loaded: NO
   - real_data_backtest_run: NO
   - synthetic_tests_run: YES
4. FINDINGS TABLE: (Tabla con ID, Gravedad, Categoría, Hallazgo e Implicancia)
5. DECISION:
6. ALLOWED_NEXT_STEP:
7. FORBIDDEN_NEXT_STEPS:
8. ARTIFACTS:
   - audit_report:
   - next_decision_prompt:
9. GITHUB:
   - branch:
   - commit_sha:
   - pushed:
