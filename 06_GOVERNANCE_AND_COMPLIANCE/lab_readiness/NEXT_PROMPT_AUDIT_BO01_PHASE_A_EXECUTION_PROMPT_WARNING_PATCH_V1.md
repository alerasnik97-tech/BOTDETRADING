# NEXT PROMPT — AUDIT OF BO01 PHASE A EXECUTION PROMPT WARNING PATCH V1

Actuá como auditor institucional destructivo read-only, Senior Python Reviewer, Risk Governance Officer y Git Safety Officer del proyecto Trading BOT.

============================================================
OBJETIVO
============================================================

Auditar destructivamente el parche de warnings aplicado al borrador del prompt técnico de ejecución de Phase A BO01 train-only real-data.

Esta auditoría es 100% de DISEÑO DEL PROMPT PARCHEADO.

NO ejecutar Python.
NO ejecutar backtest.
NO cargar datos de mercado.
NO leer archivos CSV reales.
NO usar 2025/2026.
NO tocar validation.
NO tocar holdout.
NO optimizar.
NO hacer sweeps.
NO declarar rentabilidad ni edge.

El objetivo único es verificar si los tres warnings (W-01 runner commit, W-02 temporary script, W-03 train_run fields) han sido corregidos de manera segura y sin provocar efectos colaterales en la lógica del prompt de ejecución.

============================================================
ACTIVATION GATE
============================================================

La frase exacta del owner debe aparecer como declaración autónoma:

“AUTORIZO AUDITORÍA EXTERNA READ-ONLY DEL PARCHE DE WARNINGS DEL PROMPT DE EJECUCIÓN PHASE A, SIN EJECUTAR PYTHON, SIN CARGAR DATOS DE MERCADO, SIN BACKTEST CON REAL DATA, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”

Si no aparece exactamente como declaración autónoma:

ABORTAR con:

BLOCKED_MISSING_EXPLICIT_OWNER_APPROVAL

============================================================
CONTEXTO
============================================================

Repo local:

C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo

Branch a auditar:

research/bo01-phase-a-execution-prompt-warning-patch-v1-20260518

Commit a auditar:

[Insertar commit SHA del patch]

Archivos autorizados para inspección:

1.
06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_PHASE_A_TRAIN_ONLY_REALDATA_BACKTEST_EXECUTION_PROMPT_DRAFT_V1.md

2.
06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_PHASE_A_EXECUTION_PROMPT_DESIGN_REPORT_V1.md

3.
06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_BO01_PHASE_A_EXECUTION_PROMPT_DRAFT_V1.md

4.
06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_PHASE_A_EXECUTION_PROMPT_WARNING_PATCH_REPORT_V1.md

5.
06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_BO01_PHASE_A_EXECUTION_PROMPT_WARNING_PATCH_V1.md

NO tocar ningún archivo. La auditoría debe ser 100% READ-ONLY.

============================================================
PASOS DE VERIFICACIÓN REQUERIDOS
============================================================

1. **Precheck Git**:
   Verificar que la rama activa es `research/bo01-phase-a-execution-prompt-warning-patch-v1-20260518` y no hay staged changes ni drift.

2. **Diff Scope Check**:
   Verificar mediante `git diff` contra la base `audit/bo01-phase-a-execution-prompt-draft-v1-20260518` que solo se modificaron/crearon los archivos whitelisted (los 5 markdowns).

3. **W-01 Verification**:
   - Comprobar que en Runner Gate y en el handoff se declara explícitamente el commit auditado del runner: `5bdb4bed1f829eb7e8bfe65dc30a6e2f49657d89`.

4. **W-02 Verification**:
   - Comprobar que en la output policy se separan los 9 archivos locales obligatorios del script opcional `temporary_execution_script.py`.

5. **W-03 Verification**:
   - Comprobar que en la sección SAFETY de la handoff se reemplaza `train_run: YES` por `formal_train_run: NO` y `train_only_backtest_run: YES`.

6. **Static Safety Scan**:
   - Escanear los markdowns en busca de palabras infladas o prohibidas.

============================================================
FORMATO DE REPORTING DE AUDITORÍA
============================================================

El handoff final debe contener:

1. STATUS: BO01_PHASE_A_EXECUTION_PROMPT_WARNING_PATCH_AUDIT_PASS / BLOCKED
2. BRANCH:
   - base:
   - audit_branch:
   - head:
3. SAFETY:
   - code_modified_by_audit: NO
   - tests_modified_by_audit: NO
   - data_loaded: NO
   - real_data_backtest_run: NO
4. FINDINGS TABLE: (id, severity, category, finding, evidence, implication, required_action)
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
