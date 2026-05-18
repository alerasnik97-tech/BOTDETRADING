# NEXT PROMPT — AUDIT OF BO01 PHASE A EXECUTION PROMPT DRAFT V1

Actuá como auditor institucional destructivo read-only, Senior Python Reviewer, Risk Governance Officer y Git Safety Officer del proyecto Trading BOT.

============================================================
OBJETIVO
============================================================

Auditar destructivamente el borrador (DRAFT) del prompt técnico de ejecución de Phase A del primer backtest BO01 train-only real-data.

Esta auditoría es 100% de DISEÑO DEL PROMPT.

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

El objetivo único es verificar si el borrador del prompt técnico de ejecución está suficientemente claro, riguroso, causal, libre de lookahead/leakage y seguro para permitir que el owner autorice la posterior ejecución de la fontanería de simulación (Phase A).

============================================================
ACTIVATION GATE
============================================================

La frase exacta del owner debe aparecer como declaración autónoma:

“AUTORIZO AUDITORÍA EXTERNA READ-ONLY DEL BORRADOR DEL PROMPT DE EJECUCIÓN PHASE A BO01 TRAIN-ONLY REAL-DATA, SIN EJECUTAR PYTHON, SIN CARGAR DATOS DE MERCADO, SIN BACKTEST CON REAL DATA, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”

Si no aparece exactamente como declaración autónoma:

ABORTAR con:

BLOCKED_MISSING_EXPLICIT_OWNER_APPROVAL

============================================================
CONTEXTO
============================================================

Repo local:

C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo

Branch a auditar:

research/bo01-phase-a-execution-prompt-design-v1-20260518

Commit a auditar:

4e8ddc61b2c2e3f446ef682554432ed9cd4cc741

Archivos autorizados para inspección:

1.
06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_PHASE_A_TRAIN_ONLY_REALDATA_BACKTEST_EXECUTION_PROMPT_DRAFT_V1.md

2.
06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_PHASE_A_EXECUTION_PROMPT_DESIGN_REPORT_V1.md

3.
06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_BO01_PHASE_A_EXECUTION_PROMPT_DRAFT_V1.md

NO tocar ningún archivo. La auditoría debe ser 100% READ-ONLY.

============================================================
PASOS DE VERIFICACIÓN REQUERIDOS
============================================================

1. **Precheck Git**:
   Verificar que la rama activa es `research/bo01-phase-a-execution-prompt-design-v1-20260518` y no hay staged changes ni drift.

2. **Diff Scope Check**:
   Verificar mediante `git diff` contra la base `audit/bo01-first-train-only-realdata-backtest-protocol-design-v1-20260518` que solo se modificaron/crearon los archivos whitelisted.

3. **Methodological Rigor Check**:
   - ¿El borrador limita la futura Phase A a la ventana acotada de 5 días de 2015-01-05 a 2015-01-09?
   - ¿El borrador exige comprobar programáticamente que no existan fechas de 2025 o 2026 en todo el índice temporal?
   - ¿El borrador exige generar y registrar el hash SHA256 de los CSV cargados?

4. **Runner & Execution Audit**:
   - ¿Se fija el motor a la versión auditada de `bo01_backtest_runner.py`?
   - ¿El borrador especifica las políticas `ENTRY_NEXT_CANDLE_OPEN` y `STOP_FIRST` sin ambigüedades?
   - ¿Se limita estrictamente a un máximo de 1 trade activo y 1 por día?

5. **Friction Cost Model Audit**:
   - ¿Se definen los tres perfiles de costo fijos (Base, Conservative, Stress) y se prohíbe elegir un ganador heurístico?

6. **Output & Safety Audit**:
   - ¿Los outputs de ejecución están restringidos a carpetas locales gitignored?
   - ¿Se exigen los 9 archivos locales obligatorios de control (logs de datos, comandos, contadores de diagnóstico, trades detailed y friction summaries)?
   - ¿Las abort conditions son completas e inmediatas?

7. **Static Safety Scan**:
   - Escanear los markdowns en busca de palabras infladas o prohibidas.

============================================================
FORMATO DE REPORTING DE AUDITORÍA
============================================================

El handoff final debe contener:

1. STATUS: BO01_PHASE_A_EXECUTION_PROMPT_AUDIT_PASS / BLOCKED
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
