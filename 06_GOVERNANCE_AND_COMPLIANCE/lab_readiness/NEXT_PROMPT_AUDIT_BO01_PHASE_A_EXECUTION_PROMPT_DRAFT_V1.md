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
   - ¿El borrador vincula explícitamente el runner audit commit `5bdb4bed1f829eb7e8bfe65dc30a6e2f49657d89`?
   - ¿El borrador especifica las políticas `ENTRY_NEXT_CANDLE_OPEN` y `STOP_FIRST` sin ambigüedades?
   - ¿Se limita estrictamente a un máximo de 1 trade activo y 1 por día?

5. **Friction Cost Model Audit**:
   - ¿Se definen los tres perfiles de costo fijos (Base, Conservative, Stress) y se prohíbe elegir un ganador heurístico?

6. **Output & Safety Audit**:
   - ¿Los outputs de ejecución están restringidos a carpetas locales gitignored?
   - ¿Se exigen los 9 archivos locales obligatorios de control (logs de datos, comandos, contadores de diagnóstico, trades detailed y friction summaries)?
   - ¿Se aclara que `temporary_execution_script.py` es opcional (solo obligatorio si se usa script temporal)?
   - ¿Se reemplaza el ambiguo `train_run: YES` por `formal_train_run: NO` y `train_only_backtest_run: YES` en el handoff?
   - ¿Las abort conditions son completas e inmediatas?

7. **Static Safety Scan**:
   - Escanear los markdowns en busca de palabras infladas o prohibidas.

8. **H-02 Flow Hardening Check (obligatorio)**:
   - ¿El prompt deja Phase A dividida explícitamente en Phase A-0 y Phase A-1?
   - ¿Phase A-0 NO carga datos, NO lee CSV real, NO ejecuta Python, NO ejecuta backtest?
   - ¿Phase A-0 solo genera el script (`PHASE_A_EXECUTION_SCRIPT_DRAFT.py`) sin ejecutarlo?
   - ¿El script debe auditarse (auditoría dedicada read-only) antes de ejecutarse?
   - ¿Phase A-1 recalcula y verifica el hash SHA256 del script auditado antes de correr,
     abortando con `BLOCKED_SCRIPT_HASH_MISMATCH` si no coincide?
   - ¿Se prohíbe explícitamente la ejecución directa de Phase A desde el prompt anterior
     sin pasar por Phase A-0 + auditoría del script + Phase A-1?
   - ¿Se prohíbe cerrar H-02 por simple owner acceptance, exigiendo control técnico
     auditable (script existente + auditado + hash verificado)?
   - ¿Cualquier modificación del script después de la auditoría invalida la ejecución
     y obliga a re-auditar?

8-BIS. **Activation Gate vs Flow Split Check (obligatorio)**:
   - ¿La frase de Phase A-0 autoriza únicamente generación de script sin datos?
   - ¿La frase de Phase A-1 autoriza ejecución solo después de script auditado y hash verificado?
   - ¿Está prohibido usar la frase Phase A-1 antes de tener script + auditoría aprobada + SHA256?
   - ¿Ya no existe una frase única que diga “ejecutar Phase A” como si A-0/A-1 fueran una sola fase?
   - Si existe una frase única ambigua, clasificar como `BLOCKER_ACTIVATION_GATE_NOT_RESCOPED`.

9. **H-01 Pre-Phase-B Registration Check (obligatorio)**:
   - ¿H-01 (`ema_m15_200` / `atr14` causal data-prep) queda formalmente pre-registrado
     como blocker mandatorio pre-Phase-B?
   - ¿Queda explícito que H-01 NO bloquea la fontanería de Phase A pero SÍ bloquea
     Phase B y cualquier interpretación de edge/rentabilidad?
   - ¿Se enumeran los chequeos de causalidad mínimos (resample, merge_asof, forward-fill,
     closed/label, barras M15 cerradas, sin `shift(-1)`, sin rolling centrado, sin uso
     de High/Low/Close futuros)?
   - ¿El owner decision prompt fue actualizado para retirar la opción de aceptar H-02
     por criterio subjetivo?

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
