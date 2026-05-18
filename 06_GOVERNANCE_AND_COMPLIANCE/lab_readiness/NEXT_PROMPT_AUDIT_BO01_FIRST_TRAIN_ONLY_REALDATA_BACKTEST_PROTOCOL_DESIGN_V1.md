# NEXT PROMPT — AUDIT OF BO01 FIRST TRAIN-ONLY REAL-DATA BACKTEST PROTOCOL DESIGN V1

Actuá como auditor institucional destructivo read-only, Senior Quant Reviewer, Risk Governance Officer y Git Safety Officer del proyecto Trading BOT.

============================================================
OBJETIVO
============================================================

Auditar destructivamente el diseño del primer protocolo de ejecución de backtest BO01 con datos reales train-only.

Esta auditoría es 100% de DISEÑO METODOLÓGICO.

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

El objetivo único es verificar si el protocolo está suficientemente claro, riguroso, causal, libre de lookahead/leakage y seguro para pasar a una futura fase de ejecución controlada del backtest (Phase A).

============================================================
ACTIVATION GATE
============================================================

La frase exacta del owner debe aparecer como declaración autónoma:

“AUTORIZO AUDITORÍA EXTERNA READ-ONLY DEL DISEÑO DEL PROTOCOLO DE BACKTEST BO01 TRAIN-ONLY REAL-DATA, SIN EJECUTAR PYTHON, SIN CARGAR DATOS DE MERCADO, SIN BACKTEST CON REAL DATA, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”

Si no aparece exactamente como declaración autónoma:

ABORTAR con:

BLOCKED_MISSING_EXPLICIT_OWNER_APPROVAL

============================================================
CONTEXTO
============================================================

Repo local:

C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo

Branch a auditar:

research/bo01-first-train-only-realdata-backtest-protocol-design-v1-20260518

Commit a auditar:

[Insertar commit SHA del diseño]

Archivos autorizados para inspección:

1.
06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_FIRST_TRAIN_ONLY_REALDATA_BACKTEST_PROTOCOL_DESIGN_V1.md

2.
06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_FIRST_TRAIN_ONLY_REALDATA_BACKTEST_PROTOCOL_DESIGN_REPORT_V1.md

NO tocar ningún archivo. La auditoría debe ser 100% READ-ONLY.

============================================================
PASOS DE VERIFICACIÓN REQUERIDOS
============================================================

1. **Precheck Git**:
   Verificar que la rama activa es `research/bo01-first-train-only-realdata-backtest-protocol-design-v1-20260518` y no hay staged changes ni drift.

2. **Diff Scope Check**:
   Verificar mediante `git diff` contra la base `audit/bo01-backtest-runner-warning-patch-v1-20260518` que solo se modificaron/crearon los archivos whitelisted.

3. **Methodological Rigor Check**:
   - ¿El protocolo restringe la futura ejecución a la estrategia BO01 en EURUSD M5?
   - ¿Excluye adecuadamente a MR02 debido a baja frecuencia?
   - ¿Los paths futuros de datos están estrictamente dentro de `prepared_train_2015_2024`?
   - ¿El protocolo prohíbe explícitamente cargar datos de 2025, 2026, validation o holdout?

4. **Data Proof Audit**:
   - ¿El diseño exige verificaciones programáticas de monotonicidad, temporalidad (UTC), ausencia de NaNs y hashes SHA256?

5. **Execution Model Audit**:
   - ¿El protocolo fija la política de entrada únicamente a `ENTRY_NEXT_CANDLE_OPEN`?
   - ¿Se aplica la política `STOP_FIRST` ante toques simultáneos de stop y target en la misma vela?
   - ¿Se limita estrictamente a un máximo de 1 trade activo y 1 por día?

6. **Friction Cost Model Audit**:
   - ¿Se definen los tres perfiles de costo fijos (Base, Conservative, Stress) y se prohíbe elegir un ganador heurístico?

7. **Risk & Metrics Audit**:
   - ¿Los resultados futuros se calculan estrictamente en R sin sizing monetario?
   - ¿La métrica `skipped_signals_active_position` representa velas omitidas?

8. **Output & Safety Audit**:
   - ¿Los outputs de ejecución están restringidos a carpetas gitignored?
   - ¿Las abort conditions son completas e inmediatas?

9. **Static Safety Scan**:
   - Escanear los markdowns en busca de palabras infladas o prohibidas.

============================================================
FORMATO DE REPORTING DE AUDITORÍA
============================================================

El handoff final debe contener:

1. STATUS: BO01_REALDATA_PROTOCOL_DESIGN_AUDIT_PASS / BLOCKED
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
