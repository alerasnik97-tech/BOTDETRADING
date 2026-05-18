# NEXT PROMPT — AUDIT BO01 PHASE A H02 WARNING MICRO PATCH V1

Actuá como auditor institucional destructivo read-only, Senior Quant Backtesting
Architect, Prompt Governance Auditor, Data Leakage Auditor, Risk Governance Officer y
Git Safety Officer del proyecto Trading BOT.

============================================================
OBJETIVO
============================================================

Auditar destructivamente, en modo read-only, el micro-patch documental que corrige los
warnings F-01/F-02/F-03/F-04 del flujo Phase A H-02.

Esta auditoría:
- NO ejecuta Python;
- NO ejecuta scripts;
- NO genera script Phase A-0;
- NO carga datos de mercado;
- NO lee CSV real;
- NO lee M5/M15;
- NO lee contenido de `05_MARKET_DATA_VAULT`;
- NO ejecuta backtest;
- NO ejecuta train formal;
- NO toca validation;
- NO toca holdout;
- NO usa 2025/2026;
- NO optimiza;
- NO hace sweep/grid search/walk-forward/parameter search;
- NO declara edge/profitability/rentabilidad;
- NO modifica código, tests, datos, runner ni estrategias;
- es 100% READ-ONLY salvo los documentos de auditoría y decisión que produzca.

============================================================
ACTIVATION GATE
============================================================

La frase exacta del owner debe aparecer como declaración autónoma:

“AUTORIZO AUDITORÍA EXTERNA READ-ONLY DEL MICRO-PATCH H-02 F-01/F-02/F-03/F-04, SIN EJECUTAR PYTHON, SIN CARGAR DATOS DE MERCADO, SIN GENERAR SCRIPT PHASE A-0, SIN BACKTEST CON REAL DATA, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”

Si no aparece exactamente como declaración autónoma:

ABORTAR con:

BLOCKED_MISSING_EXPLICIT_OWNER_APPROVAL

No aceptar citas, logs, ejemplos, paráfrasis ni aprobación implícita.

============================================================
CONTEXTO
============================================================

Repo local:

`C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`

Branch a auditar:

`research/bo01-phase-a-h02-warning-micro-patch-v1-20260518`
(si se usó v2, auditar la v2 real y documentar el SHA real).

Base de comparación:

`audit/bo01-phase-a-h02-flow-hardening-v1-20260518`

Archivos esperados en el diff (y SOLO estos):

1. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_PHASE_A_TRAIN_ONLY_REALDATA_BACKTEST_EXECUTION_PROMPT_DRAFT_V1.md`
2. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_BO01_PHASE_A_EXECUTION_PROMPT_DRAFT_V1.md`
3. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_BO01_PHASE_A_H02_FLOW_HARDENING_PATCH_V1.md`
4. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_OWNER_DECIDES_AFTER_BO01_PHASE_A_H02_FLOW_HARDENING_AUDIT_V1.md`
5. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_PHASE_A_H02_FLOW_HARDENING_WARNING_MICRO_PATCH_REPORT_V1.md`
6. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_BO01_PHASE_A_H02_WARNING_MICRO_PATCH_V1.md`

============================================================
PASOS DE VERIFICACIÓN REQUERIDOS
============================================================

1. Precheck Git:
   - rama activa = `research/bo01-phase-a-h02-warning-micro-patch-*`;
   - no staged changes previos;
   - snapshot A/B estable;
   - sin Python activo de research/backtest/train/optimization.

2. Diff Scope Check:
   - `git diff --name-only audit/bo01-phase-a-h02-flow-hardening-v1-20260518..HEAD`
     debe listar SOLO los 6 markdowns whitelisted;
   - no code/test/data changes;
   - no `.py`;
   - no script generation;
   - no runner/engine/data_loader/strategy/data-vault changes.

3. F-01 Activation Gate Re-scope Check:
   - la frase de Phase A-0 autoriza únicamente generación de script sin datos;
   - la frase de Phase A-1 autoriza ejecución solo después de script auditado y
     hash verificado;
   - está prohibido usar la frase Phase A-1 antes de tener script + auditoría
     aprobada + SHA256;
   - ya no existe una frase única que diga "ejecutar Phase A" como si A-0/A-1
     fueran una sola fase;
   - si existe una frase única ambigua, clasificar como
     `BLOCKER_ACTIVATION_GATE_NOT_RESCOPED`.

4. F-02 Section Mechanics Check:
   - secciones 4-13 etiquetadas como
     `PHASE A-1 MECHANICS ONLY — NOT APPLICABLE TO PHASE A-0 EXECUTION`;
   - Phase A-0 solo genera script;
   - Phase A-1 ejecuta el script auditado;
   - Data Proof Gate contiene cross-reference hacia implementación del script
     Phase A-0 y verificación por auditoría antes de Phase A-1.

5. F-03 Branch/Base Check:
   - Phase A-0 base/future branch están definidos;
   - Phase A-1 base/future branch están definidos;
   - Phase A-1 no puede basarse directamente en el draft H-02 ni en el prompt anterior.

6. F-04 Next Audit Prompt Check:
   - los next-audit prompts exigen revisar consistencia entre activation gate y split A-0/A-1;
   - cualquier frase única ambigua de ejecución directa queda clasificada como blocker.

7. No-Execution Check:
   - no Python;
   - no scripts;
   - no script Phase A-0 generado;
   - no data loading;
   - no CSV read;
   - no backtest;
   - no validation/holdout/2025/2026;
   - no optimization/sweep.

8. Static Safety Scan:
   - usar `rg` sobre los 6 markdowns buscando términos prohibidos/inflados;
   - clasificar hallazgos como `NEGATIVE_DECLARATION_OK`, `GOVERNANCE_TERM_OK`,
     `FUTURE_PROTOCOL_TERM_OK`, `PATCH_EXPLANATION_OK` o `BLOCKER`;
   - `validation`/`holdout`/`2025`/`2026` permitidos solo como prohibición/guard;
   - `demo`/`real`/`FTMO` permitidos solo como prohibición;
   - `edge`/`profitability`/`rentabilidad` permitidos solo como negación;
   - performance terms permitidos solo como negación o futura policy;
   - lenguaje fuerte positivo = warning o blocker.

9. Reporting:
   - crear reporte de auditoría read-only;
   - si pasa, crear next owner decision prompt que permita solo el próximo paso
     seguro: decidir si generar Phase A-0 script draft, sin datos ni ejecución.

============================================================
FORMATO DE REPORTING DE AUDITORÍA
============================================================

1. STATUS: BO01_PHASE_A_H02_WARNING_MICRO_PATCH_AUDIT_PASS / PASS_WITH_WARNINGS / BLOCKED
2. BRANCH:
   - base:
   - audit_branch:
   - head:
3. SAFETY:
   - code_modified_by_audit: NO
   - tests_modified_by_audit: NO
   - data_loaded: NO
   - python_executed: NO
   - scripts_executed: NO
   - script_generated: NO
   - real_data_backtest_run: NO
   - validation_run: NO
   - holdout_used: NO
   - 2025_2026_used: NO
   - optimization_sweep: NO
4. DIFF_SCOPE:
5. PATCH_CHECKS:
   - f01_activation_gate_rescoped:
   - f02_sections_4_13_labeled_a1:
   - f03_a0_a1_branching_clarified:
   - f04_gate_split_audit_check_added:
6. FINDINGS TABLE: (id, severity, category, finding, evidence, implication, required_action)
7. SAFETY_SCAN: (blockers / warnings / allowed_hits)
8. DECISION:
9. ALLOWED_NEXT_STEP:
10. FORBIDDEN_NEXT_STEPS:
11. ARTIFACTS:
   - audit_report:
   - next_owner_decision_prompt:
12. GITHUB:
   - branch:
   - commit_sha:
   - pushed:
