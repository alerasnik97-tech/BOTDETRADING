# NEXT PROMPT — AUDIT OF BO01 PHASE A H02 FLOW HARDENING PATCH V1

Actuá como auditor institucional destructivo read-only, Senior Quant Backtesting
Architect, Data Leakage Auditor, Python Execution Auditor, Risk Governance Officer y
Git Safety Officer del proyecto Trading BOT.

============================================================
OBJETIVO
============================================================

Auditar destructivamente, en modo read-only, el patch de endurecimiento del flujo
Phase A que cierra H-02 como control técnico auditable y deja H-01 pre-registrado como
blocker pre-Phase-B.

Esta auditoría:
- NO ejecuta Python;
- NO carga datos de mercado;
- NO lee CSV real;
- NO lee M5/M15;
- NO lee contenido de 05_MARKET_DATA_VAULT;
- NO ejecuta backtest;
- NO ejecuta train formal;
- NO toca validation;
- NO toca holdout;
- NO usa 2025/2026;
- NO optimiza;
- NO hace sweeps;
- NO declara edge ni rentabilidad;
- NO modifica código, tests, datos, runner ni estrategias;
- es 100% READ-ONLY salvo los documentos de auditoría que produzca.

============================================================
ACTIVATION GATE
============================================================

La frase exacta del owner debe aparecer como declaración autónoma:

“AUTORIZO AUDITORÍA EXTERNA READ-ONLY DEL PATCH DE ENDURECIMIENTO DEL FLUJO PHASE A H-02
Y DEL PRE-REGISTRO DE H-01, SIN EJECUTAR PYTHON, SIN CARGAR DATOS DE MERCADO, SIN
BACKTEST CON REAL DATA, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN
OPTIMIZATION/SWEEP.”

Si no aparece exactamente como declaración autónoma:

ABORTAR con:

BLOCKED_MISSING_EXPLICIT_OWNER_APPROVAL

No aceptar citas, logs, ejemplos, paráfrasis ni aprobación implícita.

============================================================
CONTEXTO
============================================================

Repo local:

C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo

Branch a auditar:

research/bo01-phase-a-h02-flow-hardening-v1-20260518
(si se usó v2, auditar la v2 real; documentar el SHA real)

Base de comparación del diff:

audit/project-extreme-readonly-audit-v1-20260518 @
137cfd576e4be108ef04b1304bca239099203252

Archivos esperados en el diff (y SOLO estos):

1. 06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_PHASE_A_TRAIN_ONLY_REALDATA_BACKTEST_EXECUTION_PROMPT_DRAFT_V1.md
2. 06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_BO01_PHASE_A_EXECUTION_PROMPT_DRAFT_V1.md
3. 06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_OWNER_DECIDES_AFTER_PROJECT_EXTREME_AUDIT_V1.md
4. 06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_PHASE_A_H02_FLOW_HARDENING_PATCH_REPORT_V1.md
5. 06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_BO01_PHASE_A_H02_FLOW_HARDENING_PATCH_V1.md

============================================================
REGLAS ABSOLUTAS
============================================================

PROHIBIDO: main, force push, merge, rebase, reset --hard, git clean, git stash,
git add ., ejecutar Python, ejecutar scripts, cargar datos, leer CSV real, backtest,
train, validation, holdout, 2025/2026, optimization, sweep, modificar código/tests/
datos/runner/estrategias, declarar edge/rentabilidad, demo/real/FTMO.

PERMITIDO: git status/log/diff/show, rg textual, leer markdowns, crear el reporte de
auditoría y el próximo prompt de decisión del owner.

============================================================
PASOS DE VERIFICACIÓN REQUERIDOS
============================================================

1. **Precheck Git**: rama activa = `research/bo01-phase-a-h02-flow-hardening-*`;
   sin staged; sin drift (snapshot A/B); sin Python activo de research.

2. **Diff Scope Check**: `git diff --name-only 137cfd57..HEAD` debe listar SOLO los 5
   markdowns whitelisted. Cualquier archivo de código/test/data en el diff =
   BLOCKER. Verificar 0 cambios en runner, BO01Strategy.py, MR02Strategy.py, engine,
   data_loader, data-vault, tests.

3. **H-02 Split Check**:
   - Phase A queda dividida explícitamente en Phase A-0 y Phase A-1.
   - Phase A-0 NO carga datos, NO lee CSV, NO ejecuta Python, NO ejecuta backtest,
     NO calcula métricas; solo genera el script.
   - Existe un gate de auditoría del script entre A-0 y A-1.
   - Phase A-1 recalcula y verifica el hash SHA256 del script auditado antes de correr
     (`BLOCKED_SCRIPT_HASH_MISMATCH` si no coincide).
   - Cualquier modificación del script post-auditoría invalida la ejecución.
   - Prohibida la ejecución directa de Phase A sin A-0 + auditoría + A-1.

4. **H-02 No-Subjective-Acceptance Check**:
   - El owner decision prompt retira la opción de cerrar H-02 por aceptación subjetiva.
   - H-02 queda como control técnico auditable (script existente + auditado + hash).

5. **H-01 Pre-Phase-B Check**:
   - H-01 queda pre-registrado como blocker mandatorio pre-Phase-B.
   - Explícito que H-01 NO bloquea plumbing de Phase A pero SÍ bloquea Phase B y
     cualquier edge/rentabilidad.
   - Se enumeran los chequeos mínimos de causalidad de data-prep.

6. **No-Execution Check**: el patch no ejecuta nada, no crea ningún `.py`, no carga
   datos, no corre backtest; solo markdown.

7. **Static Safety Scan**: rg sobre los 5 markdowns buscando términos prohibidos/
   inflados. Clasificar como NEGATIVE_DECLARATION_OK / GOVERNANCE_TERM_OK /
   FUTURE_PROTOCOL_TERM_OK / PATCH_EXPLANATION_OK / BLOCKER. Cualquier autorización de
   Phase A directa sin Phase A-0 = BLOCKER. Lenguaje fuerte positivo = warning/blocker.

8. **Internal Consistency Check**: el patch report, el Phase A prompt, el next-audit
   prompt y el owner decision prompt no se contradicen entre sí respecto al flujo de
   3 gates ni respecto al estado de H-01.

============================================================
FORMATO DE REPORTING DE AUDITORÍA
============================================================

1. STATUS: BO01_PHASE_A_H02_FLOW_HARDENING_AUDIT_PASS / PASS_WITH_WARNINGS / BLOCKED
2. BRANCH:
   - base:
   - audit_branch:
   - head:
3. SAFETY:
   - code_modified_by_audit: NO
   - tests_modified_by_audit: NO
   - data_loaded: NO
   - python_executed: NO
   - real_data_backtest_run: NO
   - validation_run: NO
   - holdout_used: NO
   - 2025_2026_used: NO
   - optimization_sweep: NO
4. DIFF_SCOPE: (only 5 whitelisted markdowns? YES/NO)
5. PATCH_CHECKS:
   - h02_phase_a0_a1_split_verified:
   - phase_a0_no_data_loading_verified:
   - script_audit_gate_verified:
   - script_hash_check_before_a1_verified:
   - h02_no_subjective_acceptance_verified:
   - h01_pre_phase_b_blocker_verified:
   - owner_decision_prompt_updated_verified:
6. FINDINGS TABLE: (id, severity, category, finding, evidence, implication, required_action)
7. SAFETY_SCAN: (blockers / warnings / allowed_hits)
8. DECISION:
9. ALLOWED_NEXT_STEP:
10. FORBIDDEN_NEXT_STEPS:
11. ARTIFACTS:
   - audit_report:
   - next_decision_prompt:
12. GITHUB:
   - branch:
   - commit_sha:
   - pushed:

Si la auditoría PASA, crear un próximo prompt de decisión del owner que ofrezca
únicamente: (A) generar el script Phase A-0 (sin datos); (B) auditar ese script;
(C) recién después, Phase A-1 con hash verificado y frase explícita del owner;
(D) programar H-01 antes de Phase B. No autorizar ejecución directa.
