# NEXT PROMPT — AUDIT BO01 FIRST TRAIN-ONLY BACKTEST FRAMEWORK DESIGN V1

Actuá como auditor institucional destructivo read-only, Senior Python Reviewer, Quant Infrastructure Auditor, Risk Governance Officer y Git Safety Officer del proyecto Trading BOT.

============================================================
OBJETIVO
============================================================

Auditar destructivamente el diseño del primer framework de backtest train-only para BO01.

El objetivo NO es ejecutar backtest.
El objetivo NO es cargar datos reales.
El objetivo NO es probar edge.
El objetivo NO es probar rentabilidad.

El objetivo es verificar si el diseño del framework:

1. Fue creado solo dentro del scope de documentación autorizado.
2. No contiene código ejecutable de backtest.
3. No carga datos ni ejecuta Python.
4. Diseña correctamente el modelo de costos estáticos (Base, Conservative, Stress).
5. Diseña correctamente el modelo de ejecución (vela a vela, sin lookahead, max 1 trade/día, primer señal, stop-first same-bar).
6. Diseña correctamente el modelo de riesgo (riesgo fijo 1R, sin compounding, sin Martingale).
7. Establece correctamente las abort conditions.
8. Establece la política de outputs locales gitignored y gobernanza commiteable.
9. No accede a validation, holdout ni años 2025/2026.
10. No busca parámetros ni optimiza.
11. No declara edge ni rentabilidad.

============================================================
ACTIVATION GATE
============================================================

La frase exacta del owner debe aparecer como declaración autónoma:

“AUTORIZO AUDITORÍA EXTERNA READ-ONLY DEL DISEÑO DEL FRAMEWORK DE BACKTEST BO01, SIN EJECUTAR PYTHON, SIN CARGAR DATOS DE MERCADO, SIN BACKTEST, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP.”

Si no aparece exactamente como declaración autónoma:

ABORTAR con:

BLOCKED_MISSING_EXPLICIT_OWNER_APPROVAL

============================================================
CONTEXTO A AUDITAR
============================================================

Branch de diseño:

research/bo01-first-train-only-backtest-framework-design-v1-20260518

Archivos a auditar en `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/`:

1.
BO01_FIRST_TRAIN_ONLY_BACKTEST_FRAMEWORK_DESIGN_V1.md

2.
BO01_FIRST_TRAIN_ONLY_BACKTEST_FRAMEWORK_DESIGN_REPORT_V1.md

============================================================
REGLAS ABSOLUTAS DE AUDITORÍA
============================================================

PROHIBIDO:
- NO ejecutar Python.
- NO ejecutar backtest.
- NO cargar datos de mercado.
- NO modificar código, tests, datos ni runner.
- NO git add .
- NO force push, reset --hard, rebase, clean o stash.
- NO declarar edge.
- NO declarar rentabilidad.

PERMITIDO:
- git status/diff/log/show/ls-files/check-ignore.
- rg textual / Select-String.
- Leer markdowns de diseño.
- Crear reporte de auditoría markdown.
- Crear prompt futuro de ejecución o bloqueo.
- Commit/push de reportes de auditoría autorizados.
